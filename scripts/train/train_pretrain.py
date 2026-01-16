#!/usr/bin/env python3
"""
Pretrain 数据训练（Pretraining）

功能：
- 基于 Qwen3-4B 预训练模型，只训练 Fusion 组件
- 训练组件：GatedMemoryFusion 和 memory_norm
- 冻结组件：MemoryGate（包括 keys）、Backbone、MemoryBank（通过 MemoryBankUpdater 更新）
- 使用预训练格式数据训练（纯文本，非对话格式）
- 支持分布式训练和混合精度
- 支持 Memory Bank 动态更新（通过 --enable_memory_update 启用）
"""

import os

# 设置进程名称（在 nvidia-smi 中显示）
try:
    import setproctitle
    process_name = os.environ.get('PYTHON_PROCESS_NAME', 'llama-env')
    setproctitle.setproctitle(process_name)
except ImportError:
    # 如果没有 setproctitle，尝data试使用 prctl (Linux only)
    try:
        import prctl
        process_name = os.environ.get('PYTHON_PROCESS_NAME', 'llama-env')
        prctl.set_name(process_name.encode('utf-8'))
    except (ImportError, AttributeError):
        # 如果都不可用，跳过（不影响训练）
        pass
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
import swanlab
from pathlib import Path

from utils.model_initializer import init_model, load_pretrained_memory_gate
from utils.pretrain_datasets import create_pretrain_dataloader, create_validation_dataloader
from utils.logger import Logger
from utils.memory_update_tracker import MemoryUpdateTracker
from config.memory_update import MemoryUpdateConf


def freeze_parameters_for_fusion_training(model, accelerator, train_memory_gate=False):
    """
    冻结参数：根据是否提供预训练权重决定训练策略
    
    如果 train_memory_gate=True（未提供预训练权重）：
    - 训练：gated_memory_fusion, memory_norm, memory_gate（不包括 keys）
    - 冻结：backbone, keys, memory_bank (通过 MemoryBankUpdater 更新)
    
    如果 train_memory_gate=False（提供了预训练权重）：
    - 训练：gated_memory_fusion, memory_norm
    - 冻结：backbone, memory_gate (包括 keys), memory_bank (通过 MemoryBankUpdater 更新)
    """
    if train_memory_gate:
        Logger("设置参数冻结策略（Pretrain训练：训练 Fusion + MemoryGate，冻结 Keys）", accelerator)
    else:
        Logger("设置参数冻结策略（Pretrain训练：只训练 Fusion，冻结 Keys 和 MemoryGate）", accelerator)
    
    frozen_params = 0
    trainable_params = 0
    memory_bank_params = 0
    keys_params = 0
    memory_gate_params = 0
    fusion_params = 0
    
    for name, param in model.named_parameters():
        is_keys = "keys" in name and "memory_gate" in name
        is_memory_bank = "memory_bank" in name
        is_memory_gate = "memory_gate" in name and not is_keys
        is_fusion_component = any(keyword in name for keyword in [
            "gated_memory_fusion",
            "memory_norm",
        ])
        
        if is_memory_bank:
            # Bank 不通过梯度更新，而是通过 MemoryBankUpdater 进行非梯度更新
            param.requires_grad = False
            memory_bank_params += param.numel()
            frozen_params += param.numel()
        elif is_keys:
            # Keys 完全冻结
            param.requires_grad = False
            keys_params += param.numel()
            frozen_params += param.numel()
        elif is_memory_gate:
            # MemoryGate：根据是否提供预训练权重决定是否训练
            if train_memory_gate:
                # 未提供预训练权重，需要训练 MemoryGate（但 keys 仍然冻结）
                param.requires_grad = True
                memory_gate_params += param.numel()
                trainable_params += param.numel()
            else:
                # 提供了预训练权重，冻结 MemoryGate
                param.requires_grad = False
                memory_gate_params += param.numel()
                frozen_params += param.numel()
        elif is_fusion_component:
            # 只训练 Fusion 组件
            param.requires_grad = True
            fusion_params += param.numel()
            trainable_params += param.numel()
        else:
            # 其他参数（backbone）完全冻结
            param.requires_grad = False
            frozen_params += param.numel()
    
    Logger(f"参数冻结完成: 冻结 {frozen_params / 1e6:.3f}M, 可训练 {trainable_params / 1e6:.3f}M", accelerator)
    Logger(f"  - Memory bank: {memory_bank_params / 1e6:.3f}M (通过 MemoryBankUpdater 更新)", accelerator)
    Logger(f"  - Keys: {keys_params / 1e6:.3f}M (冻结)", accelerator)
    if train_memory_gate:
        Logger(f"  - MemoryGate: {memory_gate_params / 1e6:.3f}M (可训练)", accelerator)
    else:
        Logger(f"  - MemoryGate: {memory_gate_params / 1e6:.3f}M (冻结)", accelerator)
    Logger(f"  - Fusion: {fusion_params / 1e6:.3f}M (可训练)", accelerator)
    
    return frozen_params, trainable_params


def save_trainable_components(model, save_path, accelerator, train_memory_gate=False):
    """
    保存需要训练的组件权重和重要状态（Fusion + MemoryGate + MemoryBank状态）

    Args:
        model: 模型实例
        save_path: 保存路径
        accelerator: Accelerator实例
        train_memory_gate: 是否训练了MemoryGate
    """
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 收集需要保存的参数和重要的buffer
    trainable_state_dict = {}

    # 保存可训练参数
    for name, param in model.named_parameters():
        is_keys = "keys" in name and "memory_gate" in name
        is_memory_bank = "memory_bank" in name
        is_memory_gate = "memory_gate" in name and not is_keys
        is_fusion_component = any(keyword in name for keyword in [
            "gated_memory_fusion",
            "memory_norm",
        ])

        # 保存Fusion组件（始终需要）
        if is_fusion_component:
            trainable_state_dict[name] = param.cpu().detach()

        # 保存MemoryGate（如果训练了的话）
        elif is_memory_gate and train_memory_gate:
            trainable_state_dict[name] = param.cpu().detach()

    # 保存重要的buffer（persistent=True的buffer）
    # 这些buffer包含训练过程中更新的状态
    for name, buffer in model.named_buffers():
        # 保存memory_bank和valid_mask，这些在训练过程中通过MemoryBankUpdater更新
        if name in ['memory_bank', 'valid_mask']:
            trainable_state_dict[name] = buffer.cpu().detach()
            print(f"  📦 保存重要buffer: {name} {buffer.shape}")

    # 保存到文件
    save_file = os.path.join(save_path, "trainable_components.pth")

    try:
        torch.save({
            'state_dict': trainable_state_dict,
            'train_memory_gate': train_memory_gate,
            'saved_at_step': getattr(model, '_global_step', 0),  # 如果有的话
        }, save_file)
        Logger(f"保存训练组件权重: {save_file} ({len(trainable_state_dict)} 个参数)", accelerator)
        return save_file
    except Exception as e:
        Logger(f"保存训练组件权重失败: {e}", accelerator)
        return None


def load_trainable_components(model, checkpoint_path, accelerator):
    """
    加载训练组件权重

    Args:
        model: 模型实例
        checkpoint_path: checkpoint目录路径
        accelerator: Accelerator实例

    Returns:
        bool: 是否成功加载
    """
    save_file = os.path.join(checkpoint_path, "trainable_components.pth")

    if not os.path.exists(save_file):
        Logger(f"警告: 找不到训练组件权重文件: {save_file}", accelerator)
        return False

    try:
        checkpoint = torch.load(save_file, map_location='cpu')

        # 加载权重
        state_dict = checkpoint['state_dict']
        loaded_params = 0
        loaded_buffers = 0

        # 加载可训练参数
        for name, param in model.named_parameters():
            if name in state_dict:
                param.data.copy_(state_dict[name].to(param.device))
                loaded_params += 1

        # 加载重要buffer
        for name, buffer in model.named_buffers():
            if name in state_dict:
                buffer.data.copy_(state_dict[name].to(buffer.device))
                loaded_buffers += 1

        Logger(f"成功加载训练组件权重: {loaded_params} 个参数, {loaded_buffers} 个buffer", accelerator)
        return True

    except Exception as e:
        Logger(f"加载训练组件权重失败: {e}", accelerator)
        return False


def main():
    parser = argparse.ArgumentParser(description="阶段2：训练知识融合组件")
    
    # 模型配置
    parser.add_argument("--qwen3_model_path", type=str, required=True, help="Qwen3 模型路径")
    parser.add_argument("--pretrained_memory_gate_path", type=str, default="", help="预训练 MemoryGate 权重路径（可选，如果为空则跳过加载）")
    parser.add_argument("--knowledge_num", type=int, default=100*100, help="记忆库条目数")
    parser.add_argument("--knowledge_length", type=int, default=16, help="每个记忆条目的 token 数")
    # 注意：--knowledge_dim 已移除，实际使用 Qwen3 的 hidden_size (2560)
    parser.add_argument("--num_candidates", type=int, default=8, help="候选记忆数")
    parser.add_argument("--num_selected", type=int, default=1, help="选中的记忆数")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0, help="Gumbel-Softmax 温度")
    
    # 数据配置
    parser.add_argument("--dataset_path", type=str, required=True, help="训练数据路径（JSONL格式）")
    parser.add_argument("--val_dataset_path", type=str, default=None, help="验证数据路径（可选，如果设置了 val_split_ratio 或 val_split_size 则忽略）")
    parser.add_argument("--val_split_ratio", type=float, default=0.0, help="从训练数据中分割验证集的比例（0.0-1.0），例如 0.1 表示 10%%")
    parser.add_argument("--val_split_size", type=int, default=None, help="从训练数据中分割验证集的样本数量（如果设置，则忽略 val_split_ratio）")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--cache_path", type=str, default=None, help="记忆库 cache 路径（.pt 文件），如果提供则从 cache 加载记忆库")
    parser.add_argument("--keys_path", type=str, default=None, help="Keys 文件路径（.pt 文件，可选），如果提供则从文件加载 keys 进行初始化")
    
    # 训练配置
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup 步数")
    
    # Loss 配置
    parser.add_argument("--similarity_loss_coef", type=float, default=1.0, help="Similarity Loss 基础系数（用于自适应平衡，默认1.0）")
    
    # Memory Bank 更新配置（默认值从 config/memory_update.py 读取，可通过命令行覆盖）
    parser.add_argument("--enable_memory_update", action="store_true", 
                        help=f"启用 Memory Bank 更新（如果不指定，默认使用 config/memory_update.py 中的 enable_memory_update_during_training={MemoryUpdateConf.get('enable_memory_update_during_training', False)}）")
    parser.add_argument("--memory_update_frequency", type=int, 
                        default=MemoryUpdateConf.get("memory_update_frequency", 100), 
                        help="Memory Bank 更新频率（每N步更新一次，默认从 config/memory_update.py 读取）")
    parser.add_argument("--memory_update_strategy", type=str, 
                        default=MemoryUpdateConf.get("memory_update_strategy", "lru"), 
                        help="Memory Bank 更新策略（fifo/lru/random/similarity/importance，默认从 config/memory_update.py 读取）")
    parser.add_argument("--memory_compression_rate", type=float, 
                        default=MemoryUpdateConf.get("memory_compression_rate", 0.4), 
                        help="事实提取压缩率（默认从 config/memory_update.py 读取）")
    parser.add_argument("--llmlingua_model_path", type=str, 
                        default=MemoryUpdateConf.get("llmlingua_model_path", "llmlingua-2-bert"), 
                        help="LLMLingua 模型路径（默认从 config/memory_update.py 读取）")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="checkpoints/fusion", help="输出目录")
    parser.add_argument("--save_interval", type=int, default=500, help="保存间隔（步数）")
    parser.add_argument("--swanlab_project", type=str, default="explicitlm-fusion", help="SwanLab 项目名")
    parser.add_argument("--swanlab_online", action="store_true", help="使用 SwanLab 在线模式")
    
    args = parser.parse_args()
    
    # 初始化 Accelerator
    accelerator = Accelerator()
    
    # 设置随机种子（确保数据加载顺序一致，与 SFT 一致）
    set_seed(42 + accelerator.process_index)
    
    # 初始化 SwanLab
    if accelerator.is_main_process:
        swanlab.init(
            project=args.swanlab_project,
            config=vars(args),
            mode="cloud" if args.swanlab_online else "offline"
        )
    
    Logger("=" * 60, accelerator)
    Logger("阶段2：训练知识融合组件", accelerator)
    Logger("=" * 60, accelerator)
    
    # 初始化模型
    model_args = {
        "qwen3_model_path": args.qwen3_model_path,
        "knowledge_num": args.knowledge_num,
        "knowledge_length": args.knowledge_length,
        "num_candidates": args.num_candidates,
        "num_selected": args.num_selected,
        "gumbel_temperature": args.gumbel_temperature,
        "use_moe": False,
        "dropout": 0.0,
        "cache_path": args.cache_path,  # 使用提供的 cache 路径
        "recompute_cache": False,
    }
    
    # 如果提供了 keys_path，添加到 model_args
    if args.keys_path:
        model_args["keys_path"] = args.keys_path
        Logger(f"将使用 Keys 文件进行初始化: {args.keys_path}", accelerator)
    
    model, tokenizer = init_model(model_args, accelerator)
    Logger("模型初始化完成", accelerator)
    

    has_pretrained_memory_gate = False
    if args.pretrained_memory_gate_path and os.path.exists(args.pretrained_memory_gate_path):
        try:
            load_pretrained_memory_gate(model, args.pretrained_memory_gate_path, accelerator)
            Logger("✓ MemoryGate 权重加载完成", accelerator)
            has_pretrained_memory_gate = True
        except (FileNotFoundError, Exception) as e:
            Logger(f"警告: 加载 MemoryGate 权重失败: {e}，将跳过加载并训练 MemoryGate", accelerator)
            has_pretrained_memory_gate = False
    else:
        if args.pretrained_memory_gate_path:
            Logger(f"警告: MemoryGate 权重路径不存在: {args.pretrained_memory_gate_path}，将训练 MemoryGate", accelerator)
        else:
            Logger("未提供 MemoryGate 权重路径，将训练 MemoryGate", accelerator)
        has_pretrained_memory_gate = False
    
    # 设置参数冻结策略
    # 如果未提供预训练权重，则同时训练 MemoryGate 和 Fusion
    # 如果提供了预训练权重，则只训练 Fusion
    freeze_parameters_for_fusion_training(model, accelerator, train_memory_gate=not has_pretrained_memory_gate)
    
    # 准备数据
    # 如果指定了验证集分割，从训练数据中分割
    if args.val_split_ratio > 0.0 or args.val_split_size is not None:
        Logger(f"从训练数据中分割验证集: ratio={args.val_split_ratio}, size={args.val_split_size}", accelerator)
        train_loader, val_loader = create_pretrain_dataloader(
            data_path=args.dataset_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            shuffle=True,
            num_workers=0,  # 分布式训练中设置为 0 避免死锁
            val_split_ratio=args.val_split_ratio,
            val_split_size=args.val_split_size,
        )
        Logger(f"数据集分割完成: 训练集={len(train_loader.dataset)} 样本, 验证集={len(val_loader.dataset)} 样本", accelerator)
    else:
        # 不分割，使用独立的验证数据（如果提供）
        train_loader = create_pretrain_dataloader(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        shuffle=True,
        num_workers=0,  # 分布式训练中设置为 0 避免死锁
    )
    
    val_loader = None
    if args.val_dataset_path and os.path.exists(args.val_dataset_path):
        val_loader = create_validation_dataloader(
            data_path=args.val_dataset_path,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=0,  # 分布式训练中设置为 0 避免死锁
        )
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    num_training_steps = args.epochs * len(train_loader) // args.accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # 准备
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)
    
    # 损失函数
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    # 创建输出目录
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 尝试加载最新的有效checkpoint（用于恢复训练）
    resume_step = 0
    if accelerator.is_main_process:
        # 只查找包含有效checkpoint文件的目录
        checkpoint_dirs = []
        for d in os.listdir(args.output_dir):
            if d.startswith("checkpoint_step_") and os.path.isdir(os.path.join(args.output_dir, d)):
                checkpoint_path = os.path.join(args.output_dir, d)
                pth_file = os.path.join(checkpoint_path, "trainable_components.pth")
                if os.path.exists(pth_file) and os.path.getsize(pth_file) > 0:
                    checkpoint_dirs.append(d)

        if checkpoint_dirs:
            # 找到最新的有效checkpoint
            checkpoint_dirs.sort(key=lambda x: int(x.split("_")[-1]))
            latest_checkpoint = checkpoint_dirs[-1]
            resume_step = int(latest_checkpoint.split("_")[-1])

            Logger(f"发现有效checkpoint: {latest_checkpoint}, 尝试恢复训练", accelerator)

            # 加载checkpoint
            checkpoint_path = os.path.join(args.output_dir, latest_checkpoint)
            if load_trainable_components(model, checkpoint_path, accelerator):
                Logger(f"成功从step {resume_step}恢复训练", accelerator)
            else:
                Logger(f"checkpoint加载失败，从头开始训练", accelerator)
                resume_step = 0
        else:
            Logger("未发现有效checkpoint，从头开始训练", accelerator)
    
    # 初始化 MemoryBankUpdater（如果启用）
    memory_bank_updater = None
    memory_update_tracker = None
    
    # 如果命令行没有指定 --enable_memory_update，则使用配置文件的值
    enable_memory_update = args.enable_memory_update if args.enable_memory_update else MemoryUpdateConf.get("enable_memory_update_during_training", False)
    
    # 🔴 标记是否启用 Memory Update（所有进程都需要知道）
    memory_update_enabled = enable_memory_update
    
    if enable_memory_update:
        from utils.memory_bank_updater import MemoryBankUpdater
        from utils.fact_extractor import FactExtractor
        
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            fact_extractor = FactExtractor(
                model_path=args.llmlingua_model_path,
                compression_rate=args.memory_compression_rate
            )
            
            memory_bank_updater = MemoryBankUpdater(
                model=unwrapped_model,
                tokenizer=tokenizer,
                fact_extractor=fact_extractor,
                update_strategy=args.memory_update_strategy
            )
            
            # 初始化更新追踪器（用于统计更新情况）
            total_valid_entries = unwrapped_model.valid_mask.sum().item() if hasattr(unwrapped_model, 'valid_mask') else unwrapped_model.memory_bank.shape[0]
            memory_update_tracker = MemoryUpdateTracker(
                total_valid_entries=total_valid_entries,
                update_ratio_threshold=1.0  # 不再用于 keys 重新聚类，设为 1.0 禁用
            )
            
            Logger(f"Memory Bank 更新组件初始化完成: 更新频率={args.memory_update_frequency}, 策略={args.memory_update_strategy} (默认值来自 config/memory_update.py)", accelerator)
        
        accelerator.wait_for_everyone()

    global_step = resume_step  # 从checkpoint恢复或从0开始
    best_val_loss = float('inf')
    
    # 确保所有进程在训练开始前同步
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        Logger("训练循环开始", accelerator)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_ce_loss_sum = 0.0
        epoch_baseline_loss_sum = 0.0
        steps_in_epoch = 0
        
        # 确保所有进程在epoch开始前同步
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            Logger(f"Epoch {epoch+1}/{args.epochs} 开始", accelerator)
        
        # 计算每个 epoch 的总步数（考虑梯度累积）
        total_steps_in_epoch = len(train_loader) // args.accumulation_steps
        
        # 只在主进程创建进度条
        if accelerator.is_main_process and TQDM_AVAILABLE:
            pbar = tqdm(
                total=total_steps_in_epoch,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                unit="step",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        else:
            pbar = None
        
        # 确保所有进程在开始迭代前同步
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            Logger(f"开始迭代训练数据，共 {len(train_loader)} 个batch", accelerator)

        # 计时：上一个 step 结束时间（用于估计数据加载耗时）
        prev_step_end_time = time.time()
        
        # 创建迭代器，准备开始训练
        if accelerator.is_main_process:
            Logger("正在创建数据迭代器...", accelerator)
        train_iter = iter(train_loader)
        if accelerator.is_main_process:
            Logger("数据迭代器创建完成，开始加载第一个 batch（如果 shuffle=True，首次加载可能需要时间）...", accelerator)
        
        # 使用标准的循环（与train_fusion.py一致）
        step = 0
        while True:
            # 在数据加载前开始计时
            data_load_start = time.time()
            try:
                (X, Y, loss_mask) = next(train_iter)
            except StopIteration:
                # 数据加载完成，退出循环
                if accelerator.is_main_process:
                    Logger("数据加载完成，退出循环", accelerator)
                break
            # 数据加载完成，记录时间
            data_loading_time = time.time() - data_load_start

            # 更新 step 计数（在每个迭代开始时递增，而不是在累积完成时）
            step += 1
            
            # 当前 step 开始时间（用于计算总耗时）
            step_start_time = time.time()
            
            # 第一个 step 的数据加载时间可能包含 DataLoader 初始化，标记但不影响后续计算
            if step == 0:
                # 第一个 step 的数据加载时间可能包含初始化，这是正常的
                pass
            
            # 注意：这里不加同步点，因为数据加载可能不同步
            # accelerator.wait_for_everyone()

            baseline_loss = None
            total_loss = None
            
            # 使用 accelerator.accumulate() 上下文管理器自动处理梯度累积
            # 这确保在 DeepSpeed Stage2 下，所有 rank 的 collective 操作完全同步
            with accelerator.accumulate(model):
                # 前向传播计时
                t_fwd_start = time.time()
                res = model(X)
                forward_time = time.time() - t_fwd_start
                
                # 计算 CE Loss + baseline_loss + total_loss 计时
                t_loss_start = time.time()
                ce_loss = loss_fct(
                    res.logits.view(-1, res.logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()
                
                # 处理辅助损失：相对基线损失（relative baseline loss）
                # 注意：在 DeepSpeed 下，损失会自动处理，不需要手动 gather
                baseline_loss = torch.tensor(0.0, device=ce_loss.device)
                
                if hasattr(res, 'aux_loss') and res.aux_loss is not None:
                    aux_loss = res.aux_loss
                    if isinstance(aux_loss, dict):
                        baseline_loss = aux_loss.get('baseline_loss', torch.tensor(0.0, device=ce_loss.device))
                        
                        # 确保 baseline_loss 是标量（如果不是，取 mean）
                        if isinstance(baseline_loss, torch.Tensor):
                            if baseline_loss.dim() > 0:
                                baseline_loss = baseline_loss.mean()
                
                # 自适应损失平衡（与 SFT 一致）
                epsilon = 1e-8
                ce_loss_detached = ce_loss.detach()
                baseline_loss_detached = baseline_loss.detach()
                
                # 计算自适应系数：让 baseline_loss 的贡献等于 ce_loss 的大小
                adaptive_coef = ce_loss_detached / (baseline_loss_detached + epsilon)
                
                # 使用基础系数进行微调（默认 1.0，表示完全平衡）
                base_coef = getattr(args, 'baseline_loss_coef', getattr(args, 'similarity_loss_coef', 1.0))
                
                # 总损失（与 SFT 一致，不除以 accumulation_steps，由 accelerator.accumulate() 自动处理）
                total_loss = (
                    ce_loss +
                    base_coef * baseline_loss * adaptive_coef
                )
                loss_compute_time = time.time() - t_loss_start
                
                # 反向传播计时
                t_bwd_start = time.time()
                accelerator.backward(total_loss)
                backward_time = time.time() - t_bwd_start
                # optimizer.step() 和 optimizer.zero_grad() 由 accelerator.accumulate() 自动处理
                # 只在累积步数达到 accumulation_steps 时才会调用
            
            # 本 step 总的计算时间（不含下一次数据加载）
            step_compute_time = time.time() - step_start_time
            
            # 累积统计（total_loss 不再除以 accumulation_steps，所以直接累加）
            epoch_loss_sum += total_loss.item()
            epoch_ce_loss_sum += ce_loss.item()
            # 确保正确获取 baseline_loss 的值
            if isinstance(baseline_loss, torch.Tensor):
                epoch_baseline_loss_sum += baseline_loss.item()
            else:
                epoch_baseline_loss_sum += float(baseline_loss)
            steps_in_epoch += 1
            
            # 在累积完成后执行的操作（所有进程必须同步）
            if (step + 1) % args.accumulation_steps == 0:
                update_start_time = time.time()

                # Step 1: Scheduler 更新
                if scheduler is not None:
                    scheduler.step()
                
                # Step 2: 更新 global_step
                global_step += 1
                
                # Step 3: 更新进度条（仅主进程）
                if accelerator.is_main_process and pbar is not None:
                    if steps_in_epoch > 0:
                        running_loss = epoch_loss_sum / steps_in_epoch
                        running_ce = epoch_ce_loss_sum / steps_in_epoch
                        running_baseline = epoch_baseline_loss_sum / steps_in_epoch
                        pbar.set_description(
                            f"Epoch {epoch+1}/{args.epochs} | Loss: {running_loss:.4f} | "
                            f"CE: {running_ce:.4f} | Baseline: {running_baseline:.4f}"
                        )
                    pbar.update(1)
                
                # Step 4: 记录日志（仅主进程）
                if global_step % 10 == 0 and accelerator.is_main_process:
                    running_loss = epoch_loss_sum / steps_in_epoch
                    running_ce = epoch_ce_loss_sum / steps_in_epoch
                    running_baseline = epoch_baseline_loss_sum / steps_in_epoch
                    
                    swanlab.log({
                        "train/step_loss": total_loss.item(),
                        "train/running_loss": running_loss,
                        "train/ce_loss": running_ce,
                        "train/baseline_loss": running_baseline,
                        "train/lr": scheduler.get_last_lr()[0] if scheduler is not None else 0.0,
                    }, step=global_step)
                    
                    # 保存 checkpoint
                    if global_step % args.save_interval == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step}")
                        save_trainable_components(
                            model, save_path, accelerator,
                            train_memory_gate=not has_pretrained_memory_gate
                        )
                
                # ========== Memory Bank 更新 ==========
                # 使用 memory_update_enabled 而非 memory_bank_updater（后者仅主进程有值）
                accelerator.wait_for_everyone()
                
                try:
                    if memory_update_enabled:
                        # 广播是否需要更新
                        if accelerator.is_main_process:
                            should_update_value = 1 if (args.memory_update_frequency > 0 and 
                                                       global_step % args.memory_update_frequency == 0) else 0
                            should_update_tensor = torch.tensor(should_update_value, dtype=torch.long, device=accelerator.device)
                        else:
                            should_update_tensor = torch.tensor(0, dtype=torch.long, device=accelerator.device)
                        
                        if accelerator.num_processes > 1:
                            import torch.distributed as dist
                            dist.broadcast(should_update_tensor, src=0)
                        
                        should_update = bool(should_update_tensor.item())
                        should_sync_memory_bank = False
                            
                        if should_update:
                            # 主进程执行更新
                            if accelerator.is_main_process:
                                try:
                                    decoded_texts = tokenizer.batch_decode(X.cpu().tolist(), skip_special_tokens=True)
                                    input_text = next((t for t in decoded_texts if t and t.strip()), None)
                                    
                                    if input_text:
                                        with torch.no_grad():
                                            update_result = memory_bank_updater.update_from_text(
                                                input_text, compression_rate=args.memory_compression_rate
                                            )
                                            
                                            if update_result.get("updated_count", 0) > 0:
                                                memory_update_tracker.record_update(update_result)
                                                Logger(f"✅ Memory Bank 已更新: {update_result['updated_count']} 条事实", accelerator)
                                                should_sync_memory_bank = True
                                except Exception as e:
                                    Logger(f"❌ Memory Update 失败: {e}", accelerator)
                            
                            # 多卡同步
                            if accelerator.num_processes > 1:
                                accelerator.wait_for_everyone()
                                
                                # 同步 Memory Bank
                                sync_mb_flag = torch.tensor([1 if should_sync_memory_bank else 0], dtype=torch.int, device=accelerator.device)
                                if not accelerator.is_main_process:
                                    sync_mb_flag.zero_()
                                dist.broadcast(sync_mb_flag, src=0)
                                
                                if sync_mb_flag.item() == 1:
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    mb_shape = unwrapped_model.memory_bank.shape
                                    chunk_size = 5000
                                    
                                    for start in range(0, mb_shape[0], chunk_size):
                                        end = min(start + chunk_size, mb_shape[0])
                                        if accelerator.is_main_process:
                                            chunk = unwrapped_model.memory_bank[start:end].clone().to(accelerator.device)
                                        else:
                                            chunk = torch.zeros(end - start, mb_shape[1], dtype=torch.int64, device=accelerator.device)
                                        dist.broadcast(chunk, src=0)
                                        unwrapped_model.memory_bank[start:end] = chunk.cpu()
                                    
                                    if accelerator.is_main_process:
                                        vm = unwrapped_model.valid_mask.clone().to(accelerator.device)
                                    else:
                                        vm = torch.zeros(mb_shape[0], dtype=torch.bool, device=accelerator.device)
                                    dist.broadcast(vm, src=0)
                                    unwrapped_model.valid_mask.copy_(vm.cpu())
                                    
                                    if accelerator.is_main_process:
                                        Logger("✅ Memory Bank 同步完成", accelerator)
                finally:
                    accelerator.wait_for_everyone()
                
                update_time = time.time() - update_start_time

                # 记录耗时信息
                if accelerator.is_main_process:
                    Logger(
                        f"[Time] step={global_step} | "
                        f"data_load={data_loading_time:.3f}s | "
                        f"forward={forward_time:.3f}s | "
                        f"loss={loss_compute_time:.3f}s | "
                        f"backward={backward_time:.3f}s | "
                        f"step_compute={step_compute_time:.3f}s | "
                        f"update={update_time:.3f}s",
                        accelerator,
                    )
                    if scheduler is not None:
                        swanlab.log(
                            {
                                "time/data_load": data_loading_time,
                                "time/forward": forward_time,
                                "time/loss": loss_compute_time,
                                "time/backward": backward_time,
                                "time/step_compute": step_compute_time,
                                "time/update": update_time,
                            },
                            step=global_step,
                        )

                prev_step_end_time = time.time() ######
        
        # 关闭进度条
        if pbar is not None:
            pbar.close()
        
        # Epoch 结束：计算平均损失
        avg_loss = epoch_loss_sum / steps_in_epoch
        avg_ce = epoch_ce_loss_sum / steps_in_epoch
        avg_baseline = epoch_baseline_loss_sum / steps_in_epoch
        
        Logger(f"Epoch {epoch} 完成: Loss={avg_loss:.4f}, CE={avg_ce:.4f}, Baseline={avg_baseline:.4f}", accelerator)
        
        if accelerator.is_main_process:
            swanlab.log({
                "train/epoch_loss": avg_loss,
                "train/epoch_ce_loss": avg_ce,
                "train/epoch_baseline_loss": avg_baseline,
            }, step=global_step)
        
        # 验证
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for X, Y, loss_mask in val_loader:
                    res = model(X)
                    ce_loss = loss_fct(
                        res.logits.view(-1, res.logits.size(-1)),
                        Y.view(-1)
                    ).view(Y.size())
                    ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()
                    val_loss_sum += ce_loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss_sum / val_steps if val_steps > 0 else 0.0
            Logger(f"验证损失: {avg_val_loss:.4f}", accelerator)
            
            # 确保所有进程在验证完成后同步
            accelerator.wait_for_everyone()
            
            if accelerator.is_main_process:
                swanlab.log({"val/loss": avg_val_loss}, step=global_step)
                
                # 保存最佳模型
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    unwrapped_model = accelerator.unwrap_model(model)
                    best_path = os.path.join(args.output_dir, "fusion_best.pth")
                    torch.save(unwrapped_model.state_dict(), best_path)
                    Logger(f"最佳模型已保存: {best_path} (val_loss={best_val_loss:.4f})", accelerator)
        
        # 确保所有进程在保存 checkpoint 前同步
        accelerator.wait_for_everyone()
        
        # 保存 epoch checkpoint
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            epoch_path = os.path.join(args.output_dir, f"fusion_epoch_{epoch}.pth")
            torch.save(unwrapped_model.state_dict(), epoch_path)
            Logger(f"Epoch checkpoint saved: {epoch_path}", accelerator)
        
        # 确保所有进程在 epoch 结束后同步
        accelerator.wait_for_everyone()
    
    # 训练完成
    accelerator.end_training()
    if accelerator.is_main_process:
        swanlab.finish()
        Logger("阶段2训练完成！", accelerator)


if __name__ == "__main__":
    main()

