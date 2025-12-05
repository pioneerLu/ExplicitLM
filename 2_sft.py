#!/usr/bin/env python3
"""
Hydra-Zen + DeepSpeed 监督微调（SFT）入口

功能：
- 基于预训练模型进行监督微调
- 使用对话格式数据训练
- 支持分布式训练和混合精度
"""
import os
import time
import gc
from typing import Optional, Any

# 设置进程名称（在 nvidia-smi 中显示的名称）
try:
    import setproctitle
    process_name = os.environ.get('PYTHON_PROCESS_NAME', 'llama-env')
    setproctitle.setproctitle(process_name)
except ImportError:
    # 如果没有 setproctitle，尝试使用 prctl (Linux only)
    try:
        import prctl
        process_name = os.environ.get('PYTHON_PROCESS_NAME', 'llama-env')
        prctl.set_name(process_name.encode('utf-8'))
    except (ImportError, AttributeError):
        # 如果都不可用，跳过（不影响训练）
        pass

from accelerate import Accelerator, DistributedDataParallelKwargs, DeepSpeedPlugin
from accelerate.utils import set_seed
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from hydra_zen import launch, instantiate
from config import store, _main_cfg_func          # 触发配置注册
from utils.logger import Logger
from utils.sft_datasets import create_sft_dataloader, create_sft_validation_dataloader
from utils.train_loop_sft import train_epoch_sft
from utils.model_initializer import init_model
from hydra.utils import get_original_cwd
from pathlib import Path

try:
    import swanlab
except ImportError:
    swanlab = None


def main(cfg):
    """cfg 就是 Hydra-Zen 注入的五大配置节点"""
    # ------------------------------------------------------------------
    # 第一阶段：解构配置
    # ------------------------------------------------------------------
    m_cfg = cfg.model
    d_cfg = cfg.dataset
    l_cfg = cfg.logging
    tr_cfg = cfg.training
    proj_root = Path(get_original_cwd())

    # 配置 DDP 参数：允许未使用的参数（用于部分模型组件可能不参与梯度计算的情况）
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    ds_plugin = DeepSpeedPlugin(zero_stage=tr_cfg.zero_stage)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=ds_plugin,
    )
    set_seed(tr_cfg.seed + accelerator.process_index)

    if accelerator.is_main_process:
        os.makedirs(l_cfg.out_dir, exist_ok=True)
        os.makedirs(l_cfg.save_dir, exist_ok=True)

    swanlab_run: Optional[Any] = None
    if l_cfg.use_swanlab and accelerator.is_main_process and swanlab is not None:
        mode = "cloud" if l_cfg.swanlab_online else "offline"
        Logger(f"SwanLab 模式：{mode}", accelerator)
        Logger(f"SwanLab 运行中...", accelerator)
        # 从环境变量获取API key，如果未设置则使用项目默认值
        api_key = os.environ.get("SWANLAB_API_KEY", "GtiI1qjU5lco6MKKSrRmN")
        swanlab_run = swanlab.init(
            project=l_cfg.swanlab_project,
            experiment_name=f"ExplicitLM-SFT-{tr_cfg.epochs}e-{tr_cfg.batch_size}b-{tr_cfg.learning_rate}lr",
            config=instantiate(cfg),   # 把完整配置 flatten 上传
            mode=mode,
            api_key=api_key  # 使用项目特定的API key
        )

    model, tokenizer = init_model(m_cfg)
    Logger("模型架构初始化完成", accelerator)

    if hasattr(d_cfg, 'pretrained_sft_model_path') and d_cfg.pretrained_sft_model_path:
        Logger(f"开始加载预训练权重: {d_cfg.pretrained_sft_model_path}", accelerator)

        try:
            # 加载预训练检查点
            pretrained_path = proj_root / d_cfg.pretrained_sft_model_path
            checkpoint = torch.load(pretrained_path, map_location='cpu')

            # 记录检查点中包含的memory相关参数
            memory_keys_in_checkpoint = [k for k in checkpoint.keys() if 'memory' in k.lower()]
            if memory_keys_in_checkpoint:
                Logger(f"检查点中包含 {len(memory_keys_in_checkpoint)} 个memory相关参数", accelerator)
                Logger(f"示例: {memory_keys_in_checkpoint[:3]}{'...' if len(memory_keys_in_checkpoint) > 3 else ''}", accelerator)

            # 加载模型参数（使用 strict=False 允许部分加载）
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

            if missing_keys:
                Logger(f"警告: 以下参数未在检查点中找到: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}", accelerator)
            if unexpected_keys:
                Logger(f"警告: 以下参数在检查点中但模型不需要: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}", accelerator)

            Logger(f"✓ 预训练权重加载完成（包括知识库参数）", accelerator)

            # 重新确认参数冻结状态（确保加载权重后冻结状态不变）
            Logger("🔒 重新确认参数冻结状态...", accelerator)
            frozen_params = 0
            trainable_params = 0
            memory_bank_params = 0
            
            for name, param in model.named_parameters():
                is_memory_component = any(keyword in name for keyword in [
                    "memory_gate", "gated_memory_fusion", "memory_norm"
                ])
                is_memory_bank = "memory_bank" in name
                
                if is_memory_bank:
                    param.requires_grad = False
                    memory_bank_params += param.numel()
                    frozen_params += param.numel()
                elif is_memory_component:
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
                    frozen_params += param.numel()
            
            Logger(f"参数冻结: 冻结 {frozen_params / 1e6:.3f}M, 可训练 {trainable_params / 1e6:.3f}M, Memory bank {memory_bank_params / 1e6:.3f}M", accelerator)

        except FileNotFoundError:
            Logger(f"错误: 预训练模型文件不存在: {pretrained_path}", accelerator)
            Logger("将从随机初始化开始训练（不推荐）", accelerator)
        except Exception as e:
            Logger(f"错误: 加载预训练权重时发生异常: {e}", accelerator)
            Logger("将从随机初始化开始训练（不推荐）", accelerator)
    else:
        Logger("警告: 未指定 pretrained_sft_model_path 参数", accelerator)
        Logger("SFT 训练通常需要加载预训练权重，当前将从随机初始化开始（不推荐）", accelerator)

    # 确保优化器只包含可训练的参数（memory_gate 和 fusion）
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in optimizer_params)
    total_count = sum(p.numel() for p in model.parameters())
    
    Logger(f"优化器参数: {trainable_count / 1e6:.3f}M / {total_count / 1e6:.3f}M ({trainable_count / total_count * 100:.2f}%)", accelerator)
    
    # 列出可训练的参数名称（用于验证）
    trainable_param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    Logger(f"  - 可训练参数模块: {len(trainable_param_names)} 个", accelerator)
    if len(trainable_param_names) <= 10:
        for name in trainable_param_names:
            Logger(f"    * {name}", accelerator)
    else:
        Logger(f"    * {trainable_param_names[0]} ... (共{len(trainable_param_names)}个)", accelerator)

    optimizer = torch.optim.AdamW(
        optimizer_params,
        lr=tr_cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # ------------------------------------------------------------------
    # 第七阶段：SFT 数据加载器
    # ------------------------------------------------------------------
    train_loader = create_sft_dataloader(
        data_path=str(proj_root / d_cfg.sft_dataset_path),
        tokenizer=tokenizer,
        batch_size=tr_cfg.batch_size,
        max_length=m_cfg.max_seq_len,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = create_sft_validation_dataloader(
        val_data_path=str(proj_root / d_cfg.sft_val_dataset_path),
        tokenizer=tokenizer,
        batch_size=tr_cfg.batch_size,
        max_length=m_cfg.max_seq_len,
        num_samples=200,
    )

    # ------------------------------------------------------------------
    # 第八阶段：学习率调度器
    # ------------------------------------------------------------------
    steps_per_epoch = len(train_loader) // tr_cfg.accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * steps_per_epoch * tr_cfg.epochs),
        num_training_steps=steps_per_epoch * tr_cfg.epochs,
    )

    # ------------------------------------------------------------------
    # 第九阶段：Accelerator.prepare
    # ------------------------------------------------------------------
    model, optimizer, scheduler, train_loader, val_loader = accelerator.prepare(
        model, optimizer, scheduler, train_loader, val_loader
    )
    Logger(instantiate(cfg), accelerator)

    # ------------------------------------------------------------------
    # 第十阶段：SFT 训练循环
    # ------------------------------------------------------------------
    overall_start_time = time.time()

    for epoch in range(tr_cfg.epochs):
        train_epoch_sft(
            epoch=epoch,
            accelerator=accelerator,
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            args=instantiate(cfg),   # 兼容训练循环的 args 用法
            overall_start_time=overall_start_time,
            swanlab_run=swanlab_run,
            tokenizer=tokenizer,
            eval_loader=val_loader,
        )
        gc.collect()
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 第十一阶段：收尾
    # ------------------------------------------------------------------
    if accelerator.is_main_process and swanlab_run:
        # 获取 SwanLab 实验 URL
        if l_cfg.swanlab_online:                       # 云版
            exp_url = str(swanlab_run.public.cloud.experiment_url)
        else:                                          # 本地版
            exp_url = 'local-mode'

        # 写入临时文件供脚本读取（和 pretrain 保持一致的文件名）
        with open('.swanlab_url', 'w') as f:
            f.write(exp_url)

        Logger(f"SwanLab URL 已保存: {exp_url}", accelerator)

    # ------------------------------------------------------------------
    # 第十二阶段：关闭 SwanLab
    # ------------------------------------------------------------------
    if l_cfg.use_swanlab and accelerator.is_main_process:
        if swanlab_run is not None:
            swanlab_run.finish()
            Logger("SwanLab 运行已结束", accelerator)

    Logger("SFT 训练完成！", accelerator)


if __name__ == "__main__":
    import sys
    # 提取命令行参数用于配置覆盖，跳过脚本名称
    # 接受 key=value 格式的参数
    overrides = []
    for arg in sys.argv[1:]:
        if '=' in arg and not arg.startswith('--'):
            overrides.append(arg)

    # 使用命令行覆盖启动
    launch(_main_cfg_func, main, overrides=overrides)
