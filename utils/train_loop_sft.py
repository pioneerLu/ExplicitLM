"""
SFT训练循环模块

功能：
- train_epoch_sft: SFT阶段的单个epoch训练循环
- eval_model_sft: 生成式评估函数（文本生成+准确率计算）
- judger: 文本匹配评估器
- 支持梯度累积、验证评估、模型保存
- 集成SwanLab实验追踪
"""

import time
from typing import Any, Optional, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import numpy as np

from utils.logger import Logger
from utils.train_utils import format_time
from utils.memory_update_tracker import MemoryUpdateTracker

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

try:
    import swanlab
except ImportError:
    swanlab = None

import re


def extract_accessed_indices(cosine_stats: Dict) -> List[int]:
    """
    从模型输出中提取被访问的知识索引
    
    Args:
        cosine_stats: 模型输出的 cosine_stats 字典（可能包含多层）
    
    Returns:
        被访问的知识索引列表（去重后）
    """
    all_indices = []
    for key, value in cosine_stats.items():
        if key.endswith('_actual_selected_indices'):
            # value: [bsz, seq_len] 或 [seq_len]
            if isinstance(value, torch.Tensor):
                # 展平并转换为列表
                indices = value.flatten().cpu().tolist()
                all_indices.extend(indices)
    return list(set(all_indices))  # 去重


def extract_answer_from_generation(
    generated_text: str,
    remove_thinking: bool = True,
    extract_answer_tag: bool = True
) -> str:
    """
    从生成的文本中提取实际答案
    
    处理策略：
    1. 移除所有 thinking 标签和内容（<think>, <reasoning>, <thinking> 等）
    2. 提取 <answer>...</answer> 格式的答案（如果存在）
    3. 移除明显的 thinking 前缀（如 "First, I remember...", "Let me think..." 等）
    
    Args:
        generated_text: 原始生成的文本
        remove_thinking: 是否移除 thinking 标签和内容
        extract_answer_tag: 是否提取 <answer>...</answer> 格式
    
    Returns:
        清理后的答案文本
    """
    text = generated_text.strip()
    if not text:
        return text
    
    # 1. 移除 thinking 标签及其内容（开闭标签对）
    if remove_thinking:
        thinking_pairs = [
            ('<think>', '</think>'),
            ('<reasoning>', '</reasoning>'),
            ('<thinking>', '</thinking>'),
            ('<thought>', '</thought>'),
            ('<think>', '</think>'),
        ]
        
        for open_tag, close_tag in thinking_pairs:
            # 移除开闭标签及其之间的内容
            pattern = rf'{re.escape(open_tag)}.*?{re.escape(close_tag)}'
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
            # 移除单独的标签（没有配对的）
            text = text.replace(open_tag, '').replace(close_tag, '')
        
        # 移除明显的 thinking 前缀和思考过程
        thinking_prefixes = [
            'first,', 'first of all,', 'let me', 'i need', 'i should',
            'i think', 'this is', 'well,', 'so,', 'now,', 'okay,',
            'the user', 'i remember', 'let me think', 'first i',
            'i should think', 'i need to', 'let me consider',
            'okay, the user', 'the user is asking', 'i need to recall',
            'i need to remember', 'let me recall', 'i should recall',
            'based on', 'according to', 'i know that', 'i remember that'
        ]
        
        lines = text.split('\n')
        cleaned_lines = []
        skip_thinking = True
        
        for line in lines:
            line = line.strip()
            if not line:
                if not skip_thinking:
                    cleaned_lines.append('')
                continue
            
            # 检查是否是 thinking 内容
            line_lower = line.lower()
            
            # 检查是否以 thinking 前缀开头
            is_thinking = any(
                line_lower.startswith(prefix) 
                for prefix in thinking_prefixes
            ) or any(
                line_lower.startswith(f"first, {prefix}") or 
                line_lower.startswith(f"well, {prefix}")
                for prefix in thinking_prefixes
            )
            
            # 检查是否包含常见的思考模式
            thinking_patterns = [
                'the user is asking',
                'i need to recall',
                'i need to remember',
                'let me recall',
                'i should recall',
                'based on my knowledge',
                'according to my understanding',
                'i know that',
                'i remember that',
                'so, it\'s',
                'so it\'s',
                'this means',
                'this is because'
            ]
            
            # 如果整行都是思考内容（以思考模式开头且长度较短），标记为 thinking
            if not is_thinking:
                for pattern in thinking_patterns:
                    if pattern in line_lower and len(line.split()) < 30:
                        # 检查是否主要是思考内容而不是答案
                        if any(marker in line_lower for marker in ['the user', 'i need', 'i should', 'let me']):
                            is_thinking = True
                            break
            
            # 检查是否包含问号（可能是问题而不是答案）
            if '?' in line and not any(tag in line for tag in ['<answer>', '</answer>']):
                is_thinking = True
            
            if is_thinking and skip_thinking:
                continue
            
            # 找到第一个非 thinking 内容后，停止跳过
            if not is_thinking:
                skip_thinking = False
            
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines).strip()
    
    # 2. 提取 <answer>...</answer> 格式（如果存在）
    if extract_answer_tag:
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
    
    # 3. 如果文本仍然包含 thinking 标记，尝试更智能地提取答案
    if remove_thinking:
        text_lower = text.lower()
        
        # 查找答案开始的标志（通常是定义、解释等）
        answer_start_markers = [
            ' is ',
            ' are ',
            ' refers to ',
            ' means ',
            ' can be defined as ',
            ' occurs ',
            ' happens ',
            ' takes place '
        ]
        
        # 如果文本以思考内容开头，尝试找到第一个答案标记
        if any(marker in text_lower for marker in ['the user', 'i need', 'i should', 'let me', 'okay,']):
            # 查找第一个答案标记的位置
            best_pos = len(text)
            for marker in answer_start_markers:
                pos = text_lower.find(marker)
                if pos != -1 and pos < best_pos:
                    best_pos = pos
            
            # 如果找到了答案开始位置，提取从那里开始的内容
            if best_pos < len(text):
                # 向前查找句子的开始（找到前一个句号或行首）
                sentence_start = 0
                for i in range(best_pos - 1, -1, -1):
                    if text[i] in ['.', '\n']:
                        sentence_start = i + 1
                        break
                
                # 提取答案部分
                answer_part = text[sentence_start:].strip()
                if len(answer_part) > 10:  # 确保提取的内容足够长
                    text = answer_part
        
        # 如果仍然包含 thinking 标记，尝试提取最后一部分
        if any(marker in text_lower for marker in ['think', 'reasoning', 'remember', 'the user']):
            # 尝试找到最后一个句号后的内容
            parts = text.split('.')
            if len(parts) > 1:
                # 取最后几个部分（可能是实际答案）
                text = '.'.join(parts[-2:]).strip()
    
    return text.strip()


def judger(
    generated_text: str,
    std_output: str,
    judger_mode: str
) -> bool:
    """
    评估生成文本与标准答案的匹配程度

    参数：
        generated_text: 生成的文本
        std_output: 标准答案
        judger_mode: 判断模式
            - exact: 精确匹配
            - contains: 标准答案包含在生成文本中
            - startswith: 生成文本以标准答案开头
            - endswith: 生成文本以标准答案结尾

    返回：
        是否匹配成功
    """
    gen_stripped = generated_text.strip()
    std_stripped = std_output.strip()

    if judger_mode == 'exact':
        return gen_stripped == std_stripped
    elif judger_mode == 'contains':
        return std_stripped in gen_stripped
    elif judger_mode == 'startswith':
        return gen_stripped.startswith(std_stripped)
    elif judger_mode == 'endswith':
        return gen_stripped.endswith(std_stripped)
    return False


def eval_model_sft(
    model: nn.Module,
    eval_loader: DataLoader,
    tokenizer: Any,
    accelerator: Accelerator,
    args: Any,
    judger_mode: str = "startswith"
) -> dict:
    """
    SFT阶段的生成式评估函数

    参数：
        model: 待评估模型（wrapped model）
        eval_loader: 评估数据加载器
        tokenizer: Tokenizer实例
        accelerator: Accelerator实例
        args: 配置参数
        judger_mode: 判断模式

    返回：
        性能字典，包含每个样本的结果和整体准确率

    说明：
        - 模型生成完整回复
        - 与标准答案对比（startswith/contains等）
        - 计算准确率指标
    """
    # 验证 tokenizer 配置
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer 缺少 eos_token_id 配置")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer 缺少 pad_token_id 配置")

    # 设置评估模式（在 wrapped model 上操作）
    model.eval()
    performance = {}
    total_correct = 0
    total_steps = 0

    with torch.no_grad():
        for step, batch in enumerate(eval_loader):
            try:
                # DataLoader 可能返回 tuple 或 list，需要同时支持
                if (isinstance(batch, (tuple, list)) and len(batch) == 2):
                    prompt_input, std_output = batch
                    
                    # 处理 batch_size > 1 的情况（虽然通常 batch_size=1）
                    if isinstance(prompt_input, (list, tuple)):
                        if len(prompt_input) > 0:
                            prompt_input = prompt_input[0]
                        else:
                            Logger(f"警告: prompt_input 为空，跳过 step {step}", accelerator)
                            continue
                    
                    if isinstance(std_output, (list, tuple)):
                        if len(std_output) > 0:
                            std_output = std_output[0]
                        else:
                            Logger(f"警告: std_output 为空，跳过 step {step}", accelerator)
                            continue
                    
                    if not isinstance(prompt_input, str):
                        Logger(f"警告: prompt_input 不是字符串类型 (type: {type(prompt_input)}), 跳过 step {step}", accelerator)
                        continue
                        
                    if not isinstance(std_output, str):
                        Logger(f"警告: std_output 不是字符串类型 (type: {type(std_output)}), 跳过 step {step}", accelerator)
                        continue
                else:
                    Logger(f"警告: 评估数据格式错误 (batch type: {type(batch)}, len: {len(batch) if hasattr(batch, '__len__') else 'N/A'}), 跳过 step {step}", accelerator)
                    continue
            except Exception as e:
                Logger(f"警告: 处理batch时出错 (step {step}): {e}, 跳过", accelerator)
                continue

            try:
                x = torch.tensor(
                    tokenizer(prompt_input)['input_ids'],
                    device=accelerator.device
                ).unsqueeze(0)
            except Exception as e:
                Logger(f"警告: Tokenization 失败 (step {step}): {e}", accelerator)
                continue

            unwrapped_model = accelerator.unwrap_model(model)

            try:
                generated = unwrapped_model.generate(
                    x,
                    max_new_tokens=args.training.max_new_tokens,
                    max_length=args.model.max_seq_len + args.training.max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            except Exception as e:
                Logger(f"警告: 生成失败 (step {step}): {e}", accelerator)
                continue

            try:
                generated_text = tokenizer.decode(
                    generated.squeeze()[x.shape[1]:].tolist(),
                    skip_special_tokens=True
                )
            except Exception as e:
                Logger(f"警告: 解码失败 (step {step}): {e}", accelerator)
                continue

            generated_text_raw = generated_text.strip()
            
            # 后处理：提取实际答案（移除thinking标签和内容）
            extracted_text = extract_answer_from_generation(generated_text_raw)
            
            flag = judger(extracted_text, std_output, judger_mode)

            performance[step] = {
                'prompt': prompt_input,
                'std_output': std_output,
                'generated_text_raw': generated_text_raw,  # 原始生成文本（提取前）
                'generated_text': extracted_text,  # 提取后的文本
                'result': flag,
                'judger_mode': judger_mode
            }

            total_correct += int(flag)
            total_steps += 1

            if accelerator.is_main_process and step % max(1, len(eval_loader) // args.training.show_eval_res) == 0:
                # 显示完整文本，不截断
                Logger(f"评估样例 Step {step}:", accelerator)
                Logger(f"  生成文本（完整）: {extracted_text}", accelerator)
                Logger(f"  标准答案（完整）: {std_output}", accelerator)
                Logger(f"  匹配结果: {flag}", accelerator)
                Logger("", accelerator)  # 空行分隔

    accuracy = total_correct / total_steps if total_steps > 0 else 0.0

    performance['overall'] = {
        'total_steps': total_steps,
        'total_correct': total_correct,
        'accuracy': accuracy
    }

    if accelerator.is_main_process:
        Logger(
            f"评估结果: 准确率={accuracy:.4f} ({total_correct}/{total_steps})",
            accelerator
        )

    model.train()

    return performance


def train_epoch_sft(
    epoch: int,
    accelerator: Accelerator,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Any,
    scheduler: Any,
    args: Any,
    overall_start_time: float,
    swanlab_run: Optional[Any],
    tokenizer: Any,
    eval_loader: Optional[DataLoader] = None,
    memory_bank_updater: Optional[Any] = None,  # 可选的已初始化的 memory_bank_updater
    memory_update_tracker: Optional[Any] = None,  # 可选的已初始化的 memory_update_tracker
) -> None:
    """
    SFT阶段的单个epoch训练循环

    参数：
        epoch: 当前epoch编号（从0开始）
        accelerator: Accelerator实例
        model: 训练模型（wrapped model）
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        args: 配置参数（hydra-zen结构：args.training/model/logging）
        overall_start_time: 整体训练开始时间
        swanlab_run: SwanLab运行实例
        tokenizer: Tokenizer实例
        eval_loader: 评估数据加载器（可选）

    说明：
        混合精度由DeepSpeed配置文件（ds_config.json）自动控制
        - bf16已启用，无需手动创建autocast上下文
        - 梯度累积由DeepSpeed自动处理
        - 梯度裁剪由DeepSpeed自动处理

        评估方式：
        - 使用eval_model_sft进行生成式评估
        - 计算文本生成准确率
        - 适用于SFT阶段的对话质量评估
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    epoch_start_time = time.time()
    total_steps_in_epoch = len(train_loader)
    total_training_steps = args.training.epochs * total_steps_in_epoch
    moe_path = '_moe' if args.model.use_moe else ''
    unwrapped_model = accelerator.unwrap_model(model)
    hidden_size = getattr(unwrapped_model.config, 'hidden_size', 2560)
    best_loss = float('inf')
    best_accuracy = 0.0
    last_log_time = epoch_start_time
    
    # 初始化知识更新组件（如果启用）
    memory_update_cfg = getattr(args, 'memory_update', None)
    if memory_update_cfg is None:
        memory_update_cfg = args.get('memory_update', {}) if hasattr(args, 'get') else {}
    
    # 全局标志：所有进程都需要知道是否启用 Memory Update
    memory_update_enabled = False
    if memory_update_cfg:
        if isinstance(memory_update_cfg, dict):
            memory_update_enabled = memory_update_cfg.get("enable_memory_update_during_training", False)
        elif hasattr(memory_update_cfg, 'get'):
            memory_update_enabled = memory_update_cfg.get("enable_memory_update_during_training", False)
        else:
            memory_update_enabled = getattr(memory_update_cfg, "enable_memory_update_during_training", False)
    
    if memory_update_enabled and (memory_bank_updater is None or memory_update_tracker is None):
        from utils.memory_bank_updater import MemoryBankUpdater
        from utils.fact_extractor import FactExtractor
            from config.memory_update import MemoryUpdateConf
        
        # 只在主进程初始化
        if accelerator.is_main_process:
            def get_cfg_value(key, default):
                    if isinstance(memory_update_cfg, dict):
                    return memory_update_cfg.get(key, MemoryUpdateConf.get(key, default))
                    elif hasattr(memory_update_cfg, 'get'):
                    return memory_update_cfg.get(key, MemoryUpdateConf.get(key, default))
                else:
                    return getattr(memory_update_cfg, key, MemoryUpdateConf.get(key, default))
            
                if memory_bank_updater is None:
            fact_extractor = FactExtractor(
                    model_path=get_cfg_value("llmlingua_model_path", "llmlingua-2-bert"),
                    compression_rate=get_cfg_value("memory_compression_rate", 0.4)
            )
            memory_bank_updater = MemoryBankUpdater(
                model=unwrapped_model,
                tokenizer=tokenizer,
                fact_extractor=fact_extractor,
                    update_strategy=get_cfg_value("memory_update_strategy", "lru")
            )
            
                if memory_update_tracker is None:
            total_valid_entries = unwrapped_model.valid_mask.sum().item() if hasattr(unwrapped_model, 'valid_mask') else unwrapped_model.memory_bank.shape[0]
            memory_update_tracker = MemoryUpdateTracker(
                total_valid_entries=total_valid_entries,
                update_ratio_threshold=1.0  # 不再用于 keys 重新聚类，设为 1.0 禁用
            )
            
            Logger(f"Memory Update 初始化完成", accelerator)
        
        accelerator.wait_for_everyone()

    if accelerator.is_main_process and TQDM_AVAILABLE:
        pbar = tqdm(
            total=total_steps_in_epoch,
            desc=f"Epoch {epoch+1}/{args.training.epochs}",
            unit="step",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    else:
        pbar = None

    # 计时：上一个 step 结束时间（用于估计数据加载耗时）
    prev_step_end_time = time.time()

    for step, batch_data in enumerate(train_loader):
        # 计时：当前 step 开始，估计数据加载时间
        step_start_time = time.time()
        data_loading_time = step_start_time - prev_step_end_time
        if step == 0:
            # 第一个 step 的数据加载时间包含初始化，设为0
            data_loading_time = 0.0
            prev_step_end_time = step_start_time

        # 解包 batch 数据（支持新旧格式兼容）
        if len(batch_data) == 3:
            # 旧格式：只有 (X, Y, loss_mask)
            X, Y, loss_mask = batch_data
            prompt_texts = None
        elif len(batch_data) == 4:
            # 新格式：(X, Y, loss_mask, prompt_texts)
            X, Y, loss_mask, prompt_texts = batch_data
        else:
            raise ValueError(f"意外的 batch 格式，长度: {len(batch_data)}")
        # 使用 accelerator.accumulate() 上下文管理器自动处理梯度累积
        # 这确保在 DeepSpeed Stage2 下，所有 rank 的 collective 操作完全同步
        with accelerator.accumulate(model):
            # 前向传播计时
            t_fwd_start = time.time()
            res = model(X, step=step)
            forward_time = time.time() - t_fwd_start
            
            # 记录知识使用（如果启用知识更新）
            if memory_bank_updater is not None and accelerator.is_main_process:
                accessed_indices = extract_accessed_indices(res.cosine_stats)
                if accessed_indices:
                    memory_bank_updater.record_access(accessed_indices)

            # 计算 CE Loss + baseline_loss + total_loss 计时
            t_loss_start = time.time()
            ce_loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()

            aux_loss = res.aux_loss
            # 相对基线损失（relative baseline loss），用于训练 MemoryGate
            baseline_loss = aux_loss.get('baseline_loss', torch.tensor(0.0, device=ce_loss.device))

            epsilon = 1e-8
            ce_loss_detached = ce_loss.detach()
            baseline_loss_detached = baseline_loss.detach()

            # 计算自适应系数：让 baseline_loss 的贡献等于 ce_loss 的大小
            adaptive_coef = ce_loss_detached / (baseline_loss_detached + epsilon)

            # 可选：保留一个基础系数用于微调（默认 1.0，表示完全平衡）
            base_coef = args.training.get("baseline_loss_coef", args.training.get("similarity_loss_coef", 1.0))

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

        # 注意：移除 empty_cache() 调用以避免分布式训练的 NCCL 超时问题
        # PyTorch 和现代 GPU 驱动的内存管理已经足够智能，无需手动干预
        # 如果遇到 OOM 错误，可以考虑在训练前优化批次大小或模型参数
        
        # ========== Memory Bank 更新 ==========
        # 使用 memory_update_enabled 而非 memory_bank_updater（后者仅主进程有值）
        if memory_update_enabled:
            from config.memory_update import MemoryUpdateConf
            
            def get_cfg_value(key, default):
                if memory_update_cfg:
                    if isinstance(memory_update_cfg, dict):
                        return memory_update_cfg.get(key, MemoryUpdateConf.get(key, default))
                    elif hasattr(memory_update_cfg, 'get'):
                        return memory_update_cfg.get(key, MemoryUpdateConf.get(key, default))
                else:
                        return getattr(memory_update_cfg, key, MemoryUpdateConf.get(key, default))
                return MemoryUpdateConf.get(key, default)
            
            update_frequency = get_cfg_value("memory_update_frequency", 0)
            
            accelerator.wait_for_everyone()
            
            try:
                # 广播是否需要更新
                if accelerator.is_main_process:
                    should_update_value = 1 if (update_frequency > 0 and (step + 1) % update_frequency == 0) else 0
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
                    if accelerator.is_main_process and memory_bank_updater is not None:
                try:
                    if prompt_texts is not None and len(prompt_texts) > 0:
                        input_text = prompt_texts[0]
                    else:
                                decoded_texts = tokenizer.batch_decode(X.cpu().tolist(), skip_special_tokens=True)
                                input_text = next((t for t in decoded_texts if t and t.strip()), None)
                    
                            if input_text and input_text.strip():
                    with torch.no_grad():
                        update_result = memory_bank_updater.update_from_text(
                                        input_text, compression_rate=get_cfg_value("memory_compression_rate", 0.4)
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

        # 更新进度条 - 只在主进程
        if accelerator.is_main_process and pbar is not None:
            pbar.update(1)

        if (step + 1) % args.logging.log_interval == 0 and accelerator.is_main_process:

            current_time = time.time()
            current_lr = optimizer.param_groups[0]['lr']
            epoch_elapsed_time = current_time - epoch_start_time
            epoch_steps_done = step + 1
            epoch_avg_step_time = epoch_elapsed_time / epoch_steps_done
            epoch_remaining_time = epoch_avg_step_time * (total_steps_in_epoch - epoch_steps_done)

            total_elapsed_time = current_time - overall_start_time
            total_steps_done = epoch * total_steps_in_epoch + epoch_steps_done
            total_avg_step_time = total_elapsed_time / total_steps_done if total_steps_done > 0 else 0
            total_remaining_time = total_avg_step_time * (total_training_steps - total_steps_done) if total_steps_done > 0 else 0
            interval_elapsed_time = current_time - last_log_time
            tokens_processed_interval = args.logging.log_interval * args.training.batch_size * args.model.max_seq_len
            tokens_per_sec = tokens_processed_interval / interval_elapsed_time if interval_elapsed_time > 0 else 0
            last_log_time = current_time

            unwrapped_model = accelerator.unwrap_model(model)
            try:
                memory_update_stats = unwrapped_model.get_memory_update_stats()
            except Exception as e:
                Logger(f"获取记忆更新统计失败: {e}", accelerator)
                memory_update_stats = {}

            cosine_stats = res.cosine_stats
            selected_similarities = [
                v for k, v in cosine_stats.items()
                if k.endswith('_selected_avg_similarity')
            ]
            avg_selected_similarity = np.mean(selected_similarities) if selected_similarities else 0.0

            # 相对基线损失（relative baseline loss）
            similarity_loss_log = res.aux_loss.get('baseline_loss', res.aux_loss.get('similarity_loss', torch.tensor(0.0))).item()
            
            # 计算自适应系数（用于日志）
            epsilon = 1e-8
            ce_loss_detached = ce_loss.detach()
            similarity_loss_detached = similarity_loss.detach()
            adaptive_coef_value = (ce_loss_detached / (similarity_loss_detached + epsilon)).item()
            base_coef = args.training.get("similarity_loss_coef", 1.0)
            effective_coef = base_coef * adaptive_coef_value
            
            log_dict = {
                "epoch": epoch + 1,
                "step": step + 1,
                "total_steps_in_epoch": total_steps_in_epoch,
                "train/loss_ce": ce_loss.item(),
                "train/loss_similarity": similarity_loss_log,  # 相对基线损失（relative baseline loss）
                "train/loss_baseline": similarity_loss_log,  # 别名
                "train/loss_total": total_loss.item(),
                "train/adaptive_coef": adaptive_coef_value,
                "train/effective_sim_coef": effective_coef,
                "lr": current_lr,
                "tokens_per_sec": tokens_per_sec,
                "epoch_time_left_seconds": epoch_remaining_time,
                "total_time_left_seconds": total_remaining_time,
                "train/avg_selected_similarity": avg_selected_similarity,
            }

            log_dict.update(memory_update_stats)

            epoch_eta = format_time(log_dict['epoch_time_left_seconds'])
            total_eta = format_time(log_dict['total_time_left_seconds'])

            Logger(
                f"Epoch {epoch+1}/{args.training.epochs}, Step {step+1}/{total_steps_in_epoch} | "
                f"Loss: {log_dict['train/loss_total']:.4f} (CE:{log_dict['train/loss_ce']:.4f} "
                f"Baseline:{log_dict['train/loss_baseline']:.4f}) | "  # 相对基线损失
                f"LR: {log_dict['lr']:.6f} | Speed: {log_dict['tokens_per_sec']:.0f} tok/s | "
                f"ETA: {epoch_eta} (epoch), {total_eta} (total)",
                accelerator
            )

            # 记录详细耗时信息（每10步记录一次）
            if accelerator.is_main_process and (step + 1) % 10 == 0:
                # 计算总step时间
                step_compute_time = time.time() - step_start_time

                Logger(
                    f"[Time] step={step+1} | "
                    f"data_load={data_loading_time:.3f}s | "
                    f"forward={forward_time:.3f}s | "
                    f"loss={loss_compute_time:.3f}s | "
                    f"backward={backward_time:.3f}s | "
                    f"step_compute={step_compute_time:.3f}s",
                    accelerator,
                )

                # 更新 swanlab 日志
                if args.logging.use_swanlab and swanlab_run:
                    swanlab_run.log({
                        "time/data_load": data_loading_time,
                        "time/forward": forward_time,
                        "time/loss": loss_compute_time,
                        "time/backward": backward_time,
                        "time/step_compute": step_compute_time,
                    }, step=global_step)

            # 更新进度条描述
            if pbar is not None:
                pbar.set_description(
                    f"Epoch {epoch+1}/{args.training.epochs} | Loss: {log_dict['train/loss_total']:.4f} | ETA: {total_eta}"
                )

            if args.logging.use_swanlab and swanlab_run:
                swanlab_run.log(log_dict)

            # 本次累积 step 完成，记录结束时间供下一个 step 估算数据加载时间
            prev_step_end_time = time.time()

        eval_interval = args.training.eval_interval
        start_eval = args.training.start_eval

        if (step + 1) % eval_interval == 0 and (step + 1) >= start_eval and eval_loader is not None:
            if accelerator.is_main_process:
                Logger(f"开始评估...", accelerator)

            performance = eval_model_sft(
                model=model,
                eval_loader=eval_loader,
                tokenizer=tokenizer,
                accelerator=accelerator,
                args=args,
                judger_mode=args.training.judger_mode
            )

            # 记录评估指标到SwanLab
            if accelerator.is_main_process and args.logging.use_swanlab and swanlab_run:
                swanlab_run.log({
                    "val/eval_accuracy": performance['overall']['accuracy'],
                    "val/eval_total_steps": performance['overall']['total_steps'],
                    "val/eval_total_correct": performance['overall']['total_correct']
                })

            # 基于准确率保存最佳模型
            current_accuracy = performance['overall']['accuracy']
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                if accelerator.is_main_process:
                    ckp = f'{args.logging.save_dir}/sft_best_acc_{hidden_size}{moe_path}.pth'
                    # 使用 DeepSpeed/Accelerate 方式获取模型状态字典（自动处理分片参数）
                    try:
                        # accelerator.get_state_dict 会自动收集 DeepSpeed 分片的参数
                        state_dict = accelerator.get_state_dict(model, unwrap=False)
                        # 提取模型权重（排除 optimizer 等）
                        model_state_dict = {}
                        for k, v in state_dict.items():
                            # 跳过 optimizer 和 scheduler 相关的键
                            if not any(x in k for x in ['optimizer', 'scheduler', 'lr_scheduler']):
                                # 移除可能的 'module.' 前缀
                                model_key = k[7:] if k.startswith('module.') else k
                                model_state_dict[model_key] = v.cpu() if hasattr(v, 'cpu') else v
                        
                        torch.save(model_state_dict, ckp)
                        Logger(f"最佳准确率模型已保存: {ckp} (acc={best_accuracy:.4f})", accelerator)
                    except Exception as e:
                        Logger(f"保存模型失败: {e}", accelerator)

        # 基于损失保存模型（仅主进程）
        if accelerator.is_main_process:
            loss_total = total_loss.item()
            if best_loss > loss_total:
                best_loss = loss_total
                ckp = f'{args.logging.save_dir}/sft_{hidden_size}{moe_path}.pth'

                # 使用 DeepSpeed/Accelerate 方式获取模型状态字典（自动处理分片参数）
                try:
                    # accelerator.get_state_dict 会自动收集 DeepSpeed 分片的参数
                    state_dict = accelerator.get_state_dict(model, unwrap=False)
                    # 提取模型权重（排除 optimizer 等）
                    model_state_dict = {}
                    for k, v in state_dict.items():
                        # 跳过 optimizer 和 scheduler 相关的键
                        if not any(x in k for x in ['optimizer', 'scheduler', 'lr_scheduler']):
                            # 移除可能的 'module.' 前缀
                            model_key = k[7:] if k.startswith('module.') else k
                            model_state_dict[model_key] = v.cpu() if hasattr(v, 'cpu') else v
                    
                    torch.save(model_state_dict, ckp)
                    Logger(f"最佳损失模型已保存: {ckp} (loss={best_loss:.4f})", accelerator)
                except Exception as e:
                    Logger(f"保存模型失败: {e}", accelerator)

    # 关闭进度条
    if pbar is not None:
        pbar.close()
