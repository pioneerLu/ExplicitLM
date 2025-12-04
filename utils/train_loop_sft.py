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
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

from utils.logger import Logger
from utils.train_utils import format_time

try:
    import swanlab
except ImportError:
    swanlab = None


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
            # DataLoader返回的batch格式：当batch_size=1时，可能是：
            # - tuple: (prompt_list, target_list) 其中每个都是长度为1的list
            # - 或者直接是 (prompt, target) tuple
            
            # 处理不同的batch格式
            if isinstance(batch, tuple) and len(batch) == 2:
                prompt_input, std_output = batch
                
                # 如果prompt_input是list/tuple，取第一个元素
                if isinstance(prompt_input, (list, tuple)):
                    if len(prompt_input) > 0:
                        prompt_input = prompt_input[0]
                    else:
                        Logger(f"警告: prompt_input 为空，跳过 step {step}", accelerator)
                        continue
                
                # 如果std_output是list/tuple，取第一个元素
                if isinstance(std_output, (list, tuple)):
                    if len(std_output) > 0:
                        std_output = std_output[0]
                    else:
                        Logger(f"警告: std_output 为空，跳过 step {step}", accelerator)
                        continue
                
                # 确保prompt_input是字符串
                if not isinstance(prompt_input, str):
                    Logger(f"警告: prompt_input 不是字符串类型，跳过 step {step}", accelerator)
                    continue
                    
                # 确保std_output是字符串
                if not isinstance(std_output, str):
                    Logger(f"警告: std_output 不是字符串类型，跳过 step {step}", accelerator)
                    continue
            else:
                Logger(f"警告: 评估数据格式错误，跳过 step {step}", accelerator)
                continue

            # 截断prompt_input到最大长度
            # 注意：这里假设prompt_input已经是完整的字符串，不需要再处理
            # 如果需要截断，应该在tokenization之前进行

            # Tokenize输入
            try:
                x = torch.tensor(
                    tokenizer(prompt_input)['input_ids'],
                    device=accelerator.device
                ).unsqueeze(0)
            except Exception as e:
                Logger(f"警告: Tokenization 失败 (step {step}): {e}", accelerator)
                continue

            # 生成文本（需要 unwrap 访问自定义方法）
            unwrapped_model = accelerator.unwrap_model(model)

            try:
                generated = unwrapped_model.generate(
                    x,
                    max_new_tokens=getattr(args.training, 'max_new_tokens', 50),
                    max_length=args.model.max_seq_len + getattr(args.training, 'max_new_tokens', 50),
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id
                )
            except Exception as e:
                Logger(f"警告: 生成失败 (step {step}): {e}", accelerator)
                continue

            # 解码生成的文本
            try:
                generated_text = tokenizer.decode(
                    generated.squeeze()[x.shape[1]:].tolist(),
                    skip_special_tokens=True
                )
            except Exception as e:
                Logger(f"警告: 解码失败 (step {step}): {e}", accelerator)
                continue

            # 判断是否匹配
            generated_text = generated_text.strip()
            flag = judger(generated_text, std_output, judger_mode)

            # 保存样本结果
            performance[step] = {
                'prompt': prompt_input,
                'std_output': std_output,
                'generated_text': generated_text,
                'result': flag,
                'judger_mode': judger_mode
            }

            # 累积统计
            total_correct += int(flag)
            total_steps += 1

            # 定期打印评估结果样例
            if accelerator.is_main_process and step % max(1, len(eval_loader) // getattr(args.training, 'show_eval_res', 5)) == 0:
                Logger(
                    f"评估样例 Step {step}: "
                    f"生成='{generated_text[:30]}...' | "
                    f"标准='{std_output[:30]}...' | "
                    f"匹配={flag}",
                    accelerator
                )

    # 计算整体性能
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

    # 恢复训练模式（在 wrapped model 上操作）
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
    eval_loader: Optional[DataLoader] = None
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
    best_loss = float('inf')
    best_accuracy = 0.0

    # 记录初始状态
    last_log_time = epoch_start_time

    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 前向传播（DeepSpeed自动处理bf16混合精度）
        res = model(X, step=step)

        # 计算主要损失（交叉熵损失）
        ce_loss = loss_fct(
            res.logits.view(-1, res.logits.size(-1)),
            Y.view(-1)
        ).view(Y.size())
        ce_loss = (ce_loss * loss_mask).sum() / loss_mask.sum()

        # 处理辅助损失
        similarity_loss = 0
        diversity_loss = 0

        if hasattr(res, 'aux_loss') and res.aux_loss is not None:
            aux_loss = res.aux_loss
            if isinstance(aux_loss, dict):
                # 三损失结构
                similarity_loss = aux_loss.get('similarity_loss', 0)
                diversity_loss = aux_loss.get('diversity_loss', 0)

                # 分布式训练中的损失聚合
                if isinstance(similarity_loss, torch.Tensor):
                    similarity_loss = accelerator.gather(similarity_loss).mean()
                if isinstance(diversity_loss, torch.Tensor):
                    diversity_loss = accelerator.gather(diversity_loss).mean()

        # 三损失系统：CE + Similarity + Diversity
        similarity_coef = getattr(args.training, 'similarity_loss_coef', 0.1)
        diversity_coef = getattr(args.training, 'diversity_loss_coef', 0.05)

        total_loss = (
            ce_loss +
            similarity_coef * similarity_loss +
            diversity_coef * diversity_loss
        )
        loss = total_loss / args.training.accumulation_steps

        # 反向传播
        accelerator.backward(loss)

        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()

        # Memory bank在训练时固定，推理时通过LLMLingua更新，不再需要EMA更新

        # 训练日志记录（仅主进程）
        if (step + 1) % args.logging.log_interval == 0 and accelerator.is_main_process:
            current_time = time.time()

            # 计算当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 计算时间估算
            epoch_elapsed_time = current_time - epoch_start_time
            epoch_steps_done = step + 1
            epoch_avg_step_time = epoch_elapsed_time / epoch_steps_done
            epoch_remaining_time = epoch_avg_step_time * (total_steps_in_epoch - epoch_steps_done)

            total_elapsed_time = current_time - overall_start_time
            total_steps_done = epoch * total_steps_in_epoch + epoch_steps_done
            total_avg_step_time = total_elapsed_time / total_steps_done if total_steps_done > 0 else 0
            total_remaining_time = total_avg_step_time * (total_training_steps - total_steps_done) if total_steps_done > 0 else 0

            # 计算训练速度
            interval_elapsed_time = current_time - last_log_time
            tokens_processed_interval = args.logging.log_interval * args.training.batch_size * args.model.max_seq_len
            tokens_per_sec = tokens_processed_interval / interval_elapsed_time if interval_elapsed_time > 0 else 0
            last_log_time = current_time

            # 获取记忆库更新统计（如果模型支持）
            memory_update_stats = {}
            unwrapped_model = accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, 'get_memory_update_stats'):
                try:
                    memory_update_stats = unwrapped_model.get_memory_update_stats()
                except Exception as e:
                    Logger(f"获取记忆更新统计失败: {e}", accelerator)

            # 获取余弦相似度统计
            avg_selected_similarity = 0.0
            if hasattr(res, 'cosine_stats') and res.cosine_stats is not None:
                cosine_stats = res.cosine_stats
                selected_similarities = [
                    v for k, v in cosine_stats.items()
                    if k.endswith('_selected_avg_similarity')
                ]
                if selected_similarities:
                    import numpy as np
                    avg_selected_similarity = np.mean(selected_similarities)

            # 构建日志字典
            log_dict = {
                "epoch": epoch + 1,
                "step": step + 1,
                "total_steps_in_epoch": total_steps_in_epoch,
                "train/loss_ce": ce_loss.item(),
                "train/loss_similarity": similarity_loss.item() if isinstance(similarity_loss, torch.Tensor) else similarity_loss,
                "train/loss_diversity": diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
                "train/loss_total": total_loss.item(),
                "lr": current_lr,
                "tokens_per_sec": tokens_per_sec,
                "epoch_time_left_seconds": epoch_remaining_time,
                "total_time_left_seconds": total_remaining_time,
                "train/avg_selected_similarity": avg_selected_similarity,
            }

            # 添加记忆库更新统计
            log_dict.update(memory_update_stats)

            Logger(
                f"Epoch {epoch+1}/{args.training.epochs}, Step {step+1}/{total_steps_in_epoch} | "
                f"Loss: {log_dict['train/loss_total']:.4f} (CE:{log_dict['train/loss_ce']:.4f} "
                f"Sim:{log_dict['train/loss_similarity']:.4f} Div:{log_dict['train/loss_diversity']:.4f}) | "
                f"LR: {log_dict['lr']:.6f} | Speed: {log_dict['tokens_per_sec']:.0f} tok/s",
                accelerator
            )

            # SwanLab日志记录
            if args.logging.use_swanlab and swanlab_run:
                swanlab_run.log(log_dict)

        # SFT生成式评估（定期执行）
        eval_interval = getattr(args.training, 'eval_interval', 1000)
        start_eval = getattr(args.training, 'start_eval', 100)

        if (step + 1) % eval_interval == 0:
            if (step + 1) >= start_eval and eval_loader is not None:
                if accelerator.is_main_process:
                    Logger(f"开始评估...", accelerator)

                performance = eval_model_sft(
                    model=model,
                    eval_loader=eval_loader,
                    tokenizer=tokenizer,
                    accelerator=accelerator,
                    args=args,
                    judger_mode=getattr(args.training, 'judger_mode', 'startswith')
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
                        ckp = f'{args.logging.save_dir}/sft_best_acc_{args.model.dim}{moe_path}.pth'
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), ckp)
                        Logger(f"最佳准确率模型已保存: {ckp} (acc={best_accuracy:.4f})", accelerator)

        # 基于损失保存模型（仅主进程）
        if accelerator.is_main_process:
            loss_total = loss.item() * args.training.accumulation_steps
            if best_loss > loss_total:
                best_loss = loss_total
                ckp = f'{args.logging.save_dir}/sft_{args.model.dim}{moe_path}.pth'

                # 获取解包后的模型
                unwrapped_model = accelerator.unwrap_model(model)

                # 保存模型参数
                accelerator.save(unwrapped_model.state_dict(), ckp)
                Logger(f"最佳损失模型已保存: {ckp} (loss={best_loss:.4f})", accelerator)
