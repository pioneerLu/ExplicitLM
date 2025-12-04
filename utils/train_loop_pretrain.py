"""
训练循环模块

功能：
- train_epoch: 单个epoch的训练循环
- 支持梯度累积、验证评估、模型保存
- 集成SwanLab实验追踪
"""

import os
import json
import time
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

from utils.logger import Logger
from utils.train_utils import validate_model, format_time

try:
    import swanlab
except ImportError:
    swanlab = None


def train_epoch(
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
    val_loader: Optional[DataLoader] = None,
    resume_step: int = 0  # [新增] 接收需要跳过的步数
) -> None:
    """
    单个epoch的训练循环

    参数：
        epoch: 当前epoch编号（从0开始）
        accelerator: Accelerator实例
        model: 训练模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        args: 配置参数
        overall_start_time: 整体训练开始时间
        swanlab_run: SwanLab运行实例
        tokenizer: Tokenizer实例
        val_loader: 验证数据加载器（可选）

    说明：
        混合精度由DeepSpeed配置文件（ds_config.json）自动控制
        - bf16已启用，无需手动创建autocast上下文
        - 梯度累积由DeepSpeed自动处理
        - 梯度裁剪由DeepSpeed自动处理
    """
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    epoch_start_time = time.time()
    
    # 计算总步数信息
    total_steps_in_epoch = len(train_loader)
    total_training_steps = args.training.epochs * total_steps_in_epoch
    
    moe_path = '_moe' if args.model.use_moe else ''
    best_loss = float('inf') # 注意：这里best_loss是epoch内局部最优，如果需要全局最优需要在外部维护并传入

    # [新增] 断点续训：跳过已训练的 batches
    if resume_step > 0:
        train_loader = accelerator.skip_first_batches(train_loader, num_batches=resume_step)
        if accelerator.is_main_process:
            Logger(f"Epoch {epoch}: 已跳过前 {resume_step} 个 batches 以实现续训", accelerator)

    # 记录初始状态
    last_log_time = time.time()

    # 使用 enumerate 获取当前循环的索引 step_idx
    for step_idx, (X, Y, loss_mask) in enumerate(train_loader):
        # [新增] 计算当前 epoch 内的真实 step 和全局 step
        current_step = step_idx + resume_step
        global_step = epoch * total_steps_in_epoch + current_step + 1

        # 更新学习率
        if scheduler is not None:
            scheduler.step()

        # 前向传播（DeepSpeed自动处理bf16混合精度）
        # 第一个epoch的embedding冻结处理 (使用 current_step 判断)
        if current_step == 0 and args.training.embeddings_epoch == epoch:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.freeze_embedding = True
            Logger(f"设置freeze_embedding=True (epoch {epoch}, step {current_step})", accelerator)

        res = model(X, step=current_step) # 传入正确的 step

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
        similarity_coef = getattr(args, 'similarity_loss_coef', 0.1)
        diversity_coef = getattr(args, 'diversity_loss_coef', 0.05)

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

        # ============================================================
        # [新增] 机制1：每 500 个 Global Step 保存一次完整 Checkpoint (用于续训)
        # ============================================================
        SAVE_INTERVAL = 500  # 可以改为从 args 传入: args.training.save_interval
        if global_step % SAVE_INTERVAL == 0:
            save_dir = os.path.join(args.logging.save_dir, f"checkpoint_step_{global_step}")
            if accelerator.is_main_process:
                os.makedirs(save_dir, exist_ok=True)
            
            # 等待所有进程，确保安全保存
            accelerator.wait_for_everyone()
            # 保存完整状态 (模型、优化器、LR调度器等)
            accelerator.save_state(save_dir)
            
            # 仅主进程写入元数据，记录精确的恢复位置
            if accelerator.is_main_process:
                with open(os.path.join(save_dir, "training_state.json"), "w") as f:
                    # 记录当前完成的 step，恢复时应从 current_step + 1 开始
                    json.dump({
                        "epoch": epoch, 
                        "step": current_step, 
                        "global_step": global_step
                    }, f)
                Logger(f"🔥 [Checkpoint] Step {global_step} 完整状态已保存至 {save_dir}", accelerator)

        # ============================================================
        # 验证评估和日志记录（仅主进程）
        # ============================================================
        if (current_step + 1) % args.logging.log_interval == 0 and accelerator.is_main_process:
            current_time = time.time()
            current_lr = optimizer.param_groups[0]['lr']

            # 时间估算
            epoch_elapsed = current_time - epoch_start_time
            # 当前epoch已完成的step数 (包含跳过的)
            epoch_steps_done = current_step + 1
            # 注意：如果跳过了很多步，初期估算可能不准，但会迅速收敛
            epoch_avg_time = epoch_elapsed / (epoch_steps_done - resume_step) if (epoch_steps_done - resume_step) > 0 else 0
            epoch_remaining = epoch_avg_time * (total_steps_in_epoch - epoch_steps_done)

            total_elapsed = current_time - overall_start_time
            total_steps_done = global_step
            total_avg_time = total_elapsed / total_steps_done if total_steps_done > 0 else 0
            total_remaining = total_avg_time * (total_training_steps - total_steps_done)

            # 计算训练速度
            interval_time = current_time - last_log_time
            tokens_processed = args.logging.log_interval * args.training.batch_size * args.model.max_seq_len
            tokens_per_sec = tokens_processed / interval_time if interval_time > 0 else 0
            last_log_time = current_time

            # 执行验证评估
            val_loss = None
            if val_loader is not None:
                try:
                    val_loss = validate_model(model, val_loader, loss_fct, accelerator)
                    Logger(f"验证损失: {val_loss:.4f}", accelerator)
                except Exception as e:
                    Logger(f"验证评估失败: {e}", accelerator)
                    val_loss = None

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
                "step": current_step + 1,
                "global_step": global_step,
                "train/loss_ce": ce_loss.item(),
                "train/loss_similarity": similarity_loss.item() if isinstance(similarity_loss, torch.Tensor) else similarity_loss,
                "train/loss_diversity": diversity_loss.item() if isinstance(diversity_loss, torch.Tensor) else diversity_loss,
                "train/loss_total": total_loss.item(),
                "lr": current_lr,
                "tokens_per_sec": tokens_per_sec,
                "epoch_time_left": epoch_remaining,
                "total_time_left": total_remaining,
                "train/avg_selected_similarity": avg_selected_similarity,
            }

            # 添加验证损失
            if val_loss is not None:
                log_dict["val/loss"] = val_loss

            # 添加记忆库更新统计
            log_dict.update(memory_update_stats)

            # 控制台输出
            Logger(
                f"Epoch {epoch+1}/{args.training.epochs} | Step {current_step+1}/{total_steps_in_epoch} (Global {global_step}) | "
                f"Loss: {total_loss.item():.4f} | Val: {log_dict.get('val/loss', 'N/A')} | "
                f"CE: {ce_loss.item():.4f} | Sim: {similarity_loss.item():.4f} | Div: {diversity_loss.item():.4f} | "
                f"Speed: {tokens_per_sec:.0f} tok/s | "
                f"ETA Epoch: {format_time(epoch_remaining)}",
                accelerator
            )

            # SwanLab日志记录
            if args.logging.use_swanlab and swanlab_run:
                swanlab_run.log(log_dict)

        # ============================================================
        # [原有] 机制2：保存当前 Epoch 内最佳权重 (用于推理)
        # ============================================================
        # 注意：这里仅在主进程执行，且只保存权重(state_dict)，不包含优化器状态
        if accelerator.is_main_process:
            current_loss_total = loss.item() * args.training.accumulation_steps
            if best_loss > current_loss_total:
                best_loss = current_loss_total
                # 构造保存路径，建议加上 epoch 以免不同 epoch 的最佳模型互相覆盖(可选)
                # 原路径: f'{args.logging.save_dir}/pretrain_{args.model.dim}{moe_path}.pth'
                # 建议改进路径:
                ckp_best = f'{args.logging.save_dir}/pretrain_{args.model.dim}_epoch{epoch}{moe_path}_best.pth'

                unwrapped_model = accelerator.unwrap_model(model)
                accelerator.save(unwrapped_model.state_dict(), ckp_best)
                # Logger(f"🌟 新最佳模型 (Loss {best_loss:.4f}) 已保存至 {ckp_best}", accelerator) 
                # 注：如果每个step都打印可能会太多，可以考虑只在 log_interval 时打印
