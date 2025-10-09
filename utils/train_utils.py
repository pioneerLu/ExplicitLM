"""
训练辅助工具模块

功能：
- validate_model: 验证集评估函数
- format_time: 时间格式化工具
"""

from typing import Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator


def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fct: nn.Module,
    accelerator: Accelerator
) -> float:
    """
    在验证集上评估模型

    参数：
        model: 待评估模型
        val_loader: 验证数据加载器
        loss_fct: 损失函数（CrossEntropyLoss）
        accelerator: Accelerator实例

    返回：
        平均验证损失

    说明：
        混合精度由DeepSpeed自动处理，无需手动autocast
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for X, Y, loss_mask in val_loader:
            # DeepSpeed自动处理混合精度，无需手动包装
            res = model(X)

            # 计算交叉熵损失
            ce_loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())

            # 应用loss_mask
            masked_loss = (ce_loss * loss_mask).sum()
            num_tokens = loss_mask.sum()

            # 累积损失和token数
            total_loss += masked_loss.item()
            total_tokens += num_tokens.item()

    model.train()

    # 返回平均损失
    return total_loss / total_tokens if total_tokens > 0 else 0.0


def format_time(seconds: float) -> str:
    """
    将秒数格式化为人类可读的时间字符串

    参数：
        seconds: 秒数

    返回：
        格式化的时间字符串（例如：1h 23m 45s）
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"
