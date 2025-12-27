import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedMemoryFusion(nn.Module):
    """
    简化的记忆融合模块（每层独立）
    
    使用简化的融合方式：h + memory -> Linear -> output
    参数量大幅减少（从 32.77M/层 降到 6.55M/层），同时保持每层独立性。
    公式：out = alpha * Linear(h + memory)
    """

    def __init__(self, cfg: dict) -> None:
        """
        Args:
            cfg: 模型配置字典
        """
        super().__init__()
        self.cfg = cfg
        self.dim = cfg["dim"]

        # 实际使用 dim (即 hidden_size) 作为记忆嵌入维度
        self.knowledge_dim = cfg.get("knowledge_dim", self.dim)  # 兼容性保留
        self.num_selected = cfg.get("num_selected", 1)

        self.fusion_proj = nn.Linear(self.dim, self.dim, bias=False)
        
        # 可学习的缩放因子（每层可以学习不同的缩放）
        self.alpha = nn.Parameter(torch.ones(1))
        
        # Dropout
        self.dropout = nn.Dropout(cfg.get("dropout", 0.1))

    def forward(
        self,   
        h_attn: torch.Tensor,
        selected_memory: torch.Tensor,
        similarity_scores: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            h_attn: 自注意力输出 [batch_size, seq_len, dim]
            selected_memory: 选中的记忆 [batch_size, seq_len, dim]
            similarity_scores: 相似度分数 [batch_size, seq_len]（可选，用于可选的相似度加权）
        Returns:
            memory_output: 记忆融合输出 [batch_size, seq_len, dim]
        """
        # 简化融合：直接相加后投影
        fused = h_attn + selected_memory
        memory_output = self.fusion_proj(fused)
        
        # 可选的相似度加权（如果提供了 similarity_scores）
        if similarity_scores is not None:
            # 使用相似度分数作为额外的加权因子
            # 高相似度 → 更多 memory 贡献
            # similarity_scores: [batch_size, seq_len]
            similarity_alpha = torch.sigmoid(similarity_scores).unsqueeze(-1)  # [batch_size, seq_len, 1]
            memory_output = similarity_alpha * memory_output
        
        # 应用可学习的缩放因子和 dropout
        memory_output = self.dropout(self.alpha * memory_output)
        
        return memory_output