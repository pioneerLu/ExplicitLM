"""
Qwen3ExplicitLMBlock: 基于Qwen3DecoderLayer的记忆增强Transformer块

该模块在Qwen3DecoderLayer的基础上添加了ExplicitLM的记忆库机制：
- 复用Qwen3的标准Attention和MLP
- 在MLP输出后添加记忆检索和融合机制
- 保持与ExplicitLMBlock相同的接口和损失计算
"""

from typing import Dict, Tuple, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Config,
    Cache,
)
from transformers.utils import TransformersKwargs
from typing import Unpack

from models.memory_bank.MemoryGate import MemoryGate
from models.memory_bank.GatedMemoryFusion import GatedMemoryFusion
from models.layers.RMSNorm import RMSNorm


class Qwen3ExplicitLMBlock(nn.Module):
    """
    基于Qwen3DecoderLayer的记忆增强Transformer块
    
    该模块实现了以下核心功能：
    1. 复用Qwen3的标准Attention和MLP
    2. 在MLP输出后添加记忆检索机制
    3. Gumbel-Softmax进行可微分的离散选择
    4. 计算相似度损失和多样性损失以优化记忆选择
    """

    def __init__(self, config: Qwen3Config, layer_idx: int, memory_cfg: dict) -> None:
        """
        初始化Qwen3ExplicitLMBlock

        Args:
            config: Qwen3Config配置对象
            layer_idx: 当前层的ID索引
            memory_cfg: 记忆库相关配置字典，包含knowledge_num, knowledge_dim等
        """
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.memory_cfg = memory_cfg
        self.hidden_size = config.hidden_size
        
        # 复用Qwen3DecoderLayer的核心组件
        self.qwen3_decoder = Qwen3DecoderLayer(config, layer_idx)
        
        # 记忆相关模块（仅在非MOE模式下使用）
        use_moe = memory_cfg.get("use_moe", False)
        if not use_moe:
            # 记忆查询归一化层
            self.memory_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            
            # 记忆门控和融合模块
            # 需要将Qwen3的hidden_size映射到memory_cfg的dim
            memory_cfg_with_dim = memory_cfg.copy()
            memory_cfg_with_dim["dim"] = config.hidden_size  # 使用Qwen3的hidden_size
            self.memory_gate = MemoryGate(memory_cfg_with_dim)
            self.gated_memory_fusion = GatedMemoryFusion(memory_cfg_with_dim)
            
            # Gumbel-Softmax参数
            self.gumbel_temperature = memory_cfg.get("gumbel_temperature", 1.0)
        else:
            self.memory_norm = None
            self.memory_gate = None
            self.gated_memory_fusion = None

    def gumbel_softmax_selection(
        self,
        similarity_scores: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gumbel-Softmax选择机制"""
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(similarity_scores) + 1e-20) + 1e-20)
        logits = (similarity_scores + gumbel_noise) / temperature
        soft_weights = F.softmax(logits, dim=-1)
        if hard:
            _, max_indices = soft_weights.max(dim=-1, keepdim=True)
            hard_weights = torch.zeros_like(soft_weights).scatter_(-1, max_indices, 1.0)
            selection_weights = hard_weights - soft_weights.detach() + soft_weights
            selected_indices = max_indices.squeeze(-1)
        else:
            selection_weights = soft_weights
            selected_indices = torch.argmax(soft_weights, dim=-1)
        return selection_weights, selected_indices

    def compute_diversity_loss(self, candidate_memories: torch.Tensor) -> torch.Tensor:
        """计算多样性损失"""
        bsz, seq_len, num_candidates, dim = candidate_memories.shape
        normalized_memories = F.normalize(candidate_memories, p=2, dim=-1)
        similarity_matrix = torch.matmul(normalized_memories, normalized_memories.transpose(-2, -1))
        mask = torch.eye(num_candidates, device=candidate_memories.device).bool()
        mask = mask.unsqueeze(0).unsqueeze(0).expand(bsz, seq_len, -1, -1)
        off_diagonal_similarities = similarity_matrix.masked_select(~mask)
        avg_similarity = off_diagonal_similarities.mean()
        diversity_loss = avg_similarity
        return diversity_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        memory_bank: Optional[torch.Tensor] = None,
        tok_embeddings: Optional[nn.Embedding] = None,
        collect_ema_stats: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, Union[torch.Tensor, float]]],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float], Dict[str, Union[torch.Tensor, float]], Dict[str, Union[torch.Tensor, float]]],
    ]:
        """
        前向传播
        
        Args:
            hidden_states: 输入隐藏状态
            memory_bank: 记忆库，形状为 [knowledge_num, knowledge_length]
            tok_embeddings: token嵌入层，用于解码记忆库中的token
            collect_ema_stats: 是否收集EMA统计信息
            其他参数：Qwen3DecoderLayer的标准参数
        
        Returns:
            如果collect_ema_stats=True: (output, sim_loss, div_loss, layer_stats, ema_stats, cosine_stats)
            否则: (output, sim_loss, div_loss, layer_stats, cosine_stats)
        """
        use_moe = self.memory_cfg.get("use_moe", False)
        
        # ===== 第一阶段：Qwen3标准流程 =====
        # 通过Qwen3DecoderLayer处理（Attention + MLP）
        hidden_states = self.qwen3_decoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        
        if use_moe:
            # MOE模式：不需要记忆库机制
            similarity_loss = torch.tensor(0.0, device=hidden_states.device)
            diversity_loss = torch.tensor(0.0, device=hidden_states.device)
            layer_stats = {}
            cosine_stats = {}
            ema_stats = None
            
            if collect_ema_stats:
                return hidden_states, similarity_loss, diversity_loss, layer_stats, ema_stats, cosine_stats
            else:
                return hidden_states, similarity_loss, diversity_loss, layer_stats, cosine_stats
        else:
            # ===== 第二阶段：记忆库模式 =====
            # 准备记忆查询
            h_for_memory = self.memory_norm(hidden_states)
            
            # ===== 第三阶段：候选记忆生成 =====
            candidate_indices, _candidate_scores = self.memory_gate(h_for_memory)
            bsz, seq_len, num_candidates = candidate_indices.shape
            
            # ===== 第四阶段：候选记忆解码 =====
            # 优化内存：分批处理候选记忆，避免一次性处理所有候选
            candidate_indices_flat = candidate_indices.view(-1)
            candidate_token_ids = memory_bank[candidate_indices_flat]  # [bsz*seq_len*num_candidates, knowledge_length]
            
            # 分批处理embedding，避免OOM（减小批次大小以适应显存）
            batch_size_embed = 2048  # 每批处理的候选数量
            num_total_candidates = candidate_token_ids.shape[0]
            candidate_embeddings_list = []
            
            for i in range(0, num_total_candidates, batch_size_embed):
                end_idx = min(i + batch_size_embed, num_total_candidates)
                batch_token_ids = candidate_token_ids[i:end_idx]
                batch_embeddings = tok_embeddings(batch_token_ids)  # [batch_size, knowledge_length, hidden_size]
                batch_memories = batch_embeddings.mean(dim=1)  # [batch_size, hidden_size]
                candidate_embeddings_list.append(batch_memories)
                # 及时释放中间变量
                del batch_embeddings, batch_token_ids
            
            # 合并所有批次的记忆
            candidate_memories_flat = torch.cat(candidate_embeddings_list, dim=0)  # [bsz*seq_len*num_candidates, hidden_size]
            candidate_memories = candidate_memories_flat.view(bsz, seq_len, num_candidates, self.hidden_size)
            # 释放中间变量
            del candidate_embeddings_list, candidate_memories_flat, candidate_token_ids, candidate_indices_flat
            
            # ===== 第五阶段：相似度计算 =====
            h_expanded = h_for_memory.unsqueeze(2).expand(-1, -1, num_candidates, -1)
            similarity_scores = F.cosine_similarity(h_expanded, candidate_memories, dim=-1)
            
            # ===== 第六阶段：Gumbel-Softmax选择 =====
            selection_weights, selected_indices = self.gumbel_softmax_selection(
                similarity_scores, temperature=self.gumbel_temperature, hard=True
            )
            
            # ===== 第七阶段：损失计算 =====
            selected_similarities = (similarity_scores * selection_weights).sum(dim=-1)
            similarity_loss = -selected_similarities.mean()
            diversity_loss = self.compute_diversity_loss(candidate_memories)
            
            # ===== 第八阶段：记忆融合 =====
            selected_memory = (candidate_memories * selection_weights.unsqueeze(-1)).sum(dim=2)
            memory_output = self.gated_memory_fusion(h_for_memory, selected_memory)
            
            # ===== 第九阶段：输出生成 =====
            out = hidden_states + memory_output
            
            # ===== 第十阶段：统计信息收集 =====
            layer_stats = self._compute_selection_stats(candidate_indices, selection_weights)
            cosine_stats = {
                "similarity_scores": similarity_scores,
                "selected_similarities": selected_similarities,
                "avg_similarity": similarity_scores.mean().item(),
                "max_similarity": similarity_scores.max().item(),
                "min_similarity": similarity_scores.min().item(),
                "selected_avg_similarity": selected_similarities.mean().item(),
                "selection_entropy": -torch.sum(
                    selection_weights * torch.log(selection_weights + 1e-10), dim=-1
                ).mean().item(),
            }
            
            ema_stats = None
            if collect_ema_stats and self.training:
                selected_memory_indices = candidate_indices.gather(2, selected_indices.unsqueeze(-1))
                ema_stats = {
                    "memory_indices": selected_memory_indices,
                    "memory_scores": torch.ones_like(selected_memory_indices.float()),
                    "h_for_memory": h_for_memory,
                    "selected_memory": selected_memory.unsqueeze(2),
                }
            
            if collect_ema_stats:
                return out, similarity_loss, diversity_loss, layer_stats, ema_stats, cosine_stats
            else:
                return out, similarity_loss, diversity_loss, layer_stats, cosine_stats

    def _compute_selection_stats(
        self,
        candidate_indices: torch.Tensor,
        selection_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """计算选择统计信息"""
        device = candidate_indices.device
        flat_indices = candidate_indices.view(-1)
        flat_weights = selection_weights.view(-1)
        knowledge_num = self.memory_cfg["knowledge_num"]
        # 确保memory_counts和flat_weights的数据类型一致（用于scatter_add）
        memory_counts = torch.zeros(knowledge_num, device=device, dtype=flat_weights.dtype)
        memory_counts.scatter_add_(0, flat_indices, flat_weights)
        with torch.no_grad():
            # 转换为float32进行统计计算（quantile等操作需要float/double类型）
            memory_counts_fp32 = memory_counts.float()
            coverage_rate = (memory_counts_fp32 > 0.01).float().mean().item()
            top10_threshold = torch.quantile(memory_counts_fp32, 0.9)
            hot_memories = (memory_counts_fp32 >= top10_threshold).sum().item()
            dead_memories = (memory_counts_fp32 < 0.01).sum().item()
            selection_variance = memory_counts_fp32.var().item()
            stats = {
                "coverage_rate": coverage_rate,
                "hot_memories": hot_memories,
                "dead_memories": dead_memories,
                "selection_variance": selection_variance,
                "max_selections": memory_counts_fp32.max().item(),
                "min_selections": memory_counts_fp32.min().item(),
            }
        return stats

