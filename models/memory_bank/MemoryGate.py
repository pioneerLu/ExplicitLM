import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class MemoryGate(nn.Module):
    """
    MemoryGate with direct RAG similarity search.
    Computes cosine similarity between query and all memory bank entries.
    """
    def __init__(self, cfg: dict, backbone=None, tokenizer=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone  # Shared backbone (frozen or not)
        self.tokenizer = tokenizer # Shared tokenizer
        
        # Dimensions
        self.input_dim = cfg.get("dim", 2560)          # x dimension (from backbone)
        self.query_dim = cfg.get("query_dim", 1024)    # query dimension
        self.num_candidates = cfg.get("num_candidates", 32)
        
        self.knowledge_num = cfg.get("knowledge_num", 100*100)
        
        # Query Projection: input_dim -> query_dim
        self.query_proj = nn.Linear(self.input_dim, self.query_dim, bias=False)
        
        # 相对基线损失函数配置
        self.use_relative_baseline_loss = cfg.get("use_relative_baseline_loss", True)
        self.baseline_size = cfg.get("baseline_size", 100)  # 随机采样基线样本数量
        self.baseline_aggregation = cfg.get("baseline_aggregation", "soft_max")  # "mean" 或 "soft_max"
        self.baseline_temperature = cfg.get("baseline_temperature", 0.1)  # 用于 soft_max 的温度
        self.margin_temperature = cfg.get("margin_temperature", 1.0)  # 控制 soft margin 的软度
        self.loss_type = cfg.get("relative_loss_type", "softplus")  # "softplus" 或 "sigmoid"
        self.exclude_target_from_baseline = cfg.get("exclude_target_from_baseline", True)  # 是否从基线中排除目标样本
        
        # 批量处理 memory embeddings 的批次大小（用于节省显存）
        self.embed_batch_size = cfg.get("embed_batch_size", 64)

    def forward(
        self, 
        x: torch.Tensor, 
        memory_bank: torch.Tensor,
        tok_embeddings: nn.Embedding,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Direct RAG similarity search: compute similarity between query and all memory entries.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            memory_bank: Memory bank token IDs [knowledge_num, knowledge_length]
            tok_embeddings: Token embedding layer
            valid_mask: Valid mask [knowledge_num] (optional)
            
        Returns:
            candidate_indices: [batch, seq_len, num_candidates]
            candidate_scores: [batch, seq_len, num_candidates]
        """
        bsz, seq_len, _ = x.shape
        device = x.device
        knowledge_num = memory_bank.shape[0]
        
        # 1. Project Query (mean pooling first)
        x_mean = x.mean(dim=1)  # [batch, input_dim]
        query = self.query_proj(x_mean)  # [batch, query_dim]
        query_normalized = F.normalize(query, p=2, dim=-1)  # [batch, query_dim]
        
        # 2. Compute all memory embeddings (batch processing for memory efficiency)
        memory_embeddings = self._compute_memory_embeddings(
            memory_bank, tok_embeddings, device
        )  # [knowledge_num, query_dim]
        
        # 3. Compute cosine similarity: query [batch, query_dim] @ memory [knowledge_num, query_dim].T
        # similarity: [batch, knowledge_num]
        similarity = torch.matmul(query_normalized, memory_embeddings.t())  # [batch, knowledge_num]
        
        # 4. Apply valid_mask if provided
        if valid_mask is not None:
            if valid_mask.device != device:
                valid_mask = valid_mask.to(device)
            # Mask invalid entries with -inf
            similarity = similarity.masked_fill(~valid_mask.unsqueeze(0), float('-inf'))
        
        # 5. Expand to [batch, seq_len, knowledge_num] for compatibility
        similarity = similarity.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, knowledge_num]
        
        # 6. Top-K selection
        candidate_scores, candidate_indices = similarity.topk(
            self.num_candidates, dim=-1
        )  # [batch, seq_len, num_candidates]
        
        return candidate_indices, candidate_scores
    
    def _compute_memory_embeddings(
        self,
        memory_bank: torch.Tensor,
        tok_embeddings: nn.Embedding,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute embeddings for all memory bank entries.
        Uses batch processing to save memory.
        
        Args:
            memory_bank: Memory bank token IDs [knowledge_num, knowledge_length]
            tok_embeddings: Token embedding layer
            device: Target device
            
        Returns:
            memory_embeddings: [knowledge_num, query_dim]
        """
        knowledge_num = memory_bank.shape[0]
        pad_token_id = 0  # Default pad token ID
        
        # Move memory_bank to device if needed
        if memory_bank.device != device:
            memory_bank = memory_bank.to(device, non_blocking=True)
        
        all_embeddings = []
        
        # Batch processing
        for i in range(0, knowledge_num, self.embed_batch_size):
            end_idx = min(i + self.embed_batch_size, knowledge_num)
            batch_token_ids = memory_bank[i:end_idx]  # [batch_size, knowledge_length]
            
            # Get embeddings
            batch_embeddings = tok_embeddings(batch_token_ids)  # [batch_size, knowledge_length, hidden_size]
            
            # Mean pooling (considering pad tokens)
            attention_mask = (batch_token_ids != pad_token_id).long()  # [batch_size, knowledge_length]
            mask = attention_mask.unsqueeze(-1).to(dtype=batch_embeddings.dtype)  # [batch_size, knowledge_length, 1]
            sum_hidden = (batch_embeddings * mask).sum(dim=1)  # [batch_size, hidden_size]
            len_hidden = mask.sum(dim=1).clamp(min=1e-6)  # [batch_size, 1]
            batch_mean = sum_hidden / len_hidden  # [batch_size, hidden_size]
            
            # Project to query_dim
            batch_query = self.query_proj(batch_mean)  # [batch_size, query_dim]
            batch_normalized = F.normalize(batch_query, p=2, dim=-1)  # [batch_size, query_dim]
            
            all_embeddings.append(batch_normalized)
        
        # Concatenate all embeddings
        memory_embeddings = torch.cat(all_embeddings, dim=0)  # [knowledge_num, query_dim]
        
        return memory_embeddings

    def compute_loss_single_target(
        self,
        query_embedding: torch.Tensor,
        selected_knowledge_embedding: torch.Tensor,
        memory_bank: torch.Tensor,
        selected_index: torch.Tensor,
        tok_embeddings: nn.Embedding,
        valid_mask: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """
        计算相对基线损失：基于实际 embedding 的相似度
        
        Args:
            query_embedding: 查询的 embedding [batch, seq_len, hidden_size]
            selected_knowledge_embedding: 选中知识的 embedding [batch, seq_len, hidden_size]
            memory_bank: 知识库 [knowledge_num, knowledge_length]
            selected_index: 选中的知识索引 [batch, seq_len]
            tok_embeddings: token embedding 层
            valid_mask: 有效掩码 [knowledge_num] (可选)
            pad_token_id: pad token ID
        
        Returns:
            loss: scalar tensor
        """
        # 1. 计算 query 与选中知识的相似度
        query_normalized = F.normalize(query_embedding, p=2, dim=-1)  # [batch, seq_len, hidden_size]
        selected_normalized = F.normalize(selected_knowledge_embedding, p=2, dim=-1)  # [batch, seq_len, hidden_size]
        sim_pos = torch.einsum('bsd,bsd->bs', query_normalized, selected_normalized)  # [batch, seq_len]
        
        # 2. 从知识库中随机采样基线样本，计算 query 与基线知识的相似度
        baseline = self._compute_baseline_similarity(
            query_normalized,
            memory_bank,
            selected_index,
            tok_embeddings,
            valid_mask,
            pad_token_id,
        )
        
        # 3. 计算相对基线损失
        return self._compute_relative_baseline_loss(sim_pos, baseline)
    
    def _compute_baseline_similarity(
        self,
        query_normalized: torch.Tensor,
        memory_bank: torch.Tensor,
        selected_index: torch.Tensor,
        tok_embeddings: nn.Embedding,
        valid_mask: Optional[torch.Tensor],
        pad_token_id: int,
    ) -> torch.Tensor:
        """
        计算基线相似度：从知识库中随机采样基线样本，计算 query 与基线知识的相似度
        
        Args:
            query_normalized: 归一化的查询 embedding [batch, seq_len, hidden_size]
            memory_bank: 知识库 [knowledge_num, knowledge_length]
            selected_index: 选中的知识索引 [batch, seq_len]
            tok_embeddings: token embedding 层
            valid_mask: 有效掩码 [knowledge_num] (可选)
            pad_token_id: pad token ID
        
        Returns:
            baseline: 基线相似度 [batch, seq_len]
        """
        bsz, seq_len, hidden_size = query_normalized.shape
        device = query_normalized.device
        knowledge_num = memory_bank.shape[0]
        
        # 随机采样基线样本索引
        baseline_size = min(self.baseline_size, knowledge_num)
        
        # 为所有位置统一采样基线
        sampled_indices = torch.randint(
            0, knowledge_num,
            (baseline_size,),
            device=device,
            generator=None
        )  # [baseline_size]
        
        # 如果 exclude_target=True，需要排除目标本身
        if self.exclude_target_from_baseline:
            # 获取所有唯一的目标索引
            unique_targets = torch.unique(selected_index)  # [num_unique_targets]
            # 从采样中排除目标
            mask = torch.ones(baseline_size, dtype=torch.bool, device=device)
            for target_idx in unique_targets:
                mask = mask & (sampled_indices != target_idx)
            # 如果排除后样本不足，补充随机样本
            num_valid = mask.sum().item()
            if num_valid < baseline_size:
                # 补充随机样本
                remaining_size = baseline_size - num_valid
                remaining_candidates = torch.arange(knowledge_num, device=device)
                # 排除所有目标
                for target_idx in unique_targets:
                    remaining_candidates = remaining_candidates[remaining_candidates != target_idx]
                if len(remaining_candidates) > 0:
                    additional_indices = remaining_candidates[torch.randperm(len(remaining_candidates), device=device)[:remaining_size]]
                    sampled_indices = torch.cat([sampled_indices[mask], additional_indices])
                else:
                    sampled_indices = sampled_indices[mask]
            else:
                sampled_indices = sampled_indices[mask]
        
        # 获取基线样本的 token IDs
        baseline_token_ids = memory_bank[sampled_indices]  # [baseline_size, knowledge_length]
        
        # 转换为 embedding（批量处理）
        baseline_token_ids = baseline_token_ids.to(device)
        baseline_embeddings = tok_embeddings(baseline_token_ids)  # [baseline_size, knowledge_length, hidden_size]
        
        baseline_embeddings = baseline_embeddings.to(dtype=query_normalized.dtype)
        
        # Mean pooling（考虑 pad）
        attention_mask = (baseline_token_ids != pad_token_id).long()  # [baseline_size, knowledge_length]
        mask = attention_mask.unsqueeze(-1).to(dtype=baseline_embeddings.dtype)  # [baseline_size, knowledge_length, 1]
        sum_hidden = (baseline_embeddings * mask).sum(dim=1)  # [baseline_size, hidden_size]
        len_hidden = mask.sum(dim=1).clamp(min=1e-6)  # [baseline_size, 1]
        baseline_memories = sum_hidden / len_hidden  # [baseline_size, hidden_size]
        
        # 归一化
        baseline_memories_normalized = F.normalize(baseline_memories, p=2, dim=-1)  # [baseline_size, hidden_size]
        
        # 计算 query 与基线知识的相似度
        # query_normalized: [bsz, seq_len, hidden_size]
        # baseline_memories_normalized: [baseline_size, hidden_size]
        # baseline_similarities: [bsz, seq_len, baseline_size]
        baseline_similarities = torch.einsum('bsd,kd->bsk', query_normalized, baseline_memories_normalized)  # [bsz, seq_len, baseline_size]
        
        # 聚合基线相似度
        if self.baseline_aggregation == "mean":
            baseline = baseline_similarities.mean(dim=-1)  # [bsz, seq_len]
        elif self.baseline_aggregation == "soft_max":
            # Soft max: log_sum_exp(scores / τ) * τ
            scaled_scores = baseline_similarities / self.baseline_temperature  # [bsz, seq_len, baseline_size]
            baseline = torch.logsumexp(scaled_scores, dim=-1) * self.baseline_temperature  # [bsz, seq_len]
        else:
            raise ValueError(f"Unknown baseline_aggregation: {self.baseline_aggregation}")
        
        return baseline
    
    def _compute_relative_baseline_loss(
        self,
        sim_pos: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算相对基线损失
        
        Args:
            sim_pos: query 与选中知识的相似度 [batch, seq_len]
            baseline: query 与基线知识的相似度 [batch, seq_len]
        
        Returns:
            loss: scalar tensor
        """
        # 计算优势
        margin = sim_pos - baseline  # [batch, seq_len]
        
        # Soft margin 损失
        if self.loss_type == "softplus":
            # Softplus: log(1 + exp(-margin / τ))
            scaled_margin = -margin / self.margin_temperature
            loss = F.softplus(scaled_margin)  # [batch, seq_len]
        elif self.loss_type == "sigmoid":
            # Sigmoid: sigmoid(-margin / τ) * scale
            scaled_margin = -margin / self.margin_temperature
            loss = torch.sigmoid(scaled_margin) * self.margin_temperature  # [bsz, seq_len]
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        return loss.mean()
