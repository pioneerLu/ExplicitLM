import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union
import os

class MemoryGate(nn.Module):
    """
    Simplified MemoryGate with Product Key Memory.
    Dynamically updates keys every epoch using clustering results.
    """
    def __init__(self, cfg: dict, backbone=None, tokenizer=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone  # Shared backbone (frozen or not)
        self.tokenizer = tokenizer # Shared tokenizer
        
        # Dimensions
        self.input_dim = cfg.get("dim", 2560)          # x dimension (from backbone)
        self.query_dim = cfg.get("query_dim", 1024)    # query dimension
        self.key_proj_dim = cfg.get("key_proj_dim", 512) # dimension for dot product
        self.num_candidates = cfg.get("num_candidates", 32)
        
        self.knowledge_num = cfg.get("knowledge_num", 1024*1024)
        self.num_keys = int(self.knowledge_num ** 0.5)
        
        # Ensure perfect square
        if self.num_keys * self.num_keys != self.knowledge_num:
             raise ValueError(f"knowledge_num {self.knowledge_num} must be a perfect square")

        # Query Projection: 2560 -> 1024
        # The query will be split into q1, q2 each of size 512 (if query_dim=1024)
        self.query_proj = nn.Linear(self.input_dim, self.query_dim, bias=False)
        
        # Key Projections: 2560 -> 512
        # Transforming the 2560-dim knowledge embeddings to 512-dim keys for matching
        self.row_key_proj = nn.Linear(self.input_dim, self.key_proj_dim, bias=False)
        self.col_key_proj = nn.Linear(self.input_dim, self.key_proj_dim, bias=False)
        
        # Dynamic Keys (updated every epoch)
        # register_buffer ensures they are part of state_dict but not parameters updated by optimizer
        self.register_buffer("row_keys", torch.zeros(self.num_keys, self.input_dim))
        self.register_buffer("col_keys", torch.zeros(self.num_keys, self.input_dim))
        
        # Load keys from file if specified (支持新格式和旧格式)
        if "keys_path" in cfg and cfg["keys_path"] and os.path.exists(cfg["keys_path"]):
            self._load_keys_from_file(cfg["keys_path"])
        
        # Temperature for softmax
        self.temperature = cfg.get("temperature", 0.1)
        
        # 相对基线损失函数配置
        self.use_relative_baseline_loss = cfg.get("use_relative_baseline_loss", True)
        self.baseline_size = cfg.get("baseline_size", 100)  # 随机采样基线样本数量
        self.baseline_aggregation = cfg.get("baseline_aggregation", "soft_max")  # "mean" 或 "soft_max"
        self.baseline_temperature = cfg.get("baseline_temperature", 0.1)  # 用于 soft_max 的温度
        self.margin_temperature = cfg.get("margin_temperature", 1.0)  # 控制 soft margin 的软度
        self.loss_type = cfg.get("relative_loss_type", "softplus")  # "softplus" 或 "sigmoid"
        self.exclude_target_from_baseline = cfg.get("exclude_target_from_baseline", True)  # 是否从基线中排除目标样本
    
    def _load_keys_from_file(self, keys_path: str):
        """Load keys from file (新格式：字典格式，包含 row_keys 和 col_keys)"""
        try:
            loaded_data = torch.load(keys_path, map_location="cpu")
            
            # 只支持新格式：字典格式
            if not isinstance(loaded_data, dict):
                raise ValueError(f"Keys 文件必须是字典格式: {keys_path}")
            
            if "row_keys" not in loaded_data or "col_keys" not in loaded_data:
                raise ValueError(f"Keys 字典中缺少 row_keys 或 col_keys: {keys_path}")
            
            row_keys = loaded_data["row_keys"]
            col_keys = loaded_data["col_keys"]
            
            print(f"✓ 加载 Keys: {keys_path}")
            print(f"  - Row Keys 形状: {row_keys.shape}")
            print(f"  - Col Keys 形状: {col_keys.shape}")
            if "format" in loaded_data:
                print(f"  - 格式版本: {loaded_data['format']}")
            
            # 检查维度是否匹配
            if row_keys.shape[0] != self.num_keys:
                raise ValueError(
                    f"Row keys 数量不匹配: 期望 {self.num_keys}, 得到 {row_keys.shape[0]}"
                )
            if col_keys.shape[0] != self.num_keys:
                raise ValueError(
                    f"Col keys 数量不匹配: 期望 {self.num_keys}, 得到 {col_keys.shape[0]}"
                )
            
            # 如果 keys 的维度与 input_dim 不匹配，需要投影或警告
            if row_keys.shape[1] != self.input_dim:
                print(f"⚠️  警告: Keys 维度不匹配 (期望 {self.input_dim}, 得到 {row_keys.shape[1]})")
                print(f"  将自动适配维度")
                # 简单截断或填充（临时方案）
                if row_keys.shape[1] > self.input_dim:
                    row_keys = row_keys[:, :self.input_dim]
                    col_keys = col_keys[:, :self.input_dim]
                else:
                    padding = torch.zeros(
                        self.num_keys, 
                        self.input_dim - row_keys.shape[1],
                        dtype=row_keys.dtype
                    )
                    row_keys = torch.cat([row_keys, padding], dim=1)
                    col_keys = torch.cat([col_keys, padding], dim=1)
            
            # 更新 keys
            self.row_keys.copy_(row_keys.to(self.row_keys.device))
            self.col_keys.copy_(col_keys.to(self.col_keys.device))
            
        except Exception as e:
            print(f"❌ 加载 Keys 失败: {e}")
            print(f"  将使用随机初始化的 keys")
            # 保持默认的零初始化

    def update_keys(self, row_keys: torch.Tensor, col_keys: torch.Tensor):
        """Update the knowledge keys (called every epoch after clustering)"""
        if row_keys.shape[0] != self.num_keys or row_keys.shape[1] != self.input_dim:
             if row_keys.shape[1] != self.input_dim:
                 # Check if we need to use the embedding dim instead?
                 # Conceptually: keys are centroids of embeddings. 
                 # Embeddings come from backbone -> 2560 dim. 
                 # So keys should be 2560 dim.
                 pass
        
        self.row_keys.copy_(row_keys.to(self.row_keys.device))
        self.col_keys.copy_(col_keys.to(self.col_keys.device))

    def forward(self, x: torch.Tensor, target_index: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            target_index: Target grid indices [batch, seq_len] (optional, for training loss)
            
        Returns:
            If target_index is provided (Training):
                scores_1, scores_2: [batch, seq_len, num_keys]
            Else (Inference):
                candidate_indices: [batch, seq_len, num_candidates]
                candidate_scores: [batch, seq_len, num_candidates]
        """
        # 1. Project Query (mean pooling first)
        x_mean = x.mean(dim=1)  # [batch, input_dim]
        query = self.query_proj(x_mean) # [batch, query_dim]
        
        # Split query into two parts
        q1 = query[..., :self.key_proj_dim] # [batch, key_proj_dim]
        q2 = query[..., self.key_proj_dim:] # [batch, key_proj_dim]
        
        # Normalize queries
        q1 = F.normalize(q1, p=2, dim=-1)
        q2 = F.normalize(q2, p=2, dim=-1)
        
        # 2. Project Keys
        # row_keys: [num_keys, input_dim] -> [num_keys, key_proj_dim]
        k1 = self.row_key_proj(self.row_keys) 
        k2 = self.col_key_proj(self.col_keys)
        
        # Normalize keys
        k1 = F.normalize(k1, p=2, dim=-1)
        k2 = F.normalize(k2, p=2, dim=-1)
        
        # 3. Compute Scores (Dot Product)
        # q1: [batch, key_proj_dim], k1: [num_keys, key_proj_dim] -> scores_1: [batch, num_keys]
        scores_1 = torch.matmul(q1, k1.t())  # [batch, num_keys]
        scores_2 = torch.matmul(q2, k2.t())  # [batch, num_keys]
        
        # Expand to [batch, seq_len, num_keys] for compatibility
        seq_len = x.shape[1]
        scores_1 = scores_1.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, num_keys]
        scores_2 = scores_2.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, num_keys]
        
        if target_index is not None:
            # Training Mode: Return scores for loss calculation
            return scores_1, scores_2
        else:
            # Inference Mode: Generate Candidates
            return self.generate_candidates(scores_1, scores_2)

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
    
    def compute_cross_entropy_loss(self, scores_1: torch.Tensor, scores_2: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
        """
        Compute Cross Entropy Loss for single target (保留作为备用，当前不使用)
        target_index is the flattened grid index (row * num_keys + col)
        """
        # Decompose target index to row and col indices
        target_row = target_index // self.num_keys
        target_col = target_index % self.num_keys
        
        # Flatten for cross_entropy
        # scores: [batch, seq_len, num_keys] -> [batch*seq_len, num_keys]
        # target_row: [batch, seq_len] -> [batch*seq_len]
        flat_scores_1 = scores_1.view(-1, self.num_keys) / self.temperature
        flat_scores_2 = scores_2.view(-1, self.num_keys) / self.temperature
        flat_target_row = target_row.view(-1)
        flat_target_col = target_col.view(-1)
        
        loss_row = F.cross_entropy(flat_scores_1, flat_target_row)
        loss_col = F.cross_entropy(flat_scores_2, flat_target_col)
        
        return loss_row + loss_col

    def generate_candidates(self, scores_1: torch.Tensor, scores_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate candidate indices and scores from row and col scores.
        
        Args:
            scores_1: [batch, seq_len, num_keys]
            scores_2: [batch, seq_len, num_keys]
            
        Returns:
            candidate_indices: [batch, seq_len, num_candidates]
            candidate_scores: [batch, seq_len, num_candidates]
        """
        bsz, seq_len, _ = scores_1.shape
        
        # We want final num_candidates. The most efficient way is probably just taking top-k from each and combining
        # But to be accurate we might want to check more combinations.
        # Let's use sqrt(num_candidates) * factor
        k_internal = int(self.num_candidates ** 0.5) * 2
        k_internal = max(k_internal, 8)
        
        top_scores_1, top_indices_1 = scores_1.topk(k_internal, dim=-1)
        top_scores_2, top_indices_2 = scores_2.topk(k_internal, dim=-1)
        
        # Cartesian Product
        # [b, s, k, 1] + [b, s, 1, k] -> [b, s, k, k]
        combined_scores = top_scores_1.unsqueeze(-1) + top_scores_2.unsqueeze(-2)
        
        # Combined Indices
        combined_indices = (
            top_indices_1.unsqueeze(-1) * self.num_keys + top_indices_2.unsqueeze(-2)
        )
        
        # Flatten and Top-K
        # Flatten to [b, s, k*k]
        flat_combined_scores = combined_scores.view(bsz, seq_len, -1)
        flat_combined_indices = combined_indices.view(bsz, seq_len, -1)
        
        # Final Top-K 
        final_scores, best_indices = flat_combined_scores.topk(self.num_candidates, dim=-1)
        final_indices = flat_combined_indices.gather(-1, best_indices)
        
        return final_indices, final_scores
