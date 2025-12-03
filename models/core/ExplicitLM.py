"""
ExplicitLM: 基于显式记忆增强的语言模型（基于Qwen3架构）

该模型在Qwen3架构的基础上添加了显式记忆库机制：
- 使用Qwen3的预训练backbone（Attention + MLP）
- 在MLP输出后添加记忆检索和融合机制
- 共享记忆库存储可学习的token序列
- EMA更新机制实现类似VQ-VAE的codebook更新
- 支持记忆冻结策略以保护重要知识
- 多损失优化系统（相似度损失+多样性损失）
"""

from typing import Dict, List, Optional, Union, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
    Qwen3Model,
    Qwen3RotaryEmbedding,
    Qwen3RMSNorm,
    Cache,
    DynamicCache,
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.utils import TransformersKwargs
from typing import Unpack

from models.core.Qwen3ExplicitLMBlock import Qwen3ExplicitLMBlock


class ExplicitLM(PreTrainedModel):
    """
    基于显式记忆增强的因果语言模型（基于Qwen3架构）

    该模型在Qwen3架构的基础上添加了显式记忆库机制，通过共享记忆库增强
    Transformer架构，记忆库存储token序列并通过EMA机制动态更新。
    """

    config_class = Qwen3Config

    def __init__(self, qwen3_config: Qwen3Config, memory_cfg: dict) -> None:
        """
        初始化ExplicitLM模型

        Args:
            qwen3_config: Qwen3Config配置对象，包含Qwen3的所有架构参数
            memory_cfg: 记忆库相关配置字典，包含knowledge_num, knowledge_dim等
        """
        super().__init__(qwen3_config)
        self.config = qwen3_config
        self.memory_cfg = memory_cfg
        
        # ===== 使用Qwen3的基础组件 =====
        self.vocab_size = qwen3_config.vocab_size
        self.hidden_size = qwen3_config.hidden_size
        
        # Token嵌入层（使用Qwen3的embed_tokens）
        self.embed_tokens = nn.Embedding(
            qwen3_config.vocab_size, 
            qwen3_config.hidden_size, 
            qwen3_config.pad_token_id
        )
        
        # 位置编码（使用Qwen3的rotary_emb）
        self.rotary_emb = Qwen3RotaryEmbedding(config=qwen3_config)
        
        # Transformer层堆叠（使用Qwen3ExplicitLMBlock）
        self.layers = nn.ModuleList([
            Qwen3ExplicitLMBlock(qwen3_config, layer_idx, memory_cfg)
            for layer_idx in range(qwen3_config.num_hidden_layers)
        ])
        
        # 最终归一化层（使用Qwen3的RMSNorm）
        self.norm = Qwen3RMSNorm(qwen3_config.hidden_size, eps=qwen3_config.rms_norm_eps)
        
        # 输出层（lm_head，与embed_tokens权重共享由Qwen3Config控制）
        self.lm_head = nn.Linear(qwen3_config.hidden_size, qwen3_config.vocab_size, bias=False)
        
        # 用于记忆库解码的token嵌入（与embed_tokens共享或独立）
        # 如果Qwen3Config中tie_word_embeddings=True，则共享权重
        if qwen3_config.tie_word_embeddings:
            self.tok_embeddings = self.embed_tokens
        else:
            self.tok_embeddings = nn.Embedding(
                qwen3_config.vocab_size,
                qwen3_config.hidden_size,
                qwen3_config.pad_token_id
            )

        # ===== 共享记忆库初始化（仅记忆库模式需要） =====
        use_moe = memory_cfg.get("use_moe", False)
        if not use_moe:
            # 记忆库模式：初始化 memory_bank
            knowledge_num = memory_cfg["knowledge_num"]
            knowledge_length = memory_cfg["knowledge_length"]
            
            # memory_bank存储的是token IDs（int64），不应该直接通过梯度更新
            # 使用register_buffer而不是nn.Parameter，避免DeepSpeed处理其梯度
            # memory_bank通过EMA机制更新，而不是梯度更新
            self.register_buffer(
                "memory_bank",
                torch.randint(
                    0, qwen3_config.vocab_size, (knowledge_num, knowledge_length)
                ),
                persistent=True,  # 持久化，确保保存和加载时包含
            )

            # ===== EMA更新相关缓冲区 =====
            if memory_cfg.get("use_ema_update", False):
                self.register_buffer(
                    "ema_update_count",
                    torch.zeros(knowledge_num),
                    persistent=False,
                )
                self.register_buffer(
                    "ema_step_counter",
                    torch.zeros(1, dtype=torch.long),
                    persistent=False,
                )

            # 记录上一步的记忆库状态
            self.register_buffer(
                "prev_memory_bank",
                torch.zeros_like(self.memory_bank),
                persistent=False,
            )

            # ===== 记忆冻结机制 =====
            freeze_ratio = memory_cfg.get("freeze_ratio", 0.0)
            if freeze_ratio > 0.0:
                freeze_num = int(knowledge_num * freeze_ratio)
                freeze_mask = torch.zeros(knowledge_num, dtype=torch.bool)
                freeze_mask[:freeze_num] = True
                self.register_buffer("freeze_mask", freeze_mask, persistent=False)
                print(
                    f"🔥 Memory bank freezing enabled: {freeze_num}/{knowledge_num} "
                    f"entries ({freeze_ratio*100:.1f}%) frozen",
                    flush=True,
                )
            else:
                self.register_buffer(
                    "freeze_mask",
                    torch.zeros(knowledge_num, dtype=torch.bool),
                    persistent=False,
                )
                print("🔥 Memory bank freezing disabled: all entries can be updated", flush=True)
        else:
            # MOE 模式：不需要 memory_bank
            self.memory_bank = None
            print("🔥 MOE mode enabled: using Mixture of Experts instead of memory bank", flush=True)

        # 输出容器
        self.OUT = CausalLMOutputWithPast()
        
        # 初始化权重
        self.post_init()

    def get_memory_update_stats(self) -> Dict[str, float]:
        # MOE 模式下不返回记忆库统计信息
        if self.memory_cfg.get("use_moe", False) or self.memory_bank is None:
            return {
                "memory_avg_l2_change": 0.0,
                "memory_max_l2_change": 0.0,
                "memory_cosine_similarity": 1.0,
                "memory_update_rate": 0.0,
                "memory_updated_count": 0,
            }
        with torch.no_grad():
            if hasattr(self, "prev_memory_bank") and self.prev_memory_bank.numel() > 0:
                # memory_bank存储的是token IDs（int64），不能直接计算L2距离
                # 改为计算token差异：有多少个token不同
                token_diff = (self.memory_bank != self.prev_memory_bank).sum(dim=-1).float()  # [knowledge_num]
                avg_token_diff = token_diff.mean().item()
                max_token_diff = token_diff.max().item()
                
                # 计算完全相同的记忆条目比例作为相似度
                identical_memories = (token_diff == 0).sum().item()
                similarity = identical_memories / self.memory_bank.size(0)
                
                # 更新阈值：如果至少有一个token不同，则认为更新了
                threshold = 0.5  # 至少有一个token不同
                updated_memories = (token_diff >= threshold).sum().item()
                update_rate = updated_memories / self.memory_bank.size(0)
                update_stats = {
                    "memory_avg_l2_change": avg_token_diff,  # 实际上是平均token差异数
                    "memory_max_l2_change": max_token_diff,  # 实际上是最大token差异数
                    "memory_cosine_similarity": similarity,  # 完全相同的记忆条目比例
                    "memory_update_rate": update_rate,
                    "memory_updated_count": updated_memories,
                }
            else:
                update_stats = {
                    "memory_avg_l2_change": 0.0,
                    "memory_max_l2_change": 0.0,
                    "memory_cosine_similarity": 1.0,
                    "memory_update_rate": 0.0,
                    "memory_updated_count": 0,
                }
            self.prev_memory_bank.copy_(self.memory_bank)
            return update_stats

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        collect_ema_stats: Optional[bool] = None,
        step: Optional[int] = None,  # 兼容旧接口（已废弃）
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs（兼容旧接口：可以直接传入tensor作为位置参数）
            attention_mask: 注意力掩码
            position_ids: 位置IDs
            past_key_values: KV缓存
            inputs_embeds: 输入嵌入（可选，与input_ids二选一）
            use_cache: 是否使用缓存
            cache_position: 缓存位置
            collect_ema_stats: 是否收集EMA统计信息
            step: 当前步数（兼容旧接口，已废弃，不再使用）
        """
        # 兼容旧接口：如果input_ids是第一个位置参数传入的tensor
        # Python会将位置参数作为input_ids传入，所以这里直接处理即可
        
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        # 准备注意力掩码
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if hasattr(self.config, "layer_types") and "sliding_attention" in self.config.layer_types:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        
        hidden_states = inputs_embeds
        
        # 生成位置嵌入
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # 设置EMA统计收集
        if collect_ema_stats is None:
            collect_ema_stats = self.memory_cfg.get("use_ema_update", False) and self.training
        
        total_similarity_loss = torch.tensor(0.0, device=hidden_states.device)
        total_diversity_loss = torch.tensor(0.0, device=hidden_states.device)
        total_moe_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        all_layer_stats: Dict[str, float] = {}
        all_ema_stats: Dict[str, Dict] = {}
        all_cosine_stats: Dict[str, Union[torch.Tensor, float]] = {}
        
        use_moe = self.memory_cfg.get("use_moe", False)
        
        # 通过所有层
        for layer_idx, layer in enumerate(self.layers):
            layer_attention_mask = causal_mask_mapping.get(
                getattr(layer.qwen3_decoder, "attention_type", "full_attention"),
                causal_mask_mapping["full_attention"]
            )
            
            if use_moe:
                # MOE 模式
                if collect_ema_stats:
                    hidden_states, sim_loss, div_loss, layer_stats, ema_stats, cosine_stats = layer(
                        hidden_states=hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        collect_ema_stats=True,
                        **kwargs,
                    )
                    all_ema_stats[f"layer_{layer_idx}"] = ema_stats
                else:
                    hidden_states, sim_loss, div_loss, layer_stats, cosine_stats = layer(
                        hidden_states=hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        collect_ema_stats=False,
                        **kwargs,
                    )
                if "moe_aux_loss" in layer_stats:
                    moe_aux = layer_stats["moe_aux_loss"]
                    if isinstance(moe_aux, (int, float)):
                        total_moe_aux_loss += torch.tensor(moe_aux, device=hidden_states.device)
                    elif isinstance(moe_aux, torch.Tensor):
                        total_moe_aux_loss += moe_aux
            else:
                # 记忆库模式
                if collect_ema_stats:
                    hidden_states, sim_loss, div_loss, layer_stats, ema_stats, cosine_stats = layer(
                        hidden_states=hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        memory_bank=self.memory_bank,
                        tok_embeddings=self.tok_embeddings,
                        collect_ema_stats=True,
                        **kwargs,
                    )
                    all_ema_stats[f"layer_{layer_idx}"] = ema_stats
                else:
                    hidden_states, sim_loss, div_loss, layer_stats, cosine_stats = layer(
                        hidden_states=hidden_states,
                        attention_mask=layer_attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        memory_bank=self.memory_bank,
                        tok_embeddings=self.tok_embeddings,
                        collect_ema_stats=False,
                        **kwargs,
                    )
                total_similarity_loss += sim_loss
                total_diversity_loss += div_loss
            
            for k, v in layer_stats.items():
                all_layer_stats[f"layer_{layer_idx}_{k}"] = v
            for k, v in cosine_stats.items():
                all_cosine_stats[f"layer_{layer_idx}_{k}"] = v
        
        # 最终归一化和输出
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        n_layers = len(self.layers)
        if use_moe:
            aux_loss = {
                "moe_aux_loss": total_moe_aux_loss / n_layers,
            }
        else:
            aux_loss = {
                "similarity_loss": total_similarity_loss / n_layers,
                "diversity_loss": total_diversity_loss / n_layers,
            }
        
        self.OUT.__setitem__("last_hidden_state", hidden_states)
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("aux_loss", aux_loss)
        self.OUT.__setitem__("layer_stats", all_layer_stats)
        self.OUT.__setitem__("ema_stats", all_ema_stats if collect_ema_stats else None)
        self.OUT.__setitem__("cosine_stats", all_cosine_stats)
        self.OUT.__setitem__("past_key_values", past_key_values if use_cache else None)
        return self.OUT

    # ---------------- generate / stream / ema 更新 ----------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int = 2,
        max_new_tokens: int = 1024,
        temperature: float = 0.75,
        top_p: float = 0.90,
        stream: bool = False,
        rp: float = 1.0,
        pad_token_id: int = 0,
        num_return_sequences: int = 1,
        **args,
    ) -> torch.Tensor:
        if stream:
            return self._stream(
                input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, **args
            )
        # 非流式生成逻辑与原版完全一致，此处省略篇幅
        # （直接拷贝你原来的实现即可，无额外改动）
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            for _ in range(num_return_sequences):
                out = self._stream(
                    non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, **args
                )
                tokens_list = [tokens[:, -1:] for tokens in out]
                gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
                full_sequence = torch.cat([non_pad, gen], dim=-1)
                generated.append(full_sequence)
        max_len = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [
                    seq,
                    torch.full(
                        (1, max_len - seq.size(1)),
                        pad_token_id,
                        dtype=seq.dtype,
                        device=seq.device,
                    ),
                ],
                dim=-1,
            )
            for seq in generated
        ]
        output = torch.cat(generated, dim=0)
        return output.view(input_ids.size(0) * num_return_sequences, -1)

    def _stream(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        rp: float,
        **args,
    ) -> Iterator[torch.Tensor]:
        start = input_ids.shape[1]
        while input_ids.shape[1] < start + max_new_tokens:
            out = self(input_ids, **args)
            logits = out.logits[:, -1, :]
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= temperature + 1e-9
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")
            next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_tok), dim=1)
            yield input_ids[:, start:]
            if next_tok.item() == eos_token_id:
                break

    def apply_ema_update(self, ema_stats: Dict[str, Dict]) -> Dict[str, Union[bool, int, float]]:
        # MOE 模式下不支持 EMA 更新
        if self.memory_cfg.get("use_moe", False) or self.memory_bank is None:
            return {"ema_update_applied": False, "reason": "moe_mode"}
        if not self.memory_cfg.get("use_ema_update", False):
            return {}
        self.ema_step_counter += 1
        if self.ema_step_counter % self.memory_cfg["ema_update_freq"] != 0:
            return {"ema_update_applied": False, "reason": "frequency_check_failed"}

        with torch.no_grad():
            device = self.memory_bank.device
            knowledge_num, knowledge_length = self.memory_bank.shape
            dim = self.hidden_size

            all_indices: List[torch.Tensor] = []
            all_features: List[torch.Tensor] = []
            total_selections = 0
            total_layers = 0
            for layer_ema_stats in ema_stats.values():
                if layer_ema_stats is None:
                    continue
                total_layers += 1
                memory_indices = layer_ema_stats["memory_indices"]
                h_for_memory = layer_ema_stats["h_for_memory"]
                bsz, seq_len, num_selected = memory_indices.shape
                total_selections += bsz * seq_len * num_selected
                flat_indices = memory_indices.view(-1)
                h_expanded = h_for_memory.unsqueeze(2).expand(-1, -1, num_selected, -1)
                flat_h = h_expanded.reshape(-1, dim)
                all_indices.append(flat_indices)
                all_features.append(flat_h)

            if not all_indices:
                return {"ema_update_applied": False, "reason": "no_ema_stats"}

            all_indices = torch.cat(all_indices, dim=0)
            all_features = torch.cat(all_features, dim=0)
            unique_indices, inverse_indices = torch.unique(all_indices, return_inverse=True)
            aggregated_features = torch.zeros(
                unique_indices.size(0), dim, device=device, dtype=all_features.dtype
            )
            count_per_memory = torch.zeros(
                unique_indices.size(0), device=device, dtype=all_features.dtype
            )
            aggregated_features.scatter_add_(
                0, inverse_indices.unsqueeze(1).expand(-1, dim), all_features
            )
            count_per_memory.scatter_add_(
                0, inverse_indices, torch.ones_like(inverse_indices, dtype=all_features.dtype)
            )
            avg_features = aggregated_features / count_per_memory.unsqueeze(1)

            # 减小批次大小以节省显存，避免lm_head输出过大
            batch_size = 512  # 从4096减小到512，减少lm_head的内存占用
            updated_memories = 0
            for i in range(0, unique_indices.size(0), batch_size):
                end_i = min(i + batch_size, unique_indices.size(0))
                batch_indices = unique_indices[i:end_i]
                batch_avg_features = avg_features[i:end_i]
                current_tokens_batch = self.memory_bank[batch_indices]
                current_embeddings_batch = self.tok_embeddings(
                    current_tokens_batch.view(-1)
                ).view(batch_indices.size(0), knowledge_length, dim)
                old_features_batch = current_embeddings_batch.view(
                    batch_indices.size(0), -1
                )
                expanded_new_features = batch_avg_features.repeat(1, knowledge_length)
                updated_features_batch = (
                    self.memory_cfg["ema_decay"] * old_features_batch
                    + (1 - self.memory_cfg["ema_decay"]) * expanded_new_features
                )
                updated_reshaped = updated_features_batch.view(-1, dim)
                logits_batch = self.lm_head(updated_reshaped)
                new_token_ids_batch = torch.argmax(logits_batch, dim=-1).view(
                    batch_indices.size(0), knowledge_length
                )
                unfrozen_mask_batch = ~self.freeze_mask[batch_indices]
                if unfrozen_mask_batch.any():
                    unfrozen_indices = batch_indices[unfrozen_mask_batch]
                    unfrozen_tokens = new_token_ids_batch[unfrozen_mask_batch]
                    self.memory_bank[unfrozen_indices] = unfrozen_tokens
                    updated_memories += unfrozen_indices.size(0)

            frozen_count = self.freeze_mask.sum().item()
            total_memories = knowledge_num
            update_stats = {
                "ema_update_applied": True,
                "ema_step": self.ema_step_counter.item(),
                "total_selections": total_selections,
                "total_layers": total_layers,
                "updated_memories": updated_memories,
                "update_ratio": updated_memories / knowledge_num,
                "frozen_memories": frozen_count,
                "frozen_ratio": frozen_count / total_memories,
                "ema_decay": self.memory_cfg["ema_decay"],
                "selected_memory_coverage": updated_memories / knowledge_num,
            }
            return update_stats
