"""
ExplicitLM: 基于显式记忆增强的语言模型（基于Qwen3架构）

在Qwen3基础上添加显式记忆库机制，训练时固定，推理时通过LLMLingua更新。
采用Shortcut机制确保backbone独立工作。
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
from models.memory_bank.MemoryGate import MemoryGate


class ExplicitLM(PreTrainedModel):
    """
    基于显式记忆增强的因果语言模型（基于Qwen3架构）
    
    记忆库在训练时固定，推理时通过LLMLingua动态更新。
    """

    config_class = Qwen3Config

    def __init__(self, qwen3_config: Qwen3Config, memory_cfg: dict) -> None:
        """
        Args:
            qwen3_config: Qwen3Config配置对象
            memory_cfg: 记忆库相关配置字典
        """
        super().__init__(qwen3_config)
        self.config = qwen3_config
        self.memory_cfg = memory_cfg
        
        self.vocab_size = qwen3_config.vocab_size
        self.hidden_size = qwen3_config.hidden_size
        
        self.embed_tokens = nn.Embedding(
            qwen3_config.vocab_size, 
            qwen3_config.hidden_size, 
            qwen3_config.pad_token_id
        )
        
        self.rotary_emb = Qwen3RotaryEmbedding(config=qwen3_config)
        
        # Create shared MemoryGate (all layers share the same gate)
        use_moe = memory_cfg.get("use_moe", False)
        if not use_moe:
            memory_cfg_with_dim = memory_cfg.copy()
            memory_cfg_with_dim["dim"] = qwen3_config.hidden_size
            self.shared_memory_gate = MemoryGate(memory_cfg_with_dim)
        else:
            self.shared_memory_gate = None
        
        self.layers = nn.ModuleList([
            Qwen3ExplicitLMBlock(qwen3_config, layer_idx, memory_cfg, shared_memory_gate=self.shared_memory_gate)
            for layer_idx in range(qwen3_config.num_hidden_layers)
        ])
        
        self.norm = Qwen3RMSNorm(qwen3_config.hidden_size, eps=qwen3_config.rms_norm_eps)
        self.lm_head = nn.Linear(qwen3_config.hidden_size, qwen3_config.vocab_size, bias=False)
        
        # Token嵌入用于记忆库解码
        if qwen3_config.tie_word_embeddings:
            self.tok_embeddings = self.embed_tokens
        else:
            self.tok_embeddings = nn.Embedding(
                qwen3_config.vocab_size,
                qwen3_config.hidden_size,
                qwen3_config.pad_token_id
            )

        use_moe = memory_cfg.get("use_moe", False)
        if not use_moe:
            knowledge_num = memory_cfg["knowledge_num"]
            knowledge_length = memory_cfg["knowledge_length"]
            
            # memory_bank存储token IDs，训练时固定，推理时通过LLMLingua更新
            # 存储在CPU上以节省GPU显存，使用时临时传输到GPU
            # 使用 pad_token_id 初始化（如果可用），否则使用 0
            pad_token_id = getattr(qwen3_config, 'pad_token_id', 0) or 0
            memory_bank_tensor = torch.full(
                (knowledge_num, knowledge_length), pad_token_id, dtype=torch.long
            )
            self.register_buffer(
                "memory_bank",
                memory_bank_tensor,
                persistent=True,  # 持久化，确保保存和加载时包含
            )
            # 将 memory_bank 移到 CPU 以节省 GPU 显存
            self.memory_bank = self.memory_bank.cpu()

            # 记录上一步的记忆库状态（用于统计）
            # prev_memory_bank 也存储在 CPU 上
            self.register_buffer(
                "prev_memory_bank",
                torch.zeros_like(self.memory_bank),
                persistent=False,
            )
            self.prev_memory_bank = self.prev_memory_bank.cpu()

            # 注册 freeze_mask buffer（全 False，所有条目都可以更新）
            self.register_buffer(
                "freeze_mask",
                torch.zeros(knowledge_num, dtype=torch.bool),
                persistent=False,
            )
            
            # 注册 valid_mask buffer（标记哪些条目是有效的，非空条目）
            # 如果记忆库是随机初始化的（全为 pad_token_id），则 valid_mask 全为 False
            # 加载 cache 时会被更新为实际的 valid_mask
            # 注意：随机初始化的记忆库所有条目都是 pad，所以初始时全为 False
            is_random_init = (memory_bank_tensor == pad_token_id).all()
            initial_valid_mask = torch.zeros(knowledge_num, dtype=torch.bool) if is_random_init else torch.ones(knowledge_num, dtype=torch.bool)
            self.register_buffer(
                "valid_mask",
                initial_valid_mask,
                persistent=True,  # 持久化，确保保存和加载时包含
            )
            self.valid_mask = self.valid_mask.cpu()
            
            # 注册 memory_bank 版本号（用于缓存失效检测）
            # 每次 memory_bank 更新时，版本号会递增，MemoryGate 可以检测到变化
            self.register_buffer(
                "_memory_bank_version",
                torch.tensor(0, dtype=torch.long),
                persistent=False,  # 不需要持久化，每次加载时重置为0
            )
        else:
            # MOE 模式：不需要 memory_bank
            self.memory_bank = None
            self._memory_bank_version = None
            print("🔥 MOE mode enabled: using Mixture of Experts instead of memory bank", flush=True)

        self.OUT = CausalLMOutputWithPast()
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
        step: Optional[int] = None,  # 兼容旧接口（已废弃）
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            position_ids: 位置IDs
            past_key_values: KV缓存
            inputs_embeds: 输入嵌入（可选）
            use_cache: 是否使用缓存
            cache_position: 缓存位置
        """
        
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
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # 初始化 baseline_loss（相对基线损失）
        # 注意：不能使用 requires_grad=True 的叶子变量，因为后面会用 += 进行原地操作
        # 使用 0.0 初始化，第一次累加时会自动创建计算图
        total_baseline_loss = torch.tensor(0.0, device=hidden_states.device)
        total_moe_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        all_layer_stats: Dict[str, float] = {}
        all_cosine_stats: Dict[str, Union[torch.Tensor, float]] = {}
        
        use_moe = self.memory_cfg.get("use_moe", False)
        
        # 在 embedding 阶段进行一次检索（仅非 MoE 模式）
        precomputed_candidates = None
        if not use_moe and self.shared_memory_gate is not None:
            # 使用输入 embedding 作为 query 进行检索
            # inputs_embeds: [batch, seq_len, hidden_size]
            precomputed_candidates = self.shared_memory_gate(
                inputs_embeds, self.memory_bank, self.tok_embeddings, self.valid_mask
            )
        
        for layer_idx, layer in enumerate(self.layers):
            layer_attention_mask = causal_mask_mapping.get(
                getattr(layer.qwen3_decoder, "attention_type", "full_attention"),
                causal_mask_mapping["full_attention"]
            )
            
            if use_moe:
                hidden_states, sim_loss, layer_stats, cosine_stats = layer(
                    hidden_states=hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
                if "moe_aux_loss" in layer_stats:
                    moe_aux = layer_stats["moe_aux_loss"]
                    if isinstance(moe_aux, (int, float)):
                        total_moe_aux_loss += torch.tensor(moe_aux, device=hidden_states.device)
                    elif isinstance(moe_aux, torch.Tensor):
                        total_moe_aux_loss += moe_aux
            else:
                hidden_states, sim_loss, layer_stats, cosine_stats = layer(
                    hidden_states=hidden_states,
                    attention_mask=layer_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    memory_bank=self.memory_bank,
                    valid_mask=self.valid_mask,
                    tok_embeddings=self.tok_embeddings,
                    precomputed_candidates=precomputed_candidates,
                    **kwargs,
                )
                # 使用非原地操作，避免对叶子变量进行原地操作
                total_baseline_loss = total_baseline_loss + sim_loss
            
            for k, v in layer_stats.items():
                all_layer_stats[f"layer_{layer_idx}_{k}"] = v
            for k, v in cosine_stats.items():
                all_cosine_stats[f"layer_{layer_idx}_{k}"] = v
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        n_layers = len(self.layers)
        if use_moe:
            aux_loss = {
                "moe_aux_loss": total_moe_aux_loss / n_layers,
            }
        else:
            aux_loss = {
                "baseline_loss": total_baseline_loss / n_layers,  # 相对基线损失（relative baseline loss）
            }
        
        self.OUT.__setitem__("last_hidden_state", hidden_states)
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("aux_loss", aux_loss)
        self.OUT.__setitem__("layer_stats", all_layer_stats)
        self.OUT.__setitem__("cosine_stats", all_cosine_stats)
        self.OUT.__setitem__("past_key_values", past_key_values if use_cache else None)
        return self.OUT

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
