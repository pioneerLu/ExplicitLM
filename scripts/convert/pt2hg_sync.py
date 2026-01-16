#!/usr/bin/env python3
"""
将ExplicitLM训练的checkpoint转换为HuggingFace格式并用于推理（Memory Bank 同步存储）

使用方法（直接 uv 运行）:
    # 转换checkpoint
    uv run python pt2hg_sync.py \
        --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_14500 \
        --qwen3_path Qwen_hg/Qwen3-4b \
        --output_path hf_explicitlm_model

    # 诊断checkpoint（不进行转换）
    uv run python pt2hg_sync.py \
        --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_7500 \
        --diagnose
"""

import os
import sys
import argparse
import torch
import re
import inspect
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.resolve()  # scripts/convert -> scripts -> ExplicitLM
sys.path.insert(0, str(project_root))

# 导入transformers
try:
    from transformers import (
        Qwen3Config,
        Qwen3PreTrainedModel,
        PretrainedConfig,
    )
    from transformers.modeling_outputs import CausalLMOutputWithPast
    try:
        from transformers import GenerationMixin
    except ImportError:
        GenerationMixin = object
    import torch.nn as nn
except ImportError as e:
    print(f"❌ 需要安装transformers库: {e}")
    exit(1)


class ExplicitLMConfig(PretrainedConfig):
    """ExplicitLM的配置文件，兼容Hugging Face格式"""
    model_type = "explicitlm"

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        knowledge_num=10000,
        knowledge_length=32,
        num_candidates=16,
        num_selected=1,
        gumbel_temperature=1.0,
        use_memory_gate=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.knowledge_num = knowledge_num
        self.knowledge_length = knowledge_length
        self.num_candidates = num_candidates
        self.num_selected = num_selected
        self.gumbel_temperature = gumbel_temperature
        self.use_memory_gate = use_memory_gate


class ExplicitLMForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Hugging Face兼容的ExplicitLM因果语言模型"""
    config_class = ExplicitLMConfig

    def __init__(self, config: ExplicitLMConfig):
        super().__init__(config)
        
        # 创建Qwen3配置
        qwen3_config = Qwen3Config(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attention_bias=config.attention_bias,
            attention_dropout=config.attention_dropout,
        )

        memory_cfg = {
            "knowledge_num": config.knowledge_num,
            "knowledge_length": config.knowledge_length,
            "num_candidates": config.num_candidates,
            "num_selected": config.num_selected,
            "gumbel_temperature": config.gumbel_temperature,
        }

        # 导入ExplicitLM类
        try:
            from models.core.ExplicitLM import ExplicitLM
        except ImportError:
            import sys
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)
            from models.core.ExplicitLM import ExplicitLM

        self.model = ExplicitLM(qwen3_config, memory_cfg)
        # 注意：不设置 self.lm_head，直接使用 self.model.lm_head
        # 这样可以避免 state_dict 中出现重复的 lm_head.weight 键

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """前向传播，兼容Hugging Face接口"""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        if isinstance(outputs, tuple):
            hidden_states, loss, aux_loss = outputs[:3]
            past_key_values = None
        else:
            hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
            loss = None
            aux_loss = {}

        logits = self.model.lm_head(hidden_states)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        """为生成准备输入，兼容Hugging Face"""
        import torch
        
        # 检查past_key_values是否真的包含缓存数据
        # 如果past_key_values存在但长度为0，说明这是第一次forward，不应该截断input_ids
        is_first_forward = (past_key_values is None) or (
            hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0
        )
        
        # 如果有 past_key_values 且不是第一次forward，只使用最后一个 token
        if past_key_values is not None and not is_first_forward:
            input_ids = input_ids[:, -1:]
            # 在生成过程中，需要扩展 attention_mask 以包含新token
            # create_causal_mask 需要知道完整的序列长度（包括已缓存的token）
            if attention_mask is not None:
                past_seen_tokens = past_key_values.get_seq_length()
                current_seq_len = input_ids.shape[1]  # 应该是1
                # 扩展 attention_mask 以匹配完整的序列长度
                expected_len = past_seen_tokens + current_seq_len
                if attention_mask.shape[1] < expected_len:
                    # 扩展 attention_mask
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], expected_len - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device
                        )
                    ], dim=-1)
                elif attention_mask.shape[1] > expected_len:
                    # 如果 attention_mask 太长，截断到期望长度
                    attention_mask = attention_mask[:, :expected_len]
        
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        
        # 只在 attention_mask 不为 None 时添加
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        # 确保 cache_position 与 input_ids 的长度匹配
        # 这对于第一次 forward 时特别重要，因为 cache_position 会影响 kv_length 的计算
        if "cache_position" not in kwargs or kwargs.get("cache_position") is None:
            if past_key_values is not None and not is_first_forward:
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + input_ids.shape[1], 
                    device=input_ids.device, dtype=torch.long
                )
            else:
                # 第一次 forward，cache_position 应该从 0 开始，长度为 input_ids 的长度
                cache_position = torch.arange(
                    0, input_ids.shape[1], 
                    device=input_ids.device, dtype=torch.long
                )
            model_inputs["cache_position"] = cache_position
        
        # 添加其他 kwargs（排除已处理的）
        excluded_keys = {"input_ids", "past_key_values", "attention_mask", "use_cache", "cache_position"}
        model_inputs.update({k: v for k, v in kwargs.items() if k not in excluded_keys})
        
        return model_inputs

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """自定义加载逻辑，在加载后重新建立共享权重关系"""
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._restore_shared_weights()
    
    def _restore_shared_weights(self):
        """重新建立共享权重关系（用于safetensors格式加载后）"""
        if not hasattr(self.model, 'shared_memory_gate') or self.model.shared_memory_gate is None:
            return
        for layer in self.model.layers:
            if hasattr(layer, 'memory_gate'):
                layer.memory_gate = self.model.shared_memory_gate

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
        """从预训练权重加载模型"""
        model = super(ExplicitLMForCausalLM, ExplicitLMForCausalLM).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        if isinstance(model, ExplicitLMForCausalLM):
            model._restore_shared_weights()
        return model


def map_qwen3_weight_name(qwen3_name: str) -> str:
    """将Qwen3权重名称映射到ExplicitLM格式"""
    # 移除'model.'前缀
    if qwen3_name.startswith('model.'):
        key = qwen3_name[6:]
    else:
        key = qwen3_name
    
    # 映射layers到qwen3_decoder
    if key.startswith('layers.'):
        parts = key.split('.', 2)
        if len(parts) >= 3:
            layer_idx = parts[1]
            rest = parts[2]
            return f'layers.{layer_idx}.qwen3_decoder.{rest}'
    
    # embed_tokens, norm, lm_head, rotary_emb 等直接使用
    return key


def map_explicitlm_weight_name(explicitlm_name: str) -> str:
    """将ExplicitLM组件权重名称映射到HF格式"""
    if explicitlm_name.startswith('module.'):
        return 'model.' + explicitlm_name[7:]  # 移除'module.'前缀，添加'model.'
    else:
        return 'model.' + explicitlm_name


def create_explicitlm_config(qwen3_path: str, memory_config: Optional[Dict[str, Any]] = None) -> ExplicitLMConfig:
    """基于Qwen3配置创建ExplicitLM配置"""
    try:
        qwen3_config = Qwen3Config.from_pretrained(qwen3_path)
    except Exception as e:
        print(f"⚠️ 无法加载Qwen3配置，使用默认值: {e}")
        qwen3_config = Qwen3Config()

    # 获取tokenizer以获取特殊token IDs
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, trust_remote_code=True, fix_mistral_regex=True)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        bos_token_id = tokenizer.bos_token_id
    except:
        pad_token_id = getattr(qwen3_config, 'pad_token_id', None)
        eos_token_id = getattr(qwen3_config, 'eos_token_id', None)
        bos_token_id = getattr(qwen3_config, 'bos_token_id', None)
    
    return ExplicitLMConfig(
        vocab_size=qwen3_config.vocab_size,
        hidden_size=qwen3_config.hidden_size,
        intermediate_size=getattr(qwen3_config, 'intermediate_size', 11008),
        num_hidden_layers=qwen3_config.num_hidden_layers,
        num_attention_heads=qwen3_config.num_attention_heads,
        num_key_value_heads=getattr(qwen3_config, 'num_key_value_heads', qwen3_config.num_attention_heads),
        max_position_embeddings=getattr(qwen3_config, 'max_position_embeddings', 32768),
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        bos_token_id=bos_token_id,
        rms_norm_eps=getattr(qwen3_config, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(qwen3_config, 'rope_theta', 10000.0),
        attention_bias=getattr(qwen3_config, 'attention_bias', False),
        attention_dropout=getattr(qwen3_config, 'attention_dropout', 0.0),
        knowledge_num=memory_config.get('knowledge_num', 1048576) if memory_config else 1048576,
        knowledge_length=memory_config.get('knowledge_length', 32) if memory_config else 32,
        num_candidates=memory_config.get('num_candidates', 16) if memory_config else 16,
        num_selected=memory_config.get('num_selected', 1) if memory_config else 1,
        gumbel_temperature=memory_config.get('gumbel_temperature', 1.0) if memory_config else 1.0,
        use_memory_gate=memory_config.get('use_memory_gate', True) if memory_config else True,
    )


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """加载checkpoint文件，支持多种格式"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到checkpoint路径: {checkpoint_path}")
    
    if os.path.isdir(checkpoint_path):
        components_file = os.path.join(checkpoint_path, "trainable_components.pth")
        if not os.path.exists(components_file):
            raise FileNotFoundError(f"在目录中找不到trainable_components.pth: {checkpoint_path}")
        print(f"📂 从checkpoint目录加载: {checkpoint_path}")
    elif checkpoint_path.endswith('.pth'):
        components_file = checkpoint_path
        print(f"📂 直接加载checkpoint文件: {checkpoint_path}")
    else:
        raise ValueError(f"不支持的checkpoint路径格式: {checkpoint_path}")
    
    print(f"   加载文件: {components_file}")
    checkpoint = torch.load(components_file, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        checkpoint_info = {
            'train_memory_gate': checkpoint.get('train_memory_gate', False),
            'saved_at_step': checkpoint.get('saved_at_step', 0),
        }
        print(f"✅ 检测到 trainable_components 格式")
        print(f"   加载了 {len(state_dict)} 个训练组件参数")
    elif isinstance(checkpoint, dict) and all(isinstance(k, str) for k in checkpoint.keys()):
        state_dict = checkpoint
        has_memory_gate_params = any('memory_gate' in k and 'keys' not in k for k in state_dict.keys())
        checkpoint_info = {
            'train_memory_gate': has_memory_gate_params,
            'saved_at_step': 0,
        }
        print(f"✅ 检测到完整 state_dict 格式")
        print(f"   加载了 {len(state_dict)} 个参数")
    else:
        raise ValueError(f"不支持的checkpoint格式: {type(checkpoint)}")
    
    return state_dict, checkpoint_info


def convert_to_hf_format(
    checkpoint_path: str,
    qwen3_path: str,
    output_path: str,
    keys_path: Optional[str] = None,
    memory_bank_path: Optional[str] = None,
    knowledge_num: int = 1048576,
    knowledge_length: int = 32,
    num_candidates: int = 16,
    num_selected: int = 1,
    gumbel_temperature: float = 1.0,
) -> str:
    """将checkpoint转换为HuggingFace格式"""
    print("=" * 60)
    print("🔄 开始转换ExplicitLM checkpoint到HuggingFace格式")
    print("=" * 60)
    
    # 1. 加载checkpoint
    print("\n📥 步骤1: 加载checkpoint...")
    explicitlm_state_dict, checkpoint_info = load_checkpoint(checkpoint_path)
    
    # 2. 创建ExplicitLM配置
    print("\n🏗️ 步骤2: 创建ExplicitLM配置...")
    memory_config = {
        "knowledge_num": knowledge_num,
        "knowledge_length": knowledge_length,
        "num_candidates": num_candidates,
        "num_selected": num_selected,
        "gumbel_temperature": gumbel_temperature,
        "use_memory_gate": True,
    }
    config = create_explicitlm_config(qwen3_path, memory_config)
    
    # 2.1. 添加 auto_map 属性，使 transformers 能够自动加载自定义模型
    config.auto_map = {
        "AutoConfig": "configuration_explicitlm.ExplicitLMConfig",
        "AutoModelForCausalLM": "modeling_explicitlm.ExplicitLMForCausalLM"
    }
    
    # 3. 创建HF兼容的ExplicitLM模型
    print("\n🏗️ 步骤3: 创建HuggingFace兼容的ExplicitLM模型...")
    try:
        hf_model = ExplicitLMForCausalLM(config)
        print(f"✅ 模型创建成功，包含 {len(hf_model.state_dict())} 个参数")
    except Exception as e:
        print(f"❌ 创建HF模型失败: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        raise
    
    # 4. 加载Qwen3基础权重
    print("\n📥 步骤4: 加载Qwen3基础权重...")
    try:
        qwen3_bin_path = os.path.join(qwen3_path, "pytorch_model.bin")
        if os.path.exists(qwen3_bin_path):
            qwen3_state_dict = torch.load(qwen3_bin_path, map_location='cpu')
        else:
            from transformers import Qwen3ForCausalLM
            qwen3_model = Qwen3ForCausalLM.from_pretrained(qwen3_path)
            qwen3_state_dict = qwen3_model.state_dict()
            del qwen3_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"   加载Qwen3权重: {len(qwen3_state_dict)} 个参数")
    except Exception as e:
        print(f"⚠️ 无法加载Qwen3权重，使用随机初始化: {e}")
        qwen3_state_dict = {}
    
    # 5. 合并权重
    print("\n🔗 步骤5: 合并权重...")
    hf_state_dict = hf_model.state_dict()
    
    # 复制Qwen3基础权重
    qwen3_applied = 0
    qwen3_not_found = []
    for name, weight in qwen3_state_dict.items():
        # 先尝试直接映射
        hf_name = map_qwen3_weight_name(name)
        # HF模型的state_dict键名格式是 model.xxx
        if not hf_name.startswith('model.'):
            hf_name = 'model.' + hf_name
        
        if hf_name in hf_state_dict:
            try:
                if hf_state_dict[hf_name].shape == weight.shape:
                    hf_state_dict[hf_name].copy_(weight)
                    qwen3_applied += 1
                else:
                    print(f"⚠️ 形状不匹配 {name}->{hf_name}: {hf_state_dict[hf_name].shape} vs {weight.shape}")
            except Exception as e:
                print(f"⚠️ Qwen3权重复制失败 {name}->{hf_name}: {e}")
        else:
            qwen3_not_found.append((name, hf_name))
    
    print(f"✅ 应用Qwen3基础权重: {qwen3_applied} 个")
    if qwen3_applied == 0 and qwen3_not_found:
        print(f"⚠️  未找到匹配的权重，前5个示例:")
        for qwen3_name, hf_name in qwen3_not_found[:5]:
            # 查找相似的键名
            similar_keys = [k for k in hf_state_dict.keys() if qwen3_name.split('.')[-1] in k][:3]
            print(f"  Qwen3: {qwen3_name} -> 尝试HF: {hf_name}")
            if similar_keys:
                print(f"    相似的HF键名: {similar_keys}")
    
    # 应用ExplicitLM组件权重
    explicitlm_applied = 0
    explicitlm_missing = []
    explicitlm_shape_mismatch = []
    shared_memory_gate_loaded = set()
    
    for name, weight in explicitlm_state_dict.items():
        if not isinstance(weight, torch.Tensor):
            continue
        
        # 处理共享MemoryGate权重
        shared_gate_match1 = re.match(r'^(module\.)?shared_memory_gate\.(.+)$', name)
        shared_gate_match2 = re.match(r'^(module\.)?layers\.\d+\.memory_gate\.(.+)$', name)
        shared_gate_match = shared_gate_match1 or shared_gate_match2
        
        if shared_gate_match:
            if shared_gate_match1:
                gate_submodule = shared_gate_match1.group(2)
            else:
                gate_submodule = shared_gate_match2.group(2)
            
            hf_shared_gate_name = f'model.shared_memory_gate.{gate_submodule}'
            
            if hf_shared_gate_name in shared_memory_gate_loaded:
                continue
            
            if hf_shared_gate_name in hf_state_dict:
                try:
                    if hf_state_dict[hf_shared_gate_name].shape == weight.shape:
                        hf_state_dict[hf_shared_gate_name].copy_(weight)
                        shared_memory_gate_loaded.add(hf_shared_gate_name)
                        explicitlm_applied += 1
                        if explicitlm_applied <= 5:
                            print(f"  ✅ {name} -> {hf_shared_gate_name} (共享MemoryGate)")
                        continue
                    else:
                        explicitlm_shape_mismatch.append(f"{name}->{hf_shared_gate_name}: {hf_state_dict[hf_shared_gate_name].shape} vs {weight.shape}")
                except Exception as e:
                    print(f"⚠️ 共享MemoryGate权重应用失败 {name}->{hf_shared_gate_name}: {e}")
            else:
                explicitlm_missing.append(f"{name} -> {hf_shared_gate_name} (共享MemoryGate未找到)")
        
        # 标准映射
        hf_name = map_explicitlm_weight_name(name)
        
        if hf_name in hf_state_dict:
            try:
                if hf_state_dict[hf_name].shape == weight.shape:
                    hf_state_dict[hf_name].copy_(weight)
                    explicitlm_applied += 1
                    if explicitlm_applied <= 5:
                        print(f"  ✅ {name} -> {hf_name}")
                else:
                    explicitlm_shape_mismatch.append(f"{name}->{hf_name}: {hf_state_dict[hf_name].shape} vs {weight.shape}")
            except Exception as e:
                print(f"⚠️ ExplicitLM权重应用失败 {name}->{hf_name}: {e}")
        else:
            # 尝试备用映射
            alt_name = name.replace('module.', '') if name.startswith('module.') else name
            alt_hf_name = 'model.' + alt_name if not alt_name.startswith('model.') else alt_name
            
            if alt_hf_name in hf_state_dict:
                if hf_state_dict[alt_hf_name].shape == weight.shape:
                    hf_state_dict[alt_hf_name].copy_(weight)
                    explicitlm_applied += 1
                    if explicitlm_applied <= 5:
                        print(f"  ✅ {name} -> {alt_hf_name} (备用映射)")
                else:
                    explicitlm_missing.append(f"{name} -> {hf_name}")
            else:
                explicitlm_missing.append(f"{name} -> {hf_name}")
    
    print(f"✅ 应用ExplicitLM训练权重: {explicitlm_applied} 个")
    if shared_memory_gate_loaded:
        print(f"  其中共享MemoryGate权重: {len(shared_memory_gate_loaded)} 个（已去重）")
    
    if explicitlm_missing:
        print(f"⚠️ 有 {len(explicitlm_missing)} 个权重未找到对应（前10个）:")
        for item in explicitlm_missing[:10]:
            print(f"     {item}")
    
    if explicitlm_shape_mismatch:
        print(f"⚠️ 有 {len(explicitlm_shape_mismatch)} 个权重形状不匹配（前5个）:")
        for item in explicitlm_shape_mismatch[:5]:
            print(f"     {item}")
    
    # 6. 加载权重到模型
    print("\n📥 步骤6: 加载合并后的权重到模型...")
    try:
        missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    except Exception as e:
        print(f"❌ 加载权重到模型失败: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        raise
    
    if missing_keys:
        print(f"⚠️ 有 {len(missing_keys)} 个权重缺失（通常是backbone权重，已从Qwen3加载）")
    
    if unexpected_keys:
        print(f"⚠️ 有 {len(unexpected_keys)} 个权重未使用")
    
    # 7. 加载memory_bank（如果提供）
    if memory_bank_path and os.path.exists(memory_bank_path):
        print(f"\n📥 步骤7: 加载Memory Bank数据...")
        try:
            memory_bank_data = torch.load(memory_bank_path, map_location='cpu')
            
            if isinstance(memory_bank_data, dict):
                memory_bank = memory_bank_data.get('memory_bank', memory_bank_data.get('processed_tensor'))
                valid_mask = memory_bank_data.get('valid_mask', None)
            else:
                memory_bank = memory_bank_data
                valid_mask = None
            
            if memory_bank is not None and hasattr(hf_model.model, 'memory_bank'):
                if memory_bank.shape[0] > knowledge_num:
                    memory_bank = memory_bank[:knowledge_num]
                elif memory_bank.shape[0] < knowledge_num:
                    pad_token_id = getattr(config, 'pad_token_id', 0) or 0
                    padding = torch.full(
                        (knowledge_num - memory_bank.shape[0], knowledge_length),
                        pad_token_id,
                        dtype=memory_bank.dtype
                    )
                    memory_bank = torch.cat([memory_bank, padding], dim=0)
                
                hf_model.model.memory_bank.data.copy_(memory_bank)
                print(f"✅ Memory Bank已更新: {memory_bank.shape}")
                
                if valid_mask is not None and hasattr(hf_model.model, 'valid_mask'):
                    if valid_mask.shape[0] > knowledge_num:
                        valid_mask = valid_mask[:knowledge_num]
                    elif valid_mask.shape[0] < knowledge_num:
                        padding_mask = torch.zeros(knowledge_num - valid_mask.shape[0], dtype=torch.bool)
                        valid_mask = torch.cat([valid_mask, padding_mask], dim=0)
                    hf_model.model.valid_mask.data.copy_(valid_mask)
                    print(f"✅ Valid Mask已更新: {valid_mask.sum().item()}/{valid_mask.shape[0]} 有效")
        except Exception as e:
            print(f"⚠️ 加载Memory Bank失败: {e}")
    
    # 8. 保存为HuggingFace格式
    print(f"\n💾 步骤8: 保存为HuggingFace格式: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    print("   保存模型权重和配置...")
    # 直接使用 PyTorch 原生保存，支持共享权重（Memory Gate）
    # 不使用 safetensors，因为它不支持共享张量
    hf_model.save_pretrained(output_path, safe_serialization=False)
    print("   ✅ 已保存为 PyTorch 格式（支持共享权重）")
    
    # 保存tokenizer
    print("   保存tokenizer...")
    from transformers import AutoTokenizer
    # 修复 Mistral 正则警告
    tokenizer = AutoTokenizer.from_pretrained(qwen3_path, trust_remote_code=True, fix_mistral_regex=True)
    tokenizer.save_pretrained(output_path)
    
    # 保存checkpoint信息
    checkpoint_meta = {
        "checkpoint_path": checkpoint_path,
        "knowledge_num": knowledge_num,
        "knowledge_length": knowledge_length,
        "num_candidates": num_candidates,
        "num_selected": num_selected,
        "gumbel_temperature": gumbel_temperature,
        "train_memory_gate": checkpoint_info.get('train_memory_gate', False),
        "saved_at_step": checkpoint_info.get('saved_at_step', 0),
        "qwen3_path": qwen3_path,
    }
    
    import json
    with open(os.path.join(output_path, "checkpoint_info.json"), 'w', encoding='utf-8') as f:
        json.dump(checkpoint_meta, f, indent=2, ensure_ascii=False)
    
    # 保存模型定义文件，使AutoModelForCausalLM能够识别
    _save_modeling_file(output_path)
    
    print(f"\n✅ 转换完成！")
    print(f"📁 HF模型保存路径: {output_path}")
    print(f"\n💡 使用方法:")
    print(f"   from transformers import AutoTokenizer, AutoModelForCausalLM")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{output_path}')")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{output_path}', trust_remote_code=True)")
    
    return output_path


def _save_modeling_file(output_path: str):
    """保存模型定义文件到输出目录，使AutoModelForCausalLM能够识别explicitlm模型类型
    使用 inspect 模块获取类源代码，使模型文件夹完全独立
    """
    # 1. 创建 configuration_explicitlm.py
    config_source = inspect.getsource(ExplicitLMConfig)
    config_content = f'''"""
ExplicitLM配置定义 - 用于HuggingFace自动加载配置
此文件由 pt2hg_sync.py 自动生成，包含完整的类定义
"""
from transformers import PretrainedConfig

{config_source}
'''
    
    config_path = os.path.join(output_path, "configuration_explicitlm.py")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"   ✅ 配置定义文件已保存: {config_path}")
    
    # 2. 创建 modeling_explicitlm.py
    # 获取 ExplicitLMForCausalLM 的源代码
    model_source = inspect.getsource(ExplicitLMForCausalLM)
    
    # 修改源代码中的导入路径，使其能够找到 ExplicitLM
    # 使用正则表达式替换整个 try-except 导入块
    import re
    
    # 匹配 try-except 块（包含 ExplicitLM 导入）
    # 模式：匹配从 "try:" 开始到对应的 "except" 块结束的整个块
    pattern = r'(\s+)# 导入ExplicitLM类\s*\n\s+try:\s*\n\s+from models\.core\.ExplicitLM import ExplicitLM\s*\n\s+except ImportError:\s*\n(?:\s+.*\n)*?\s+from models\.core\.ExplicitLM import ExplicitLM'
    
    def replace_import_block(match):
        indent = match.group(1)
        indent_str = indent
        
        replacement = f'''{indent_str}# 导入ExplicitLM类及其依赖
{indent_str}# 需要从项目路径导入，因为 ExplicitLM 依赖于 models.core 和 models.memory_bank
{indent_str}import sys
{indent_str}from pathlib import Path
{indent_str}import os
{indent_str}
{indent_str}# 方法1: 从环境变量获取项目根目录
{indent_str}explicitlm_root = os.environ.get("EXPLICITLM_ROOT", None)
{indent_str}if explicitlm_root and os.path.exists(explicitlm_root):
{indent_str}    if str(explicitlm_root) not in sys.path:
{indent_str}        sys.path.insert(0, str(explicitlm_root))
{indent_str}
{indent_str}# 方法2: 从当前文件向上查找项目根目录
{indent_str}if explicitlm_root is None:
{indent_str}    current_file = Path(__file__).resolve()
{indent_str}    # 尝试向上查找项目根目录（包含 models 目录的目录）
{indent_str}    for potential_root in [current_file.parent.parent, current_file.parent.parent.parent]:
{indent_str}        if potential_root and (potential_root / "models" / "core" / "ExplicitLM.py").exists():
{indent_str}            explicitlm_root = potential_root
{indent_str}            if str(explicitlm_root) not in sys.path:
{indent_str}                sys.path.insert(0, str(explicitlm_root))
{indent_str}            break
{indent_str}
{indent_str}# 执行导入
{indent_str}try:
{indent_str}    from models.core.ExplicitLM import ExplicitLM
{indent_str}except ImportError as e:
{indent_str}    raise ImportError(f"无法找到 ExplicitLM 模块。请设置 EXPLICITLM_ROOT 环境变量指向项目根目录（当前尝试的路径: {{explicitlm_root}}）。错误: {{e}}")'''
        return replacement
    
    # 尝试使用正则表达式替换
    modified_model_source = re.sub(pattern, replace_import_block, model_source, flags=re.MULTILINE)
    
    # 如果正则表达式没有匹配到，使用逐行处理的方法
    if modified_model_source == model_source:
        model_source_lines = model_source.split('\n')
        modified_source_lines = []
        i = 0
        in_try_except_block = False
        try_indent = 0
        
        while i < len(model_source_lines):
            line = model_source_lines[i]
            line_stripped = line.strip()
            
            # 检测 try 块开始（包含 ExplicitLM 导入的 try 块）
            if line_stripped == 'try:' and i + 1 < len(model_source_lines):
                next_line = model_source_lines[i + 1]
                if 'from models.core.ExplicitLM import ExplicitLM' in next_line:
                    # 找到包含 ExplicitLM 导入的 try 块
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    
                    # 替换为新的导入逻辑
                    modified_source_lines.append(f'{indent_str}# 导入ExplicitLM类及其依赖')
                    modified_source_lines.append(f'{indent_str}# 需要从项目路径导入，因为 ExplicitLM 依赖于 models.core 和 models.memory_bank')
                    modified_source_lines.append(f'{indent_str}import sys')
                    modified_source_lines.append(f'{indent_str}from pathlib import Path')
                    modified_source_lines.append(f'{indent_str}import os')
                    modified_source_lines.append(f'{indent_str}')
                    modified_source_lines.append(f'{indent_str}# 方法1: 从环境变量获取项目根目录')
                    modified_source_lines.append(f'{indent_str}explicitlm_root = os.environ.get("EXPLICITLM_ROOT", None)')
                    modified_source_lines.append(f'{indent_str}if explicitlm_root and os.path.exists(explicitlm_root):')
                    modified_source_lines.append(f'{indent_str}    if str(explicitlm_root) not in sys.path:')
                    modified_source_lines.append(f'{indent_str}        sys.path.insert(0, str(explicitlm_root))')
                    modified_source_lines.append(f'{indent_str}')
                    modified_source_lines.append(f'{indent_str}# 方法2: 从当前文件向上查找项目根目录')
                    modified_source_lines.append(f'{indent_str}if explicitlm_root is None:')
                    modified_source_lines.append(f'{indent_str}    current_file = Path(__file__).resolve()')
                    modified_source_lines.append(f'{indent_str}    # 尝试向上查找项目根目录（包含 models 目录的目录）')
                    modified_source_lines.append(f'{indent_str}    for potential_root in [current_file.parent.parent, current_file.parent.parent.parent]:')
                    modified_source_lines.append(f'{indent_str}        if potential_root and (potential_root / "models" / "core" / "ExplicitLM.py").exists():')
                    modified_source_lines.append(f'{indent_str}            explicitlm_root = potential_root')
                    modified_source_lines.append(f'{indent_str}            if str(explicitlm_root) not in sys.path:')
                    modified_source_lines.append(f'{indent_str}                sys.path.insert(0, str(explicitlm_root))')
                    modified_source_lines.append(f'{indent_str}            break')
                    modified_source_lines.append(f'{indent_str}')
                    modified_source_lines.append(f'{indent_str}# 执行导入')
                    modified_source_lines.append(f'{indent_str}try:')
                    modified_source_lines.append(f'{indent_str}    from models.core.ExplicitLM import ExplicitLM')
                    modified_source_lines.append(f'{indent_str}except ImportError as e:')
                    modified_source_lines.append(f'{indent_str}    raise ImportError(f"无法找到 ExplicitLM 模块。请设置 EXPLICITLM_ROOT 环境变量指向项目根目录（当前尝试的路径: {{explicitlm_root}}）。错误: {{e}}")')
                    
                    # 跳过原始的 try-except 块
                    in_try_except_block = True
                    try_indent = indent
                    i += 1
                    continue
            
            if in_try_except_block:
                # 跳过 try-except 块内的所有行
                if line_stripped.startswith('except'):
                    # 找到 except 块，继续跳过直到 except 块结束
                    except_indent = len(line) - len(line.lstrip())
                    i += 1
                    while i < len(model_source_lines):
                        next_line = model_source_lines[i]
                        if not next_line.strip():
                            i += 1
                            continue
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= except_indent and next_line.strip():
                            in_try_except_block = False
                            break
                        i += 1
                    continue
                else:
                    # 仍在 try-except 块中，跳过
                    i += 1
                    continue
            
            modified_source_lines.append(line)
            i += 1
        
        modified_model_source = '\n'.join(modified_source_lines)
    
    modeling_content = f'''"""
ExplicitLM模型定义 - 用于HuggingFace自动加载模型
此文件由 pt2hg_sync.py 自动生成，包含完整的类定义
"""
import os
import sys
from pathlib import Path

# 导入transformers
try:
    from transformers import (
        Qwen3Config,
        Qwen3PreTrainedModel,
        PretrainedConfig,
    )
    from transformers.modeling_outputs import CausalLMOutputWithPast
    try:
        from transformers import GenerationMixin
    except ImportError:
        GenerationMixin = object
    import torch
    import torch.nn as nn
except ImportError as e:
    raise ImportError(f"需要安装transformers库: {{e}}")

# 导入配置类（从同一目录，使用相对导入）
try:
    from .configuration_explicitlm import ExplicitLMConfig
except ImportError:
    # 如果相对导入失败，尝试动态导入
    import importlib.util
    config_path = Path(__file__).parent / "configuration_explicitlm.py"
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("configuration_explicitlm", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        ExplicitLMConfig = config_module.ExplicitLMConfig
    else:
        raise ImportError(f"无法找到 configuration_explicitlm.py 文件: {config_path}")

{modified_model_source}
'''
    
    modeling_path = os.path.join(output_path, "modeling_explicitlm.py")
    with open(modeling_path, 'w', encoding='utf-8') as f:
        f.write(modeling_content)
    print(f"   ✅ 模型定义文件已保存: {modeling_path}")


def create_inference_script(output_path: str):
    """创建推理示例脚本"""
    # 获取项目根目录（相对于output_path）
    # output_path通常是相对于当前工作目录的路径，我们需要找到项目根目录
    project_root = Path(__file__).parent.resolve()
    
    script_content = f'''#!/usr/bin/env python3
"""
ExplicitLM推理示例
"""
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 设置EXPLICITLM_ROOT环境变量（如果未设置）
    if "EXPLICITLM_ROOT" not in os.environ:
        # 方法1: 从当前脚本所在目录向上查找项目根目录
        current_file = Path(__file__).resolve()
        # 从inference_example.py向上到项目根目录
        # 如果inference_example.py在hf_model_path/目录下，需要向上两级
        project_root = current_file.parent.parent  # 从hf_model_path/向上到ExplicitLM/
        
        # 检查是否是项目根目录（包含models目录）
        if (project_root / "models" / "core" / "ExplicitLM.py").exists():
            os.environ["EXPLICITLM_ROOT"] = str(project_root)
            print(f"✅ 自动设置 EXPLICITLM_ROOT={{project_root}}")
        else:
            # 方法2: 尝试从output_path推断
            # 如果output_path是绝对路径，尝试找到项目根目录
            model_path = "{output_path}"
            if os.path.isabs(model_path):
                # 如果是绝对路径，向上查找
                model_dir = Path(model_path).parent
                for potential_root in [model_dir.parent, model_dir.parent.parent]:
                    if (potential_root / "models" / "core" / "ExplicitLM.py").exists():
                        os.environ["EXPLICITLM_ROOT"] = str(potential_root)
                        print(f"✅ 自动设置 EXPLICITLM_ROOT={{potential_root}}")
                        break
            else:
                # 如果是相对路径，尝试从当前工作目录查找
                cwd = Path.cwd()
                if (cwd / "models" / "core" / "ExplicitLM.py").exists():
                    os.environ["EXPLICITLM_ROOT"] = str(cwd)
                    print(f"✅ 自动设置 EXPLICITLM_ROOT={{cwd}}")
                else:
                    # 尝试使用已知的项目根目录
                    known_root = Path(r"{project_root}")
                    if known_root.exists() and (known_root / "models" / "core" / "ExplicitLM.py").exists():
                        os.environ["EXPLICITLM_ROOT"] = str(known_root)
                        print(f"✅ 自动设置 EXPLICITLM_ROOT={{known_root}}")
                    else:
                        print(f"⚠️  无法自动找到项目根目录，请手动设置 EXPLICITLM_ROOT 环境变量")
                        print(f"   当前脚本: {{current_file}}")
                        print(f"   尝试的路径: {{project_root}}, {{cwd}}, {{known_root}}")
    
    model_path = "{output_path}"
    
    print(f"加载模型: {{model_path}}")
    # trust_remote_code=True 会自动加载模型目录中的 modeling_explicitlm.py
    # fix_mistral_regex=True 修复 Mistral 正则警告
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, fix_mistral_regex=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    prompt = "你好，请介绍一下人工智能"
    print(f"输入: {{prompt}}")
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"回复: {{response}}")

if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(output_path, "inference_example.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"📝 推理示例脚本已创建: {script_path}")


def diagnose_checkpoint(checkpoint_path: str):
    """诊断checkpoint文件，帮助定位问题"""
    print("\n" + "=" * 60)
    print("🔍 Checkpoint诊断")
    print("=" * 60)
    
    try:
        state_dict, checkpoint_info = load_checkpoint(checkpoint_path)
        
        print(f"\n✅ Checkpoint加载成功")
        print(f"   参数数量: {len(state_dict)}")
        print(f"   train_memory_gate: {checkpoint_info.get('train_memory_gate', 'N/A')}")
        print(f"   saved_at_step: {checkpoint_info.get('saved_at_step', 'N/A')}")
        
        print(f"\n📊 权重名称分析:")
        fusion_keys = [k for k in state_dict.keys() if 'gated_memory_fusion' in k]
        norm_keys = [k for k in state_dict.keys() if 'memory_norm' in k]
        gate_keys = [k for k in state_dict.keys() if 'memory_gate' in k and 'keys' not in k]
        buffer_keys = [k for k in state_dict.keys() if k in ['memory_bank', 'valid_mask']]
        
        shared_gate_pattern1 = re.compile(r'^(module\.)?shared_memory_gate\.')
        shared_gate_pattern2 = re.compile(r'^(module\.)?layers\.\d+\.memory_gate\.')
        
        shared_gate_keys = [k for k in gate_keys if shared_gate_pattern1.match(k) or shared_gate_pattern2.match(k)]
        unique_shared_gate_keys = set()
        for k in shared_gate_keys:
            match1 = re.match(r'^(module\.)?shared_memory_gate\.(.+)$', k)
            match2 = re.match(r'^(module\.)?layers\.\d+\.memory_gate\.(.+)$', k)
            if match1:
                unique_shared_gate_keys.add(match1.group(2))
            elif match2:
                unique_shared_gate_keys.add(match2.group(2))
        
        print(f"   GatedMemoryFusion: {len(fusion_keys)} 个")
        print(f"   MemoryNorm: {len(norm_keys)} 个")
        print(f"   MemoryGate: {len(gate_keys)} 个")
        if shared_gate_keys:
            print(f"     - 检测到共享MemoryGate格式: {len(shared_gate_keys)} 个（来自不同层）")
            print(f"     - 唯一MemoryGate子模块: {len(unique_shared_gate_keys)} 个")
            print(f"     - 示例子模块: {list(unique_shared_gate_keys)[:3]}")
        print(f"   Buffers: {len(buffer_keys)} 个")
        
        print(f"\n📋 权重名称格式检查:")
        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
        has_model_prefix = any(k.startswith('model.') for k in state_dict.keys())
        has_layers_prefix = any(k.startswith('layers.') for k in state_dict.keys())
        
        print(f"   包含 'module.' 前缀: {has_module_prefix}")
        print(f"   包含 'model.' 前缀: {has_model_prefix}")
        print(f"   包含 'layers.' 前缀: {has_layers_prefix}")
        
        return True
    except Exception as e:
        print(f"\n❌ Checkpoint诊断失败: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="将ExplicitLM checkpoint转换为HuggingFace格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用checkpoint目录
  python pt2hg_sync.py \\
      --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_7500 \\
      --qwen3_path Qwen_hg/Qwen3-4b \\
      --output_path hf_explicitlm_model
  
  # 直接使用pth文件
  python pt2hg_sync.py \\
      --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_7500/trainable_components.pth \\
      --qwen3_path Qwen_hg/Qwen3-4b \\
      --output_path hf_explicitlm_model
  
  # 诊断checkpoint（不进行转换）
  python pt2hg_sync.py \\
      --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_7500 \\
      --diagnose
        """
    )
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="checkpoint文件或目录路径")
    parser.add_argument("--qwen3_path", type=str, required=True,
                       help="Qwen3基础模型路径")
    parser.add_argument("--output_path", type=str, required=True,
                       help="输出HF模型路径")
    parser.add_argument("--keys_path", type=str, default=None,
                       help="Keys文件路径（可选）")
    parser.add_argument("--memory_bank_path", type=str, default=None,
                       help="Memory Bank文件路径（可选）")
    parser.add_argument("--knowledge_num", type=int, default=1048576,
                       help="记忆库大小")
    parser.add_argument("--knowledge_length", type=int, default=32,
                       help="记忆条目长度")
    parser.add_argument("--num_candidates", type=int, default=16,
                       help="候选记忆数")
    parser.add_argument("--num_selected", type=int, default=1,
                       help="选中记忆数")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0,
                       help="Gumbel-Softmax温度")
    parser.add_argument("--diagnose", action="store_true",
                       help="仅诊断checkpoint，不进行转换")
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_checkpoint(args.checkpoint_path)
        return
    
    try:
        output_path = convert_to_hf_format(
            checkpoint_path=args.checkpoint_path,
            qwen3_path=args.qwen3_path,
            output_path=args.output_path,
            keys_path=args.keys_path,
            memory_bank_path=args.memory_bank_path,
            knowledge_num=args.knowledge_num,
            knowledge_length=args.knowledge_length,
            num_candidates=args.num_candidates,
            num_selected=args.num_selected,
            gumbel_temperature=args.gumbel_temperature,
        )
        
        create_inference_script(output_path)
        
        print("\n" + "=" * 60)
        print("🎉 转换完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
