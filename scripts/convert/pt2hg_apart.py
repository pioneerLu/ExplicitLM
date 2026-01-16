#!/usr/bin/env python3
"""
将ExplicitLM训练的checkpoint转换为HuggingFace格式（Memory Bank独立存储版本）

与pt2hg_sync.py的区别：
- Memory Bank不保存在模型文件中，单独保存为memory_bank.pt
- 支持动态加载和切换不同的Memory Bank
- 模型文件更小，加载更快

使用方法:
    uv run python pt2hg_apart.py \
        --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_14500 \
        --qwen3_path Qwen_hg/Qwen3-4b \
        --output_path hf_explicitlm_model_apart \
        --memory_bank_path data/memory_bank.pt  
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
    """ExplicitLM的配置文件，兼容Hugging Face格式（Memory Bank独立存储版本）"""
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
        memory_bank_path: Optional[str] = None,  # Memory Bank文件路径（相对或绝对路径）
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
        self.memory_bank_path = memory_bank_path  # Memory Bank独立存储路径


class ExplicitLMForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Hugging Face兼容的ExplicitLM因果语言模型（Memory Bank独立存储版本）"""
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
        
        # 在__init__阶段，config._name_or_path可能还未设置，所以不在这里加载
        # Memory Bank会在from_pretrained时加载

    def _get_input_device(self):
        """智能获取模型输入应该放置的设备（支持 device_map="auto" 和多卡）"""
        # 方法1: 如果模型使用了 device_map，从 device_map 获取第一个设备
        if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
            first_device = next(iter(self.model.hf_device_map.values()))
            if isinstance(first_device, (list, tuple)):
                first_device = first_device[0]
            return torch.device(first_device)
        
        # 方法2: 尝试获取 embedding 层所在的设备（输入总是先经过 embedding）
        try:
            if hasattr(self.model, 'get_input_embeddings'):
                embedding_layer = self.model.get_input_embeddings()
                if hasattr(embedding_layer, 'weight'):
                    return embedding_layer.weight.device
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                return self.model.model.embed_tokens.weight.device
        except Exception:
            pass
        
        # 方法3: 使用第一个参数所在的设备（兜底方案）
        return next(self.model.parameters()).device
    
    def load_memory_bank(self, path: str):
        """从文件加载Memory Bank"""
        # 首先尝试直接路径
        if os.path.exists(path):
            pass  # 路径有效，继续
        else:
            # 尝试相对路径（相对于模型目录）
            model_dir = None
            if hasattr(self.config, '_name_or_path') and self.config._name_or_path:
                # 从config获取模型目录
                if isinstance(self.config._name_or_path, str):
                    model_dir = Path(self.config._name_or_path)
                    if model_dir.is_file():
                        model_dir = model_dir.parent
                else:
                    model_dir = Path(self.config._name_or_path)
            
            if model_dir and model_dir.exists():
                abs_path = model_dir / path
                if abs_path.exists():
                    path = str(abs_path)
                else:
                    # 如果还是找不到，尝试模型目录下的memory_bank.pt（默认位置）
                    default_path = model_dir / "memory_bank.pt"
                    if default_path.exists():
                        path = str(default_path)
                    else:
                        print(f"⚠️ Memory Bank文件不存在: {path} 和 {default_path}，将使用默认初始化")
                        return
            else:
                # 如果无法确定模型目录，尝试当前目录和常见路径
                current_dir = Path.cwd()
                potential_paths = [
                    current_dir / path,
                    current_dir / "memory_bank.pt",
                ]
                found = False
                for potential_path in potential_paths:
                    if potential_path.exists():
                        path = str(potential_path)
                        found = True
                        break
                if not found:
                    print(f"⚠️ Memory Bank文件不存在: {path}，将使用默认初始化")
                    return
        
        try:
            data = torch.load(path, map_location='cpu')
            if isinstance(data, dict):
                memory_bank = data.get('memory_bank', data.get('processed_tensor'))
                valid_mask = data.get('valid_mask', None)
            else:
                memory_bank = data
                valid_mask = None
            
            if memory_bank is not None and hasattr(self.model, 'memory_bank'):
                # 检查形状是否匹配
                if memory_bank.shape[0] > self.model.memory_bank.shape[0]:
                    memory_bank = memory_bank[:self.model.memory_bank.shape[0]]
                elif memory_bank.shape[0] < self.model.memory_bank.shape[0]:
                    pad_token_id = getattr(self.config, 'pad_token_id', 0) or 0
                    padding = torch.full(
                        (self.model.memory_bank.shape[0] - memory_bank.shape[0], memory_bank.shape[1]),
                        pad_token_id,
                        dtype=memory_bank.dtype
                    )
                    memory_bank = torch.cat([memory_bank, padding], dim=0)
                
                self.model.memory_bank.data.copy_(memory_bank.cpu())
                print(f"✅ Memory Bank已加载: {memory_bank.shape} from {path}")
                
                # 处理 valid_mask：如果不存在，自动生成（基于非全pad条目）
                if valid_mask is None:
                    pad_token_id = getattr(self.config, 'pad_token_id', 0) or 0
                    is_all_pad = (memory_bank == pad_token_id).all(dim=-1)
                    valid_mask = ~is_all_pad
                    print(f"  ⚠️  未找到 valid_mask，自动生成: {valid_mask.sum().item()}/{len(valid_mask)} 个有效条目")
                
                if hasattr(self.model, 'valid_mask'):
                    if valid_mask.shape[0] > self.model.valid_mask.shape[0]:
                        valid_mask = valid_mask[:self.model.valid_mask.shape[0]]
                    elif valid_mask.shape[0] < self.model.valid_mask.shape[0]:
                        padding_mask = torch.zeros(self.model.valid_mask.shape[0] - valid_mask.shape[0], dtype=torch.bool)
                        valid_mask = torch.cat([valid_mask, padding_mask], dim=0)
                    self.model.valid_mask.data.copy_(valid_mask.cpu())
                    print(f"✅ Valid Mask已加载: {valid_mask.sum().item()}/{valid_mask.shape[0]} 有效")
        except Exception as e:
            print(f"⚠️ 加载Memory Bank失败: {e}")

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """前向传播，兼容Hugging Face接口"""
        # 智能获取模型输入设备（支持 device_map="auto" 和多卡场景）
        input_device = self._get_input_device()
        
        # 确保输入在正确的设备上
        if input_ids is not None and input_ids.device != input_device:
            input_ids = input_ids.to(input_device)
        if attention_mask is not None and attention_mask.device != input_device:
            attention_mask = attention_mask.to(input_device)
        # 确保 kwargs 中的 tensor 也在正确的设备上
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.device != input_device:
                kwargs[key] = value.to(input_device)
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        
        # ExplicitLM.forward 返回的是 CausalLMOutputWithPast，已经包含 logits
        if isinstance(outputs, CausalLMOutputWithPast):
            # 直接使用 outputs 中的 logits，不需要重新计算
            return outputs
        elif isinstance(outputs, tuple):
            # 兼容旧接口（tuple格式）
            hidden_states, loss, aux_loss = outputs[:3]
            past_key_values = None
            logits = self.model.lm_head(hidden_states)
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
                attentions=None,
            )
        else:
            # 兜底：如果 outputs 不是预期格式，尝试提取信息
            hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
            past_key_values = outputs.past_key_values if hasattr(outputs, 'past_key_values') else None
            logits = outputs.logits if hasattr(outputs, 'logits') else self.model.lm_head(hidden_states)
            loss = getattr(outputs, 'loss', None)
            
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
        is_first_forward = (past_key_values is None) or (
            hasattr(past_key_values, 'get_seq_length') and past_key_values.get_seq_length() == 0
        )
        
        if past_key_values is not None and not is_first_forward:
            input_ids = input_ids[:, -1:]
            if attention_mask is not None:
                past_seen_tokens = past_key_values.get_seq_length()
                current_seq_len = input_ids.shape[1]
                expected_len = past_seen_tokens + current_seq_len
                if attention_mask.shape[1] < expected_len:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], expected_len - attention_mask.shape[1]),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device
                        )
                    ], dim=-1)
                elif attention_mask.shape[1] > expected_len:
                    attention_mask = attention_mask[:, :expected_len]
        
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        
        if "cache_position" not in kwargs or kwargs.get("cache_position") is None:
            if past_key_values is not None and not is_first_forward:
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + input_ids.shape[1], 
                    device=input_ids.device, dtype=torch.long
                )
            else:
                cache_position = torch.arange(
                    0, input_ids.shape[1], 
                    device=input_ids.device, dtype=torch.long
                )
            model_inputs["cache_position"] = cache_position
        
        excluded_keys = {"input_ids", "past_key_values", "attention_mask", "use_cache", "cache_position"}
        model_inputs.update({k: v for k, v in kwargs.items() if k not in excluded_keys})
        
        return model_inputs

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """自定义加载逻辑，排除memory_bank和valid_mask"""
        # 过滤掉memory_bank和valid_mask
        filtered_state_dict = {k: v for k, v in state_dict.items() 
                              if 'memory_bank' not in k and 'valid_mask' not in k}
        super()._load_from_state_dict(filtered_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        self._restore_shared_weights()
    
    def _restore_shared_weights(self):
        """重新建立共享权重关系"""
        if not hasattr(self.model, 'shared_memory_gate') or self.model.shared_memory_gate is None:
            return
        for layer in self.model.layers:
            if hasattr(layer, 'memory_gate'):
                layer.memory_gate = self.model.shared_memory_gate

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
        """从预训练权重加载模型"""
        device_map = kwargs.get('device_map', None)
        if device_map is not None:
            # 临时移除 device_map，先加载模型，然后再手动应用
            kwargs_no_device_map = {k: v for k, v in kwargs.items() if k != 'device_map'}
            model = super(ExplicitLMForCausalLM, ExplicitLMForCausalLM).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs_no_device_map
            )
            # 应用 device_map（处理 memory_bank 和 valid_mask）
            model = _apply_device_map(model, device_map, kwargs.get('torch_dtype'))
        else:
            model = super(ExplicitLMForCausalLM, ExplicitLMForCausalLM).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )
        if isinstance(model, ExplicitLMForCausalLM):
            model._restore_shared_weights()
            
            # 解析文件路径
            model_dir = Path(pretrained_model_name_or_path).resolve()
            if not model_dir.exists():
                model_dir = Path(pretrained_model_name_or_path)
            
            def resolve_path(filename: str, config_attr: str = None) -> Optional[str]:
                """解析文件路径：优先模型目录，其次 config，最后默认"""
                if model_dir.exists() and (model_dir / filename).exists():
                    return str(model_dir / filename)
                if config_attr and hasattr(model.config, config_attr):
                    config_path = getattr(model.config, config_attr)
                    if os.path.exists(config_path):
                        return os.path.abspath(config_path)
                return None
            
            # 确定文件路径
            memory_bank_path = resolve_path("memory_bank.pt", "memory_bank_path") or "memory_bank.pt"
            
            # 加载 Memory Bank
            model.load_memory_bank(memory_bank_path)
            
            # 如果使用了 device_map，将 memory_bank 和 valid_mask 移到正确的设备
            if device_map is not None and hasattr(model.model, 'memory_bank') and hasattr(model.model, 'valid_mask'):
                target_device = next(model.model.parameters()).device
                if model.model.memory_bank.device != target_device:
                    model.model.memory_bank = model.model.memory_bank.to(target_device)
                if model.model.valid_mask.device != target_device:
                    model.model.valid_mask = model.model.valid_mask.to(target_device)
        return model

    def save_pretrained(self, save_directory, **kwargs):
        """保存模型，但不包含memory_bank和valid_mask"""
        # 获取原始state_dict
        original_state_dict = self.state_dict()
        
        # 排除memory_bank和valid_mask
        filtered_state_dict = {k: v for k, v in original_state_dict.items() 
                              if 'memory_bank' not in k and 'valid_mask' not in k}
        
        # 临时替换state_dict方法
        original_state_dict_method = self.state_dict
        def filtered_state_dict_method(*args, **kwargs):
            return filtered_state_dict
        self.state_dict = filtered_state_dict_method
        
        try:
            # 调用父类保存方法
            super().save_pretrained(save_directory, **kwargs)
        finally:
            # 恢复原始方法
            self.state_dict = original_state_dict_method


def map_qwen3_weight_name(qwen3_name: str) -> str:
    """将Qwen3权重名称映射到ExplicitLM格式"""
    if qwen3_name.startswith('model.'):
        key = qwen3_name[6:]
    else:
        key = qwen3_name
    
    if key.startswith('layers.'):
        parts = key.split('.', 2)
        if len(parts) >= 3:
            layer_idx = parts[1]
            rest = parts[2]
            return f'layers.{layer_idx}.qwen3_decoder.{rest}'
    
    return key


def map_explicitlm_weight_name(explicitlm_name: str) -> str:
    """将ExplicitLM组件权重名称映射到HF格式"""
    if explicitlm_name.startswith('module.'):
        return 'model.' + explicitlm_name[7:]
    else:
        return 'model.' + explicitlm_name


def create_explicitlm_config(qwen3_path: str, memory_config: Optional[Dict[str, Any]] = None, memory_bank_path: Optional[str] = None) -> ExplicitLMConfig:
    """基于Qwen3配置创建ExplicitLM配置"""
    try:
        qwen3_config = Qwen3Config.from_pretrained(qwen3_path)
    except Exception as e:
        print(f"⚠️ 无法加载Qwen3配置，使用默认值: {e}")
        qwen3_config = Qwen3Config()

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
        knowledge_num=memory_config.get('knowledge_num', 10000) if memory_config else 10000,
        knowledge_length=memory_config.get('knowledge_length', 32) if memory_config else 32,
        num_candidates=memory_config.get('num_candidates', 16) if memory_config else 16,
        num_selected=memory_config.get('num_selected', 1) if memory_config else 1,
        gumbel_temperature=memory_config.get('gumbel_temperature', 1.0) if memory_config else 1.0,
        use_memory_gate=memory_config.get('use_memory_gate', True) if memory_config else True,
        memory_bank_path=memory_bank_path,  # Memory Bank路径
    )


def _load_memory_bank_file(memory_bank_path: str) -> tuple:
    """加载 Memory Bank 文件，返回 (memory_bank, valid_mask, metadata)"""
    mb_data = torch.load(memory_bank_path, map_location='cpu')
    if isinstance(mb_data, dict):
        mb_tensor = mb_data.get('memory_bank', mb_data.get('processed_tensor'))
        mb_valid_mask = mb_data.get('valid_mask', None)
        mb_metadata = mb_data.get('metadata', {})
    else:
        mb_tensor = mb_data
        mb_valid_mask = None
        mb_metadata = {}
    return mb_tensor, mb_valid_mask, mb_metadata


def _extract_dataset_name(memory_bank_path: str, metadata: dict = None) -> Optional[str]:
    """从 metadata 或文件路径提取 dataset_name"""
    if metadata and metadata.get('dataset_name'):
        return metadata['dataset_name']
    try:
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(memory_bank_path)))
        return parent_dir if parent_dir else None
    except:
        return None


def _resize_memory_bank(memory_bank: torch.Tensor, knowledge_num: int, knowledge_length: int, pad_token_id: int) -> torch.Tensor:
    """调整 Memory Bank 大小以匹配 knowledge_num"""
    if memory_bank.shape[0] > knowledge_num:
        return memory_bank[:knowledge_num]
    elif memory_bank.shape[0] < knowledge_num:
        padding = torch.full(
            (knowledge_num - memory_bank.shape[0], knowledge_length),
            pad_token_id,
            dtype=memory_bank.dtype
        )
        return torch.cat([memory_bank, padding], dim=0)
    return memory_bank


def _resize_valid_mask(valid_mask: torch.Tensor, knowledge_num: int) -> torch.Tensor:
    """调整 Valid Mask 大小以匹配 knowledge_num"""
    if valid_mask.shape[0] > knowledge_num:
        return valid_mask[:knowledge_num]
    elif valid_mask.shape[0] < knowledge_num:
        padding_mask = torch.zeros(knowledge_num - valid_mask.shape[0], dtype=torch.bool)
        return torch.cat([valid_mask, padding_mask], dim=0)
    return valid_mask


def _copy_weight_safe(target_dict: dict, target_name: str, source_weight: torch.Tensor) -> bool:
    """安全地复制权重，返回是否成功"""
    if target_name in target_dict:
        try:
            if target_dict[target_name].shape == source_weight.shape:
                target_dict[target_name].copy_(source_weight)
                return True
        except:
            pass
    return False


def _apply_device_map(model, device_map, torch_dtype=None):
    """应用 device_map，处理 memory_bank 和 valid_mask"""
    from accelerate import dispatch_model, infer_auto_device_map
    
    # 将 memory_bank 和 valid_mask 移到 CPU（避免被 device_map 检查）
    if hasattr(model.model, 'memory_bank') and model.model.memory_bank is not None:
        model.model.memory_bank = model.model.memory_bank.cpu()
    if hasattr(model.model, 'valid_mask') and model.model.valid_mask is not None:
        model.model.valid_mask = model.model.valid_mask.cpu()
    
    # 转换 device_map="auto" 为实际设备映射字典
    if device_map == "auto":
        try:
            if torch_dtype is None:
                torch_dtype = next(model.parameters()).dtype
            actual_device_map = infer_auto_device_map(model, max_memory=None, dtype=torch_dtype)
            # 移除 memory_bank 和 valid_mask（会被手动处理）
            if isinstance(actual_device_map, dict):
                actual_device_map = {k: v for k, v in actual_device_map.items() 
                                   if 'memory_bank' not in k and 'valid_mask' not in k}
            device_map = actual_device_map
        except Exception as e:
            print(f"⚠️  无法自动推断设备映射: {e}，使用手动分配")
            device_map = None
    
    # 应用 device_map
    if device_map:
        try:
            model = dispatch_model(model, device_map=device_map)
        except Exception as e:
            print(f"⚠️  dispatch_model 失败: {e}，使用手动分配")
            device_map = None
    
    # 获取目标设备
    if device_map and hasattr(model.model, "hf_device_map") and model.model.hf_device_map:
        first_device = next(iter(model.model.hf_device_map.values()))
        if isinstance(first_device, (list, tuple)):
            first_device = first_device[0]
        target_device = torch.device(first_device)
    else:
        target_device = next(model.model.parameters()).device
    
    # 将 memory_bank 和 valid_mask 移到目标设备
    if hasattr(model.model, 'memory_bank') and model.model.memory_bank is not None:
        model.model.memory_bank = model.model.memory_bank.to(target_device)
    if hasattr(model.model, 'valid_mask') and model.model.valid_mask is not None:
        model.model.valid_mask = model.model.valid_mask.to(target_device)
    
    return model


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
    memory_bank_path: Optional[str] = None,
    knowledge_num: int = 10000,
    knowledge_length: int = 32,
    num_candidates: int = 16,
    num_selected: int = 1,
    gumbel_temperature: float = 1.0,
) -> str:
    """将checkpoint转换为HuggingFace格式（Memory Bank独立存储）"""
    print("=" * 60)
    print("🔄 开始转换ExplicitLM checkpoint到HuggingFace格式（Memory Bank独立存储）")
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
    
    # 设置memory_bank_path
    # 注意：无论是否提供memory_bank_path，最终都会保存到模型目录下的memory_bank.pt
    # 所以config中统一使用相对路径"memory_bank.pt"
    config_memory_bank_path = "memory_bank.pt"
    
    config = create_explicitlm_config(qwen3_path, memory_config, config_memory_bank_path)
    
    # 2.1. 添加 auto_map 属性
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
    
    # 5. 合并权重（排除memory_bank和valid_mask）
    print("\n🔗 步骤5: 合并权重（排除Memory Bank）...")
    hf_state_dict = hf_model.state_dict()
    
    # 复制Qwen3基础权重
    qwen3_applied = 0
    for name, weight in qwen3_state_dict.items():
        hf_name = map_qwen3_weight_name(name)
        if not hf_name.startswith('model.'):
            hf_name = 'model.' + hf_name
        if _copy_weight_safe(hf_state_dict, hf_name, weight):
            qwen3_applied += 1
    print(f"✅ 应用Qwen3基础权重: {qwen3_applied} 个")
    
    # 应用ExplicitLM组件权重（排除memory_bank和valid_mask）
    explicitlm_applied = 0
    shared_memory_gate_loaded = set()
    
    for name, weight in explicitlm_state_dict.items():
        if not isinstance(weight, torch.Tensor) or 'memory_bank' in name or 'valid_mask' in name:
            continue
        
        # 处理共享MemoryGate权重
        shared_gate_match = re.match(r'^(module\.)?(shared_memory_gate|layers\.\d+\.memory_gate)\.(.+)$', name)
        if shared_gate_match:
            hf_shared_gate_name = f'model.shared_memory_gate.{shared_gate_match.group(3)}'
            if hf_shared_gate_name not in shared_memory_gate_loaded:
                if _copy_weight_safe(hf_state_dict, hf_shared_gate_name, weight):
                    shared_memory_gate_loaded.add(hf_shared_gate_name)
                    explicitlm_applied += 1
                    continue
        
        # 标准映射
        hf_name = map_explicitlm_weight_name(name)
        if _copy_weight_safe(hf_state_dict, hf_name, weight):
            explicitlm_applied += 1
    
    print(f"✅ 应用ExplicitLM训练权重: {explicitlm_applied} 个")
    
    # 6. 加载权重到模型
    print("\n📥 步骤6: 加载合并后的权重到模型...")
    try:
        missing_keys, unexpected_keys = hf_model.load_state_dict(hf_state_dict, strict=False)
    except Exception as e:
        print(f"❌ 加载权重到模型失败: {e}")
        import traceback
        print(f"错误详情:\n{traceback.format_exc()}")
        raise
    
    # 7. 加载并保存Memory Bank（如果提供）
    # 先创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    memory_bank_saved = False
    if memory_bank_path and os.path.exists(memory_bank_path):
        print(f"\n📥 步骤7: 加载Memory Bank数据...")
        try:
            memory_bank, valid_mask, mb_metadata = _load_memory_bank_file(memory_bank_path)
            
            if memory_bank is not None and hasattr(hf_model.model, 'memory_bank'):
                pad_token_id = getattr(config, 'pad_token_id', 0) or 0
                memory_bank = _resize_memory_bank(memory_bank, knowledge_num, knowledge_length, pad_token_id)
                
                if valid_mask is not None:
                    valid_mask = _resize_valid_mask(valid_mask, knowledge_num)
                
                # 保存到独立文件
                output_memory_bank_path = os.path.join(output_path, "memory_bank.pt")
                torch.save({
                    'memory_bank': memory_bank.cpu(),
                    'valid_mask': valid_mask.cpu() if valid_mask is not None else None,
                    'metadata': {
                        'knowledge_num': memory_bank.shape[0],
                        'knowledge_length': memory_bank.shape[1],
                        'source_path': memory_bank_path,
                        **mb_metadata,
                    }
                }, output_memory_bank_path)
                print(f"✅ Memory Bank已保存到独立文件: {output_memory_bank_path}")
                
                # 加载到模型
                hf_model.model.memory_bank.data.copy_(memory_bank)
                print(f"✅ Memory Bank已加载到模型: {memory_bank.shape}")
                
                if valid_mask is not None and hasattr(hf_model.model, 'valid_mask'):
                    hf_model.model.valid_mask.data.copy_(valid_mask)
                    print(f"✅ Valid Mask已加载: {valid_mask.sum().item()}/{valid_mask.shape[0]} 有效")
                
                memory_bank_saved = True
        except Exception as e:
            print(f"⚠️ 加载Memory Bank失败: {e}")
    
    # 8. 保存为HuggingFace格式
    print(f"\n💾 步骤8: 保存为HuggingFace格式: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    
    print("   保存模型权重和配置（不包含Memory Bank）...")
    # 使用自定义的save_pretrained方法，排除memory_bank
    hf_model.save_pretrained(output_path, safe_serialization=False)
    print("   ✅ 已保存为 PyTorch 格式（Memory Bank已独立存储）")
    
    # 如果Memory Bank未保存，创建一个空的占位文件
    if not memory_bank_saved:
        output_memory_bank_path = os.path.join(output_path, "memory_bank.pt")
        pad_token_id = getattr(config, 'pad_token_id', 0) or 0
        empty_memory_bank = torch.full(
            (knowledge_num, knowledge_length), pad_token_id, dtype=torch.long
        )
        empty_valid_mask = torch.zeros(knowledge_num, dtype=torch.bool)
        memory_bank_save_data = {
            'memory_bank': empty_memory_bank,
            'valid_mask': empty_valid_mask,
            'metadata': {
                'knowledge_num': knowledge_num,
                'knowledge_length': knowledge_length,
                'note': 'Empty memory bank, initialized with pad tokens',
            }
        }
        torch.save(memory_bank_save_data, output_memory_bank_path)
        print(f"   ✅ 已创建空的Memory Bank文件: {output_memory_bank_path}")
    
    # 保存tokenizer
    print("   保存tokenizer...")
    from transformers import AutoTokenizer
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
        "memory_bank_storage": "separate",  # 标记为独立存储
    }
    
    import json
    with open(os.path.join(output_path, "checkpoint_info.json"), 'w', encoding='utf-8') as f:
        json.dump(checkpoint_meta, f, indent=2, ensure_ascii=False)
    
    # 保存模型定义文件
    _save_modeling_file(output_path)
    
    # 复制核心代码到模型目录（自包含模式）
    _copy_core_files_to_model_dir(output_path)
    
    print(f"\n✅ 转换完成！")
    print(f"📁 HF模型保存路径: {output_path}")
    print(f"📁 Memory Bank独立文件: {os.path.join(output_path, 'memory_bank.pt')}")
    if keys_saved:
        print(f"📁 Keys 文件: {os.path.join(output_path, 'keys.pt')}")
    print(f"\n💡 使用方法:")
    print(f"   from transformers import AutoTokenizer, AutoModelForCausalLM")
    print(f"   tokenizer = AutoTokenizer.from_pretrained('{output_path}')")
    print(f"   model = AutoModelForCausalLM.from_pretrained('{output_path}', trust_remote_code=True)")
    print(f"   # Memory Bank会自动从 {os.path.join(output_path, 'memory_bank.pt')} 加载")
    if keys_saved:
        print(f"   # Keys 文件位于: {os.path.join(output_path, 'keys.pt')}")
    print(f"   # 或手动加载: model.load_memory_bank('custom_memory_bank.pt')")
    
    return output_path


def _save_modeling_file(output_path: str):
    """保存模型定义文件，包含load_memory_bank方法"""
    # 1. 创建 configuration_explicitlm.py
    config_source = inspect.getsource(ExplicitLMConfig)
    config_content = f'''"""
ExplicitLM配置定义 - 用于HuggingFace自动加载配置（Memory Bank独立存储版本）
此文件由 pt2hg_apart.py 自动生成
"""
from transformers import PretrainedConfig
from typing import Optional

{config_source}
'''
    
    config_path = os.path.join(output_path, "configuration_explicitlm.py")
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"   ✅ 配置定义文件已保存: {config_path}")
    
    # 2. 创建 modeling_explicitlm.py（包含load_memory_bank方法）
    model_source = inspect.getsource(ExplicitLMForCausalLM)
    
    # 修改导入部分
    import re
    pattern = r'(\s+)# 导入ExplicitLM类\s*\n\s+try:\s*\n\s+from models\.core\.ExplicitLM import ExplicitLM\s*\n\s+except ImportError:\s*\n(?:\s+.*\n)*?\s+from models\.core\.ExplicitLM import ExplicitLM'
    
    def replace_import_block(match):
        indent = match.group(1)
        indent_str = indent
        
        replacement = f'''{indent_str}# 导入策略：优先从模型目录导入（自包含模式），然后回退到项目根目录
{indent_str}import sys
{indent_str}from pathlib import Path
{indent_str}import os
{indent_str}
{indent_str}# 策略1: 尝试从模型目录的models子目录导入（自包含模式）
{indent_str}current_file = Path(__file__).resolve()
{indent_str}model_dir = current_file.parent
{indent_str}local_models_path = model_dir / "models" / "core" / "ExplicitLM.py"
{indent_str}explicitlm_root = None
{indent_str}import_success = False
{indent_str}
{indent_str}if local_models_path.exists():
{indent_str}    # 自包含模式：从模型目录导入
{indent_str}    if str(model_dir) not in sys.path:
{indent_str}        sys.path.insert(0, str(model_dir))
{indent_str}    try:
{indent_str}        from models.core.ExplicitLM import ExplicitLM
{indent_str}        import_success = True
{indent_str}    except ImportError:
{indent_str}        pass
{indent_str}
{indent_str}# 策略2: 如果自包含模式失败，尝试通过EXPLICITLM_ROOT环境变量
{indent_str}if not import_success:
{indent_str}    explicitlm_root = os.environ.get("EXPLICITLM_ROOT", None)
{indent_str}    if explicitlm_root and os.path.exists(explicitlm_root):
{indent_str}        if str(explicitlm_root) not in sys.path:
{indent_str}            sys.path.insert(0, str(explicitlm_root))
{indent_str}        try:
{indent_str}            from models.core.ExplicitLM import ExplicitLM
{indent_str}            import_success = True
{indent_str}        except ImportError:
{indent_str}            pass
{indent_str}
{indent_str}# 策略3: 自动查找项目根目录
{indent_str}if not import_success and explicitlm_root is None:
{indent_str}    for potential_root in [current_file.parent.parent, current_file.parent.parent.parent]:
{indent_str}        if potential_root and (potential_root / "models" / "core" / "ExplicitLM.py").exists():
{indent_str}            explicitlm_root = potential_root
{indent_str}            if str(explicitlm_root) not in sys.path:
{indent_str}                sys.path.insert(0, str(explicitlm_root))
{indent_str}            try:
{indent_str}                from models.core.ExplicitLM import ExplicitLM
{indent_str}                import_success = True
{indent_str}                break
{indent_str}            except ImportError:
{indent_str}                pass
{indent_str}
{indent_str}# 如果所有策略都失败，抛出错误
{indent_str}if not import_success:
{indent_str}    error_msg = (
{indent_str}        f"无法找到 ExplicitLM 模块。\\n"
{indent_str}        f"尝试的路径:\\n"
{indent_str}        f"  1. 模型目录 (自包含模式): {{model_dir / 'models' / 'core' / 'ExplicitLM.py'}}\\n"
{indent_str}    )
{indent_str}    if explicitlm_root:
{indent_str}        error_msg += f"  2. EXPLICITLM_ROOT: {{explicitlm_root}}\\n"
{indent_str}    error_msg += (
{indent_str}        f"  3. 自动查找项目根目录: 未找到\\n\\n"
{indent_str}        f"解决方案:\\n"
{indent_str}        f"  - 如果这是自包含模型，请确保模型目录包含 models/core/ExplicitLM.py\\n"
{indent_str}        f"  - 或者设置 EXPLICITLM_ROOT 环境变量指向项目根目录\\n"
{indent_str}        f"  - 或者将模型目录放在项目根目录下"
{indent_str}    )
{indent_str}    raise ImportError(error_msg)'''
        return replacement
    
    modified_model_source = re.sub(pattern, replace_import_block, model_source, flags=re.MULTILINE)
    
    # 如果正则没有匹配，使用逐行处理
    if modified_model_source == model_source:
        model_source_lines = model_source.split('\n')
        modified_source_lines = []
        i = 0
        in_try_except_block = False
        
        while i < len(model_source_lines):
            line = model_source_lines[i]
            line_stripped = line.strip()
            
            if line_stripped == 'try:' and i + 1 < len(model_source_lines):
                next_line = model_source_lines[i + 1]
                if 'from models.core.ExplicitLM import ExplicitLM' in next_line:
                    indent = len(line) - len(line.lstrip())
                    indent_str = ' ' * indent
                    
                    modified_source_lines.append(f'{indent_str}# 导入策略：优先从模型目录导入（自包含模式），然后回退到项目根目录')
                    modified_source_lines.append(f'{indent_str}import sys')
                    modified_source_lines.append(f'{indent_str}from pathlib import Path')
                    modified_source_lines.append(f'{indent_str}import os')
                    modified_source_lines.append(f'{indent_str}')
                    modified_source_lines.append(f'{indent_str}# 策略1: 尝试从模型目录的models子目录导入（自包含模式）')
                    modified_source_lines.append(f'{indent_str}current_file = Path(__file__).resolve()')
                    modified_source_lines.append(f'{indent_str}model_dir = current_file.parent')
                    modified_source_lines.append(f'{indent_str}local_models_path = model_dir / "models" / "core" / "ExplicitLM.py"')
                    modified_source_lines.append(f'{indent_str}explicitlm_root = None')
                    modified_source_lines.append(f'{indent_str}import_success = False')
                    modified_source_lines.append(f'{indent_str}')
                    modified_source_lines.append(f'{indent_str}if local_models_path.exists():')
                    modified_source_lines.append(f'{indent_str}    # 自包含模式：从模型目录导入')
                    modified_source_lines.append(f'{indent_str}    if str(model_dir) not in sys.path:')
                    modified_source_lines.append(f'{indent_str}        sys.path.insert(0, str(model_dir))')
                    modified_source_lines.append(f'{indent_str}    try:')
                    modified_source_lines.append(f'{indent_str}        from models.core.ExplicitLM import ExplicitLM')
                    modified_source_lines.append(f'{indent_str}        import_success = True')
                    modified_source_lines.append(f'{indent_str}    except ImportError:')
                    modified_source_lines.append(f'{indent_str}        pass')
                    modified_source_lines.append(f'{indent_str}')
                    modified_source_lines.append(f'{indent_str}# 策略2: 如果自包含模式失败，尝试通过EXPLICITLM_ROOT环境变量')
                    modified_source_lines.append(f'{indent_str}if not import_success:')
                    modified_source_lines.append(f'{indent_str}    explicitlm_root = os.environ.get("EXPLICITLM_ROOT", None)')
                    modified_source_lines.append(f'{indent_str}    if explicitlm_root and os.path.exists(explicitlm_root):')
                    modified_source_lines.append(f'{indent_str}        if str(explicitlm_root) not in sys.path:')
                    modified_source_lines.append(f'{indent_str}            sys.path.insert(0, str(explicitlm_root))')
                    modified_source_lines.append(f'{indent_str}        try:')
                    modified_source_lines.append(f'{indent_str}            from models.core.ExplicitLM import ExplicitLM')
                    modified_source_lines.append(f'{indent_str}            import_success = True')
                    modified_source_lines.append(f'{indent_str}        except ImportError:')
                    modified_source_lines.append(f'{indent_str}            pass')
                    modified_source_lines.append(f'{indent_str}')
                    modified_source_lines.append(f'{indent_str}# 策略3: 自动查找项目根目录')
                    modified_source_lines.append(f'{indent_str}if not import_success and explicitlm_root is None:')
                    modified_source_lines.append(f'{indent_str}    for potential_root in [current_file.parent.parent, current_file.parent.parent.parent]:')
                    modified_source_lines.append(f'{indent_str}        if potential_root and (potential_root / "models" / "core" / "ExplicitLM.py").exists():')
                    modified_source_lines.append(f'{indent_str}            explicitlm_root = potential_root')
                    modified_source_lines.append(f'{indent_str}            if str(explicitlm_root) not in sys.path:')
                    modified_source_lines.append(f'{indent_str}                sys.path.insert(0, str(explicitlm_root))')
                    modified_source_lines.append(f'{indent_str}            try:')
                    modified_source_lines.append(f'{indent_str}                from models.core.ExplicitLM import ExplicitLM')
                    modified_source_lines.append(f'{indent_str}                import_success = True')
                    modified_source_lines.append(f'{indent_str}                break')
                    modified_source_lines.append(f'{indent_str}            except ImportError:')
                    modified_source_lines.append(f'{indent_str}                pass')
                    modified_source_lines.append(f'{indent_str}')
                    modified_source_lines.append(f'{indent_str}# 如果所有策略都失败，抛出错误')
                    modified_source_lines.append(f'{indent_str}if not import_success:')
                    modified_source_lines.append(f'{indent_str}    error_msg = (')
                    modified_source_lines.append(f'{indent_str}        f"无法找到 ExplicitLM 模块。\\n"')
                    modified_source_lines.append(f'{indent_str}        f"尝试的路径:\\n"')
                    modified_source_lines.append(f'{indent_str}        f"  1. 模型目录 (自包含模式): {{model_dir / \'models\' / \'core\' / \'ExplicitLM.py\'}}\\n"')
                    modified_source_lines.append(f'{indent_str}    )')
                    modified_source_lines.append(f'{indent_str}    if explicitlm_root:')
                    modified_source_lines.append(f'{indent_str}        error_msg += f"  2. EXPLICITLM_ROOT: {{explicitlm_root}}\\n"')
                    modified_source_lines.append(f'{indent_str}    error_msg += (')
                    modified_source_lines.append(f'{indent_str}        f"  3. 自动查找项目根目录: 未找到\\n\\n"')
                    modified_source_lines.append(f'{indent_str}        f"解决方案:\\n"')
                    modified_source_lines.append(f'{indent_str}        f"  - 如果这是自包含模型，请确保模型目录包含 models/core/ExplicitLM.py\\n"')
                    modified_source_lines.append(f'{indent_str}        f"  - 或者设置 EXPLICITLM_ROOT 环境变量指向项目根目录\\n"')
                    modified_source_lines.append(f'{indent_str}        f"  - 或者将模型目录放在项目根目录下"')
                    modified_source_lines.append(f'{indent_str}    )')
                    modified_source_lines.append(f'{indent_str}    raise ImportError(error_msg)')
                    
                    in_try_except_block = True
                    i += 1
                    continue
            
            if in_try_except_block:
                if line_stripped.startswith('except'):
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
                    i += 1
                    continue
            
            modified_source_lines.append(line)
            i += 1
        
        modified_model_source = '\n'.join(modified_source_lines)
    
    # 获取辅助函数的源代码
    helper_functions = None
    try:
        helper_functions = f'''
# 辅助函数（由 pt2hg_apart.py 自动生成）
{inspect.getsource(_extract_dataset_name)}

{inspect.getsource(_apply_device_map)}
'''
        print(f"   ✅ 已获取辅助函数源代码（通过inspect.getsource）")
    except Exception as e:
        print(f"   ⚠️  警告: 无法通过inspect获取辅助函数源代码: {e}，使用备用版本")
        # 如果获取源代码失败，手动编写关键函数
        helper_functions = '''
def _apply_device_map(model, device_map, torch_dtype=None):
    """应用 device_map，处理 memory_bank 和 valid_mask"""
    from accelerate import dispatch_model, infer_auto_device_map
    
    if hasattr(model.model, 'memory_bank') and model.model.memory_bank is not None:
        model.model.memory_bank = model.model.memory_bank.cpu()
    if hasattr(model.model, 'valid_mask') and model.model.valid_mask is not None:
        model.model.valid_mask = model.model.valid_mask.cpu()
    
    if device_map == "auto":
        try:
            if torch_dtype is None:
                torch_dtype = next(model.parameters()).dtype
            actual_device_map = infer_auto_device_map(model, max_memory=None, dtype=torch_dtype)
            if isinstance(actual_device_map, dict):
                actual_device_map = {k: v for k, v in actual_device_map.items() 
                                   if 'memory_bank' not in k and 'valid_mask' not in k}
            device_map = actual_device_map
        except Exception as e:
            print(f"⚠️  无法自动推断设备映射: {e}，使用手动分配")
            device_map = None
    
    if device_map:
        try:
            model = dispatch_model(model, device_map=device_map)
        except Exception as e:
            print(f"⚠️  dispatch_model 失败: {e}，使用手动分配")
            device_map = None
    
    if device_map and hasattr(model.model, "hf_device_map") and model.model.hf_device_map:
        first_device = next(iter(model.model.hf_device_map.values()))
        if isinstance(first_device, (list, tuple)):
            first_device = first_device[0]
        target_device = torch.device(first_device)
    else:
        target_device = next(model.model.parameters()).device
    
    if hasattr(model.model, 'memory_bank') and model.model.memory_bank is not None:
        model.model.memory_bank = model.model.memory_bank.to(target_device)
    if hasattr(model.model, 'valid_mask') and model.model.valid_mask is not None:
        model.model.valid_mask = model.model.valid_mask.to(target_device)
    
    return model

def _extract_dataset_name(memory_bank_path: str, metadata: dict = None) -> Optional[str]:
    """从 metadata 或文件路径提取 dataset_name"""
    if metadata and metadata.get('dataset_name'):
        return metadata['dataset_name']
    try:
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(memory_bank_path)))
        return parent_dir if parent_dir else None
    except:
        return None

def _auto_generate_keys(model, mb_file_path, model_dir, knowledge_num):
    """自动生成 Keys 文件"""
    from util_py.generate_keys_from_memory_bank import generate_keys_from_token_ids, load_memory_bank_batch
    
    print(f"⚠️  Keys 文件不存在，基于 Memory Bank 自动生成: {mb_file_path}")
    output_keys_path = str(model_dir / "keys.pt") if model_dir.exists() else "keys.pt"
    
    try:
        mb_tensor, mb_valid_mask, mb_metadata = load_memory_bank_batch(mb_file_path)
        dataset_name = _extract_dataset_name(mb_file_path, mb_metadata)
        
        embedding_layer = None
        try:
            if hasattr(model, 'get_input_embeddings'):
                embedding_layer = model.get_input_embeddings()
            elif hasattr(model.model, 'embed_tokens'):
                embedding_layer = model.model.embed_tokens
        except:
            pass
        
        qwen3_path = None
        checkpoint_info_path = model_dir / "checkpoint_info.json"
        if checkpoint_info_path.exists():
            try:
                import json
                with open(checkpoint_info_path, 'r') as f:
                    qwen3_path = json.load(f).get("qwen3_path")
            except:
                pass
        
        if embedding_layer or (qwen3_path and os.path.exists(qwen3_path)):
            generate_keys_from_token_ids(
                memory_bank=mb_tensor,
                valid_mask=mb_valid_mask,
                embedding_layer=embedding_layer,
                qwen_model_path=qwen3_path if embedding_layer is None else None,
                output_keys_path=output_keys_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=32,
                knowledge_num=knowledge_num,
                memory_bank_path=mb_file_path,
                dataset_name=dataset_name,
            )
            print(f"✅ Keys 已自动生成: {output_keys_path}")
            return output_keys_path
        else:
            print(f"⚠️  无法获取 embedding 层或 Qwen3 路径，无法自动生成 Keys")
    except Exception as e:
        print(f"⚠️  自动生成 Keys 失败: {e}")
        import traceback
        print(traceback.format_exc())
    
    return None

def _load_keys(model, keys_path):
    """加载 Keys 文件"""
    if not hasattr(model.model, 'shared_memory_gate') or model.model.shared_memory_gate is None:
        return
    
    if not keys_path or not os.path.exists(keys_path):
        print(f"⚠️  Keys 文件不存在，将使用随机初始化的 keys")
        return
    
    try:
        keys_data = torch.load(keys_path, map_location='cpu')
        if isinstance(keys_data, dict):
            row_keys = keys_data.get("row_keys")
            col_keys = keys_data.get("col_keys")
            if row_keys is not None and col_keys is not None:
                model.model.shared_memory_gate.update_keys(row_keys, col_keys)
                print(f"✅ Keys 已加载: {keys_path}")
                if keys_metadata := keys_data.get("metadata", {}):
                    print(f"⚠️  提醒: 请确认 Keys 与 Memory Bank 匹配")
                    print(f"   Keys 来源: dataset={keys_metadata.get('dataset_name', 'unknown')}, mb_path={keys_metadata.get('memory_bank_path', 'unknown')}")
            else:
                print(f"⚠️  Keys 文件格式不正确，将使用随机初始化的 keys")
        else:
            print(f"⚠️  Keys 文件格式不正确，将使用随机初始化的 keys")
    except Exception as e:
        print(f"⚠️  加载 Keys 失败: {e}，将使用随机初始化的 keys")
'''
    
    # 确保 helper_functions 被设置
    if helper_functions is None:
        print(f"   ⚠️  错误: helper_functions 未设置，使用备用版本")
        helper_functions = '''
# 辅助函数（由 pt2hg_apart.py 自动生成 - 备用版本）
def _extract_dataset_name(memory_bank_path: str, metadata: dict = None):
    """从 metadata 或文件路径提取 dataset_name"""
    if metadata and metadata.get('dataset_name'):
        return metadata['dataset_name']
    try:
        parent_dir = os.path.basename(os.path.dirname(os.path.abspath(memory_bank_path)))
        return parent_dir if parent_dir else None
    except:
        return None

def _apply_device_map(model, device_map, torch_dtype=None):
    """应用 device_map，处理 memory_bank 和 valid_mask"""
    from accelerate import dispatch_model, infer_auto_device_map
    
    if hasattr(model.model, 'memory_bank') and model.model.memory_bank is not None:
        model.model.memory_bank = model.model.memory_bank.cpu()
    if hasattr(model.model, 'valid_mask') and model.model.valid_mask is not None:
        model.model.valid_mask = model.model.valid_mask.cpu()
    
    if device_map == "auto":
        try:
            if torch_dtype is None:
                torch_dtype = next(model.parameters()).dtype
            actual_device_map = infer_auto_device_map(model, max_memory=None, dtype=torch_dtype)
            if isinstance(actual_device_map, dict):
                actual_device_map = {k: v for k, v in actual_device_map.items() 
                                   if 'memory_bank' not in k and 'valid_mask' not in k}
            device_map = actual_device_map
        except Exception as e:
            print(f"⚠️  无法自动推断设备映射: {e}，使用手动分配")
            device_map = None
    
    if device_map:
        try:
            model = dispatch_model(model, device_map=device_map)
        except Exception as e:
            print(f"⚠️  dispatch_model 失败: {e}，使用手动分配")
            device_map = None
    
    if device_map and hasattr(model.model, "hf_device_map") and model.model.hf_device_map:
        first_device = next(iter(model.model.hf_device_map.values()))
        if isinstance(first_device, (list, tuple)):
            first_device = first_device[0]
        target_device = torch.device(first_device)
    else:
        target_device = next(model.model.parameters()).device
    
    if hasattr(model.model, 'memory_bank') and model.model.memory_bank is not None:
        model.model.memory_bank = model.model.memory_bank.to(target_device)
    if hasattr(model.model, 'valid_mask') and model.model.valid_mask is not None:
        model.model.valid_mask = model.model.valid_mask.to(target_device)
    
    return model

def _auto_generate_keys(model, mb_file_path, model_dir, knowledge_num):
    """自动生成 Keys 文件"""
    from util_py.generate_keys_from_memory_bank import generate_keys_from_token_ids, load_memory_bank_batch
    
    print(f"⚠️  Keys 文件不存在，基于 Memory Bank 自动生成: {mb_file_path}")
    output_keys_path = str(model_dir / "keys.pt") if model_dir.exists() else "keys.pt"
    
    try:
        mb_tensor, mb_valid_mask, mb_metadata = load_memory_bank_batch(mb_file_path)
        dataset_name = _extract_dataset_name(mb_file_path, mb_metadata)
        
        embedding_layer = None
        try:
            if hasattr(model, 'get_input_embeddings'):
                embedding_layer = model.get_input_embeddings()
            elif hasattr(model.model, 'embed_tokens'):
                embedding_layer = model.model.embed_tokens
        except:
            pass
        
        qwen3_path = None
        checkpoint_info_path = model_dir / "checkpoint_info.json"
        if checkpoint_info_path.exists():
            try:
                import json
                with open(checkpoint_info_path, 'r') as f:
                    qwen3_path = json.load(f).get("qwen3_path")
            except:
                pass
        
        if embedding_layer or (qwen3_path and os.path.exists(qwen3_path)):
            generate_keys_from_token_ids(
                memory_bank=mb_tensor,
                valid_mask=mb_valid_mask,
                embedding_layer=embedding_layer,
                qwen_model_path=qwen3_path if embedding_layer is None else None,
                output_keys_path=output_keys_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=32,
                knowledge_num=knowledge_num,
                memory_bank_path=mb_file_path,
                dataset_name=dataset_name,
            )
            print(f"✅ Keys 已自动生成: {output_keys_path}")
            return output_keys_path
        else:
            print(f"⚠️  无法获取 embedding 层或 Qwen3 路径，无法自动生成 Keys")
    except Exception as e:
        print(f"⚠️  自动生成 Keys 失败: {e}")
        import traceback
        print(traceback.format_exc())
    
    return None

def _load_keys(model, keys_path):
    """加载 Keys 文件"""
    if not hasattr(model.model, 'shared_memory_gate') or model.model.shared_memory_gate is None:
        return
    
    if not keys_path or not os.path.exists(keys_path):
        print(f"⚠️  Keys 文件不存在，将使用随机初始化的 keys")
        return
    
    try:
        keys_data = torch.load(keys_path, map_location='cpu')
        if isinstance(keys_data, dict):
            row_keys = keys_data.get("row_keys")
            col_keys = keys_data.get("col_keys")
            if row_keys is not None and col_keys is not None:
                model.model.shared_memory_gate.update_keys(row_keys, col_keys)
                print(f"✅ Keys 已加载: {keys_path}")
                if keys_metadata := keys_data.get("metadata", {}):
                    print(f"⚠️  提醒: 请确认 Keys 与 Memory Bank 匹配")
                    print(f"   Keys 来源: dataset={keys_metadata.get('dataset_name', 'unknown')}, mb_path={keys_metadata.get('memory_bank_path', 'unknown')}")
            else:
                print(f"⚠️  Keys 文件格式不正确，将使用随机初始化的 keys")
        else:
            print(f"⚠️  Keys 文件格式不正确，将使用随机初始化的 keys")
    except Exception as e:
        print(f"⚠️  加载 Keys 失败: {e}，将使用随机初始化的 keys")
'''
    
    # 验证 helper_functions 不为空
    if not helper_functions or not helper_functions.strip():
        raise RuntimeError("helper_functions 为空，无法生成 modeling_explicitlm.py")
    
    print(f"   ✅ helper_functions 已准备就绪（长度: {len(helper_functions)} 字符）")
    
    modeling_content = f'''"""
ExplicitLM模型定义 - 用于HuggingFace自动加载模型（Memory Bank独立存储版本）
此文件由 pt2hg_apart.py 自动生成
"""
import os
import sys
from pathlib import Path
from typing import Optional

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

# 导入配置类
try:
    from .configuration_explicitlm import ExplicitLMConfig
except ImportError:
    import importlib.util
    config_path = Path(__file__).parent / "configuration_explicitlm.py"
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("configuration_explicitlm", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        ExplicitLMConfig = config_module.ExplicitLMConfig
    else:
        raise ImportError(f"无法找到 configuration_explicitlm.py 文件: {config_path}")

{helper_functions}

{modified_model_source}
'''
    
    modeling_path = os.path.join(output_path, "modeling_explicitlm.py")
    with open(modeling_path, 'w', encoding='utf-8') as f:
        f.write(modeling_content)
    print(f"   ✅ 模型定义文件已保存: {modeling_path}")


def _copy_core_files_to_model_dir(output_path: str):
    """复制核心代码文件到模型目录（实现自包含）"""
    import shutil
    from pathlib import Path
    
    # 项目根目录 (scripts/convert -> scripts -> ExplicitLM)
    project_root = Path(__file__).parent.parent.parent.resolve()
    
    # 目标目录
    target_models_dir = Path(output_path) / "models"
    target_core_dir = target_models_dir / "core"
    target_memory_bank_dir = target_models_dir / "memory_bank"
    target_layers_dir = target_models_dir / "layers"
    target_util_dir = Path(output_path) / "util_py"
    target_utils_dir = Path(output_path) / "utils"
    
    # 创建目录结构
    target_core_dir.mkdir(parents=True, exist_ok=True)
    target_memory_bank_dir.mkdir(parents=True, exist_ok=True)
    target_layers_dir.mkdir(parents=True, exist_ok=True)
    target_util_dir.mkdir(parents=True, exist_ok=True)
    target_utils_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建 __init__.py 文件
    (target_models_dir / "__init__.py").touch()
    (target_core_dir / "__init__.py").touch()
    (target_memory_bank_dir / "__init__.py").touch()
    (target_layers_dir / "__init__.py").touch()
    (target_util_dir / "__init__.py").touch()
    (target_utils_dir / "__init__.py").touch()
    
    # 需要复制的模型核心文件
    files_to_copy = {
        "models/core/ExplicitLM.py": target_core_dir / "ExplicitLM.py",
        "models/core/Qwen3ExplicitLMBlock.py": target_core_dir / "Qwen3ExplicitLMBlock.py",
        "models/memory_bank/MemoryGate.py": target_memory_bank_dir / "MemoryGate.py",
        "models/memory_bank/GatedMemoryFusion.py": target_memory_bank_dir / "GatedMemoryFusion.py",
        "models/layers/RMSNorm.py": target_layers_dir / "RMSNorm.py",
        # util_py 工具（用于 switch_memory_bank.py 生成 keys）
        "util_py/generate_keys_from_memory_bank.py": target_util_dir / "generate_keys_from_memory_bank.py",
        # utils 工具（generate_keys_from_memory_bank.py 需要）
        "utils/clustering.py": target_utils_dir / "clustering.py",
    }
    
    # 复制文件
    copied_count = 0
    for src_rel_path, dst_path in files_to_copy.items():
        src_path = project_root / src_rel_path
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"   ⚠️  警告: 源文件不存在: {src_path}")
    
    if copied_count > 0:
        print(f"   ✅ 已复制 {copied_count} 个核心代码文件到模型目录（自包含模式）")
        print(f"   📁 核心代码位置: {target_models_dir}")
        print(f"   📁 工具脚本位置: {target_util_dir}")
        if target_utils_dir.exists() and (target_utils_dir / "clustering.py").exists():
            print(f"   📁 工具库位置: {target_utils_dir}")


def create_switch_memory_bank_script(output_path: str):
    """创建切换 Memory Bank 的脚本（自包含版本）"""
    
    script_content = '''#!/usr/bin/env python3
"""
切换 Memory Bank 并自动生成 Keys

使用方法：
    cd <模型目录>
    python switch_memory_bank.py --input new_memory_bank.pt

    # 从外部路径加载：
    python switch_memory_bank.py --input /path/to/memory_bank.pt
"""
import os
import sys
import argparse
import shutil
from pathlib import Path

# 确保模型目录在 Python 路径中
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import torch


def load_memory_bank_file(path: str):
    """加载 Memory Bank 文件"""
    data = torch.load(path, map_location='cpu')
    if isinstance(data, dict):
        mb = data.get('memory_bank', data.get('processed_tensor'))
        mask = data.get('valid_mask', None)
        meta = data.get('metadata', {})
    else:
        mb = data
        mask = None
        meta = {}
    return mb, mask, meta


def generate_keys(memory_bank, valid_mask, output_path, device="cuda"):
    """基于 Memory Bank 生成 Keys"""
    print(f"🔑 正在生成 Keys...")
    
    # 尝试导入 generate_keys 工具
    try:
        # 方法1：从模型目录内的 util_py 导入（如果存在）
        util_path = MODEL_DIR / "util_py"
        if util_path.exists() and str(util_path.parent) not in sys.path:
            sys.path.insert(0, str(util_path.parent))
        
        from util_py.generate_keys_from_memory_bank import generate_keys_from_token_ids
        
        # 读取 checkpoint_info 获取 qwen3_path
        import json
        checkpoint_info_path = MODEL_DIR / "checkpoint_info.json"
        qwen3_path = None
        if checkpoint_info_path.exists():
            with open(checkpoint_info_path, 'r') as f:
                qwen3_path = json.load(f).get("qwen3_path")
        
        # 获取 embedding 层（优先使用本地模型）
        embedding_layer = None
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_DIR), 
                trust_remote_code=True, 
                torch_dtype=torch.float32,
                local_files_only=True  # 只使用本地文件
            )
            embedding_layer = model.get_input_embeddings()
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"   ⚠️ 无法加载模型获取 embedding: {e}")
        
        generate_keys_from_token_ids(
            memory_bank=memory_bank,
            valid_mask=valid_mask,
            embedding_layer=embedding_layer,
            qwen_model_path=qwen3_path if embedding_layer is None else None,
            output_keys_path=output_path,
            device=device,
            batch_size=32,
        )
        print(f"✅ Keys 已生成: {output_path}")
        return True
        
    except ImportError as e:
        print(f"⚠️  无法导入 generate_keys 工具: {e}")
        print(f"   请手动生成 Keys 或复制已有的 Keys 文件")
        return False


def main():
    parser = argparse.ArgumentParser(description="切换 Memory Bank 并自动生成 Keys")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="新的 Memory Bank 文件路径")
    parser.add_argument("--no-keys", action="store_true",
                        help="不自动生成 Keys")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="生成 Keys 使用的设备")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    
    if not input_path.exists():
        print(f"❌ Memory Bank 文件不存在: {input_path}")
        sys.exit(1)
    
    print(f"📂 模型目录: {MODEL_DIR}")
    print(f"📥 输入文件: {input_path}")
    
    # 加载新的 Memory Bank
    memory_bank, valid_mask, metadata = load_memory_bank_file(str(input_path))
    print(f"✅ 已加载 Memory Bank: {memory_bank.shape}")
    
    # 如果没有 valid_mask，自动生成
    if valid_mask is None:
        # 假设 pad_token_id = 0
        is_all_pad = (memory_bank == 0).all(dim=-1)
        valid_mask = ~is_all_pad
        print(f"   自动生成 valid_mask: {valid_mask.sum().item()}/{len(valid_mask)} 有效")
    
    # 保存到模型目录
    output_mb_path = MODEL_DIR / "memory_bank.pt"
    save_data = {
        'memory_bank': memory_bank,
        'valid_mask': valid_mask,
        'metadata': {
            **metadata,
            'source_path': str(input_path),
        }
    }
    torch.save(save_data, output_mb_path)
    print(f"✅ Memory Bank 已保存: {output_mb_path}")
    
    # 生成 Keys
    if not args.no_keys:
        output_keys_path = MODEL_DIR / "keys.pt"
        generate_keys(memory_bank, valid_mask, str(output_keys_path), args.device)
    
    print(f"\\n🎉 完成！现在可以运行 inference_example.py 进行推理")


if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(output_path, "switch_memory_bank.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"📝 Memory Bank 切换脚本已创建: {script_path}")


def create_inference_script(output_path: str):
    """创建推理示例脚本（自包含版本，用户拿到后直接能运行）"""
    
    script_content = '''#!/usr/bin/env python3
"""
ExplicitLM 推理示例（自包含版本）

使用方法：
    cd <模型目录>
    python inference_example.py

    # 或指定其他 memory bank：
    python inference_example.py --memory_bank other_memory_bank.pt
"""
import os
import sys
import argparse
from pathlib import Path

# 确保模型目录在 Python 路径中（自包含模式）
MODEL_DIR = Path(__file__).resolve().parent
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    parser = argparse.ArgumentParser(description="ExplicitLM 推理示例")
    parser.add_argument("--memory_bank", type=str, default=None,
                        help="Memory Bank 文件路径（可选，默认使用模型目录下的 memory_bank.pt）")
    parser.add_argument("--prompt", type=str, default="你好，请介绍一下人工智能",
                        help="输入提示")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="最大生成长度")
    parser.add_argument("--device", type=str, default=None, help="设备（默认自动选择）")
    args = parser.parse_args()
    
    # 模型路径就是当前脚本所在目录
    model_path = str(MODEL_DIR)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 设置 EXPLICITLM_ROOT 环境变量，让缓存的代码能够找到模型目录中的 models/core/ExplicitLM.py
    # 这对于 HuggingFace 的 trust_remote_code 缓存机制很重要
    os.environ["EXPLICITLM_ROOT"] = str(MODEL_DIR)
    
    print(f"📂 模型目录: {model_path}")
    print(f"🔧 使用设备: {device}")
    
    # 加载 tokenizer 和模型（仅使用本地文件，不尝试从 HuggingFace Hub 下载）
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True,
        local_files_only=True  # 只使用本地文件
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map={"": device} if device.startswith("cuda") else None,
        local_files_only=True  # 只使用本地文件
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 如果指定了自定义 memory_bank，加载它
    if args.memory_bank:
        mb_path = Path(args.memory_bank)
        if not mb_path.is_absolute():
            mb_path = MODEL_DIR / mb_path
        if mb_path.exists():
            print(f"📥 加载自定义 Memory Bank: {mb_path}")
            model.load_memory_bank(str(mb_path))
        else:
            print(f"⚠️  Memory Bank 不存在: {mb_path}")
    
    model.eval()
    
    prompt = args.prompt
    print(f"\\n💬 输入: {prompt}")
    
    # 使用 apply_chat_template 格式化输入（如果 tokenizer 支持）
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        messages = [{"role": "user", "content": prompt}]
        # Qwen3 支持 enable_thinking 参数，设为 False 禁用 <think> 标签
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False  # 禁用 thinking 模式
            )
        except TypeError:
            # 如果 tokenizer 不支持 enable_thinking 参数，回退到默认行为
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        formatted_prompt = prompt
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    
    # 只解码新生成的部分（排除输入）
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"🤖 回复: {response}")


if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(output_path, "inference_example.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"📝 推理示例脚本已创建: {script_path}")


def main():
    # 默认路径配置
    DEFAULT_QWEN3_PATH = "Qwen_hg/Qwen3-4b"
    DEFAULT_CHECKPOINT_DIR = "checkpoints/fusion_pretrain"
    DEFAULT_MEMORY_BANK_DIR = "data/pt_factorys/outputs/memory_banks"
    DEFAULT_OUTPUT_DIR = "hf_models"
    
    parser = argparse.ArgumentParser(
        description="将ExplicitLM checkpoint转换为HuggingFace格式（Memory Bank独立存储）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  # 基本用法（使用默认 Qwen3 路径）
  python pt2hg_apart.py --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_14500

  # 指定 Memory Bank
  python pt2hg_apart.py --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_14500 \\
      --memory_bank_path data/pt_factorys/outputs/memory_banks/medqa.pt

  # 完整参数
  python pt2hg_apart.py \\
      --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_14500 \\
      --qwen3_path Qwen_hg/Qwen3-4b \\
      --output_path hf_models/step_14500_medqa \\
      --memory_bank_path data/pt_factorys/outputs/memory_banks/medqa.pt

默认路径:
  Qwen3 模型: {DEFAULT_QWEN3_PATH}
  Checkpoint 目录: {DEFAULT_CHECKPOINT_DIR}
  Memory Bank 目录: {DEFAULT_MEMORY_BANK_DIR}
"""
    )
    parser.add_argument("--checkpoint_path", "-c", type=str, required=True,
                       help="checkpoint文件或目录路径")
    parser.add_argument("--qwen3_path", "-q", type=str, default=DEFAULT_QWEN3_PATH,
                       help=f"Qwen3基础模型路径（默认: {DEFAULT_QWEN3_PATH}）")
    parser.add_argument("--output_path", "-o", type=str, default=None,
                       help="输出HF模型路径（默认: 根据checkpoint名自动生成）")
    parser.add_argument("--memory_bank_path", "-m", type=str, default=None,
                       help="Memory Bank文件路径（可选，会单独保存）")
    parser.add_argument("--knowledge_num", type=int, default=1048576,
                       help="记忆库大小（默认: 1048576）")
    parser.add_argument("--knowledge_length", type=int, default=32,
                       help="记忆条目长度（默认: 32）")
    parser.add_argument("--num_candidates", type=int, default=16,
                       help="候选记忆数（默认: 16）")
    parser.add_argument("--num_selected", type=int, default=1,
                       help="选中记忆数（默认: 1）")
    parser.add_argument("--gumbel_temperature", type=float, default=1.0,
                       help="Gumbel-Softmax温度（默认: 1.0）")
    
    args = parser.parse_args()
    
    # 获取 ExplicitLM 项目根目录 (scripts/convert -> scripts -> ExplicitLM)
    explicitlm_root = Path(__file__).parent.parent.parent.resolve()
    
    # 自动生成输出路径（如果未指定）
    if args.output_path is None:
        checkpoint_name = Path(args.checkpoint_path).name
        if args.memory_bank_path:
            mb_name = Path(args.memory_bank_path).stem
            args.output_path = f"{DEFAULT_OUTPUT_DIR}/{checkpoint_name}_{mb_name}"
        else:
            args.output_path = f"{DEFAULT_OUTPUT_DIR}/{checkpoint_name}"
        print(f"📁 自动生成输出路径: {args.output_path}")
    
    # 确保输出路径在 ExplicitLM 目录下
    output_path_obj = Path(args.output_path)
    if not output_path_obj.is_absolute():
        # 相对路径：直接添加到 ExplicitLM 目录下
        args.output_path = str(explicitlm_root / args.output_path)
    else:
        # 绝对路径：检查是否在 ExplicitLM 目录下
        try:
            args.output_path = str(Path(args.output_path).resolve())
            # 检查是否在 ExplicitLM 目录下
            if not str(args.output_path).startswith(str(explicitlm_root)):
                print(f"⚠️  警告: 输出路径不在 ExplicitLM 目录下，将调整为 ExplicitLM 目录下")
                relative_path = Path(args.output_path).name
                args.output_path = str(explicitlm_root / relative_path)
        except:
            # 如果解析失败，使用相对路径方式
            args.output_path = str(explicitlm_root / Path(args.output_path).name)
    
    # 确保路径是 ExplicitLM 目录下的子目录
    args.output_path = str(Path(args.output_path).resolve())
    if not str(args.output_path).startswith(str(explicitlm_root)):
        # 如果仍然不在 ExplicitLM 目录下，强制使用 ExplicitLM 目录
        relative_path = Path(args.output_path).name
        args.output_path = str(explicitlm_root / relative_path)
        print(f"⚠️  已调整输出路径到 ExplicitLM 目录下")
    
    print(f"📁 最终输出路径（ExplicitLM 目录下）: {args.output_path}")
    
    try:
        output_path = convert_to_hf_format(
            checkpoint_path=args.checkpoint_path,
            qwen3_path=args.qwen3_path,
            output_path=args.output_path,
            memory_bank_path=args.memory_bank_path,
            knowledge_num=args.knowledge_num,
            knowledge_length=args.knowledge_length,
            num_candidates=args.num_candidates,
            num_selected=args.num_selected,
            gumbel_temperature=args.gumbel_temperature,
        )
        
        create_inference_script(output_path)
        create_switch_memory_bank_script(output_path)
        
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

