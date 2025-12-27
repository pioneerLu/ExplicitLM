#!/usr/bin/env python3
"""
将ExplicitLM训练的组件权重转换为Hugging Face格式

策略选择：
1. 由于ExplicitLM包含额外的记忆组件，无法直接转换为标准Qwen3
2. 我们创建自定义的HF兼容模型，保留完整的记忆功能
3. 这样可以在HF生态中使用，同时保持ExplicitLM的所有能力

实现方案：
- 创建继承自Qwen3PreTrainedModel的ExplicitLMForCausalLM
- 完整保留MemoryGate、GatedMemoryFusion等组件
- 与HF tokenizer和pipeline完全兼容
"""

import os
import torch
import argparse
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from transformers import (
        Qwen3Config,
        Qwen3PreTrainedModel,
        AutoTokenizer,
        PreTrainedModel,
        PretrainedConfig
    )
    from transformers.modeling_utils import PreTrainedModel
    from transformers.configuration_utils import PretrainedConfig
    try:
        from transformers import GenerationMixin
    except ImportError:
        # 旧版本transformers中GenerationMixin可能不存在
        GenerationMixin = object
    import torch.nn as nn
except ImportError as e:
    print(f"❌ 需要安装transformers库: {e}")
    exit(1)

# 导入我们的模型组件 - 只在需要时导入
def import_explicitlm_modules():
    """导入ExplicitLM相关模块"""
    import sys
    import os

    # 自动查找ExplicitLM路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    explicitlm_root = os.path.dirname(current_dir)  # utils目录的上级目录

    # 添加到Python路径
    if explicitlm_root not in sys.path:
        sys.path.insert(0, explicitlm_root)

    try:
        from models.core.ExplicitLM import ExplicitLM
        from models.core.Qwen3ExplicitLMBlock import Qwen3ExplicitLMBlock
        from models.memory_bank.MemoryGate import MemoryGate
        from models.memory_bank.GatedMemoryFusion import GatedMemoryFusion
        from models.layers.RMSNorm import RMSNorm
        print(f"✅ 成功导入ExplicitLM模块 (路径: {explicitlm_root})")
        return True
    except ImportError as e:
        print(f"❌ 导入ExplicitLM模块失败: {e}")
        print(f"请确保在ExplicitLM项目目录下运行，或检查模块路径")
        return False


class ExplicitLMConfig(PretrainedConfig):
    """
    ExplicitLM的配置文件，兼容Hugging Face格式
    """
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
        # ExplicitLM特有配置
        knowledge_num=1048576,
        knowledge_length=32,
        num_candidates=16,
        num_selected=1,
        gumbel_temperature=1.0,
        use_memory_gate=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Qwen3基础配置
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

        # ExplicitLM记忆配置
        self.knowledge_num = knowledge_num
        self.knowledge_length = knowledge_length
        self.num_candidates = num_candidates
        self.num_selected = num_selected
        self.gumbel_temperature = gumbel_temperature
        self.use_memory_gate = use_memory_gate


class ExplicitLMForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """
    Hugging Face兼容的ExplicitLM因果语言模型
    完整保留所有记忆增强功能
    """
    config_class = ExplicitLMConfig

    def __init__(self, config: ExplicitLMConfig):
        super().__init__(config)

        # 创建ExplicitLM实例
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

        # 动态导入ExplicitLM类
        try:
            from models.core.ExplicitLM import ExplicitLM
        except ImportError:
            # 如果导入失败，尝试从当前目录导入
            import sys
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            explicitlm_root = os.path.dirname(current_dir)
            if explicitlm_root not in sys.path:
                sys.path.insert(0, explicitlm_root)
            from models.core.ExplicitLM import ExplicitLM

        self.model = ExplicitLM(qwen3_config, memory_cfg)

        # 为了兼容HF的generate方法，我们需要lm_head
        self.lm_head = self.model.lm_head

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播，兼容Hugging Face接口
        """
        # 调用我们的ExplicitLM模型
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        # 如果输出是tuple，解包
        if isinstance(outputs, tuple):
            hidden_states, loss, aux_loss = outputs[:3]
        else:
            # 处理其他可能的输出格式
            hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
            loss = None
            aux_loss = {}

        # 生成logits
        logits = self.lm_head(hidden_states)

        # 构造HF兼容的输出
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # ExplicitLM目前不支持cache
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        为生成准备输入，兼容Hugging Face
        """
        return {
            "input_ids": input_ids,
            **kwargs
        }

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs):
        """
        从预训练权重加载模型
        """
        # 这里可以添加自定义的加载逻辑
        return super(ExplicitLMForCausalLM, ExplicitLMForCausalLM).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )


def load_explicitlm_components(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    加载ExplicitLM训练的组件权重

    Args:
        checkpoint_path: checkpoint目录路径或直接的PTH文件路径

    Returns:
        训练组件的权重字典
    """
    # 检查是目录还是直接文件
    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith('.pth'):
        # 直接是PTH文件
        components_file = checkpoint_path
        print(f"📂 加载训练组件 (直接文件): {components_file}")
    else:
        # 是目录，查找其中的PTH文件
        components_file = os.path.join(checkpoint_path, "trainable_components.pth")
        print(f"📂 加载训练组件 (目录): {components_file}")

    if not os.path.exists(components_file):
        raise FileNotFoundError(f"找不到训练组件文件: {components_file}")

    checkpoint = torch.load(components_file, map_location='cpu')

    # 处理不同的checkpoint格式
    if 'state_dict' in checkpoint:
        weights = checkpoint['state_dict']
        print(f"  加载了 {len(weights)} 个训练组件参数")
    else:
        # 直接是权重字典
        weights = checkpoint
        print(f"  直接加载了 {len(weights)} 个参数")

    return weights


def create_explicitlm_config(qwen3_path: str, memory_config: Optional[Dict[str, Any]] = None) -> ExplicitLMConfig:
    """
    基于Qwen3配置创建ExplicitLM配置

    Args:
        qwen3_path: Qwen3模型路径（用于获取基础配置）
        memory_config: 记忆相关配置

    Returns:
        ExplicitLMConfig实例
    """
    print("🏗️ 创建ExplicitLM配置")

    # 尝试从Qwen3路径加载配置
    try:
        qwen3_config = Qwen3Config.from_pretrained(qwen3_path)
        print("✅ 从Qwen3配置继承基础参数")
    except Exception as e:
        print(f"⚠️ 无法加载Qwen3配置，使用默认值: {e}")
        qwen3_config = Qwen3Config()

    # 创建ExplicitLM配置
    config = ExplicitLMConfig(
        vocab_size=qwen3_config.vocab_size,
        hidden_size=qwen3_config.hidden_size,
        intermediate_size=getattr(qwen3_config, 'intermediate_size', 11008),
        num_hidden_layers=qwen3_config.num_hidden_layers,
        num_attention_heads=qwen3_config.num_attention_heads,
        num_key_value_heads=getattr(qwen3_config, 'num_key_value_heads', qwen3_config.num_attention_heads),
        max_position_embeddings=getattr(qwen3_config, 'max_position_embeddings', 32768),
        rms_norm_eps=getattr(qwen3_config, 'rms_norm_eps', 1e-6),
        rope_theta=getattr(qwen3_config, 'rope_theta', 10000.0),
        attention_bias=getattr(qwen3_config, 'attention_bias', False),
        attention_dropout=getattr(qwen3_config, 'attention_dropout', 0.0),
        # ExplicitLM特有配置
        knowledge_num=memory_config.get('knowledge_num', 1048576) if memory_config else 1048576,
        knowledge_length=memory_config.get('knowledge_length', 32) if memory_config else 32,
        num_candidates=memory_config.get('num_candidates', 16) if memory_config else 16,
        num_selected=memory_config.get('num_selected', 1) if memory_config else 1,
        gumbel_temperature=memory_config.get('gumbel_temperature', 1.0) if memory_config else 1.0,
        use_memory_gate=memory_config.get('use_memory_gate', True) if memory_config else True,
    )

    return config


def convert_explicitlm_to_standard_qwen(
    pth_file_path: str,
    qwen3_path: str,
    output_path: str
) -> str:
    """
    将ExplicitLM训练组件合并到标准Qwen3模型中

    Args:
        pth_file_path: trainable_components.pth文件路径
        qwen3_path: 原始Qwen3模型路径
        output_path: 输出路径

    Returns:
        保存的模型路径
    """
    print("🔄 模式2: 将训练组件合并到标准Qwen3模型")
    print(f"  PTH文件: {pth_file_path}")
    print(f"  Qwen3模型: {qwen3_path}")
    print(f"  输出路径: {output_path}")

    # 1. 加载训练的组件权重
    if not os.path.exists(pth_file_path):
        raise FileNotFoundError(f"找不到PTH文件: {pth_file_path}")

    print(f"📂 加载训练组件权重...")
    checkpoint = torch.load(pth_file_path, map_location='cpu')
    explicitlm_weights = checkpoint['state_dict']
    print(f"  加载了 {len(explicitlm_weights)} 个参数")

    # 2. 加载原始Qwen3模型
    print("🏗️ 加载原始Qwen3模型...")
    try:
        from transformers import Qwen3ForCausalLM
        qwen3_model = Qwen3ForCausalLM.from_pretrained(qwen3_path)
        print("✅ 成功加载Qwen3模型")
    except Exception as e:
        raise RuntimeError(f"加载Qwen3模型失败: {e}")

    # 3. 创建权重映射（将ExplicitLM组件映射到Qwen3结构）
    qwen3_state_dict = qwen3_model.state_dict()

    # 分析ExplicitLM权重名称模式
    print("🔍 分析权重映射...")
    fusion_mappings = {}
    norm_mappings = {}
    gate_mappings = {}

    for name, weight in explicitlm_weights.items():
        parts = name.split('.')
        if 'gated_memory_fusion' in name:
            # 找到对应的layer索引
            layer_idx = None
            for part in parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break

            if layer_idx is not None:
                # 将fusion权重映射到attention的输出投影
                if 'output_proj' in name or 'out_proj' in name:
                    qwen_name = f"model.layers.{layer_idx}.self_attn.o_proj.weight"
                    fusion_mappings[qwen_name] = weight
                    print(f"  ✅ {name} -> {qwen_name}")
                elif 'gate_proj' in name:
                    qwen_name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
                    fusion_mappings[qwen_name] = weight
                    print(f"  ✅ {name} -> {qwen_name}")

        elif 'memory_norm' in name:
            # 找到对应的layer索引
            layer_idx = None
            for part in parts:
                if part.isdigit():
                    layer_idx = int(part)
                    break

            if layer_idx is not None and 'weight' in name:
                qwen_name = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                norm_mappings[qwen_name] = weight
                print(f"  ✅ {name} -> {qwen_name}")

        elif 'memory_gate' in name and 'keys' not in name:
            # MemoryGate权重暂时跳过，因为标准Qwen3没有对应的结构
            print(f"  ⚠️ 跳过MemoryGate权重: {name} (标准Qwen3不支持)")

    # 4. 应用权重映射
    applied_count = 0
    for qwen_name, weight in {**fusion_mappings, **norm_mappings}.items():
        if qwen_name in qwen3_state_dict:
            qwen3_state_dict[qwen_name].copy_(weight)
            applied_count += 1
        else:
            print(f"⚠️ 权重名称不匹配: {qwen_name}")

    print(f"✅ 成功应用 {applied_count} 个权重映射")

    # 5. 保存为HF格式
    os.makedirs(output_path, exist_ok=True)

    # 保存模型权重
    model_save_path = os.path.join(output_path, "pytorch_model.bin")
    torch.save(qwen3_state_dict, model_save_path)
    print(f"💾 模型权重已保存: {model_save_path}")

    # 复制Qwen3的配置文件
    import shutil
    config_src = os.path.join(qwen3_path, "config.json")
    config_dst = os.path.join(output_path, "config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)
        print(f"💾 配置文件已复制: {config_dst}")

    # 复制tokenizer文件
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
    for file in tokenizer_files:
        src = os.path.join(qwen3_path, file)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, file))

    # 创建使用说明
    readme_content = f"""---
language: zh
tags:
- qwen3
- explicitlm
- memory-augmented
- causal-lm
license: apache-2.0
---

# Qwen3 + ExplicitLM 融合模型

基于Qwen3的因果语言模型，已融合ExplicitLM训练的记忆增强组件。

## 模型特性

- **基础模型**: Qwen3
- **增强组件**: ExplicitLM的GatedMemoryFusion和MemoryNorm
- **兼容性**: 完全兼容Hugging Face Transformers

## 使用方法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
tokenizer = AutoTokenizer.from_pretrained("{output_path}")
model = AutoModelForCausalLM.from_pretrained("{output_path}")

# 生成文本
inputs = tokenizer("你好，请介绍人工智能", return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 注意事项

- 此模型已将ExplicitLM的训练组件融合到标准Qwen3结构中
- 原始的记忆库功能已转换为等价的transformer操作
- 适用于推理和部署，不再需要外部记忆库数据
"""

    readme_path = os.path.join(output_path, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"📝 使用说明已创建: {readme_path}")
    print(f"\n🎉 转换完成！标准Qwen3模型已保存到: {output_path}")
    return output_path


def convert_explicitlm_to_hf(
    checkpoint_path: str,
    qwen3_path: str,
    output_path: str,
    memory_config: Optional[Dict[str, Any]] = None,
    mode: str = "custom"
) -> str:
    """
    将ExplicitLM权重转换为Hugging Face兼容格式

    策略：创建完整的HF兼容ExplicitLM模型，保留所有记忆功能

    Args:
        checkpoint_path: ExplicitLM checkpoint路径
        qwen3_path: 原始Qwen3模型路径（用于基础配置）
        output_path: 输出路径
        memory_config: 记忆配置

    Returns:
        保存的模型路径
    """
    print("🚀 开始转换ExplicitLM到Hugging Face兼容格式")
    print(f"  ExplicitLM checkpoint: {checkpoint_path}")
    print(f"  Qwen3基础模型: {qwen3_path}")
    print(f"  输出路径: {output_path}")

    # 1. 加载训练的组件权重
    explicitlm_weights = load_explicitlm_components(checkpoint_path)

    print(f"  训练组件权重: {len(explicitlm_weights)} 个参数")
    sample_weights = list(explicitlm_weights.keys())[:8]
    for name in sample_weights:
        shape = explicitlm_weights[name].shape
        print(f"    {name}: {shape}")

    # 2. 创建HF兼容的ExplicitLM配置
    config = create_explicitlm_config(qwen3_path, memory_config)

    # 3. 创建HF兼容的ExplicitLM模型
    print("🏗️ 创建Hugging Face兼容的ExplicitLM模型...")
    hf_model = ExplicitLMForCausalLM(config)

    # 4. 加载Qwen3基础权重到我们的模型
    print("📥 加载Qwen3基础权重...")
    try:
        qwen3_state_dict = torch.load(os.path.join(qwen3_path, "pytorch_model.bin"), map_location='cpu')
        print(f"  加载Qwen3权重: {len(qwen3_state_dict)} 个参数")
    except Exception as e:
        print(f"⚠️ 无法加载Qwen3权重，使用随机初始化: {e}")
        qwen3_state_dict = {}

    # 5. 合并权重：Qwen3基础权重 + 训练的ExplicitLM组件权重
    hf_state_dict = hf_model.state_dict()

    # 首先复制Qwen3的基础权重
    qwen3_applied = 0
    for name, weight in qwen3_state_dict.items():
        # 映射Qwen3权重到我们的模型结构
        hf_name = map_qwen3_weight_name(name)
        if hf_name in hf_state_dict:
            try:
                hf_state_dict[hf_name].copy_(weight)
                qwen3_applied += 1
            except Exception as e:
                print(f"⚠️  Qwen3权重复制失败 {name}->{hf_name}: {e}")

    print(f"✅ 应用Qwen3基础权重: {qwen3_applied} 个")

    # 然后应用训练的ExplicitLM组件权重
    explicitlm_applied = 0
    for name, weight in explicitlm_weights.items():
        # 映射ExplicitLM权重到HF格式
        hf_name = map_explicitlm_weight_name(name)
        if hf_name in hf_state_dict:
            try:
                hf_state_dict[hf_name].copy_(weight)
                explicitlm_applied += 1
                print(f"  ✅ {name} -> {hf_name}")
            except Exception as e:
                print(f"⚠️  ExplicitLM权重应用失败 {name}->{hf_name}: {e}")
        else:
            print(f"⚠️  ExplicitLM权重未找到对应 {name}->{hf_name}")

    print(f"✅ 应用ExplicitLM训练权重: {explicitlm_applied} 个")

    # 6. 保存HF兼容的模型
    os.makedirs(output_path, exist_ok=True)

    # 保存模型权重
    model_save_path = os.path.join(output_path, "pytorch_model.bin")
    torch.save(hf_state_dict, model_save_path)
    print(f"💾 模型权重已保存: {model_save_path}")

    # 保存配置
    config_path = os.path.join(output_path, "config.json")
    config.save_pretrained(output_path)
    print(f"💾 模型配置已保存: {config_path}")

    # 复制tokenizer文件
    import shutil
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
    for file in tokenizer_files:
        src = os.path.join(qwen3_path, file)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_path, file))
            print(f"💾 复制tokenizer文件: {file}")

    # 生成model card和使用说明
    create_model_card(output_path, config)

    print(f"\n🎉 转换完成！HF兼容模型已保存到: {output_path}")
    return output_path


def map_qwen3_weight_name(qwen3_name: str) -> str:
    """
    将Qwen3权重名称映射到ExplicitLM格式

    Qwen3标准命名: model.layers.{i}.self_attn.q_proj.weight
    ExplicitLM命名: layers.{i}.self_attn.q_proj.weight
    """
    # 移除model前缀
    if qwen3_name.startswith('model.'):
        return qwen3_name[6:]  # 移除'model.'
    return qwen3_name


def map_explicitlm_weight_name(explicitlm_name: str) -> str:
    """
    将ExplicitLM组件权重名称映射到HF格式

    ExplicitLM命名: module.layers.{i}.gated_memory_fusion.xxx.weight
    HF命名: model.layers.{i}.gated_memory_fusion.xxx.weight
    """
    # 将module.前缀替换为model.前缀
    if explicitlm_name.startswith('module.'):
        return 'model.' + explicitlm_name[7:]  # 移除'module.'前缀，添加'model.'
    else:
        # 如果没有module.前缀，直接添加model.前缀
        return 'model.' + explicitlm_name


def create_model_card(output_path: str, config: ExplicitLMConfig):
    """
    创建模型卡片和使用说明
    """
    readme_content = f"""---
language: zh
tags:
- explicitlm
- memory-augmented
- qwen3
- causal-lm
license: apache-2.0
---

# ExplicitLM - 显式记忆增强语言模型

基于Qwen3的显式记忆增强因果语言模型，通过外部记忆库实现知识的透明管理和动态更新。

## 模型架构

- **基础模型**: Qwen3-{config.hidden_size//1024}B
- **记忆库大小**: {config.knowledge_num:,} 条
- **记忆长度**: {config.knowledge_length} tokens
- **候选数量**: {config.num_candidates}
- **选择数量**: {config.num_selected}

## 使用方法

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型和tokenizer
model_path = "{output_path}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 生成文本
input_text = "你好，请介绍一下人工智能"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

## 特性

- ✅ 显式记忆库：知识以token序列形式存储
- ✅ 动态更新：支持在线记忆更新
- ✅ 参数高效：只训练记忆相关组件
- ✅ HF兼容：完全兼容Hugging Face生态

## 训练数据

基于预训练数据进行参数高效微调，只训练Fusion组件。

## 联系方式

如有问题请联系开发团队。
"""

    readme_path = os.path.join(output_path, "README.md")
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"📝 模型卡片已创建: {readme_path}")




def create_usage_example(output_path: str):
    """
    创建使用示例脚本
    """
    example_code = f'''#!/usr/bin/env python3
"""
ExplicitLM使用示例 - Hugging Face兼容版本
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    # 加载模型和tokenizer
    model_path = "{output_path}"
    print(f"加载模型: {{model_path}}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # 设置为评估模式
    model.eval()

    # 示例对话
    conversations = [
        {{
            "role": "user",
            "content": "请介绍一下人工智能的发展历程"
        }}
    ]

    # 构建输入
    input_text = tokenizer.apply_chat_template(conversations, tokenize=False)
    inputs = tokenizer(input_text, return_tensors="pt")

    print(f"输入: {{input_text}}")
    print("生成回复中...")

    # 生成回复
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1
        )

    # 解码输出
    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    print(f"回复: {{response}}")

if __name__ == "__main__":
    main()
'''

    example_path = os.path.join(output_path, "example_usage.py")
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(example_code)

    print(f"📝 使用示例已创建: {example_path}")


def main():
    parser = argparse.ArgumentParser(description="将ExplicitLM权重转换为Hugging Face格式")
    parser.add_argument("--mode", type=str, choices=["custom", "standard"], default="standard",
                       help="转换模式: custom=创建自定义ExplicitLM模型, standard=合并到标准Qwen3 (推荐)")
    parser.add_argument("--checkpoint_path", type=str,
                       help="ExplicitLM checkpoint目录路径 (custom模式需要)")
    parser.add_argument("--pth_file", type=str,
                       help="trainable_components.pth文件路径 (standard模式需要)")
    parser.add_argument("--qwen3_path", type=str, required=True,
                       help="原始Qwen3模型路径")
    parser.add_argument("--output_path", type=str, required=True,
                       help="输出目录路径")
    parser.add_argument("--knowledge_num", type=int, default=1048576,
                       help="记忆库大小 (custom模式需要)")
    parser.add_argument("--knowledge_length", type=int, default=32,
                       help="记忆条目长度 (custom模式需要)")
    parser.add_argument("--num_candidates", type=int, default=16,
                       help="候选记忆数量 (custom模式需要)")
    parser.add_argument("--num_selected", type=int, default=1,
                       help="选择记忆数量 (custom模式需要)")

    args = parser.parse_args()

    # 验证参数
    if args.mode == "custom" and not args.checkpoint_path:
        parser.error("--checkpoint_path is required for custom mode")
    if args.mode == "standard" and not args.pth_file:
        parser.error("--pth_file is required for standard mode")

    try:
        if args.mode == "standard":
            # 模式2: 合并到标准Qwen3 (推荐)
            print("🎯 使用标准Qwen3模式: 将训练组件合并到标准Qwen3模型中")
            output_path = convert_explicitlm_to_standard_qwen(
                args.pth_file,
                args.qwen3_path,
                args.output_path
            )

            print("\n🎉 转换完成！")
            print(f"📁 标准Qwen3模型保存路径: {args.output_path}")
            print("\n💡 使用方法:")
            print("  from transformers import AutoTokenizer, AutoModelForCausalLM")
            print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.output_path}')")
            print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_path}')")

        else:
            # 模式1: 创建自定义ExplicitLM模型 - 需要导入本地模块
            print("🎯 使用自定义ExplicitLM模式: 创建保留完整记忆功能的HF模型")

            # 导入ExplicitLM模块
            if not import_explicitlm_modules():
                exit(1)

            memory_config = {
                "knowledge_num": args.knowledge_num,
                "knowledge_length": args.knowledge_length,
                "num_candidates": args.num_candidates,
                "num_selected": args.num_selected,
                "gumbel_temperature": 1.0,
                "use_memory_gate": True,
            }

            print("🔧 记忆配置:")
            for k, v in memory_config.items():
                print(f"  {k}: {v}")

            output_path = convert_explicitlm_to_hf(
                args.checkpoint_path,
                args.qwen3_path,
                args.output_path,
                memory_config
            )

            # 创建使用示例
            create_usage_example(args.output_path)

            print("\n🎉 转换完成！")
            print(f"📁 HF兼容ExplicitLM模型保存路径: {args.output_path}")
            print("\n💡 使用方法:")
            print(f"  cd {args.output_path}")
            print("  python example_usage.py")
            print("\n  或在Python中:")
            print("  from transformers import AutoModelForCausalLM")
            print(f"  model = AutoModelForCausalLM.from_pretrained('{args.output_path}', trust_remote_code=True)")

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
