# ExplicitLM到Hugging Face格式转换工具

这个工具可以将训练好的ExplicitLM模型转换为Hugging Face兼容格式，提供两种转换模式：

## 🎯 推荐使用模式

### 🔥 **推荐：自定义ExplicitLM模式** (保留完整记忆功能)
如果你想要**保留完整的ExplicitLM记忆增强功能**，请使用 `custom` 模式：

```bash
python utils/convert_to_hf.py --mode custom [其他参数...]
```

**为什么推荐custom模式**：
- ✅ **100%功能保留**：完整的记忆库、GatedMemoryFusion、MemoryGate
- ✅ **HF生态兼容**：支持transformers库的所有功能
- ✅ **训练效果完整**：不损失任何ExplicitLM的训练成果

### 📋 两种转换模式详解

#### **模式1: 标准Qwen3模式**
- **功能**: 将训练的ExplicitLM组件合并到标准的Qwen3模型中
- **优势**: 生成标准的HF Qwen3模型，完全兼容HF生态
- **适用**: 推理、部署、分享到HF Hub
- **特点**: 记忆功能转换为等价的transformer操作

#### **模式2: 自定义ExplicitLM模式** ⭐⭐⭐⭐⭐
- **功能**: 创建保留完整记忆功能的自定义HF模型
- **优势**: 保持ExplicitLM的所有记忆增强能力
- **适用**: 需要完整记忆功能的场景
- **特点**: 需要trust_remote_code=True，但功能完整

## 功能特点

- ✅ **完整保留功能**：保持所有记忆增强能力
- ✅ **HF完全兼容**：支持AutoModelForCausalLM等HF接口
- ✅ **自动配置生成**：自动生成合适的模型配置
- ✅ **Tokenizer集成**：自动复制tokenizer文件
- ✅ **使用示例**：生成完整的使用示例代码

## 快速开始

### 🎯 直接转换命令 (保留完整记忆功能)

```bash
cd /data2/zengzheni/lvchangwei/new_repo/ExplicitLM

# 一键转换：保留完整ExplicitLM记忆功能
python utils/convert_to_hf.py \
    --mode custom \
    --checkpoint_path out/trainable_components.pth \
    --qwen3_path /data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b \
    --output_path hf_explicitlm_model \
    --knowledge_num 1048576 \
    --knowledge_length 32 \
    --num_candidates 16 \
    --num_selected 1
```

**转换完成后使用**：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("hf_explicitlm_model")
model = AutoModelForCausalLM.from_pretrained("hf_explicitlm_model", trust_remote_code=True)
```

## 使用方法

### 1. 准备文件

确保你有以下文件：
- **训练好的组件权重**：`trainable_components.pth` (如: `out/trainable_components.pth`)
- **原始Qwen3模型**：`/data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b/`

### 2. 运行转换

#### **选项1：标准Qwen3模式** (部署友好)
```bash
cd /data2/zengzheni/lvchangwei/new_repo/ExplicitLM

python utils/convert_to_hf.py \
    --mode standard \
    --pth_file /data2/zengzheni/lvchangwei/new_repo/ExplicitLM/out/trainable_components.pth \
    --qwen3_path /data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b \
    --output_path hf_standard_qwen3_model
```

#### **选项2：自定义ExplicitLM模式** (保留完整记忆功能 ⭐推荐)
```bash
cd /data2/zengzheni/lvchangwei/new_repo/ExplicitLM

python utils/convert_to_hf.py \
    --mode custom \
    --checkpoint_path /data2/zengzheni/lvchangwei/new_repo/ExplicitLM/out/trainable_components.pth \
    --qwen3_path /data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b \
    --output_path hf_explicitlm_model \
    --knowledge_num 1048576 \
    --knowledge_length 32 \
    --num_candidates 16 \
    --num_selected 1
```

### 4. 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | ❌ | standard | 转换模式: standard/custom |
| `--pth_file` | ✅ (standard) | - | trainable_components.pth文件路径 |
| `--checkpoint_path` | ✅ (custom) | - | ExplicitLM checkpoint目录或PTH文件路径 |
| `--qwen3_path` | ✅ | - | 原始Qwen3模型路径 |
| `--output_path` | ✅ | - | 输出目录路径 |
| `--knowledge_num` | ❌ | 1048576 | 记忆库大小 (custom模式需要) |
| `--knowledge_length` | ❌ | 32 | 记忆条目长度 (custom模式需要) |
| `--num_candidates` | ❌ | 16 | 候选记忆数量 (custom模式需要) |
| `--num_selected` | ❌ | 1 | 选择记忆数量 (custom模式需要) |

## 输出文件结构

```
hf_explicitlm_model/
├── config.json              # 模型配置
├── pytorch_model.bin        # 模型权重
├── tokenizer.json           # Tokenizer配置
├── tokenizer_config.json    # Tokenizer配置
├── special_tokens_map.json  # 特殊token映射
├── vocab.json              # 词汇表
├── merges.txt              # BPE合并规则（如适用）
├── README.md               # 模型卡片
└── example_usage.py        # 使用示例
```

## 使用转换后的模型

### Python代码使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载模型
model_path = "hf_explicitlm_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 生成文本
input_text = "请介绍一下人工智能"
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

### 运行示例脚本

```bash
cd hf_explicitlm_model
python example_usage.py
```

## 技术细节

### 转换策略

1. **完整架构保留**：不进行权重投影或简化，而是创建完整的HF兼容ExplicitLM模型
2. **自动权重映射**：自动将Qwen3基础权重和ExplicitLM训练权重合并
3. **配置兼容性**：生成与HF生态完全兼容的配置

### 权重合并逻辑

- **Qwen3基础权重**：embed_tokens, layers.*.self_attn.*, layers.*.mlp.*, norm, lm_head
- **ExplicitLM训练权重**：gated_memory_fusion.*, memory_norm.*, memory_gate.*（如果训练了）
- **记忆库**：在推理时动态加载，不包含在模型权重中

### 兼容性保证

- ✅ 支持 `AutoModelForCausalLM.from_pretrained()`
- ✅ 支持 `AutoTokenizer.from_pretrained()`
- ✅ 支持 `model.generate()` 方法
- ✅ 支持HF的聊天模板和对话格式

## 故障排除

### 常见问题

1. **找不到checkpoint文件**
   ```
   确保checkpoint_path指向正确的目录，如：
   checkpoints/fusion_pretrain/checkpoint_step_500/
   ```

2. **Qwen3模型路径错误**
   ```
   确保qwen3_path包含pytorch_model.bin和config.json
   ```

3. **显存不足**
   ```
   转换过程需要加载完整模型，建议使用足够显存的GPU
   ```

4. **Tokenizer文件缺失**
   ```
   脚本会自动复制tokenizer文件，如果缺失某些文件可能影响使用
   ```

### 验证转换结果

转换完成后，可以运行以下代码验证：

```python
# 检查模型结构
print(model)
print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 测试基本功能
inputs = tokenizer("Hello", return_tensors="pt")
outputs = model(**inputs)
print(f"输出形状: {outputs.logits.shape}")
```

## 注意事项

1. **推理时需要记忆库**：虽然模型权重不包含记忆库，但在实际使用时需要准备记忆数据
2. **训练权重优先级**：ExplicitLM训练的组件权重会覆盖对应的Qwen3基础权重
3. **配置一致性**：确保记忆配置参数与训练时保持一致

## 相关链接

- [ExplicitLM项目主页](../../README.md)
- [Hugging Face Transformers文档](https://huggingface.co/docs/transformers/index)
- [Qwen3模型文档](https://huggingface.co/Qwen)
