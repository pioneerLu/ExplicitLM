# 转换脚本说明

## 概述

将训练产出的 checkpoint 转换为 HuggingFace 格式，支持两种模式：
- **sync**：Memory Bank 嵌入模型权重（单文件部署）
- **apart**：Memory Bank 独立存储（支持热切换）

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `pt2hg_sync.py` | Memory Bank 嵌入模型权重 |
| `pt2hg_apart.py` | Memory Bank 独立存储，支持热切换 |
| `convert_conversations_to_labeled.py` | 对话数据标注转换 |

## 使用示例

### pt2hg_sync（嵌入式）

```bash
cd ExplicitLM

uv run python scripts/convert/pt2hg_sync.py \
    --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_9500 \
    --qwen3_path Qwen_hg/Qwen3-4b \
    --output_path hf_models/explicitlm_sync \
    --memory_bank_path data/pt_factorys/outputs/memory_banks/medqa.pt
```

### pt2hg_apart（独立式，推荐）

```bash
cd ExplicitLM

uv run python scripts/convert/pt2hg_apart.py \
    --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_14500 \
    --qwen3_path Qwen_hg/Qwen3-4b \
    --output_path hf_models/explicitlm_apart \
    --memory_bank_path data/pt_factorys/outputs/memory_banks/medqa.pt
```

**apart 模式特点**：
- 自动生成 `memory_bank.pt`、`keys.pt`
- 自动复制核心代码实现自包含
- 生成 `switch_memory_bank.py` 用于热切换 Memory Bank

### 切换 Memory Bank（apart 模式）

```bash
cd hf_models/explicitlm_apart

python switch_memory_bank.py --input /path/to/new_memory_bank.pt
```

> ⚠️ 切换 Memory Bank 后会自动重新生成 Keys，确保检索正确。

## 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--checkpoint_path` | 是 | checkpoint 目录路径 |
| `--qwen3_path` | 是 | Qwen3 基础模型路径 |
| `--output_path` | 是 | 输出 HuggingFace 模型路径 |
| `--memory_bank_path` | 否 | Memory Bank 文件路径 |

## 推理示例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "hf_models/explicitlm_apart",
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("hf_models/explicitlm_apart")

# 推理
inputs = tokenizer("问题：...", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

