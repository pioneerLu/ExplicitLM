# ExplicitLM 转换与推理速查（面向技术人员）

## 脚本选型

- **`pt2hg_apart.py`**（推荐）: Memory Bank 独立存储，便于替换/裁剪，**自包含可分发**
- **`pt2hg_sync.py`**: Memory Bank 写入权重文件（单目录）

## 快速转换（pt2hg_apart.py）

### 最简用法
```bash
# 使用默认 Qwen3 路径，自动生成输出目录
uv run python pt2hg_apart.py -c checkpoints/fusion_pretrain/checkpoint_step_14500
# 输出: ExplicitLM/hf_models/checkpoint_step_14500
```

### 指定 Memory Bank
```bash
uv run python pt2hg_apart.py \
    -c checkpoints/fusion_pretrain/checkpoint_step_14500 \
    -m data/pt_factorys/outputs/memory_banks/medqa.pt
# 输出: ExplicitLM/hf_models/checkpoint_step_14500_medqa
```

### 完整参数
```bash
uv run python pt2hg_apart.py \
    -c checkpoints/fusion_pretrain/checkpoint_step_14500 \
    -q Qwen_hg/Qwen3-4b \
    -o my_model \
    -m data/pt_factorys/outputs/memory_banks/medqa.pt \
    --keys_path data/keys.pt  # 可选：不指定则自动生成
```

**参数说明**:
| 参数 | 短名 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint_path` | `-c` | 必填 | checkpoint 路径 |
| `--qwen3_path` | `-q` | `Qwen_hg/Qwen3-4b` | Qwen3 基础模型 |
| `--output_path` | `-o` | 自动生成 | 输出目录（始终在 ExplicitLM 目录下） |
| `--memory_bank_path` | `-m` | 可选 | Memory Bank 文件 |
| `--keys_path` | - | 自动生成 | Keys 文件 |

## 生成的模型目录结构

```
ExplicitLM/hf_models/checkpoint_step_14500_medqa/
├── config.json
├── pytorch_model*.bin
├── tokenizer*
├── memory_bank.pt          # Memory Bank
├── keys.pt                 # Keys（自动生成）
├── inference_example.py    # 自包含推理脚本
├── switch_memory_bank.py   # 切换 Memory Bank 脚本
├── models/                 # 核心代码（自包含）
└── util_py/                # 工具脚本
```

**注意**: 所有输出路径都会自动保存在 `ExplicitLM` 目录下，无论指定相对路径还是绝对路径。

## 用户拿到模型后的使用

### 直接推理
```bash
# 进入模型目录（在 ExplicitLM 目录下）
cd ExplicitLM/my_model  # 或 ExplicitLM/hf_models/checkpoint_step_14500_medqa
python inference_example.py

# 指定 prompt
python inference_example.py --prompt "什么是深度学习？"
```

### 切换 Memory Bank
```bash
cd <模型目录>
python switch_memory_bank.py --input /path/to/new_memory_bank.pt
# 自动替换 memory_bank.pt 并重新生成 keys.pt
```

## Memory Bank + Keys 配对规范

**重要**: Memory Bank 和 Keys 必须配对，否则检索异常。

- **转换时**: 自动生成配对的 Keys
- **切换时**: 使用 `switch_memory_bank.py` 自动处理

### 手动生成 Keys（可选）
```bash
uv run python util_py/generate_keys_from_memory_bank.py \
    --memory-bank-path data/pt_factorys/outputs/memory_banks/medqa.pt \
    --output-keys-path data/pt_factorys/outputs/keys/medqa_keys.pt \
    --qwen-model-path Qwen_hg/Qwen3-4b
```

## 同步存储版本（pt2hg_sync.py）

```bash
uv run python pt2hg_sync.py \
    --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_7500 \
    --qwen3_path Qwen_hg/Qwen3-4b \
    --output_path hf_explicitlm_model_sync \
    --memory_bank_path data/cache/parquet_extract/memory_bank_batches/kb_parquet.pt
```

## 常见问题

- **路径缺失**: 确认 `checkpoint_path`、`memory_bank_path` 存在
- **显存不足**: 使用 `torch_dtype=torch.float16` 或 `device_map="auto"`
- **Keys 不匹配**: 使用 `switch_memory_bank.py` 重新生成
