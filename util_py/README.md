# 工具脚本说明

## convert_extract_data_to_sft.py

将 `train_data_with_extract.json` 转换为 SFT 训练格式的脚本。

### 功能

1. 读取 JSON 数组格式的数据
2. 为每个样本生成 UUID
3. 转换为对话格式的 JSONL（包含 conversations 和 uuid）
4. 将 extract_support 转换为 token IDs，生成 knowledge_cache.pt
5. 8:2 划分训练/验证集
6. 保存 UUID 映射文件

### 使用方法

```bash
cd /data2/zengzheni/lvchangwei/new_repo

# 使用默认参数
python3 util_py/convert_extract_data_to_sft.py

# 或指定参数
python3 util_py/convert_extract_data_to_sft.py \
    --input ExplicitLM/data/train_data_with_extract.json \
    --qwen-model-path /data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b \
    --output-dir ExplicitLM/sft_data \
    --cache-path ExplicitLM/data/cache/train_data_with_extract_cache.pt \
    --knowledge-num 1048576 \
    --knowledge-length 32 \
    --train-ratio 0.8
```

### 输出文件

- `ExplicitLM/sft_data/train_data_with_extract_sft_train.jsonl` - 训练集（80%）
- `ExplicitLM/sft_data/train_data_with_extract_sft_val.jsonl` - 验证集（20%）
- `ExplicitLM/data/cache/train_data_with_extract_cache.pt` - 记忆库缓存（形状: [1048576, 32]）
- `ExplicitLM/data/cache/train_data_with_extract_cache_mapping.json` - UUID 映射文件

### 参数说明

- `--input`: 输入的 JSON 文件路径（默认: `ExplicitLM/data/train_data_with_extract.json`）
- `--qwen-model-path`: Qwen3 模型路径，用于加载 tokenizer（默认: `/data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b`）
- `--output-dir`: 输出目录（默认: `ExplicitLM/sft_data`）
- `--cache-path`: 记忆库缓存文件路径（默认: `ExplicitLM/data/cache/train_data_with_extract_cache.pt`）
- `--mapping-path`: UUID 映射文件路径（默认: `ExplicitLM/data/cache/train_data_with_extract_cache_mapping.json`）
- `--knowledge-num`: 记忆库条目数（默认: 1048576）
- `--knowledge-length`: 每个条目的 token 数（默认: 32）
- `--train-ratio`: 训练集比例（默认: 0.8）

