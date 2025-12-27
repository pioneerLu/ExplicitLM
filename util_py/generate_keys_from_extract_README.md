# 基于 extract_support 生成 Keys 脚本说明

## 📋 功能

从 `train_data_with_extract.json` 提取所有 `extract_support`，将其作为知识库生成新的 Keys，确保 Keys 和 Cache 基于相同的知识库。

## 🔧 依赖安装

脚本需要以下 Python 包：

```bash
# 使用 pip + 清华源（推荐，速度快）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch sentence-transformers scikit-learn tqdm numpy

# 或永久配置清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch sentence-transformers scikit-learn tqdm numpy

# 或使用 uv（推荐）
uv pip install torch sentence-transformers scikit-learn tqdm numpy
```

## 🚀 使用方法

### 基础用法

```bash
cd /data2/zengzheni/lvchangwei/new_repo

# 方法 1: 使用本地模型（推荐，如果已下载）
python3 util_py/generate_keys_from_extract.py \
    --input data/train_data_with_extract.json \
    --kb-output data/knowledge_base/extract_support_kb.json \
    --keys-output data/keys_extract.pt \
    --local-model-path /data2/zengzheni/lvchangwei/new_repo/BAAI

```

### 参数说明

- `--input`: 输入的 JSON 文件路径（包含 extract_support）
  - 默认: `ExplicitLM/data/train_data_with_extract.json`

- `--kb-output`: 输出的知识库 JSON 文件路径
  - 默认: `ExplicitLM/data/knowledge_base/extract_support_kb.json`

- `--keys-output`: 输出的 Keys 文件路径
  - 默认: `ExplicitLM/data/keys_extract.pt`

- `--model-name`: 嵌入模型名称（HuggingFace 名称）
  - 默认: `BAAI/bge-base-en-v1.5`
  - 其他选项: `sentence-transformers/all-MiniLM-L6-v2` 等

- `--local-model-path`: 本地模型路径
  - 如果指定且模型存在，将使用本地模型（无需下载）
  - 示例: `ExplicitLM/models/embedding_models/bge-base-en-v1.5`

- `--download-model`: 下载模型到本地
  - 需要配合 `--local-model-path` 使用
  - 首次运行时使用，后续可直接使用本地模型

- `--device`: 设备（cuda/cpu）
  - 默认: 自动选择（优先使用 GPU）

- `--batch-size`: 批处理大小
  - 默认: 32

## 📊 输出

### 1. 知识库文件

`ExplicitLM/data/knowledge_base/extract_support_kb.json`

包含所有 `extract_support` 的 JSON 数组，格式：
```json
[
  {
    "sentence": "Mesophiles moderate,. 37°C,., yogurt, beer.",
    "uuid": "extract_0",
    "subject": "",
    "predicate": "",
    "object": ""
  },
  ...
]
```

### 2. Keys 文件

`ExplicitLM/data/keys_extract.pt`

PyTorch tensor，形状: `[2, √N, 768]`
- `[0]`: Row Keys `[√N, 768]`
- `[1]`: Col Keys `[√N, 768]`

其中 N 是知识库条目数（调整为完全平方数）。

## 🔄 后续步骤

### 1. 更新训练脚本

编辑 `ExplicitLM/scripts/run_sft.sh`，更新 keys_path：

```bash
model.keys_path="data/keys_extract.pt"
```

### 2. 验证配置

确保：
- ✅ Keys 和 Cache 基于相同的知识库（extract_support）
- ✅ Keys 形状匹配配置：`[2, 1024, 768]`（对于 knowledge_num=1048576）
- ✅ knowledge_dim=1536（嵌入维度 × 2）

## ⚠️ 注意事项

1. **知识库大小调整**: 脚本会自动将知识库大小调整为完全平方数（Product Key Memory 的要求）
   - 例如：11,679 → 10,816 (104²)

2. **内存使用**: 生成 Keys 需要加载嵌入模型和编码所有句子，可能需要较多内存

3. **GPU 加速**: 如果使用 GPU，可以显著加速嵌入编码过程

4. **模型下载**: 首次运行会下载 `BAAI/bge-base-en-v1.5` 模型（约 400MB）

## 🔍 故障排除

### 问题 1: ModuleNotFoundError

**错误**: `ModuleNotFoundError: No module named 'xxx'`

**解决**: 安装缺失的依赖包
```bash
pip install torch sentence-transformers scikit-learn tqdm numpy
```

### 问题 2: CUDA out of memory

**错误**: GPU 内存不足

**解决**: 
- 使用 CPU: `--device cpu`
- 减小 batch_size: `--batch-size 16`

### 问题 3: 知识库大小不匹配

**错误**: Keys 形状与配置不匹配

**解决**: 检查 knowledge_num 是否为完全平方数，脚本会自动调整

## 📝 示例输出

```
============================================================
🚀 基于 extract_support 生成 Keys
============================================================

步骤 1: 提取 extract_support 作为知识库
------------------------------------------------------------
📖 读取输入文件: ExplicitLM/data/train_data_with_extract.json
📊 总样本数: 11679
✅ 提取了 11679 个非空的 extract_support
📝 调整知识库大小: 11679 -> 10816 (完全平方数)
💾 保存知识库: ExplicitLM/data/knowledge_base/extract_support_kb.json
  ✅ 已保存 10816 个知识库条目

步骤 2: 基于知识库生成 Keys
------------------------------------------------------------
📦 加载嵌入模型: BAAI/bge-base-en-v1.5
  ✅ 嵌入维度: 768
📖 加载知识库: ExplicitLM/data/knowledge_base/extract_support_kb.json
  ✅ 加载了 10816 个句子
🔨 编码知识库为嵌入向量...
  ✅ 嵌入向量形状: (10816, 768)
🔨 执行 Residual Quantization...
  📊 聚类数: 104 (√10816)
  📍 步骤 1: 粗粒度聚类 (Row Keys)...
    ✅ Row Keys 形状: (104, 768)
  📍 步骤 2: 计算残差...
  📍 步骤 3: 细粒度聚类 (Col Keys)...
    ✅ Col Keys 形状: (104, 768)
  📍 步骤 4: 组合 Keys...
    ✅ Keys 形状: torch.Size([2, 104, 768])
💾 保存 Keys: ExplicitLM/data/keys_extract.pt
  ✅ Keys 已保存

============================================================
✅ 完成！
============================================================
知识库: ExplicitLM/data/knowledge_base/extract_support_kb.json (10816 个条目)
Keys: ExplicitLM/data/keys_extract.pt (形状: torch.Size([2, 104, 768]))

📝 下一步:
  1. 更新 run_sft.sh 中的 keys_path 为: ExplicitLM/data/keys_extract.pt
  2. 确保 Keys 和 Cache 基于相同的知识库
============================================================
```
