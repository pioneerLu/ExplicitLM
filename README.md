# ExplicitLM - Qwen3 显式记忆增强语言模型

ExplicitLM 基于 Qwen3-4B，通过显式记忆库实现知识的透明管理和动态更新。

## 核心特性

- **Qwen3 基础架构**：基于 Qwen3-4B 预训练模型
- **显式记忆库**：知识以 token 序列形式显式存储
- **均值池化查询**：使用序列均值池化生成查询向量
- **多阶段训练**：分阶段训练 MemoryGate、Fusion 和联合微调
- **参数高效训练**：冻结 Qwen3 主模型参数，只训练记忆相关组件

## 项目结构

```
ExplicitLM/
├── models/                          # 模型架构
│   ├── core/
│   │   ├── ExplicitLM.py           # 主模型类
│   │   └── Qwen3ExplicitLMBlock.py # Qwen3 Transformer Block（记忆增强）
│   └── memory_bank/
│       ├── MemoryGate.py           # 记忆门控层（Product Key Memory）
│       └── GatedMemoryFusion.py   # 门控记忆融合模块
├── config/                          # Hydra-Zen 配置
├── utils/                           # 工具模块
├── scripts/                         # 工具脚本
│   ├── run_sft.sh                  # SFT 训练启动脚本
│   └── run_router.sh               # Router 训练启动脚本
├── train_router.py                 # 阶段1：MemoryGate 训练
├── train_fusion.py                 # 阶段2：知识融合组件训练
├── train_joint.py                  # 阶段3：联合微调
├── 1_pretrain.py                   # 预训练入口
├── 2_sft.py                        # 监督微调入口
└── pyproject.toml                  # 项目依赖配置
```

## 快速开始

### 环境准备

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd ExplicitLM
uv sync
```

## 训练流程

### 阶段1：MemoryGate 训练

训练 MemoryGate 组件，学习从查询中检索相关记忆。

**数据准备：**

首先需要将 conversations 格式的数据转换为带真实标签的 query 格式：

```bash
uv run python convert_conversations_to_labeled.py \
    --conversations_path data/train.jsonl \
    --kb_path data/knowledge_base/sentence_trex_data.json \
    --output_path data/train_labeled.jsonl \
    --model_name BAAI/bge-base-en-v1.5 \
    --top_k 32 \
    --batch_size 32
```

该脚本会：
- 从知识库加载句子并编码
- 使用 K-Means 训练 Product Key Memory 的 keys
- 对查询进行 embedding 并检索最相关的知识库条目
- 生成包含 `target_indices` 和 `target_scores` 的数据文件
- 保存 `meta.json` 和 `keys.pt` 文件

**开始训练：**

```bash
bash scripts/run_router.sh
```

或手动运行：
```bash
export CUDA_VISIBLE_DEVICES=5,6
uv run python train_router.py \
    --data_path data/train_labeled.jsonl \
    --model_name /data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b \
    --output_dir checkpoints/router \
    --batch_size 1 \
    --lr 1e-4 \
    --epochs 3 \
    --knowledge_num 65536 \
    --knowledge_dim 2048 \
    --num_candidates 32 \
    --max_length 128 \
    --temperature 0.5 \
    --swanlab_project explicitlm-router
```

**关键参数：**
- `--data_path`: 训练数据路径（query 格式，需包含 `target_indices` 和 `target_scores`）
- `--model_name`: Qwen3 模型路径
- `--knowledge_num`: 记忆库条目数（需为完全平方数，默认 65536，通常从 `meta.json` 自动读取）
- `--output_dir`: 输出目录（MemoryGate 权重会保存为 `memory_gate_epoch_X.pth`）

### 阶段2：知识融合训练

加载预训练的 MemoryGate（冻结），训练 GatedMemoryFusion 和 memory_norm。

```bash
python train_fusion.py \
    --qwen3_model_path /path/to/Qwen3-4b \
    --pretrained_memory_gate_path checkpoints/router/memory_gate.pth \
    --dataset_path data/database/merged_pretrain.jsonl \
    --val_dataset_path data/benchmarks/eval_data.json \
    --knowledge_num 65536 \
    --knowledge_length 16 \
    --knowledge_dim 128 \
    --num_candidates 8 \
    --batch_size 8 \
    --accumulation_steps 16 \
    --lr 1e-4 \
    --epochs 3 \
    --warmup_steps 100 \
    --ce_loss_coef 1.0 \
    --output_dir checkpoints/fusion \
    --swanlab_project explicitlm-fusion
```

### 阶段3：联合微调

联合训练 MemoryGate 和 GatedMemoryFusion，使用完整的三损失系统。

```bash
python train_joint.py \
    --qwen3_model_path /path/to/Qwen3-4b \
    --pretrained_memory_gate_path checkpoints/router/memory_gate.pth \
    --pretrained_fusion_path checkpoints/fusion/fusion_weights.pth \
    --dataset_path data/database/merged_pretrain.jsonl \
    --val_dataset_path data/benchmarks/eval_data.json \
    --knowledge_num 65536 \
    --knowledge_length 16 \
    --num_candidates 8 \
    --batch_size 8 \
    --accumulation_steps 16 \
    --lr 1e-4 \
    --epochs 3 \
    --ce_loss_coef 1.0 \
    --similarity_loss_coef 0.1 \
    --diversity_loss_coef 0.05 \
    --output_dir checkpoints/joint \
    --swanlab_project explicitlm-joint
```

### SFT 训练

使用监督微调优化下游任务表现。

```bash
bash scripts/run_sft.sh
```

或直接运行：
```bash
python 2_sft.py \
    +model.qwen3_model_path=/path/to/Qwen3-4b \
    model.cache_path=data/cache/knowledge_cache.pt \
    model.recompute_cache=False \
    model.max_seq_len=256 \
    dataset.sft_dataset_path=sft_data/omcq_trex_sft.jsonl \
    dataset.pretrained_sft_model_path=out/pretrain_latest.pth \
    dataset.sft_val_dataset_path=data/benchmarks/eval_data.json \
    training.learning_rate=5e-5 \
    training.batch_size=1 \
    training.accumulation_steps=128 \
    training.epochs=3 \
    training.zero_stage=3 \
    logging.out_dir=out \
    logging.save_dir=out
```

## 核心配置参数

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `qwen3_model_path` | Qwen3-4B 模型路径 | **必需** |
| `knowledge_num` | 记忆库条目总数（需为完全平方数） | 65536 |
| `knowledge_length` | 每个记忆条目的 token 数 | 16 |
| `knowledge_dim` | 记忆嵌入向量维度 | 128 |

### 记忆检索参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_candidates` | 最终输出的候选数 | 8 |
| `num_selected` | 选中的条目数 | 1 |
| `gumbel_temperature` | Gumbel-Softmax 温度 | 1.0 |

## 训练策略

### 参数冻结

- **Qwen3 主模型**：完全冻结
- **记忆库（Memory Bank）**：训练时固定
- **记忆组件训练策略**：
  - **阶段1**：只训练 `memory_gate`
  - **阶段2**：只训练 `gated_memory_fusion` 和 `memory_norm`（MemoryGate 冻结）
  - **阶段3**：联合训练 `memory_gate`、`gated_memory_fusion` 和 `memory_norm`

### 损失函数

- **阶段1（Router）**：Soft Label Loss
- **阶段2（Fusion）**：主要使用 CE Loss
- **阶段3（Joint）**：CE Loss + Similarity Loss + Diversity Loss
- **SFT**：CE Loss + Similarity Loss + Diversity Loss

## 数据格式

### Router 训练数据（JSONL）

Router 训练需要使用带真实标签的 query 格式数据。如果原始数据是 conversations 格式，需要先使用 `convert_conversations_to_labeled.py` 进行转换。

**Query 格式（必需）**
```jsonl
{"query": "问题或文本", "target_indices": [1, 5, 10], "target_scores": [0.9, 0.8, 0.7]}
{"query": "另一个问题", "target_indices": [3, 7], "target_scores": [0.95, 0.85]}
```

- `query`: 查询文本
- `target_indices`: 目标记忆条目的索引列表（通过 FAISS 检索生成）
- `target_scores`: 对应的相似度分数（可选，默认为 1.0）

### 预训练数据（JSONL）

```jsonl
{"text": "人工智能是计算机科学的一个分支..."}
```

### SFT 数据（JSONL）

```jsonl
{
  "conversations": [
    {"role": "user", "content": "问题内容"},
    {"role": "assistant", "content": "回答内容"}
  ]
}
```

## 注意事项

1. **模型路径配置**：确保 Qwen3-4B 模型路径正确
2. **记忆库大小**：`knowledge_num` 必须是完全平方数（如 1024, 4096, 65536）
3. **训练顺序**：建议按照阶段1 → 阶段2 → 阶段3 → SFT 的顺序进行训练
4. **Router 训练数据准备**：训练前必须先运行 `convert_conversations_to_labeled.py` 生成带真实标签的数据
5. **GPU 内存**：如果遇到 OOM，可以减小 `batch_size`、`knowledge_num` 或调整 DeepSpeed 配置
6. **CUDA_VISIBLE_DEVICES**：必须在运行脚本**之前**设置（在导入 torch 之前），推荐使用启动脚本

## SwanLab 配置

SwanLab API Key 可通过以下方式设置（优先级从高到低）：
1. 命令行参数 `--swanlab_api_key`
2. 环境变量 `SWANLAB_API_KEY`
3. `.env` 文件中的 `SWANLAB_API_KEY`
