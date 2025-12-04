# ExplicitLM - Qwen3 显式记忆增强语言模型

ExplicitLM 是一个创新的语言模型架构，通过引入显式记忆库（Memory Bank）解决传统语言模型知识更新困难和可解释性不足的问题。本项目基于 **Qwen3-4B** 模型，将知识以 token 序列的形式显式存储在共享记忆库中，通过可微分的检索和门控机制实现知识的透明管理、动态更新和端到端训练。

## 🎯 核心特性

- **Qwen3 基础架构**：基于 Qwen3-4B 预训练模型，保持强大的语言理解能力
- **显式记忆库**：将知识以 token 序列形式显式存储，支持直接查看和修改
- **均值池化查询**：使用序列均值池化生成查询向量，提高记忆检索效率
- **参数高效训练**：冻结 Qwen3 主模型参数，只训练记忆融合组件
- **Shortcut 机制**：即使没有相关知识，backbone 也能正常工作，确保模型鲁棒性

## 📁 项目结构

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
│   ├── model.py                    # 模型配置
│   ├── dataset.py                  # 数据集配置
│   ├── training.py                 # 训练配置
│   └── logging.py                  # 日志配置
├── utils/                           # 工具模块
│   ├── model_initializer.py        # 模型初始化
│   ├── pretrain_datasets.py        # 预训练数据加载
│   ├── sft_datasets.py             # SFT 数据加载
│   ├── train_loop_pretrain.py      # 预训练循环
│   └── train_loop_sft.py           # SFT 训练循环
├── scripts/                         # 工具脚本
│   ├── run_sft.sh                  # SFT 训练启动脚本
│   └── convert_omcq_to_sft.py     # 数据转换脚本
├── docs/                            # 文档
│   ├── SFT_TRAINING.md            # SFT 训练指南
│   └── GIT_SETUP.md               # Git 使用指南
├── 1_pretrain.py                   # 预训练入口（Hydra-Zen）
├── 2_sft.py                        # 监督微调入口（Hydra-Zen）
└── pyproject.toml                  # 项目依赖配置
```

## 🚀 快速开始

### 环境准备

```bash
# 1. 安装 uv 包管理器（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安装项目依赖
cd ExplicitLM
uv sync

# 3. 准备模型文件
# - Qwen3-4B 模型：下载到指定路径（如 /path/to/Qwen3-4b）
```

### 训练模型

#### 预训练

```bash
python 1_pretrain.py \
    model.qwen3_model_path=/path/to/Qwen3-4b \
    model.knowledge_num=1048576 \
    model.knowledge_length=16 \
    model.knowledge_dim=128 \
    dataset.dataset_path=data/database/merged_pretrain.jsonl \
    training.batch_size=48 \
    training.learning_rate=2e-4 \
    training.epochs=3
```

#### 监督微调（SFT）

```bash
# 方式1：使用启动脚本（推荐）
bash scripts/run_sft.sh

# 方式2：直接运行
python 2_sft.py \
    model.qwen3_model_path=/path/to/Qwen3-4b \
    model.pretrained_sft_model_path=out/pretrain_latest.pth \
    dataset.sft_dataset_path=sft_data/omcq_trex_sft.jsonl \
    training.batch_size=1 \
    training.accumulation_steps=128 \
    training.epochs=3
```

详细训练指南请参考 [SFT 训练文档](docs/SFT_TRAINING.md)。

## 🔧 核心配置参数

### 模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `qwen3_model_path` | Qwen3-4B 模型路径 | **必需** |
| `knowledge_num` | 记忆库条目总数（需为完全平方数） | 1048576 |
| `knowledge_length` | 每个记忆条目的 token 数 | 16 |
| `knowledge_dim` | 记忆嵌入向量维度 | 128 |

### 记忆检索参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_candidates` | 最终输出的候选数 | 8 |
| `num_candidates_internal` | 内部检索数量 | 128 |
| `num_selected` | 选中的条目数 | 1 |
| `gumbel_temperature` | Gumbel-Softmax 温度 | 1.0 |

### 知识库初始化

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `cache_path` | 预处理知识库缓存路径 | `data/cache/knowledge_cache.pt` |
| `recompute_cache` | 是否重新计算缓存 | False |

## 📊 训练策略

### 参数冻结

- **Qwen3 主模型**：完全冻结（`requires_grad=False`）
- **记忆库（Memory Bank）**：训练时固定，推理时可通过 LLMLingua 更新
- **记忆融合组件**：可训练
  - `memory_gate`：记忆门控层（Product Key Memory）
  - `gated_memory_fusion`：门控融合模块（含 Shortcut 机制）
  - `memory_norm`：记忆归一化层

### 记忆检索机制

1. **均值池化查询**：对序列进行均值池化生成查询向量
2. **两阶段检索**：
   - 第一阶段：使用 Product Key Memory 检索候选记忆
   - 第二阶段：每个位置独立计算相似度并选择最相关的记忆
3. **门控融合**：根据相似度动态控制记忆贡献（Shortcut 机制）

## 📝 数据格式

### 预训练数据（JSONL）

```jsonl
{"text": "人工智能是计算机科学的一个分支..."}
{"text": "机器学习是人工智能的核心技术..."}
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

## 🛠️ 开发指南

### 添加新的记忆更新策略

在 `utils/memory_bank_updater.py` 中实现新的更新策略。

### 自定义事实提取器

在 `utils/fact_extractor.py` 中扩展提取逻辑。

## 📚 相关文档

- [SFT 训练指南](docs/SFT_TRAINING.md)：详细的监督微调流程
- [Git 使用指南](docs/GIT_SETUP.md)：Git 仓库管理

## ⚠️ 注意事项

1. **模型路径配置**：确保 Qwen3-4B 模型路径正确，包含 `config.json` 和权重文件
2. **记忆库大小**：`knowledge_num` 必须是完全平方数（如 1024, 4096, 1048576）
3. **内存要求**：记忆库大小为 `knowledge_num × knowledge_length`，需要足够内存
4. **训练模式**：当前版本冻结 Qwen3 主模型，只训练记忆融合组件
5. **DeepSpeed**：推荐使用 DeepSpeed ZeRO-3 进行分布式训练

## 🔬 实验建议

1. **小规模测试**：先用小规模记忆库（如 1024 条目）验证流程
2. **渐进式训练**：先训练少量 epoch 观察损失变化
3. **参数调优**：根据任务调整 `num_candidates`、`gumbel_temperature` 等参数

## 📄 许可证

本项目采用研究性许可，仅供学术研究使用。

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen)：基础语言模型
- [LLMLingua](https://github.com/microsoft/LLMLingua)：文本压缩技术
