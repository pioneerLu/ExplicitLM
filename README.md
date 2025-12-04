# ExplicitLM - Qwen3 显式记忆增强语言模型

ExplicitLM 是一个创新的语言模型架构，通过引入显式记忆库（Memory Bank）解决传统语言模型知识更新困难和可解释性不足的问题。本项目基于 **Qwen3-4B** 模型，将知识以 token 序列的形式显式存储在共享记忆库中，通过可微分的检索和门控机制实现知识的透明管理、动态更新和端到端训练。

## 🎯 核心特性

- **Qwen3 基础架构**：基于 Qwen3-4B 预训练模型，保持强大的语言理解能力
- **显式记忆库**：将知识以 token 序列形式显式存储，支持直接查看和修改
- **动态知识更新**：训练时记忆库固定，推理时通过 LLMLingua 事实提取实现记忆库的动态更新
- **双路推理**：主路生成文本，辅路提取事实并更新知识库
- **参数高效训练**：冻结 Qwen3 主模型参数，只训练记忆融合组件
- **Shortcut 机制**：即使没有相关知识，backbone 也能正常工作，确保模型鲁棒性

## 📁 项目结构

```
ExplicitLM/
├── models/                          # 模型架构
│   ├── core/
│   │   ├── ExplicitLM.py           # 主模型类
│   │   └── Qwen3ExplicitLMBlock.py # Qwen3 Transformer Block
│   ├── memory_bank/
│   │   ├── MemoryGate.py           # 记忆门控层
│   │   └── GatedMemoryFusion.py    # 门控记忆融合模块
│   └── qwen_tokenizer/             # Qwen tokenizer 支持
├── utils/                           # 工具模块
│   ├── model_initializer.py        # 模型初始化（支持cache加载）
│   ├── dual_path_inference.py      # 双路推理包装器
│   ├── fact_extractor.py           # 事实提取器
│   ├── memory_bank_updater.py      # 记忆库更新器
│   ├── pretrain_datasets.py        # 数据加载器
│   └── logger.py                    # 日志工具
├── data/                            # 数据目录
│   ├── cache/                       # 知识库缓存
│   │   └── knowledge_cache.pt      # 预处理的记忆库数据
│   ├── database/                    # 训练数据
│   └── benchmarks/                  # 验证数据
├── 1_pretrain.py                    # 预训练入口（Hydra-Zen）
├── 2_sft.py                         # 监督微调入口
├── pyproject.toml                   # 项目依赖配置
└── README.md                        # 本文档
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
# - LLMLingua 模型：下载到指定路径（如 /path/to/llmlingua-2-bert）
#   下载方法：运行 bert/get_model.py 或从 HuggingFace 下载
```

### 基本使用

#### 1. 初始化模型（使用 cache 知识库）

```python
import sys
sys.path.insert(0, '.')
from utils.model_initializer import init_model

# 配置参数（请将路径替换为实际路径）
args = {
    'qwen3_model_path': '/path/to/Qwen3-4b',  # Qwen3模型路径（必需）
    'knowledge_num': 1024 * 1024,            # 记忆库条目数（1048576）
    'knowledge_length': 16,                  # 每个记忆条目的token数
    'knowledge_dim': 128,                     # 记忆嵌入维度
    'use_moe': False,                        # 是否使用MOE模式
    'num_candidates': 8,                     # 候选记忆条目数
    'num_selected': 1,                       # 选中的记忆条目数
    # 使用预处理的cache知识库
    'cache_path': 'data/cache/knowledge_cache.pt',
    'recompute_cache': False,                # 不重新计算，直接使用cache
}

# 初始化模型
model, tokenizer = init_model(args, accelerator=None)
model.eval()
```

#### 2. 使用双路推理进行生成

```python
from utils.dual_path_inference import DualPathInference
from utils.fact_extractor import FactExtractor

# 初始化事实提取器（需要指定LLMLingua模型路径）
fact_extractor = FactExtractor(
    model_path='/path/to/llmlingua-2-bert',  # LLMLingua模型路径
    compression_rate=0.4,  # 压缩到40%，保留60%的关键信息
)

# 方式1：显式传入事实提取器（推荐）
dual_path = DualPathInference(
    model=model,
    tokenizer=tokenizer,
    fact_extractor=fact_extractor,  # 传入事实提取器
    enable_fact_extraction=True,     # 启用事实提取
    fact_update_frequency=1,        # 每次推理都更新
    update_strategy='fifo',         # FIFO更新策略
)

# 方式2：直接传入LLMLingua模型路径（自动创建FactExtractor）
# dual_path = DualPathInference(
#     model=model,
#     tokenizer=tokenizer,
#     enable_fact_extraction=True,
#     fact_update_frequency=1,
#     update_strategy='fifo',
#     llmlingua_model_path='/path/to/llmlingua-2-bert',  # LLMLingua模型路径
#     compression_rate=0.4,
# )

# 生成文本
input_text = "什么是人工智能？"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(model.device)

result = dual_path.generate(
    input_ids,
    input_text=input_text,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

print(result['generated_text'])
```

#### 3. 训练模型

```bash
# 使用 Hydra-Zen 配置系统进行预训练
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

> **注意**：请将 `/path/to/Qwen3-4b` 和 `/path/to/llmlingua-2-bert` 替换为实际的模型路径

## 🔧 配置参数说明

### 核心模型参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `qwen3_model_path` | Qwen3-4B 模型路径 | **必需** |
| `knowledge_num` | 记忆库条目总数 | 1048576 |
| `knowledge_length` | 每个记忆条目的 token 数 | 16 |
| `knowledge_dim` | 记忆嵌入向量维度 | 128 |

### 记忆检索参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_candidates` | 第一阶段检索的候选数 | 8 |
| `num_selected` | 第二阶段选中的条目数 | 1 |
| `gumbel_temperature` | Gumbel-Softmax 温度 | 1.0 |

### 记忆更新机制

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `freeze_ratio` | 冻结的记忆条目比例 | 0.2 |

**注意**：记忆库在训练时固定，推理时通过 LLMLingua 进行动态更新。

### 知识库初始化参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `cache_path` | 预处理知识库缓存路径 | `data/cache/knowledge_cache.pt` |
| `recompute_cache` | 是否重新计算缓存 | False |
| `database_init_path` | 原始知识库数据路径 | None |

## 📊 参数冻结策略

ExplicitLM 采用参数高效训练策略：

- **Qwen3 主模型**：完全冻结（`requires_grad=False`）
- **记忆库（Memory Bank）**：训练时固定（`requires_grad=False`），推理时通过 LLMLingua 更新
- **记忆融合组件**：可训练
  - `memory_gate`：记忆门控层
  - `gated_memory_fusion`：门控融合模块（包含 Shortcut 机制）
  - `memory_norm`：记忆归一化层

这种策略确保：
- 保持 Qwen3 的预训练知识
- 只训练记忆融合机制
- 大幅减少可训练参数数量
- 通过 Shortcut 机制确保 backbone 独立工作能力

## 🔄 双路推理机制

### 主路：文本生成
- 使用 Qwen3 主模型进行标准文本生成
- 融合显式记忆库中的相关知识
- 通过 Shortcut 机制和门控权重控制知识融合强度
- **Shortcut 机制**：即使没有相关知识或相似度较低，backbone 仍能正常工作

### 辅路：事实提取与更新
- 使用 LLMLingua 压缩技术提取关键事实
- 将提取的事实更新到记忆库
- 支持多种更新策略（FIFO、相似度替换等）

### Shortcut 机制详解

Shortcut 机制确保模型鲁棒性：
- **公式**：`output = hidden_states + alpha * memory_output`
- **alpha 计算**：基于相似度分数动态计算，范围 [0, 1]
  - 相似度高 → alpha 接近 1 → memory 贡献大
  - 相似度低 → alpha 接近 0 → memory 贡献小，backbone 独立工作
- **优势**：即使知识库中没有相关信息，模型仍能基于 Qwen3 的预训练知识正常回答

## 📝 数据格式

### 训练数据格式（JSONL）

```jsonl
{"text": "人工智能是计算机科学的一个分支..."}
{"text": "机器学习是人工智能的核心技术..."}
```

### 知识库数据格式（JSON）

```json
[
    {"sentence": "人工智能是计算机科学的一个分支"},
    {"sentence": "机器学习是人工智能的核心技术"},
    ...
]
```

## 🛠️ 开发指南

### 添加新的记忆更新策略

在 `utils/memory_bank_updater.py` 中实现新的更新策略：

```python
def update_memory_bank_custom(self, facts, memory_indices):
    # 实现自定义更新逻辑
    pass
```

### 自定义事实提取器

在 `utils/fact_extractor.py` 中扩展提取逻辑：

```python
class CustomFactExtractor(FactExtractor):
    def extract_facts(self, text):
        # 实现自定义提取逻辑
        pass
```

## 📚 相关文档

- [实验工作流指南](docs/experiment_workflow.md)：详细的实验管理流程
- [DVC 数据版本管理](docs/dvc_guide.md)：数据版本控制指南
- [uv 包管理器](docs/uv.md)：依赖管理工具使用

## ⚠️ 注意事项

1. **模型路径配置**：
   - **Qwen3-4B 模型**：确保路径正确，包含 `config.json` 和权重文件
   - **LLMLingua 模型**：需要下载 LLMLingua-2-BERT 模型，可通过 `bert/get_model.py` 下载或从 HuggingFace 获取
   - 所有路径请使用绝对路径或相对于项目根目录的路径
2. **内存要求**：记忆库大小为 `knowledge_num × knowledge_length`，需要足够内存
3. **训练模式**：当前版本冻结 Qwen3 主模型，只训练记忆融合组件
4. **生成质量**：使用记忆库时，需要先训练记忆融合组件才能获得良好效果
5. **LLMLingua 依赖**：如果启用事实提取功能，需要安装 `llmlingua` 包：`pip install llmlingua`

## 🔬 实验建议

1. **小规模测试**：先用小规模记忆库（如 1024 条目）验证流程
2. **渐进式训练**：先训练少量 epoch 观察损失变化
3. **参数调优**：根据任务调整 `num_candidates`、`gumbel_temperature` 等参数
4. **对比实验**：对比使用/不使用记忆库的效果差异

## 📄 许可证

本项目采用研究性许可，仅供学术研究使用。

## 🙏 致谢

- [Qwen](https://github.com/QwenLM/Qwen)：基础语言模型
- [LLMLingua](https://github.com/microsoft/LLMLingua)：文本压缩技术
