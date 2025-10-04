# ExplicitLM

ExplicitLM是一个创新的语言模型架构，通过引入显式记忆库（Memory Bank）解决传统语言模型知识更新困难和可解释性不足的问题。与传统模型将知识隐式存储在参数中不同，ExplicitLM将知识以人类可读的形式存储在显式记忆库中，实现了知识的透明管理、动态更新和可解释推理。该架构采用可微分的两阶段检索机制，在保持端到端训练能力的同时，显著提升了模型的可解释性和答案准确性。

## 目录结构

```
ExplicitLM/                     # 项目根目录
├── models/                     # 模型架构实现
│   ├── core/                  # 核心Transformer组件
│   ├── memory_bank/           # Memory Bank层实现
│   ├── retrieval/             # 两阶段检索机制
│   └── layers/                # 自定义层实现
├── data/                       # 数据管理
│   ├── raw/                   # 原始数据集
│   ├── processed/             # 预处理后数据
│   ├── knowledge_base/        # 显式知识库
│   └── benchmarks/            # 评估数据集
├── evaluation/                 # 评估和分析
├── experiments/                # 实验管理
│   ├── configs/               # 实验配置文件
│   ├── scripts/               # 实验运行脚本
│   └── results/               # 实验结果存储
├── scripts/                    # 实用脚本
│   ├── data_processing/       # 数据处理脚本
│   ├── model_analysis/        # 模型分析脚本
│   └── deployment/            # 部署相关脚本
├── utils/                      # 通用工具函数
├── tests/                      # 单元测试
├── docs/                       # 项目文档
├── checkpoints/                # 模型检查点
├── logs/                       # 训练和实验日志
└── visualization/              # 可视化工具
    ├── memory_analysis/       # Memory Bank可视化
    └── attention_maps/        # 注意力机制可视化
```

## 超参数配置系统

ExplicitLM采用基于优先级的超参数配置系统，所有超参数定义在`experiments/configs/LMConfig.py`中。系统通过`utils/config_utils.py`的`setup_config()`函数自动解析命令行参数并与配置文件合并，在main.py中只需一行调用即可完成配置。配置优先级遵循"命令行参数优先于配置文件默认值"的原则，确保实验参数的灵活性和可追溯性。特别地，系统不使用argparse的default参数，避免默认值掩盖潜在错误。

超参数按功能模块划分为七个语意分组。**基本模型架构参数**定义Transformer的核心结构，包括模型维度(dim=512)、层数(n_layers=8)、注意力头数(n_heads=16)等，控制模型的基础容量和计算复杂度。**知识库相关配置**是ExplicitLM的核心创新，knowledge_num(1024×1024)定义显式记忆库的条目数量，knowledge_length(16)指定每个记忆条目的token长度，knowledge_dim(128)设置记忆向量的嵌入维度，这三个参数共同决定了模型的知识存储容量。**EMA更新相关配置**借鉴VQ-VAE的指数移动平均机制实现记忆库的动态更新，use_ema_update启用渐进式知识融合，ema_decay(0.9)控制新旧知识的平衡速度，freeze_ratio(0.2)允许冻结部分稳定记忆以防止过度修改。**实验1.4.10配置**实现优化的Gumbel-Softmax记忆选择机制，num_candidates(16)定义候选搜索空间，gumbel_temperature控制选择的随机性，是当前主要研究方向。其他分组包括可选的MOE混合专家架构、DB数据库功能和三元组提取配置，为知识图谱集成提供支持。

使用示例：`python main.py --dim 1024 --n_layers 12 --use_moe`可覆盖模型规模并启用MOE；`python main.py --ema_decay 0.95 --num_candidates 32`可调整EMA机制和候选数量。所有参数均可通过`python main.py --help`查看完整说明。

## 工具选择
1. 我们使用uv作为我们的python包管理器 - [查看uv使用指南](docs/uv.md)
2. 我们使用DVC作为数据管理工具 - [查看DVC使用指南](docs/dvc.md)
3. 我们使用Git作为版本控制工具 - [查看Git使用指南](docs/git.md)
