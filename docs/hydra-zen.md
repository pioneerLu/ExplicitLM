# Hydra-Zen 使用指南

## 简介

Hydra-Zen 是 Hydra 配置系统的增强版本，专门为机器学习项目提供了类型安全、可组合的配置管理能力。在 ExplicitLM 项目中，Hydra-Zen 用于统一管理模型、数据集、训练和日志等所有配置项。

## 为什么要使用 Hydra-Zen？

- **配置复用**：通过配置文件模板和参数覆盖，轻松创建不同的实验配置
- **类型安全**：提供静态类型检查，减少配置错误
- **可组合性**：配置项可以灵活组合，支持复杂的配置结构
- **版本控制**：每个实验的完整配置快照，确保实验可重现
- **命令行覆盖**：支持运行时动态修改配置参数

## 核心概念

### 1. 配置存储（Store）

Hydra-Zen 使用配置存储来管理所有可用的配置：

```python
from hydra_zen import store

store(model_config, name="lmconfig", package="_global_")
```

### 2. 配置结构

ExplicitLM 项目的配置分为四大模块：

- **model**: 模型结构相关配置
- **dataset**: 数据集相关配置
- **training**: 训练过程相关配置
- **logging**: 日志和监控相关配置

### 3. 配置继承与覆盖

- 通过继承基础配置，创建特定实验的配置
- 使用命令行参数动态覆盖配置项：`param.subparam=value`

## 项目中的配置结构

### Model 配置 (`config/model.py`)

```python
# 模型架构参数
dim: int = 1024                    # 模型维度
n_layers: int = 12                 # Transformer层数
n_heads: int = 16                  # 注意力头数
vocab_size: int = 50257            # 词汇表大小
max_seq_len: int = 2048            # 最大序列长度

# 内存库参数
knowledge_num: int = 1048576       # 记忆库条目数量
knowledge_length: int = 64         # 每个记忆条目的长度
knowledge_dim: int = 1024          # 记忆条目的维度

# EMA更新参数
use_ema_update: bool = True        # 是否使用EMA更新
ema_decay: float = 0.999           # EMA衰减率
ema_update_freq: int = 100         # EMA更新频率

# Gumbel-Softmax参数
num_candidates: int = 100          # 候选记忆条目数
num_selected: int = 10             # 选择的记忆条目数
gumbel_temperature: float = 1.0    # Gumbel-Softmax温度
```

### Dataset 配置 (`config/dataset.py`)

```python
# 数据集路径和参数
dataset_path: str = "data/database/merged_pretrain.jsonl"
val_dataset_path: str = "data/benchmarks/eval_data.json"

# 数据加载参数
batch_size: int = 48               # 批处理大小
num_workers: int = 4               # 数据加载进程数
shuffle: bool = True               # 是否打乱训练数据
```

### Training 配置 (`config/training.py`)

```python
# 训练参数
epochs: int = 3                    # 训练轮数
learning_rate: float = 2e-4        # 学习率
weight_decay: float = 0.1          # 权重衰减
warmup_steps: int = 1000           # 预热步数

# DeepSpeed配置
zero_stage: int = 2                # DeepSpeed ZeRO阶段
mixed_precision: str = "bf16"      # 混合精度训练

# 其他训练参数
seed: int = 42                     # 随机种子
accumulation_steps: int = 1        # 梯度累积步数
save_steps: int = 1000             # 保存间隔步数
```

### Logging 配置 (`config/logging.py`)

```python
# 日志和监控参数
use_swanlab: bool = True           # 是否使用SwanLab监控
log_interval: int = 10             # 日志记录间隔
checkpoint_dir: str = "checkpoints" # 检查点保存目录
```

## 配置使用方法

### 1. 启动训练脚本

使用 Hydra-Zen 启动训练时，通过 `launch` 函数：

```python
from hydra_zen import launch
from config import _main_cfg_func

def main(cfg):
    # cfg 包含所有配置节点
    model_cfg = cfg.model
    dataset_cfg = cfg.dataset
    # ... 训练逻辑

if __name__ == "__main__":
    launch(_main_cfg_func, main)
```

### 2. 命令行参数覆盖

运行训练脚本时，可以使用 Hydra-Zen 的参数覆盖语法：

```bash
# 基本用法
python 1_pretrain.py training.epochs=10 model.knowledge_num=1048576

# 多个参数
python 1_pretrain.py training.epochs=5 model.dim=512 training.learning_rate=1e-4

# 使用实验脚本
./experiments/scripts/exp_001_hydra_zen.sh
```

### 3. 在实验脚本中使用

在实验脚本中，通过 `TRAIN_ARGS` 变量指定 Hydra-Zen 参数：

```bash
# 实验配置
EXP_ID="exp_001_hydra_zen"
EXP_DESC="基线实验 Hydra-Zen配置版 knowledge_num=1M epochs=10"

# Hydra-Zen 配置覆盖参数
TRAIN_ARGS="training.epochs=10 model.knowledge_num=1048576 model.dim=512 training.learning_rate=2e-4"
```

## 实验管理集成

### 1. 配置版本控制

每个实验的完整配置都被记录在 JSON 文件中，包含：

- 代码版本 (Git commit hash)
- 数据集版本
- 所有配置参数
- 复现实验的完整命令

### 2. 配置解析

实验脚本会解析 Hydra-Zen 风格的参数（key=value 格式）：

```bash
# 参数格式
"training.epochs=10 model.knowledge_num=1048576"

# 解析后转换为JSON格式存储
{
  "training": {
    "epochs": 10
  },
  "model": {
    "knowledge_num": 1048576
  }
}
```

## 最佳实践

### 1. 配置命名约定

- 使用小写字母和下划线分隔
- 参数名应清晰描述其用途
- 按功能模块组织参数

### 2. 实验配置管理

- 为每个实验创建独立的配置脚本
- 保留所有配置的历史记录
- 使用有意义的实验描述

### 3. 运行目录注意事项
- **工作目录管理**：使用 `hydra.utils.get_original_cwd()` 来获取原始工作目录而不是 `os.getcwd()`
- **相对路径处理**：配置中的相对路径会被相对于 Hydra 输出目录解析，必要时使用绝对路径
- **输出路径设置**：使用 `hydra.job.chdir=True` 可以让程序在作业目录中运行
- **文件路径一致性**：确保在集群环境下不同节点的路径结构一致

#### 示例代码：正确使用get_original_cwd()

```python
from hydra.utils import get_original_cwd
from pathlib import Path

def main(cfg):
    # ❌ 错误方式：使用当前工作目录（可能在输出目录中）
    # data_path = "data/database/merged_pretrain.jsonl"

    # ✅ 正确方式：使用原始工作目录
    original_cwd = get_original_cwd()
    data_path = Path(original_cwd) / "data/database/merged_pretrain.jsonl"

    # 或者更简洁的写法
    data_path = Path(get_original_cwd()) / "data/database/merged_pretrain.jsonl"

    # 处理数据加载等操作...
```

#### 示例代码：处理输出目录

```python
from hydra.utils import get_original_cwd
from pathlib import Path
import os

def main(cfg):
    # 获取原始工作目录
    original_cwd = get_original_cwd()

    # 获取Hydra的输出目录
    hydra_output_dir = os.getcwd()

    # 在原始项目目录中读取数据
    data_file = Path(original_cwd) / "data/train.json"

    # 在Hydra输出目录中保存结果
    result_file = Path(hydra_output_dir) / "results/model_output.pt"

    print(f"原始工作目录: {original_cwd}")
    print(f"Hydra输出目录: {hydra_output_dir}")
    print(f"数据文件路径: {data_file}")
    print(f"结果文件路径: {result_file}")
```

#### 命令行使用示例：

```bash
# 设置工作目录切换（推荐用于集群环境）
python 1_pretrain.py hydra.job.chdir=True

# 或在配置中设置
python 1_pretrain.py +hydra.job.chdir=True
```

### 4. 参数验证

- 在配置类中定义合理的默认值
- 验证参数的有效性和范围
- 使用类型注解提高代码可读性

## 常见用例示例

### 示例1：调整模型大小

```bash
# 小模型实验
python 1_pretrain.py model.dim=256 model.n_layers=4 training.batch_size=32

# 大模型实验
python 1_pretrain.py model.dim=2048 model.n_layers=24 training.batch_size=16
```

### 示例2：调整训练参数

```bash
# 不同学习率实验
python 1_pretrain.py training.learning_rate=1e-3 training.epochs=5

# 不同记忆库大小实验
python 1_pretrain.py model.knowledge_num=524288 model.num_candidates=50
```

### 示例3：使用实验脚本

```bash
#!/bin/bash
# 实验ID
EXP_ID="exp_001_hydra_zen"
EXP_DESC="基线实验 Hydra-Zen配置版"

# Hydra-Zen 配置覆盖参数
TRAIN_ARGS="training.epochs=10 model.knowledge_num=1048576 model.dim=64 model.n_layers=2 training.batch_size=1 training.learning_rate=2e-4"

# 调用核心运行脚本
source experiments/scripts/_run_experiment_core_hydra_zen.sh
```

## 故障排除

### 1. 参数解析错误

如果遇到参数解析错误，请检查：

- 参数格式是否为 `key=value` 格式
- 参数名称是否正确（注意嵌套结构）
- 参数值是否为有效类型

### 2. 配置不生效

- 确认参数覆盖语法正确
- 检查配置层次结构
- 验证配置文件是否正确加载

### 3. 性能问题

- 监控内存使用情况
- 调整批处理大小和模型参数
- 使用 DeepSpeed 优化内存使用

通过 Hydra-Zen，ExplicitLM 项目实现了灵活、可重现的实验管理，每个实验的配置都清晰记录，确保研究结果的可靠性和可复现性。