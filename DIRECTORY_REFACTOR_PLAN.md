# ExplicitLM 目录重构方案

## 一、当前问题分析

### 1.1 根目录混乱
根目录下有太多脚本文件：
- 训练脚本: `train_pretrain.py`, `train_sft.py`
- 转换脚本: `pt2hg_apart.py`, `pt2hg_sync.py`
- 测试脚本: `test_*.py` (4个)
- 工具脚本: `chat_example.py`, `convert_conversations_to_labeled.py`
- 配置文件: `accelerate_config.yaml`, `ds_config.json`

### 1.2 目录职责不清
- `utils/` 和 `util_py/` 功能重叠
- 模型输出目录分散：`checkpoints/`, `out/`, `gate_ckpt/`
- 数据目录分散：`data/`, `sft_data/`, `parquet_series/`

### 1.3 可删除的废弃文件
| 文件/目录 | 原因 |
|-----------|------|
| `download_qwen_model.py` | 模型下载功能不再使用 |
| `scripts/convert_omcq_to_sft.py` | OMCQ 数据转换已无用 |
| `uptodate/` | 旧版本开发代码 |

### 1.4 无需处理的目录
| 目录 | 原因 |
|------|------|
| `my_model/` | 转换脚本 `pt2hg_apart.py` 生成的输出目录，不属于项目代码 |
| `hf_explicitlm_model/` | 同上，转换脚本输出 |
| `Qwen_hg/` | 外部模型，保持不动 |
| `llmlingua-2-bert/` | 外部模型，保持不动 |

## 二、功能模块划分

根据项目实际功能，划分为以下模块：

### 2.1 训练模块 (Training)
包含训练脚本和所有训练相关的工具。

**核心组件:**
- 训练脚本: `train_pretrain.py`, `train_sft.py`
- 模型初始化: `model_initializer.py`
- 数据集加载: `pretrain_datasets.py`, `sft_datasets.py`
- 训练循环: `train_loop_sft.py`, `train_utils.py`
- **Memory Bank 管理** (属于训练):
  - `memory_bank_updater.py`
  - `memory_update_tracker.py`
  - `keys_recluster.py`
  - `clustering.py`

### 2.2 推理模块 (Inference)
包含推理和对话相关脚本。

**核心组件:**
- `chat_example.py`
- `dual_path_inference.py`

### 2.3 模型转换模块 (Convert)
将训练好的 checkpoint 转换为 HuggingFace 格式。

**核心组件:**
- `pt2hg_apart.py` (推荐)
- `pt2hg_sync.py`

### 2.4 数据预处理模块 (Data Preprocessing)
训练前的数据准备工作。

**核心组件:**
- `convert_conversations_to_labeled.py`
- `util_py/generate_keys_from_memory_bank.py`
- `util_py/extract_parquet_to_memory_bank.py`
- `util_py/convert_extract_data_to_sft.py`
- `data/pt_factorys/scripts/*.py`

### 2.5 测试模块 (Testing)
模型测试和评估脚本。

**核心组件:**
- `test_checkpoint_comparison.py`
- `test_eval.py`
- `test_inference_baseline.py`
- `test_inference_update.py`

### 2.6 核心模型代码 (Models)
模型架构定义（保持不变）。

**核心组件:**
- `models/core/ExplicitLM.py`
- `models/core/Qwen3ExplicitLMBlock.py`
- `models/memory_bank/MemoryGate.py`
- `models/memory_bank/GatedMemoryFusion.py`

## 三、推荐的目录结构

```
ExplicitLM/
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
│
├── models/                    # 核心模型代码（保持不变）
│   ├── core/
│   ├── memory_bank/
│   └── layers/
│
├── config/                    # 配置模块（保持不变）
│
├── training/                  # 训练模块（新建，统一训练相关代码）
│   ├── train_pretrain.py     # 从根目录移入
│   ├── train_sft.py          # 从根目录移入
│   ├── utils/                # 从 utils/ 移入训练相关工具
│   │   ├── model_initializer.py
│   │   ├── datasets.py       # 合并 pretrain_datasets.py + sft_datasets.py
│   │   ├── train_loop.py     # 原 train_loop_sft.py
│   │   ├── train_utils.py
│   │   ├── memory_bank_updater.py  # Memory Bank 管理属于训练
│   │   ├── memory_update_tracker.py
│   │   ├── keys_recluster.py
│   │   └── clustering.py
│   └── scripts/              # 从 scripts/ 移入训练启动脚本
│       ├── run_sft.sh
│       └── run_fusion_pretrain.sh
│
├── inference/                 # 推理模块（新建）
│   ├── chat.py               # 原 chat_example.py
│   └── dual_path.py          # 原 dual_path_inference.py
│
├── convert/                   # 模型转换模块（新建）
│   ├── pt2hg_apart.py        # 从根目录移入
│   ├── pt2hg_sync.py         # 从根目录移入
│   └── scripts/
│       └── convert_checkpoint_to_hf.sh
│
├── data_preprocessing/        # 数据预处理模块（新建）
│   ├── convert_conversations_to_labeled.py
│   ├── generate_keys.py      # 合并 keys 生成相关脚本
│   ├── extract_memory_bank.py
│   ├── convert_to_sft.py
│   └── factory/              # 从 data/pt_factorys/scripts/ 移入
│       ├── jsonl_to_pt.py
│       ├── prepare_unified_data.py
│       ├── universal_fact_extractor.py
│       └── add_source_ids.py
│
├── tests/                     # 测试模块（新建）
│   ├── test_checkpoint_comparison.py
│   ├── test_eval.py
│   ├── test_inference_baseline.py
│   └── test_inference_update.py
│
├── utils/                     # 通用工具（精简后）
│   ├── logger.py
│   ├── fact_extractor.py
│   └── dataset_processor/
│
├── data/                      # 数据目录（重新组织）
│   ├── raw/                  # 原始数据
│   │   ├── knowledge_base/
│   │   └── benchmarks/
│   ├── cache/                # 缓存（Memory Bank, Keys 等）
│   ├── sft/                  # SFT 数据（原 sft_data/）
│   └── pretrain/             # 预训练数据（Parquet）
│
├── outputs/                   # 统一输出目录（新建）
│   ├── checkpoints/          # 训练 checkpoint（原 checkpoints/）
│   │   ├── fusion_pretrain/
│   │   ├── sft/
│   │   └── gate/            # 原 gate_ckpt/
│   └── logs/                 # 日志（原 logs/）
│
├── configs/                   # 配置文件（从根目录移入）
│   ├── accelerate_config.yaml
│   └── ds_config.json
│
├── docs/                      # 文档目录（保持不变）
│
├── examples/                  # 示例代码（保持不变）
│
└── external/                  # 外部资源（可选）
    └── Qwen_hg/              # 或保持在根目录
```

## 四、简化版方案（推荐先执行）

如果完整重构工作量太大，建议分阶段执行：

### 阶段 1：清理废弃文件（立即可做）
```bash
cd /data2/zengzheni/lvchangwei/new_repo/ExplicitLM

# 删除废弃文件
rm -f download_qwen_model.py
rm -f scripts/convert_omcq_to_sft.py
rm -rf uptodate/
```

### 阶段 2：统一输出目录
```bash
# 创建统一输出目录
mkdir -p outputs/checkpoints outputs/logs

# 移动现有输出
mv checkpoints/* outputs/checkpoints/ 2>/dev/null || true
mv gate_ckpt outputs/checkpoints/gate 2>/dev/null || true
mv logs/* outputs/logs/ 2>/dev/null || true

# 清理空目录
rmdir checkpoints gate_ckpt logs 2>/dev/null || true
```

### 阶段 3：整理根目录脚本
```bash
# 创建功能目录
mkdir -p training/scripts
mkdir -p convert
mkdir -p tests

# 移动训练脚本
mv train_pretrain.py training/
mv train_sft.py training/
mv scripts/run_sft.sh training/scripts/
mv scripts/run_fusion_pretrain.sh training/scripts/

# 移动转换脚本
mv pt2hg_apart.py convert/
mv pt2hg_sync.py convert/
mv scripts/convert_checkpoint_to_hf.sh convert/

# 移动测试脚本
mv test_*.py tests/

# 移动配置文件
mkdir -p configs
mv accelerate_config.yaml configs/
mv ds_config.json configs/
```

### 阶段 4：合并 utils 和 util_py
```bash
# 将 util_py 合并到 data_preprocessing
mkdir -p data_preprocessing
mv util_py/*.py data_preprocessing/
mv convert_conversations_to_labeled.py data_preprocessing/

# 移动数据工厂脚本
mv data/pt_factorys/scripts/*.py data_preprocessing/
```

### 阶段 5：更新导入路径
需要更新所有 Python 文件中的导入路径（这是工作量最大的部分）。

## 五、导入路径更新示例

### 训练脚本
```python
# 更新前
from utils.model_initializer import init_model
from utils.sft_datasets import create_sft_dataloader

# 更新后
from training.utils.model_initializer import init_model
from training.utils.datasets import create_sft_dataloader
```

### 配置文件路径
```bash
# 更新前
accelerate launch --config_file accelerate_config.yaml train_sft.py

# 更新后
accelerate launch --config_file configs/accelerate_config.yaml training/train_sft.py
```

## 六、总结

### 核心改动点
1. **删除废弃文件**: `download_qwen_model.py`, `scripts/convert_omcq_to_sft.py`, `uptodate/`
2. **统一输出目录**: `checkpoints/`, `gate_ckpt/`, `logs/` → `outputs/`
3. **整理根目录脚本**: 按功能分类到 `training/`, `convert/`, `tests/`
4. **合并工具目录**: `utils/` + `util_py/` 按功能重新分配
5. **Memory Bank 管理归属训练**: `memory_bank_updater.py` 等放入 `training/utils/`

### 不动的部分
- `models/`: 核心模型代码
- `config/`: 配置模块
- `Qwen_hg/`, `llmlingua-2-bert/`: 外部模型
- `my_model/`, `hf_explicitlm_model/`: 转换脚本输出（自动生成）
- `docs/`, `examples/`: 文档和示例

### 建议执行顺序
1. 阶段 1（立即）→ 阶段 2 → 阶段 3 → 阶段 4
2. 阶段 5 在功能稳定后再执行（导入路径更新工作量大）
