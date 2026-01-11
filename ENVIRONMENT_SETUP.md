# 环境配置迁移指南

本文档说明如何在新的机器上配置与当前环境相同的训练环境，特别是 FAISS GPU 支持。

## 📋 前置要求

### 系统要求
- Linux 系统（推荐 Ubuntu 20.04+）
- NVIDIA GPU（已测试 A800）
- CUDA 驱动版本：545.23.06 或更高（目标设备：545.23.06）
- CUDA Toolkit：12.3（与 PyTorch 兼容）

### 软件要求
- Conda/Miniconda（用于管理 Python 环境）
- Python 3.12
- Git
- uv（Python 包管理器，用于管理项目依赖）

## 📦 使用 uv 迁移文件清单

如果使用 `uv` 在新机器上复现环境，需要迁移以下文件：

**快速打包**: 运行 `./package_for_migration.sh` 脚本自动打包所有必须迁移的文件。

### ✅ 必须迁移的文件

#### 1. 依赖配置文件（核心）
- **`pyproject.toml`** - uv 的项目依赖配置文件，包含所有 Python 包依赖
- **`uv.lock`** - 锁定的依赖版本文件（如果存在），确保依赖版本一致性

#### 2. 源代码文件
- **`models/`** - 模型定义代码
- **`utils/`** - 工具函数（包括 `clustering.py`, `memory_bank_updater.py`, `fact_extractor.py` 等）
- **`config/`** - 配置文件目录（包括 `memory_update.py` 等）
- **`scripts/`** - 训练和运行脚本
- **`train_pretrain.py`** - 主训练脚本
- **`train_sft.py`** - SFT 训练脚本
- 其他 Python 源代码文件（`*.py`）

#### 3. 配置文件
- **`accelerate_config.yaml`** - Accelerate 分布式训练配置
- **`ds_config.json`** - DeepSpeed 配置（如果使用）
- **`.gitignore`** - Git 忽略规则（可选）

#### 4. 文档文件（可选但推荐）
- **`README.md`** - 项目说明
- **`ENVIRONMENT_SETUP.md`** - 本环境配置文档

### ❌ 不需要迁移的文件

以下文件/目录**不需要**迁移，可以在新机器上重新生成：

- **`.venv/`** - 虚拟环境目录（使用 `uv sync` 重新创建）
- **`__pycache__/`** - Python 字节码缓存
- **`*.pyc`**, **`*.pyo`** - Python 编译文件
- **`*.log`** - 日志文件
- **`checkpoints/`**, **`gate_ckpt/`** - 模型检查点（如果需要可以单独迁移）
- **`out/`**, **`outputs/`** - 训练输出目录
- **`logs/`**, **`swanlog/`** - 日志目录
- **`experiments/`** - 实验输出
- **`data/cache/`** - 数据缓存
- **`.hydra/`** - Hydra 配置缓存
- **`tokenizer_cache/`** - Tokenizer 缓存

### 📋 快速迁移命令

#### 方法 1: 使用 Git（推荐）

如果项目已纳入 Git 版本控制：

```bash
# 在新机器上克隆仓库
git clone <repository_url>
cd ExplicitLM

# 或者如果已有仓库，拉取最新代码
git pull origin main
```

#### 方法 2: 使用 rsync（保留文件权限）

```bash
# 从旧机器同步必要文件到新机器
rsync -av --exclude='.venv' \
          --exclude='__pycache__' \
          --exclude='*.log' \
          --exclude='checkpoints' \
          --exclude='out' \
          --exclude='logs' \
          --exclude='swanlog' \
          --exclude='experiments' \
          --exclude='data/cache' \
          --exclude='.hydra' \
          /path/to/old/ExplicitLM/ \
          user@new-machine:/path/to/new/ExplicitLM/
```

#### 方法 3: 使用 tar 打包

```bash
# 在旧机器上打包
cd /path/to/ExplicitLM
tar -czf explicitlm-migration.tar.gz \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.log' \
    --exclude='checkpoints' \
    --exclude='out' \
    --exclude='logs' \
    --exclude='swanlog' \
    --exclude='experiments' \
    --exclude='data/cache' \
    --exclude='.hydra' \
    pyproject.toml uv.lock models/ utils/ config/ scripts/ \
    train_pretrain.py train_sft.py *.md *.yaml *.json

# 传输到新机器后解压
tar -xzf explicitlm-migration.tar.gz -C /path/to/new/ExplicitLM/
```

### 🔄 迁移后的环境重建步骤

1. **安装 uv**（如果未安装）：

   **方法 1: 官方安装脚本（需要访问 GitHub/astral.sh）**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
   **方法 2: 使用国内镜像（推荐，无需翻墙）**
   ```bash
   # 使用国内镜像站点
   curl -LsSf https://uv.ifcfg.cn/install.sh | sh
   ```
   
   **方法 3: 使用 pip 安装（如果网络允许）**
   ```bash
   pip install uv
   ```
   
   **方法 4: 使用 conda 安装（如果有 conda 包）**
   ```bash
   conda install -c conda-forge uv
   ```
   
   **方法 5: 手动下载二进制文件（如果以上方法都不可用）**
   ```bash
   # 访问 https://github.com/astral-sh/uv/releases 下载对应平台的二进制文件
   # Linux x86_64 示例：
   wget https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
   tar -xzf uv-x86_64-unknown-linux-gnu.tar.gz
   sudo mv uv /usr/local/bin/  # 或添加到 PATH
   ```
   
   **验证安装：**
   ```bash
   uv --version
   ```
   
   **⚠️ 注意：** 如果服务器无法访问 GitHub 或 astral.sh，推荐使用方法 2（国内镜像）或方法 3（pip 安装）。

2. **创建 conda 环境**（参考下面的步骤 1-4）

3. **使用 uv 同步依赖**：
   ```bash
   cd /path/to/ExplicitLM
   conda activate qwen3
   
   # 使用 conda 环境的 Python 创建虚拟环境
   uv venv --python $(which python)
   
   # 同步依赖（根据 pyproject.toml 和 uv.lock）
   uv sync
   
   # ⚠️ 重要：删除 .venv 中的 numpy（使用 conda 环境的）
   source .venv/bin/activate
   pip uninstall -y numpy 2>/dev/null || true
   rm -rf .venv/lib/python3.12/site-packages/numpy* 2>/dev/null || true
   ```

4. **验证环境**（参考下面的验证步骤）

## 🔧 步骤 1: 创建 Conda 环境

```bash
# 创建名为 qwen3 的 conda 环境，Python 3.12
conda create -n qwen3 python=3.12 -y
conda activate qwen3
```

## 🔧 步骤 2: 安装 PyTorch 和基础依赖

**⚠️ 重要说明：CUDA 版本兼容性**

- **目标设备 CUDA 版本：** 12.3（驱动版本 545.23.06）
- **PyTorch CUDA 版本：** 使用 `pytorch-cuda=12.1`（PyTorch 使用 12.1 表示支持 CUDA 12.x，包括 12.3）
- **FAISS GPU：** 会自动匹配 PyTorch 的 CUDA 版本

```bash
# 激活环境
conda activate qwen3

# 安装 PyTorch（CUDA 12.3 版本）
# 注意：对于 CUDA 12.3，使用 pytorch-cuda=12.1（PyTorch 使用 12.1 表示支持 CUDA 12.x）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 验证 PyTorch CUDA 支持
python -c "import torch; print('✅ CUDA 可用:', torch.cuda.is_available()); print('✅ CUDA 版本:', torch.version.cuda); print('✅ GPU 数量:', torch.cuda.device_count())"
```

**预期输出：**
```
✅ CUDA 可用: True
✅ CUDA 版本: 12.1  # PyTorch 显示的版本，实际支持 CUDA 12.3
✅ GPU 数量: 8  # 或你的 GPU 数量
```

## 🔧 步骤 3: 安装 FAISS GPU（关键步骤）

**⚠️ 重要：必须从 pytorch channel 安装，确保 CUDA 版本匹配**

```bash
conda activate qwen3

# 卸载可能存在的旧版本
conda uninstall -y faiss faiss-gpu faiss-cpu libfaiss 2>/dev/null || true

# 从 pytorch channel 安装 faiss-gpu（会自动匹配 CUDA 版本）
conda install -c pytorch faiss-gpu -y

# 验证安装
python -c "import faiss; print('✅ FAISS GPU 数量:', faiss.get_num_gpus())"
```

**预期输出：**
```
✅ FAISS GPU 数量: 8  # 或你的 GPU 数量
```

## 🔧 步骤 4: 安装 NumPy（版本兼容性）

**⚠️ 关键：FAISS GPU 需要 NumPy < 2.0**

```bash
conda activate qwen3

# 安装 NumPy 1.x（与 faiss-gpu 兼容）
conda install "numpy<2" -y

# 验证版本
python -c "import numpy; print('NumPy 版本:', numpy.__version__)"
```

**预期输出：**
```
NumPy 版本: 1.26.4  # 或类似的 1.x 版本
```

## 🔧 步骤 5: 使用 uv 安装项目依赖

**⚠️ 重要：必须先完成步骤 1-4（创建 conda 环境并安装 faiss-gpu）**

```bash
# 进入项目目录
cd /path/to/ExplicitLM

# 激活 conda 环境
conda activate qwen3

# 使用 conda 环境的 Python 创建虚拟环境
uv venv --python $(which python)

# 激活虚拟环境
source .venv/bin/activate

# 同步依赖（根据 pyproject.toml 和 uv.lock）
uv sync

# ⚠️ 关键步骤：删除 .venv 中的 numpy，使用 conda 环境的
pip uninstall -y numpy 2>/dev/null || true
rm -rf .venv/lib/python3.12/site-packages/numpy* 2>/dev/null || true

# 验证 numpy 来自 conda 环境
python -c "import numpy; print('NumPy 路径:', numpy.__file__); assert 'conda' in numpy.__file__ or 'qwen3' in numpy.__file__, 'NumPy 应该来自 conda 环境'"
```

**说明：**
- `uv sync` 会根据 `pyproject.toml` 和 `uv.lock`（如果存在）安装所有依赖
- 必须删除 `.venv` 中的 numpy，因为 faiss-gpu 需要与 conda 环境中的 numpy 1.x 版本兼容
- 其他依赖（如 torch、transformers 等）可以从 `.venv` 安装，但 numpy 和 faiss 必须使用 conda 环境的

## 🔧 步骤 6: 配置代码以访问 Conda 环境的包

由于项目使用 `.venv` 虚拟环境，但 `faiss-gpu` 和 `numpy` 安装在 conda 环境中，需要在代码中自动加载 conda 环境的包。

### 方法：修改 `utils/clustering.py`

在 `utils/clustering.py` 文件开头添加以下代码，自动加载 conda 环境的包：

```python
import sys
import os

# 自动加载 conda 环境的包（faiss-gpu 和 numpy）
# 方法 1: 使用环境变量（推荐）
conda_prefix = os.environ.get("CONDA_PREFIX", "")
if conda_prefix:
    conda_site_packages = os.path.join(conda_prefix, "lib", "python3.12", "site-packages")
    if os.path.exists(conda_site_packages) and conda_site_packages not in sys.path:
        sys.path.insert(0, conda_site_packages)

# 方法 2: 使用硬编码路径（如果方法 1 不工作）
# conda_site_packages = os.path.expanduser("~/miniconda3/envs/qwen3/lib/python3.12/site-packages")
# if os.path.exists(conda_site_packages) and conda_site_packages not in sys.path:
#     sys.path.insert(0, conda_site_packages)

import numpy as np
import faiss
```

**说明：**
- 这段代码会在导入 `numpy` 和 `faiss` 之前，将 conda 环境的 site-packages 添加到 Python 路径
- 优先使用 `CONDA_PREFIX` 环境变量（conda 自动设置），更灵活
- 如果 `CONDA_PREFIX` 未设置，可以回退到硬编码路径

## ✅ 验证安装

运行以下命令验证所有组件：

```bash
conda activate qwen3
cd /path/to/ExplicitLM
source .venv/bin/activate  # 如果使用 .venv

# 测试 1: PyTorch CUDA
python -c "import torch; print('✅ PyTorch CUDA:', torch.cuda.is_available(), 'GPU数量:', torch.cuda.device_count())"

# 测试 2: FAISS GPU
python -c "import sys; import os; conda_path = os.path.expanduser('~/miniconda3/envs/qwen3/lib/python3.12/site-packages'); sys.path.insert(0, conda_path); import faiss; print('✅ FAISS GPU数量:', faiss.get_num_gpus())"

# 测试 3: 项目模块
python -c "from utils.clustering import perform_clustering; print('✅ clustering 模块正常')"
```

## 🐛 常见问题排查

### 问题 1: FAISS 检测不到 GPU

**症状：** `faiss.get_num_gpus()` 返回 0

**解决方案：**
1. 检查 CUDA 版本匹配：
   ```bash
   nvidia-smi  # 查看驱动版本（应显示 CUDA Version: 12.3）
   python -c "import torch; print('PyTorch CUDA 版本:', torch.version.cuda)"  # 查看 PyTorch CUDA 版本
   python -c "import torch; print('CUDA 可用:', torch.cuda.is_available())"  # 验证 CUDA 是否可用
   ```
   
   注意：PyTorch 的 `pytorch-cuda=12.1` 表示支持 CUDA 12.x（包括 12.3），这是正常的。
2. 确保从 `pytorch` channel 安装，而不是 `conda-forge`：
   ```bash
   conda uninstall -y faiss-gpu
   conda install -c pytorch faiss-gpu -y
   ```

### 问题 2: NumPy 版本冲突

**症状：** `ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**解决方案：**
1. 确保 conda 环境中使用 NumPy 1.x：
   ```bash
   conda activate qwen3
   conda install "numpy<2" -y
   ```
2. 删除 `.venv` 中的 NumPy：
   ```bash
   pip uninstall -y numpy
   rm -rf .venv/lib/python3.12/site-packages/numpy*
   ```

### 问题 3: 缺少共享库

**症状：** `ImportError: libxxx.so: cannot open shared object file`

**解决方案：**
1. 安装缺失的库：
   ```bash
   conda install -c conda-forge openblas -y
   ```
2. 创建符号链接（如果需要）：
   ```bash
   cd $CONDA_PREFIX/lib
   # 根据错误信息创建相应的符号链接
   ```

### 问题 4: Conda 锁文件冲突

**症状：** `LockError: Failed to acquire lock`

**解决方案：**
```bash
# 清理锁文件
find $CONDA_PREFIX -name "*.lock" -delete
find ~/miniconda3/pkgs -name "*.lock" -delete

# 终止其他 conda 进程
ps aux | grep conda | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null || true
```

## 📦 完整安装脚本

创建一个自动化安装脚本 `setup_environment.sh`：

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "环境配置脚本"
echo "=========================================="

# 1. 创建 conda 环境
echo "步骤 1: 创建 conda 环境..."
conda create -n qwen3 python=3.12 -y
conda activate qwen3

# 2. 安装 PyTorch
echo "步骤 2: 安装 PyTorch..."
# 对于 CUDA 12.3，使用 pytorch-cuda=12.1（PyTorch 使用 12.1 表示支持 CUDA 12.x）
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. 安装 FAISS GPU
echo "步骤 3: 安装 FAISS GPU..."
conda install -c pytorch faiss-gpu -y

# 4. 安装 NumPy 1.x
echo "步骤 4: 安装 NumPy..."
conda install "numpy<2" -y

# 5. 安装其他依赖
echo "步骤 5: 安装其他依赖..."
conda install -c conda-forge openblas -y

# 6. 验证
echo "步骤 6: 验证安装..."
python -c "import torch; print('✅ PyTorch CUDA:', torch.cuda.is_available())"
python -c "import faiss; print('✅ FAISS GPU数量:', faiss.get_num_gpus())"
python -c "import numpy; print('✅ NumPy版本:', numpy.__version__)"

echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
```

## 📝 环境变量配置

在训练脚本中，确保设置以下环境变量：

```bash
export CUDA_VISIBLE_DEVICES=0,1,3  # 根据实际情况调整
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800
```

## 🔍 快速检查清单

迁移到新机器后，使用以下清单验证：

- [ ] Conda 环境 `qwen3` 已创建（Python 3.12）
- [ ] PyTorch 可以检测到 GPU
- [ ] FAISS GPU 数量 > 0
- [ ] NumPy 版本 < 2.0
- [ ] 项目依赖已安装
- [ ] `.venv` 配置正确（如果使用）
- [ ] 训练脚本可以正常运行

## 📚 参考信息

### 当前环境配置（参考）

- **Conda 环境：** qwen3
- **Python 版本：** 3.12.0
- **CUDA 驱动版本：** 545.23.06
- **CUDA Toolkit：** 12.3
- **PyTorch CUDA：** 12.1（PyTorch 使用 12.1 表示支持 CUDA 12.x）
- **FAISS GPU：** 1.12.0（pytorch channel）
- **NumPy：** 1.26.4
- **GPU：** NVIDIA A800-SXM4-40GB × 8

### 关键路径

- Conda 环境路径：`~/miniconda3/envs/qwen3/`
- Site-packages：`~/miniconda3/envs/qwen3/lib/python3.12/site-packages/`
- 项目虚拟环境：`/path/to/ExplicitLM/.venv/`

## 💡 提示

1. **uv 安装网络问题**：
   - 如果无法访问 `astral.sh` 或 `github.com`，可以使用国内镜像：`curl -LsSf https://uv.ifcfg.cn/install.sh | sh`
   - 或者使用 `pip install uv`（如果 pip 源可用）
   - 如果都不行，可以手动下载二进制文件从 GitHub releases 页面
   - **推荐：** 优先尝试国内镜像，通常无需翻墙即可安装

2. **优先使用 conda 安装**：对于科学计算包（如 faiss、numpy），优先使用 conda 而不是 pip，可以自动处理依赖关系。

3. **CUDA 版本匹配**：确保 PyTorch、FAISS 和系统 CUDA 版本兼容。
   - CUDA 12.3 驱动可以使用 PyTorch 的 `pytorch-cuda=12.1`（表示支持 CUDA 12.x）
   - FAISS GPU 会自动匹配 PyTorch 的 CUDA 版本

4. **NumPy 版本**：FAISS GPU 目前只支持 NumPy 1.x，不要升级到 2.x。

5. **虚拟环境隔离**：`.venv` 用于项目特定的包，但 faiss-gpu 和 numpy 应该从 conda 环境共享。

---

**最后更新：** 2024-12-27  
**维护者：** 根据实际环境配置记录


