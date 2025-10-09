# 实验记录文件说明

## 概述
本目录包含所有实验的元数据记录文件（JSON格式），每个文件对应一次完整实验。

## 文件命名规则
- 格式：`<实验ID>.json`
- 示例：`exp_001.json`, `exp_002.json`

## 记录文件结构

### 1. experiment - 实验基本信息
- `id`: 实验唯一标识符
- `description`: 实验描述（中文）
- `timestamp`: 实验开始时间（UTC）
- `script`: 执行脚本名称
- `command`: 完整训练命令

### 2. versions - 版本信息
- `code_commit`: 代码Git commit完整哈希
- `code_commit_short`: 代码commit短哈希（8位）
- `data.dataset_commit`: 训练数据集对应的Git commit哈希
- `data.val_dataset_commit`: 验证数据集对应的Git commit哈希
- `data.embedding_commit`: 预训练嵌入版本（可选）
- `data.database_init_commit`: 知识库初始化版本（可选）
- `data.cache_commit`: 缓存数据版本（可选）
- `checkpoint_dvc`: DVC元文件路径
- `checkpoint_hash`: DVC文件MD5哈希（权重版本标识）
- `checkpoint_hash_short`: 哈希短版本

### 3. hyperparameters - 超参数
记录所有命令行传入的超参数（自动提取）

### 4. results - 实验结果
- `swanlab_url`: SwanLab实验追踪页面URL
- `checkpoint_dir`: 模型权重保存目录

### 5. environment - 运行环境
- `python_version`: Python版本
- `cuda_version`: CUDA版本
- `num_gpus`: GPU数量

### 6. reproduction - 复现指令
- `code_checkout`: Git代码恢复命令
- `data_checkout_steps`: DVC数据恢复命令（细粒度，针对各数据集）
- `checkpoint_pull`: DVC权重拉取命令
- `full_command`: 完整复现命令（一键复制）

## 如何使用记录文件复现实验

### 完全复现（包含权重）
```bash
# 1. 查看记录文件
cat experiments/records/exp_001.json

# 2. 恢复代码版本
git checkout <code_commit>

# 3. 恢复各数据集版本（细粒度）
# 从记录文件的data_checkout_steps中复制执行
git checkout <dataset_commit> && dvc checkout data/database.dvc && git checkout -
git checkout <val_dataset_commit> && dvc checkout data/benchmarks.dvc && git checkout -

# 4. 拉取模型权重
dvc pull checkpoints/exp_001.dvc

# 5. （可选）重新训练
accelerate launch 1_pretrain.py <参数从记录文件复制>
```

### 仅查看实验配置
```bash
# 使用jq工具美化查看
cat experiments/records/exp_001.json | jq '.hyperparameters'

# 查看实验对比
for f in experiments/records/*.json; do
    echo "$(basename $f): $(jq -r '.hyperparameters.knowledge_num' $f)"
done
```

## 记录文件管理

### 命名规范
- **基线实验**: `exp_baseline.json`
- **消融实验**: `exp_ablation_<feature>.json`
- **对比实验**: `exp_compare_<variant>.json`
- **编号实验**: `exp_001.json`, `exp_002.json`

### 版本控制
- ✅ 记录文件必须提交到Git
- ✅ 文件名不可修改（与实验ID绑定）
- ❌ 不要手动编辑记录文件

### 查询技巧
```bash
# 查找所有使用knowledge_num=2M的实验
grep -l '"knowledge_num": 2097152' experiments/records/*.json

# 按时间排序实验
ls -lt experiments/records/*.json

# 查找失败实验（SwanLab URL为N/A）
grep -l '"swanlab_url": "N/A"' experiments/records/*.json

# 查看特定参数的使用情况
jq '.hyperparameters.epochs' experiments/records/*.json
```

## 数据版本管理说明

### 细粒度数据版本控制
本项目采用细粒度数据版本管理，每个数据集可以独立指定版本：

**必需数据集**：
- `dataset_commit`: 训练数据集版本（data/database/）
- `val_dataset_commit`: 验证数据集版本（data/benchmarks/）

**可选数据集**：
- `embedding_commit`: 预训练嵌入权重版本（如果使用）
- `database_init_commit`: 知识库初始化数据版本（如果使用）
- `cache_commit`: 缓存数据版本（如果使用）

### 版本指定方式
```bash
# 方式1：使用当前数据版本
DATASET_VERSION=""          # 留空表示使用当前版本
VAL_DATASET_VERSION=""

# 方式2：指定历史版本
DATASET_VERSION="abc123"    # 使用特定commit的数据
VAL_DATASET_VERSION="def456"

# 方式3：混合策略
DATASET_VERSION="abc123"    # 训练数据使用历史版本
VAL_DATASET_VERSION=""      # 验证数据使用当前版本
```

## 实验脚本示例

### 创建新实验
```bash
# 创建实验脚本
cat > experiments/scripts/exp_001.sh <<'EOF'
#!/bin/bash
EXP_ID="exp_001"
EXP_DESC="基线实验 knowledge_num=1M epochs=10"
DATASET_VERSION=""          # 使用当前训练数据版本
VAL_DATASET_VERSION=""      # 使用当前验证数据版本
EMBEDDING_VERSION=""        # 可选
DATABASE_VERSION=""         # 可选
CACHE_VERSION=""            # 可选
TRAIN_ARGS="--epochs 10 --knowledge_num 1048576 --dim 512 --n_layers 8"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_run_experiment_core.sh"
EOF

# 赋予执行权限
chmod +x experiments/scripts/exp_001.sh

# 运行实验
./experiments/scripts/exp_001.sh
```

### 使用特定数据版本
```bash
# 从已有实验获取数据版本
PREV_DATASET_V=$(jq -r '.versions.data.dataset_commit' experiments/records/exp_001.json)
PREV_VAL_V=$(jq -r '.versions.data.val_dataset_commit' experiments/records/exp_001.json)

# 创建新实验（使用相同数据）
cat > experiments/scripts/exp_002.sh <<EOF
#!/bin/bash
EXP_ID="exp_002"
EXP_DESC="扩大Memory Bank到2M条目"
DATASET_VERSION="$PREV_DATASET_V"
VAL_DATASET_VERSION="$PREV_VAL_V"
EMBEDDING_VERSION=""
DATABASE_VERSION=""
CACHE_VERSION=""
TRAIN_ARGS="--epochs 10 --knowledge_num 2097152 --knowledge_dim 256"
SCRIPT_DIR="\$(cd \"\$(dirname \"\${BASH_SOURCE[0]}\")\" && pwd)"
source "\${SCRIPT_DIR}/_run_experiment_core.sh"
EOF
```

## 故障排查

### 常见问题

**Q: 实验记录文件生成失败**
- 检查是否在Git仓库中：`git status`
- 检查DVC是否初始化：`ls -la .dvc/`
- 检查必需变量是否定义：`EXP_ID`, `EXP_DESC`, `TRAIN_ARGS`

**Q: 数据版本切换失败**
- 验证commit hash是否有效：`git log --oneline`
- 检查DVC文件是否存在：`ls data/*.dvc`
- 确认DVC远程存储连接正常：`dvc remote list`

**Q: SwanLab URL为N/A**
- 确认训练时使用了`--use_swanlab`参数
- 检查1_pretrain.py是否已添加URL导出功能
- 查看`.swanlab_url`临时文件是否生成

**Q: 复现实验时数据不一致**
- 严格按照`data_checkout_steps`顺序执行
- 确保每个数据集都切换到正确的commit
- 使用`dvc checkout`而非`git checkout`恢复数据文件

## 最佳实践

### 实验设计
1. **明确实验目标**：每个实验应有清晰的研究问题
2. **单变量原则**：每次只改变一个关键参数
3. **基线对比**：始终与基线实验进行对比
4. **数据版本锁定**：对比实验使用相同数据版本

### 记录管理
1. **及时记录**：实验完成后立即检查记录文件
2. **定期备份**：定期push到远程Git仓库
3. **文档补充**：在实验脚本中添加详细注释
4. **结果归档**：重要实验的SwanLab链接单独记录

### 协作规范
1. **命名约定**：团队统一实验ID命名规则
2. **代码同步**：实验前先pull最新代码
3. **数据共享**：通过DVC远程存储共享数据
4. **结果讨论**：定期review实验记录和结果
