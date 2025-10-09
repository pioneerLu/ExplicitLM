#!/bin/bash
################################################################################
# ExplicitLM集群实验后续脚本（登陆节点执行）
# 用途：在登陆节点完成DVC追踪和Git提交
#
# 调用方式：由实验脚本source调用
# 执行环境：登陆节点（有网络，无GPU）
################################################################################

set -e
set -o pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 验证必需变量
if [ -z "$EXP_ID" ]; then
    log_error "缺少实验ID！"
    echo "需要先定义 EXP_ID 或加载状态文件"
    exit 1
fi

log_info "========================================="
log_info "【集群模式 - 后续阶段】"
log_info "实验ID: $EXP_ID"
log_info "========================================="

# 路径定义
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
STATE_FILE="${PROJECT_ROOT}/.cluster_state_${EXP_ID}"
SWANLAB_URL_FILE="${PROJECT_ROOT}/.swanlab_url_${EXP_ID}"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${EXP_ID}"
RECORD_FILE="${PROJECT_ROOT}/experiments/records/${EXP_ID}.json"
META_FILE="${PROJECT_ROOT}/.experiment_meta_${EXP_ID}"

################################################################################
# 步骤1: 加载状态
################################################################################
log_info "步骤1/5: 加载状态..."

if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    log_success "状态已加载"
else
    log_error "未找到状态文件: $STATE_FILE"
    log_info "请先运行前置脚本"
    exit 1
fi

################################################################################
# 步骤2: 读取SwanLab URL
################################################################################
log_info "步骤2/5: 读取SwanLab URL..."

if [ -f "$SWANLAB_URL_FILE" ]; then
    SWANLAB_URL=$(cat "$SWANLAB_URL_FILE")
    log_success "SwanLab URL: $SWANLAB_URL"
elif [ -f "${PROJECT_ROOT}/.swanlab_url" ]; then
    SWANLAB_URL=$(cat "${PROJECT_ROOT}/.swanlab_url")
    log_success "SwanLab URL: $SWANLAB_URL"
else
    SWANLAB_URL="N/A"
    log_warning "未找到SwanLab URL"
fi

################################################################################
# 步骤3: 追踪checkpoint到DVC
################################################################################
log_info "步骤3/5: 追踪checkpoint到DVC..."

if [ ! -d "$CHECKPOINT_DIR" ]; then
    log_error "Checkpoint目录不存在: $CHECKPOINT_DIR"
    log_info "请确保已将checkpoint从计算节点同步回来"
    exit 1
fi

log_info "Checkpoint文件:"
ls -lh "$CHECKPOINT_DIR"

# DVC追踪
dvc add "$CHECKPOINT_DIR"

CHECKPOINT_DVC="${CHECKPOINT_DIR}.dvc"

if [ -f "$CHECKPOINT_DVC" ]; then
    CHECKPOINT_HASH=$(grep "md5:" "$CHECKPOINT_DVC" | awk '{print $2}')
    log_success "DVC追踪完成 (Hash: ${CHECKPOINT_HASH:0:8})"
else
    log_error "DVC文件生成失败: $CHECKPOINT_DVC"
    exit 1
fi

################################################################################
# 步骤4: 生成实验记录文件
################################################################################
log_info "步骤4/5: 生成实验记录文件..."

# 提取训练参数为JSON
PARAMS_JSON=$(python3 -c "
import sys, json
args = '$TRAIN_ARGS'.split()
params = {}
i = 0
while i < len(args):
    if args[i].startswith('--'):
        key = args[i][2:]
        if i + 1 < len(args) and not args[i+1].startswith('--'):
            value = args[i+1]
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            params[key] = value
            i += 2
        else:
            params[key] = True
            i += 1
    else:
        i += 1
print(json.dumps(params, indent=2))
" 2>/dev/null || echo "{}")

# 获取环境信息
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',' || echo "N/A")
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")

# 生成记录文件
cat > "$RECORD_FILE" <<EOF
{
  "experiment": {
    "id": "$EXP_ID",
    "description": "$EXP_DESC",
    "timestamp": "$TIMESTAMP",
    "mode": "cluster",
    "script": "cluster_experiment.sh",
    "command": "accelerate launch 1_pretrain.py --out_dir $CHECKPOINT_DIR $TRAIN_ARGS"
  },
  "versions": {
    "code_commit": "$CODE_COMMIT",
    "code_commit_short": "${CODE_COMMIT:0:8}",
    "data": {
      "dataset_commit": "$DATABASE_COMMIT",
      "dataset_commit_short": "${DATABASE_COMMIT:0:8}",
      "val_dataset_commit": "$BENCHMARKS_COMMIT",
      "val_dataset_commit_short": "${BENCHMARKS_COMMIT:0:8}",
      "embedding_commit": "${EMBEDDINGS_COMMIT:-N/A}",
      "embedding_commit_short": "${EMBEDDINGS_COMMIT:0:8}",
      "database_init_commit": "${DATABASE_INIT_COMMIT:-N/A}",
      "database_init_commit_short": "${DATABASE_INIT_COMMIT:0:8}",
      "cache_commit": "${CACHE_COMMIT:-N/A}",
      "cache_commit_short": "${CACHE_COMMIT:0:8}"
    },
    "checkpoint_dvc": "$CHECKPOINT_DVC",
    "checkpoint_hash": "$CHECKPOINT_HASH",
    "checkpoint_hash_short": "${CHECKPOINT_HASH:0:8}"
  },
  "hyperparameters": $PARAMS_JSON,
  "results": {
    "swanlab_url": "$SWANLAB_URL",
    "checkpoint_dir": "$CHECKPOINT_DIR"
  },
  "environment": {
    "python_version": "$PYTHON_VERSION",
    "cuda_version": "$CUDA_VERSION",
    "num_gpus": $NUM_GPUS
  },
  "reproduction": {
    "code_checkout": "git checkout $CODE_COMMIT",
    "data_checkout_steps": [
      "git checkout $DATABASE_COMMIT && dvc checkout data/database.dvc && git checkout -",
      "git checkout $BENCHMARKS_COMMIT && dvc checkout data/benchmarks.dvc && git checkout -"
    ],
    "checkpoint_pull": "dvc pull ${CHECKPOINT_DVC}",
    "full_command": "# 1. 恢复代码版本\\ngit checkout $CODE_COMMIT\\n\\n# 2. 恢复各数据集版本\\ngit checkout $DATABASE_COMMIT && dvc checkout data/database.dvc && git checkout -\\ngit checkout $BENCHMARKS_COMMIT && dvc checkout data/benchmarks.dvc && git checkout -\\n\\n# 3. 运行训练\\naccelerate launch 1_pretrain.py --out_dir $CHECKPOINT_DIR $TRAIN_ARGS"
  }
}
EOF

log_success "实验记录已生成: $RECORD_FILE"

echo ""
log_info "========== 实验记录内容 =========="
cat "$RECORD_FILE" | python3 -m json.tool 2>/dev/null || cat "$RECORD_FILE"
log_info "=================================="
echo ""

################################################################################
# 步骤5: Git提交所有变更
################################################################################
log_info "步骤5/5: 提交到Git..."

echo ""
log_info "将要提交的变更："
git status --short
echo ""

git add -A
git commit -m "exp: ${EXP_ID} - ${EXP_DESC}"

log_success "所有变更已提交到Git"

################################################################################
# 清理临时文件
################################################################################
log_info "清理临时文件..."
rm -f "$SWANLAB_URL_FILE"
rm -f "${PROJECT_ROOT}/.swanlab_url"
rm -f "$META_FILE"
rm -f "$STATE_FILE"
log_success "清理完成"

echo ""
log_success "========================================="
log_success "   实验 ${EXP_ID} 全部完成！"
log_success "========================================="
echo ""
log_info "📋 记录文件: $RECORD_FILE"
log_info "🔬 SwanLab URL: $SWANLAB_URL"
log_info "💾 Checkpoint: $CHECKPOINT_DIR"
log_info "🏷️  代码版本: ${CODE_COMMIT:0:8}"
log_info "📊 数据版本: ${DATABASE_COMMIT:0:8}"
log_info "🔐 权重哈希: ${CHECKPOINT_HASH:0:8}"
echo ""
log_info "复现命令详见: $RECORD_FILE (reproduction字段)"
echo ""
log_success "========================================="
