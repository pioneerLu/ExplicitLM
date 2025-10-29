#!/bin/bash
################################################################################
# ExplicitLM集群实验前置脚本（登陆节点执行）
# 用途：在登陆节点完成数据同步和Git记录
#
# 调用方式：由实验脚本source调用，需要预先定义相同的变量
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
if [ -z "$EXP_ID" ] || [ -z "$EXP_DESC" ] || [ -z "$TRAIN_ARGS" ]; then
    log_error "缺少必需变量！"
    echo "需要在调用脚本中定义："
    echo "  EXP_ID=\"实验ID\""
    echo "  EXP_DESC=\"实验描述\""
    echo "  DATASET_VERSION=\"训练数据集版本(可选)\""
    echo "  VAL_DATASET_VERSION=\"验证数据集版本(可选)\""
    echo "  TRAIN_ARGS=\"训练参数\""
    exit 1
fi

log_info "========================================="
log_info "【集群模式 - 前置阶段】"
log_info "实验ID: $EXP_ID"
log_info "实验描述: $EXP_DESC"
log_info "========================================="

# 路径定义
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${EXP_ID}"
RECORD_FILE="${PROJECT_ROOT}/experiments/records/${EXP_ID}.json"
META_FILE="${PROJECT_ROOT}/.experiment_meta_${EXP_ID}"
STATE_FILE="${PROJECT_ROOT}/.cluster_state_${EXP_ID}"

################################################################################
# 步骤1: 前置检查
################################################################################
log_info "步骤1/5: 前置检查..."

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    log_error "当前不在Git仓库中！"
    exit 1
fi

if [ ! -d "${PROJECT_ROOT}/.dvc" ]; then
    log_error "DVC未初始化！"
    exit 1
fi

if [ -f "$RECORD_FILE" ]; then
    log_error "实验ID ${EXP_ID} 已存在！"
    log_info "现有记录文件: $RECORD_FILE"
    read -p "是否覆盖？(y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log_info "取消实验"
        exit 0
    fi
fi

mkdir -p "${PROJECT_ROOT}/experiments/records"
mkdir -p "$CHECKPOINT_DIR"

log_success "前置检查通过"

################################################################################
# 步骤2: 记录代码版本
################################################################################
log_info "步骤2/5: 记录代码版本..."

CODE_COMMIT=$(git rev-parse HEAD)
log_success "代码版本: ${CODE_COMMIT:0:8}"

if ! git diff --quiet || ! git diff --cached --quiet; then
    log_warning "检测到未提交的变更"
    git status --short
fi

################################################################################
# 步骤3: 智能数据同步（仅同步变更的数据集）
################################################################################
log_info "步骤3/5: 智能数据同步..."

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# 智能同步函数：检查版本是否变更，仅同步变更的数据集
smart_sync_dataset() {
    local dataset_name=$1
    local target_version=$2
    local dvc_file="data/${dataset_name}.dvc"

    # 检查DVC文件是否存在
    if [ ! -f "$dvc_file" ]; then
        log_warning "  - ${dataset_name}: DVC文件不存在 ($dvc_file)，跳过同步"
        eval "${dataset_name^^}_COMMIT=\"N/A\""
        return 0
    fi

    # 如果目标版本为空，使用当前版本
    if [ -z "$target_version" ]; then
        target_version=$(git log -1 --format="%H" -- "$dvc_file" 2>/dev/null || echo "$CODE_COMMIT")
    fi

    # 获取当前本地数据对应的版本
    local current_version=$(git log -1 --format="%H" -- "$dvc_file" 2>/dev/null || echo "")

    # 比较版本
    if [ "$target_version" == "$current_version" ]; then
        log_info "  - ${dataset_name}: 版本未变更 (${target_version:0:8})，跳过同步"
        eval "${dataset_name^^}_COMMIT=\"$target_version\""
        return 0
    fi

    # 版本不同，需要同步
    log_warning "  - ${dataset_name}: 版本变更 ${current_version:0:8} → ${target_version:0:8}，开始同步..."

    # 首先暂存当前更改，防止冲突
    local stash_output=$(git stash push -m "Stash by _run_experiment_cluster_pre.sh for $dvc_file" -- "$dvc_file" 2>&1 || true)
    local stash_needed=$?
    
    # 切换到目标版本
    if ! git checkout "$target_version" --quiet; then
        log_error "    无法切换到版本 $target_version"
        # 恢复之前的更改
        if [ "$stash_needed" -eq 0 ] && [ -n "$stash_output" ] && echo "$stash_output" | grep -q "Saved"; then
            git stash pop --quiet 2>/dev/null || true
        fi
        eval "${dataset_name^^}_COMMIT=\"FAILED\""
        return 1
    fi

    # DVC checkout该数据集
    if dvc checkout "$dvc_file"; then
        log_success "    同步完成"
    else
        log_error "    同步失败，可能需要dvc pull"
        log_info "    尝试执行: dvc pull $dvc_file"
        if dvc pull "$dvc_file"; then
            log_success "    DVC pull完成"
        else
            log_error "    DVC pull失败，请检查网络和远程存储"
        fi
    fi

    # 记录版本
    eval "${dataset_name^^}_COMMIT=\"$target_version\""

    # 切回当前分支
    if ! git checkout "$CURRENT_BRANCH" --quiet; then
        log_error "    无法切回当前分支 $CURRENT_BRANCH"
        # 恢复之前的更改
        if [ "$stash_needed" -eq 0 ] && [ -n "$stash_output" ] && echo "$stash_output" | grep -q "Saved"; then
            git stash pop --quiet 2>/dev/null || true
        fi
        return 1
    fi

    # 恢复之前暂存的更改（如果有的话）
    if [ "$stash_needed" -eq 0 ] && [ -n "$stash_output" ] && echo "$stash_output" | grep -q "Saved"; then
        git stash pop --quiet 2>/dev/null || true
    fi
    
    return 0
}

# 同步必需数据集
if ! smart_sync_dataset "database" "$DATASET_VERSION"; then
    log_error "数据库同步失败，终止实验"
    exit 1
fi

if ! smart_sync_dataset "benchmarks" "$VAL_DATASET_VERSION"; then
    log_error "基准测试数据同步失败，终止实验"
    exit 1
fi

# 同步可选数据集
[ -n "$EMBEDDING_VERSION" ] && smart_sync_dataset "embeddings" "$EMBEDDING_VERSION"
[ -n "$DATABASE_VERSION" ] && smart_sync_dataset "database_init" "$DATABASE_VERSION"
[ -n "$CACHE_VERSION" ] && smart_sync_dataset "cache" "$CACHE_VERSION"

log_success "数据同步完成"

################################################################################
# 步骤4: 记录实验元数据
################################################################################
log_info "步骤4/5: 记录实验元数据..."

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$META_FILE" <<EOF
{
  "experiment": {
    "id": "$EXP_ID",
    "description": "$EXP_DESC",
    "timestamp": "$TIMESTAMP"
  },
  "versions": {
    "code_commit": "$CODE_COMMIT",
    "data": {
      "dataset_commit": "${DATABASE_COMMIT:-N/A}",
      "val_dataset_commit": "${BENCHMARKS_COMMIT:-N/A}",
      "embedding_commit": "${EMBEDDINGS_COMMIT:-N/A}",
      "database_init_commit": "${DATABASE_INIT_COMMIT:-N/A}",
      "cache_commit": "${CACHE_COMMIT:-N/A}"
    }
  },
  "command": "accelerate launch 1_pretrain.py --out_dir $CHECKPOINT_DIR $TRAIN_ARGS"
}
EOF

log_success "元数据已记录: $META_FILE"

################################################################################
# 步骤5: 保存状态供后续阶段使用
################################################################################
log_info "步骤5/5: 保存状态信息..."

cat > "$STATE_FILE" <<EOF
# 集群实验状态文件 - ${EXP_ID}
# 生成时间: $TIMESTAMP

# 实验配置
export EXP_ID="$EXP_ID"
export EXP_DESC="$EXP_DESC"
export TRAIN_ARGS="$TRAIN_ARGS"

# 版本信息
export CODE_COMMIT="$CODE_COMMIT"
export DATABASE_COMMIT="${DATABASE_COMMIT:-N/A}"
export BENCHMARKS_COMMIT="${BENCHMARKS_COMMIT:-N/A}"
export EMBEDDINGS_COMMIT="${EMBEDDINGS_COMMIT:-N/A}"
export DATABASE_INIT_COMMIT="${DATABASE_INIT_COMMIT:-N/A}"
export CACHE_COMMIT="${CACHE_COMMIT:-N/A}"

# 路径信息
export PROJECT_ROOT="$PROJECT_ROOT"
export CHECKPOINT_DIR="$CHECKPOINT_DIR"
export RECORD_FILE="$RECORD_FILE"
export META_FILE="$META_FILE"
export TIMESTAMP="$TIMESTAMP"
EOF

log_success "状态已保存: $STATE_FILE"

echo ""
log_success "========================================="
log_success "   前置阶段完成！"
log_success "========================================="
echo ""
log_info "📋 下一步操作："
log_info "1. 将代码和数据同步到计算节点（如需要）"
log_info "2. 在计算节点运行训练脚本："
log_info "   source ${PROJECT_ROOT}/experiments/scripts/${EXP_ID}_train.sh"
echo ""
log_info "📝 状态文件已保存，供训练和后续阶段使用"
