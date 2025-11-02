#!/bin/bash
################################################################################
# ExplicitLM集群实验训练脚本（计算节点执行）
# 用途：在计算节点执行训练
#
# 调用方式：由实验脚本source调用
# 执行环境：计算节点（无网络，有GPU）
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
log_info "【集群模式 - 训练阶段】"
log_info "实验ID: $EXP_ID"
log_info "========================================="

# 路径定义
# In cluster environment, git might not be available, so we use a fallback
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    log_info "Git not available, using PROJECT_ROOT: $PROJECT_ROOT"
fi
STATE_FILE="${PROJECT_ROOT}/.cluster_state_${EXP_ID}"
SWANLAB_URL_FILE="${PROJECT_ROOT}/.swanlab_url_${EXP_ID}"

################################################################################
# 加载状态文件
################################################################################
log_info "加载状态文件..."

if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    log_success "状态已加载: $STATE_FILE"
    log_info "  - 实验ID: $EXP_ID"
    log_info "  - 代码版本: ${CODE_COMMIT:0:8}"
    log_info "  - 训练参数: $TRAIN_ARGS"
else
    log_warning "未找到状态文件: $STATE_FILE"
    log_info "使用当前环境变量（确保已正确设置）"

    if [ -z "$TRAIN_ARGS" ]; then
        log_error "缺少训练参数 TRAIN_ARGS"
        exit 1
    fi
fi

# 设置checkpoint目录路径 (确保无论是否加载状态文件都有此变量)
# CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints/${EXP_ID}"

################################################################################
# 运行训练
################################################################################
log_info "开始训练..."

# 清理旧的SwanLab URL文件（将在post阶段重新生成）
rm -f "$SWANLAB_URL_FILE"
rm -f "${PROJECT_ROOT}/.swanlab_url"

# 使用pre脚本中生成的FINAL_COMMAND
if [ -n "$FINAL_COMMAND" ]; then
    if [[ "$TRAIN_ARGS" == *"--use_swanlab"* ]] || [[ "$TRAIN_ARGS" == *"use_swanlab=True"* ]]; then
        TRAIN_CMD="$FINAL_COMMAND logging.swanlab_online=False"
    else 
        TRAIN_CMD="$FINAL_COMMAND"
    fi
    log_info "使用预构建的训练命令: $TRAIN_CMD"
fi

log_info "执行命令: $TRAIN_CMD"
echo ""

# 运行训练 with better error handling
if eval $TRAIN_CMD; then
    log_success "训练完成"
else
    log_error "训练失败！"
    exit 1
fi

log_success "训练完成"


################################################################################
# 检查checkpoint生成
################################################################################
log_info "检查checkpoint生成..."

if [ ! -d "$CHECKPOINT_DIR" ]; then
    log_error "Checkpoint目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

log_info "生成的checkpoint文件:"
ls -lh "$CHECKPOINT_DIR"
log_success "Checkpoint生成完成"

echo ""
log_success "========================================="
log_success "   训练阶段完成！"
log_success "========================================="
echo ""
log_info "📋 下一步操作："
log_info "1. 将checkpoint同步回登陆节点（如需要）"
log_info "2. 在登陆节点运行后续脚本："
log_info "   source ${PROJECT_ROOT}/experiments/scripts/${EXP_ID}_post.sh"
echo ""
log_info "📝 训练结果已保存，等待后续处理"
