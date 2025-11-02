#!/bin/bash
################################################################################
# ExplicitLM集群实验训练脚本（计算节点执行）
# 用途：在计算节点执行模型训练，是集群实验的核心执行阶段
#
# 调用方式：由实验脚本source调用，需要先加载前置阶段的状态文件
# 执行环境：计算节点（通常无网络连接，有GPU计算资源）
################################################################################

set -e
set -o pipefail

# 颜色定义 - 用于终端输出着色，提升日志可读性
RED='\033[0;31m'      # 红色 - 错误信息
GREEN='\033[0;32m'    # 绿色 - 成功信息
YELLOW='\033[1;33m'   # 黄色 - 警告信息
BLUE='\033[0;34m'     # 蓝色 - 普通信息
NC='\033[0m'          # 重置颜色

# 日志函数 - 格式化输出不同类型的信息
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

# 验证必需变量 - 确保实验标识符已正确定义
if [ -z "$EXP_ID" ]; then
    log_error "缺少实验ID！"
    echo "需要先定义 EXP_ID 或加载前置阶段生成的状态文件"
    exit 1
fi

# 显示训练阶段启动信息
log_info "========================================="
log_info "【集群模式 - 训练阶段】"
log_info "实验ID: $EXP_ID"
log_info "========================================="

# 路径定义 - 设置关键文件和目录路径
# 集群环境中Git可能不可用，提供备用路径获取方式
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    # Git可用时，使用Git获取项目根目录
    PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
else
    # Git不可用时，通过脚本路径推断项目根目录
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    log_info "Git不可用，使用推断路径: $PROJECT_ROOT"
fi

# 状态文件和SwanLab URL文件路径
STATE_FILE="${PROJECT_ROOT}/.cluster_state_${EXP_ID}"
SWANLAB_URL_FILE="${PROJECT_ROOT}/.swanlab_url_${EXP_ID}"

################################################################################
# 步骤1: 加载前置阶段状态文件 - 恢复实验配置和参数
################################################################################
log_info "步骤1/4: 加载状态文件..."

# 检查并加载前置阶段生成的状态文件
if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    log_success "状态已加载: $STATE_FILE"
    log_info "  - 实验ID: $EXP_ID"
    log_info "  - 代码版本: ${CODE_COMMIT:0:8}"
    log_info "  - 训练参数: $TRAIN_ARGS"
else
    # 状态文件不存在时的处理逻辑
    log_warning "未找到状态文件: $STATE_FILE"
    log_info "使用当前环境变量（确保已正确设置）"

    # 验证关键训练参数是否存在
    if [ -z "$TRAIN_ARGS" ]; then
        log_error "缺少训练参数 TRAIN_ARGS"
        exit 1
    fi
fi

# CHECKPOINT_DIR已在状态文件中定义，此处无需重复设置
# 保留注释说明：确保输出目录路径的一致性

################################################################################
# 步骤2: 执行模型训练 - 核心训练逻辑
################################################################################
log_info "步骤2/4: 开始模型训练..."

# 构建最终训练命令 - 基于前置阶段预配置的命令
if [ -n "$FINAL_COMMAND" ]; then
    # 检查是否使用SwanLab实验跟踪
    if [[ "$TRAIN_ARGS" == *"--use_swanlab"* ]] || [[ "$TRAIN_ARGS" == *"use_swanlab=True"* ]]; then
        # 集群环境通常无网络，禁用SwanLab在线模式
        TRAIN_CMD="$FINAL_COMMAND logging.swanlab_online=False"
        log_info "检测到SwanLab参数，已禁用在线模式（集群环境）"
    else
        TRAIN_CMD="$FINAL_COMMAND"
    fi
    log_info "使用预构建的训练命令: $TRAIN_CMD"
else
    log_error "未找到预构建的训练命令"
    exit 1
fi

# 显示即将执行的完整命令
log_info "执行命令: $TRAIN_CMD"
echo ""

# 执行训练过程 - 增强错误处理和状态监控
if eval $TRAIN_CMD; then
    log_success "模型训练完成"
else
    log_error "训练失败！请检查错误日志"
    exit 1
fi

log_success "训练阶段执行完成"


################################################################################
# 步骤3: 验证训练输出 - 检查模型checkpoint和结果文件
################################################################################
log_info "步骤3/4: 验证训练输出..."

# 检查checkpoint输出目录是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    log_error "Checkpoint目录不存在: $CHECKPOINT_DIR"
    log_error "训练可能未正确生成输出文件"
    exit 1
fi

# 列出生成的checkpoint文件，验证训练结果
log_info "生成的checkpoint文件:"
ls -lh "$CHECKPOINT_DIR" || {
    log_warning "无法列出目录内容或目录为空"
}

# 验证关键文件是否存在（可根据需要添加具体文件检查）
if [ -z "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
    log_warning "Checkpoint目录为空，请确认训练是否正常完成"
else
    log_success "Checkpoint文件生成完成"
fi

################################################################################
# 步骤4: 训练阶段总结和后续指导
################################################################################
echo ""
log_success "========================================="
log_success "   集群训练阶段完成！"
log_success "========================================="
echo ""

# 提供详细的下一步操作指导
log_info "📋 下一步操作指导："
log_info "1. 将checkpoint数据和训练日志同步回登陆节点（取决于集群环境配置）"
log_info "2. 在登陆节点运行实验后处理脚本："
log_info "   source ${PROJECT_ROOT}/experiments/scripts/${EXP_ID}_post.sh"
log_info "   或使用: ./experiments/scripts/${EXP_ID}_post.sh post"
echo ""

# 重要提醒和状态确认
log_info "📝 重要提醒："
log_info "  - 训练结果已保存至: $CHECKPOINT_DIR"
log_info "  - 请确保数据完整性后再进行后续处理"
log_info "  - 后续脚本将完成DVC跟踪和Git提交"
