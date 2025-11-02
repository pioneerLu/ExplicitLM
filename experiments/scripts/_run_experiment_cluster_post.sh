#!/bin/bash
################################################################################
# ExplicitLM集群实验后处理脚本（登陆节点执行）
# 用途：在登陆节点完成DVC追踪、SwanLab同步和Git提交，是集群实验的收尾阶段
#
# 调用方式：由实验脚本source调用，需要先加载前置和训练阶段的状态文件
# 执行环境：登陆节点（有网络连接，无GPU计算资源）
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

# 显示后处理阶段启动信息
log_info "========================================="
log_info "【集群模式 - 后处理阶段】"
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

# 关键文件路径定义
STATE_FILE="${PROJECT_ROOT}/.cluster_state_${EXP_ID}"
META_FILE="${PROJECT_ROOT}/.experiment_meta_${EXP_ID}"
# SWANLAB_URL_FILE和RECORD_FILE将在加载状态文件后根据状态信息设置

################################################################################
# 步骤1: 加载集群实验状态 - 恢复前置和训练阶段的配置信息
################################################################################
log_info "步骤1/7: 加载集群实验状态..."

# 检查并加载训练阶段生成的状态文件
if [ -f "$STATE_FILE" ]; then
    source "$STATE_FILE"
    log_success "集群状态已加载"

    # 设置SwanLab URL文件路径（位于checkpoint目录下）
    SWANLAB_URL_FILE="${CHECKPOINT_DIR}/.swanlab_url"
    log_info "SwanLab URL文件路径: $SWANLAB_URL_FILE"

    # 设置实验记录文件路径，使用状态文件中的时间戳确保文件名唯一性
    if [ -n "$TIMESTAMP_FILENAME" ]; then
        RECORD_FILE="${PROJECT_ROOT}/experiments/records/${EXP_ID}_${TIMESTAMP_FILENAME}.json"
    else
        # 备用方案：如果状态中没有时间戳，使用当前时间生成
        TIMESTAMP_FILENAME=$(date +"%Y%m%d_%H%M%S")
        RECORD_FILE="${PROJECT_ROOT}/experiments/records/${EXP_ID}_${TIMESTAMP_FILENAME}.json"
        log_warning "状态文件中无时间戳，使用当前时间生成"
    fi
    log_info "实验记录文件路径: $RECORD_FILE"
else
    log_error "未找到集群状态文件: $STATE_FILE"
    log_info "请确保已按顺序完成前置和训练阶段"
    exit 1
fi

################################################################################
# 步骤2: 初始化SwanLab URL - 预设默认值，实际URL将在同步步骤中获取
################################################################################
log_info "步骤2/7: 初始化SwanLab URL..."

# SwanLab URL将在后续的sync_swanlab_data函数中生成和更新
# 此处先设置为N/A，避免在同步步骤之前就读取空值
SWANLAB_URL="N/A"
log_info "SwanLab URL将在后续数据同步步骤中获取"

################################################################################
# 步骤3: 验证checkpoint目录 - 确认训练输出已从计算节点同步回来
################################################################################
log_info "步骤3/7: 验证checkpoint目录..."

# 检查checkpoint输出目录是否存在
if [ ! -d "$CHECKPOINT_DIR" ]; then
    log_error "Checkpoint目录不存在: $CHECKPOINT_DIR"
    log_info "请确保已将checkpoint数据从计算节点同步回登陆节点"
    exit 1
fi

# 列出checkpoint文件内容，验证数据完整性
log_info "Checkpoint文件列表:"
ls -lh "$CHECKPOINT_DIR" || {
    log_warning "无法列出目录内容或目录为空"
}

# DVC管理策略说明：跳过checkpoint的单独DVC追踪
# 因为整个outputs目录已被DVC统一管理，避免重复追踪
log_info "跳过checkpoint的单独DVC追踪（整个outputs目录已被DVC统一管理）"
CHECKPOINT_DVC=""
CHECKPOINT_HASH="outputs_managed"

################################################################################
# 步骤4.5: 查找Hydra输出目录 - 定位训练生成的Hydra配置和日志文件
################################################################################
find_hydra_output_dir() {
    log_info "步骤4.5/7: 查找Hydra输出目录..."

    # 查找包含.hydra子目录的最新输出目录
    # 在outputs目录中搜索包含.hydra子目录的文件夹，按时间倒序排列
    local hydra_dirs=$(find "${PROJECT_ROOT}/outputs" -name ".hydra" -type d -printf "%h\n" 2>/dev/null | sort -r | head -n 1)

    if [ -n "$hydra_dirs" ] && [ -d "$hydra_dirs" ]; then
        HYDRA_OUTPUT_DIR="$hydra_dirs"
        log_success "找到Hydra输出目录: $HYDRA_OUTPUT_DIR"

        # 将实验记录文件复制到Hydra输出目录，便于集中管理
        if [ -f "$RECORD_FILE" ]; then
            cp "$RECORD_FILE" "$HYDRA_OUTPUT_DIR/experiment_record_${EXP_ID}.json"
            log_info "实验记录已复制到Hydra目录: $HYDRA_OUTPUT_DIR/experiment_record_${EXP_ID}.json"
        fi

        # 对Hydra输出目录进行DVC追踪
        track_hydra_output
    else
        log_warning "未找到Hydra输出目录，可能训练未使用Hydra配置"
        HYDRA_OUTPUT_DIR=""
    fi
}

################################################################################
# 步骤4.6: 追踪Hydra输出目录到DVC - 对Hydra输出进行版本控制
################################################################################
track_hydra_output() {
    if [ -n "$HYDRA_OUTPUT_DIR" ] && [ -d "$HYDRA_OUTPUT_DIR" ]; then
        log_info "步骤4.6/7: 追踪Hydra输出目录到DVC..."
        log_info "Hydra输出目录内容:"
        ls -la "$HYDRA_OUTPUT_DIR"

        # 为Hydra输出目录创建DVC追踪，实现版本控制
        log_info "开始对Hydra输出目录进行DVC追踪: $HYDRA_OUTPUT_DIR"
        if dvc add "$HYDRA_OUTPUT_DIR" 2>/dev/null; then
            # 成功创建DVC追踪
            HYDRA_OUTPUT_DVC="${HYDRA_OUTPUT_DIR}.dvc"
            if [ -f "$HYDRA_OUTPUT_DVC" ]; then
                # 提取DVC文件中的MD5哈希值用于版本标识
                HYDRA_OUTPUT_HASH=$(grep "md5:" "$HYDRA_OUTPUT_DVC" | awk '{print $3}')
                log_success "Hydra输出DVC追踪完成 (Hash: ${HYDRA_OUTPUT_HASH:0:8})"
                log_info "生成的DVC元文件: $HYDRA_OUTPUT_DVC"
            else
                log_warning "DVC追踪完成但未找到元文件: $HYDRA_OUTPUT_DVC"
            fi
        else
            # DVC追踪失败的备用处理
            log_warning "DVC追踪失败，可能目录已被管理或无内容变更: $HYDRA_OUTPUT_DIR"
            # 检查是否已经存在DVC追踪文件
            HYDRA_OUTPUT_DVC="${HYDRA_OUTPUT_DIR}.dvc"
            if [ -f "$HYDRA_OUTPUT_DVC" ]; then
                HYDRA_OUTPUT_HASH=$(grep "md5:" "$HYDRA_OUTPUT_DVC" | awk '{print $3}')
                log_info "使用现有DVC追踪 (Hash: ${HYDRA_OUTPUT_HASH:0:8})"
            fi
        fi
    else
        log_info "Hydra输出目录不存在，跳过DVC追踪"
    fi
}

################################################################################
# 步骤4: 查找并追踪Hydra输出目录 - 定位和版本控制训练配置文件
################################################################################
# 查找并追踪Hydra输出目录
find_hydra_output_dir

################################################################################
# 步骤5: 生成实验记录文件 - 创建完整的实验元数据记录
################################################################################
log_info "步骤5/7: 生成实验记录文件..."

# 提取训练参数为JSON格式，增强错误处理能力
if command -v python3 >/dev/null 2>&1; then
    # 使用Python脚本解析命令行参数并转换为结构化JSON
    PARAMS_JSON=$(python3 -c "
import sys, json, shlex
try:
    # 安全分割命令行参数
    args = shlex.split('$TRAIN_ARGS')
    params = {}
    i = 0
    # 遍历参数并构建键值对字典
    while i < len(args):
        if args[i].startswith('--'):
            # 处理长选项参数
            key = args[i][2:]
            # 检查是否存在对应的值参数
            if i + 1 < len(args) and not args[i+1].startswith('--') and not args[i+1].startswith('-'):
                value = args[i+1]
                # 尝试将值转换为适当的数据类型
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # 检查布尔值
                    if value.lower() in ('true', 'yes', 'on'):
                        value = True
                    elif value.lower() in ('false', 'no', 'off'):
                        value = False
                    # 其他情况保持为字符串
                params[key] = value
                i += 2
            else:
                # 无值参数视为布尔标志
                params[key] = True
                i += 1
        else:
            i += 1
    # 输出格式化的JSON
    print(json.dumps(params, indent=2))
except Exception:
    # 出现异常时返回空对象
    print('{}')
" 2>/dev/null || echo "{}")
else
    # Python3不可用时的备用方案
    PARAMS_JSON="{}"
    log_warning "Python3不可用，超参数将为空"
fi

# 获取环境信息 - 收集Python、CUDA和GPU配置信息
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>/dev/null | awk '{print $2}' || echo "unknown")
else
    PYTHON_VERSION="N/A"
    log_warning "Python3不可用"
fi

if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',' || echo "unknown")
else
    CUDA_VERSION="N/A"
    log_info "nvcc不可用"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
else
    NUM_GPUS=0
    log_info "nvidia-smi不可用"
fi

# 生成实验记录文件 - 创建包含完整实验信息的JSON文件
cat > "$RECORD_FILE" <<EOF
{
  "experiment": {
    "id": "$EXP_ID",
    "description": "$EXP_DESC",
    "timestamp": "$TIMESTAMP",
    "record_filename": "$(basename "$RECORD_FILE")",
    "mode": "cluster",
    "script": "cluster_experiment.sh",
    "command": "accelerate launch 1_pretrain.py --out_dir $CHECKPOINT_DIR $TRAIN_ARGS"
  },
  "versions": {
    "code_commit": "$CODE_COMMIT",
    "code_commit_short": "${CODE_COMMIT:0:8}",
    "data": {
      "dataset_commit": "${DATABASE_COMMIT:-N/A}",
      "dataset_commit_short": "${DATABASE_COMMIT:0:8}",
      "val_dataset_commit": "${BENCHMARKS_COMMIT:-N/A}",
      "val_dataset_commit_short": "${BENCHMARKS_COMMIT:0:8}",
      "embedding_commit": "${EMBEDDINGS_COMMIT:-N/A}",
      "embedding_commit_short": "${EMBEDDINGS_COMMIT:0:8}",
      "database_init_commit": "${DATABASE_INIT_COMMIT:-N/A}",
      "database_init_commit_short": "${DATABASE_INIT_COMMIT:0:8}",
      "cache_commit": "${CACHE_COMMIT:-N/A}",
      "cache_commit_short": "${CACHE_COMMIT:0:8}"
    },
    "checkpoint_info": {
      "checkpoint_dir": "$CHECKPOINT_DIR",
      "dvc_managed": "true",
      "note": "整个outputs目录已被DVC管理，跳过单独追踪"
    },
    "hydra_output": {
      "hydra_output_dir": "${HYDRA_OUTPUT_DIR:-N/A}",
      "hydra_output_dvc": "${HYDRA_OUTPUT_DVC:-N/A}",
      "hydra_output_hash": "${HYDRA_OUTPUT_HASH:-N/A}",
      "hydra_output_hash_short": "${HYDRA_OUTPUT_HASH:0:8}"
    }
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
    "checkpoint_pull": "dvc pull outputs (整个outputs目录包含checkpoint)",
    "hydra_output_pull": "${HYDRA_OUTPUT_DVC:+dvc pull ${HYDRA_OUTPUT_DVC}}",
    "full_command": "# 1. 恢复代码版本\\ngit checkout $CODE_COMMIT\\n\\n# 2. 恢复各数据集版本\\ngit checkout $DATABASE_COMMIT && dvc checkout data/database.dvc && git checkout -\\ngit checkout $BENCHMARKS_COMMIT && dvc checkout data/benchmarks.dvc && git checkout -\\n\\n# 3. 拉取输出目录\\ndvc pull outputs\\n\\n# 4. 运行训练\\naccelerate launch 1_pretrain.py --out_dir $CHECKPOINT_DIR $TRAIN_ARGS"
  }
}
EOF

log_success "实验记录已生成: $RECORD_FILE"

# 在CHECKPOINT_DIR下也生成一份副本，便于本地查看和备份
CHECKPOINT_RECORD_FILE="${CHECKPOINT_DIR}/experiment_record_${EXP_ID}_${TIMESTAMP_FILENAME}.json"
if [ -d "$CHECKPOINT_DIR" ]; then
    cp "$RECORD_FILE" "$CHECKPOINT_RECORD_FILE"
    log_success "实验记录副本已生成: $CHECKPOINT_RECORD_FILE"
else
    log_warning "CHECKPOINT_DIR目录不存在，无法生成副本: $CHECKPOINT_DIR"
fi

echo ""
log_info "========== 实验记录内容 =========="
if command -v python3 >/dev/null 2>&1; then
    cat "$RECORD_FILE" | python3 -m json.tool 2>/dev/null || cat "$RECORD_FILE"
else
    cat "$RECORD_FILE"
    log_warning "Python3不可用，无法格式化JSON输出"
fi
log_info "=================================="
echo ""

################################################################################
# 步骤6: 同步SwanLab实验数据 - 将本地实验结果上传到SwanLab平台
################################################################################
sync_swanlab_data() {
    log_info "步骤6/7: 检查并同步SwanLab实验数据..."

    # 检查是否在训练参数中启用了SwanLab实验跟踪功能
    if [[ "$TRAIN_ARGS" == *"--use_swanlab"* ]] || [[ "$TRAIN_ARGS" == *"use_swanlab=True"* ]]; then
        # 检查系统中是否安装了swanlab命令行工具
        if command -v swanlab &> /dev/null; then
            # 查找SwanLab数据目录（通常在Hydra输出目录中的swanlog子目录）
            if [ -n "$HYDRA_OUTPUT_DIR" ] && [ -d "$HYDRA_OUTPUT_DIR/swanlog" ]; then
                log_info "找到SwanLab日志目录: $HYDRA_OUTPUT_DIR/swanlog"

                # 查找具体的run-*目录（SwanLab为每次运行创建的独立子目录）
                local run_dir=$(find "$HYDRA_OUTPUT_DIR/swanlog" -name "run-*" -type d | head -n 1)

                if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
                    log_info "找到SwanLab运行目录: $run_dir"

                    # 尝试执行SwanLab数据同步命令并捕获完整输出
                    log_info "执行SwanLab数据同步命令..."
                    local sync_output
                    if sync_output=$(swanlab sync "$run_dir" 2>&1); then
                        log_success "SwanLab数据同步成功"

                        # 从同步命令的输出中提取实验查看URL
                        local extracted_url
                        extracted_url=$(echo "$sync_output" | grep -oE 'https?://[^[:space:]]+' | head -n 1 || echo "")

                        if [ -n "$extracted_url" ]; then
                            # 成功提取到URL，保存到全局变量和文件中
                            SWANLAB_URL="$extracted_url"
                            # 将URL保存到CHECKPOINT_DIR下的URL文件，便于后续查看
                            echo "$SWANLAB_URL" > "$SWANLAB_URL_FILE"
                            log_success "SwanLab实验URL已保存到: $SWANLAB_URL_FILE"
                            log_info "在线查看URL: $SWANLAB_URL"
                        else
                            # 未能提取到URL，设置为默认值
                            log_warning "未能从同步输出中提取实验URL"
                            SWANLAB_URL="N/A"
                            echo "$SWANLAB_URL" > "$SWANLAB_URL_FILE"
                        fi

                        # 显示同步输出的详细信息（前5行）
                        log_info "同步输出详情:"
                        echo "$sync_output" | head -5
                    else
                        # SwanLab同步命令执行失败
                        log_warning "SwanLab同步失败，可能没有网络连接或实验数据为空"
                        SWANLAB_URL="N/A"
                        echo "$SWANLAB_URL" > "$SWANLAB_URL_FILE"
                        # 显示错误输出信息（前3行）
                        log_info "错误输出详情:"
                        echo "$sync_output" | head -3
                    fi
                else
                    # 未找到SwanLab运行目录
                    log_warning "未找到SwanLab运行目录 (run-*)，跳过数据同步"
                    SWANLAB_URL="N/A"
                    echo "$SWANLAB_URL" > "$SWANLAB_URL_FILE"
                fi
            else
                # 未找到SwanLab日志目录
                log_info "未找到SwanLab日志目录，跳过数据同步"
                SWANLAB_URL="N/A"
                echo "$SWANLAB_URL" > "$SWANLAB_URL_FILE"
            fi
        else
            # 系统中未安装swanlab命令
            log_info "SwanLab命令行工具未安装，跳过数据同步"
            SWANLAB_URL="N/A"
            echo "$SWANLAB_URL" > "$SWANLAB_URL_FILE"
        fi
    else
        # 训练参数中未启用SwanLab
        log_info "训练参数中未启用SwanLab，跳过数据同步"
        SWANLAB_URL="N/A"
        echo "$SWANLAB_URL" > "$SWANLAB_URL_FILE"
    fi
}

sync_swanlab_data

################################################################################
# 步骤7: Git提交所有变更 - 将实验结果和配置纳入版本控制
################################################################################
log_info "步骤7/7: 提交实验变更到Git..."

echo ""
log_info "将要提交的变更内容："
if git status --short 2>/dev/null; then
    echo ""
else
    log_warning "Git不可用或不在Git仓库中，无法显示变更状态"
    echo ""
fi

# 添加所有变更到Git暂存区
if git add -A 2>/dev/null; then
    # 创建包含实验ID和描述的Git提交
    if git commit -m "exp: ${EXP_ID} - ${EXP_DESC}" 2>/dev/null; then
        log_success "所有实验变更已成功提交到Git"
        log_info "本次Commit包含以下内容："
        log_info "  - 实验记录文件 (Records目录): $RECORD_FILE"
        log_info "  - 实验记录文件 (Checkpoint目录): $CHECKPOINT_RECORD_FILE"
        log_info "  - Checkpoint目录: $CHECKPOINT_DIR (已包含在outputs目录的DVC管理中)"
        if [ -n "$HYDRA_OUTPUT_DIR" ] && [ -f "${HYDRA_OUTPUT_DIR}.dvc" ]; then
            log_info "  - Hydra输出DVC元文件: ${HYDRA_OUTPUT_DIR}.dvc"
        fi
        log_info "  - 其他代码变更 (如有)"
    else
        log_error "Git提交失败，请检查Git配置和权限"
        exit 1
    fi
else
    log_error "Git add失败，请检查Git配置和权限"
    exit 1
fi

log_success "所有实验变更已成功提交到Git"

################################################################################
# 步骤8: 清理临时文件 - 移除实验过程中产生的临时状态文件
################################################################################
log_info "步骤8/8: 清理实验临时文件..."

# 删除前置和训练阶段生成的临时状态文件
rm -f "$META_FILE"
rm -f "$STATE_FILE"

# 保留SwanLab URL文件在CHECKPOINT_DIR中供后续参考，不进行删除
log_success "临时文件清理完成"

# 输出实验完成总结信息
echo ""
log_success "========================================="
log_success "   集群实验 ${EXP_ID} 全部完成！"
log_success "========================================="
echo ""

# 显示关键文件和版本信息，便于后续查看和使用
log_info "📋 关键文件路径："
log_info "  - 实验记录文件 (Records目录): $RECORD_FILE"
log_info "  - 实验记录文件 (Checkpoint目录): $CHECKPOINT_RECORD_FILE"
if [ -n "$HYDRA_OUTPUT_DIR" ]; then
    log_info "  - 实验记录文件 (Hydra目录): $HYDRA_OUTPUT_DIR/experiment_record_${EXP_ID}.json"
fi

log_info "🔬 实验结果信息："
log_info "  - SwanLab实验URL: $SWANLAB_URL"
log_info "  - SwanLab URL文件: $SWANLAB_URL_FILE"
log_info "  - Checkpoint输出目录: $CHECKPOINT_DIR"

log_info "🏷️  版本控制信息："
log_info "  - 代码版本 (Commit): ${CODE_COMMIT:0:8}"
log_info "  - 数据集版本 (Commit): ${DATABASE_COMMIT:0:8}"
log_info "  - Checkpoint已包含在outputs目录的DVC管理中"

if [ -n "$HYDRA_OUTPUT_HASH" ]; then
    log_info "📦 Hydra输出版本 (Hash): ${HYDRA_OUTPUT_HASH:0:8}"
fi

echo ""
log_info "📝 重要提醒：完整的复现命令和参数详见实验记录文件"
log_info "   文件路径: $RECORD_FILE"
log_info "   查看方式: cat $RECORD_FILE | jq '.reproduction'"
echo ""
log_success "========================================="
