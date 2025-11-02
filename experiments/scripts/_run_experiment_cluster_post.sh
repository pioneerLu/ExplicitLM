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
# In cluster environment, git might not be available, so we use a fallback
if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    log_info "Git not available, using PROJECT_ROOT: $PROJECT_ROOT"
fi
STATE_FILE="${PROJECT_ROOT}/.cluster_state_${EXP_ID}"
SWANLAB_URL_FILE="${PROJECT_ROOT}/.swanlab_url_${EXP_ID}"
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
# 步骤3: 验证checkpoint目录
################################################################################
log_info "步骤3/6: 验证checkpoint目录..."

if [ ! -d "$CHECKPOINT_DIR" ]; then
    log_error "Checkpoint目录不存在: $CHECKPOINT_DIR"
    log_info "请确保已将checkpoint从计算节点同步回来"
    exit 1
fi

log_info "Checkpoint文件:"
ls -lh "$CHECKPOINT_DIR"

# 跳过checkpoint的单独DVC追踪，因为整个outputs目录已被DVC管理
log_info "跳过checkpoint的DVC追踪（整个outputs目录已被DVC管理）"
CHECKPOINT_DVC=""
CHECKPOINT_HASH="outputs_managed"

################################################################################
# Find Hydra output directory after training
################################################################################
find_hydra_output_dir() {
    log_info "步骤4.5/6: 查找Hydra输出目录..."

    # Look for the most recent output directory containing .hydra folder
    # Search in outputs directory for folders with .hydra subdirectory
    local hydra_dirs=$(find "${PROJECT_ROOT}/outputs" -name ".hydra" -type d -printf "%h\n" 2>/dev/null | sort -r | head -n 1)

    if [ -n "$hydra_dirs" ] && [ -d "$hydra_dirs" ]; then
        HYDRA_OUTPUT_DIR="$hydra_dirs"
        log_success "找到Hydra输出目录: $HYDRA_OUTPUT_DIR"

        # Copy the record file to Hydra output directory
        if [ -f "$RECORD_FILE" ]; then
            cp "$RECORD_FILE" "$HYDRA_OUTPUT_DIR/experiment_record_${EXP_ID}.json"
            log_info "实验记录已复制到: $HYDRA_OUTPUT_DIR/experiment_record_${EXP_ID}.json"
        fi

        # Track Hydra output directory to DVC
        track_hydra_output
    else
        log_warning "未找到Hydra输出目录，使用默认目录"
        HYDRA_OUTPUT_DIR=""
    fi
}

################################################################################
# 追踪Hydra输出目录
################################################################################
track_hydra_output() {
    if [ -n "$HYDRA_OUTPUT_DIR" ] && [ -d "$HYDRA_OUTPUT_DIR" ]; then
        log_info "步骤4.6/6: 追踪Hydra输出目录到DVC..."
        log_info "Hydra输出目录内容:"
        ls -la "$HYDRA_OUTPUT_DIR"

        # 为整个实验目录创建DVC追踪
        log_info "开始DVC追踪实验目录: $HYDRA_OUTPUT_DIR"
        if dvc add "$HYDRA_OUTPUT_DIR" 2>/dev/null; then
            HYDRA_OUTPUT_DVC="${HYDRA_OUTPUT_DIR}.dvc"
            if [ -f "$HYDRA_OUTPUT_DVC" ]; then
                HYDRA_OUTPUT_HASH=$(grep "md5:" "$HYDRA_OUTPUT_DVC" | awk '{print $3}')
                log_success "Hydra输出DVC追踪完成 (Hash: ${HYDRA_OUTPUT_HASH:0:8})"
                log_info "DVC文件: $HYDRA_OUTPUT_DVC"
            else
                log_warning "DVC追踪完成但未找到元文件: $HYDRA_OUTPUT_DVC"
            fi
        else
            log_warning "DVC追踪失败，可能目录已被管理或无变更: $HYDRA_OUTPUT_DIR"
            # 检查是否已经有.dvc文件
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
# 步骤5: 生成实验记录文件
################################################################################
# Find and track Hydra output directory
find_hydra_output_dir

log_info "步骤5/6: 生成实验记录文件..."

# 提取训练参数为JSON with better error handling
if command -v python3 >/dev/null 2>&1; then
    PARAMS_JSON=$(python3 -c "
import sys, json, shlex
try:
    args = shlex.split('$TRAIN_ARGS')
    params = {}
    i = 0
    while i < len(args):
        if args[i].startswith('--'):
            key = args[i][2:]
            if i + 1 < len(args) and not args[i+1].startswith('--') and not args[i+1].startswith('-'):
                value = args[i+1]
                # Try to convert to appropriate type
                try:
                    if '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Check for boolean values
                    if value.lower() in ('true', 'yes', 'on'):
                        value = True
                    elif value.lower() in ('false', 'no', 'off'):
                        value = False
                    # Keep as string otherwise
                params[key] = value
                i += 2
            else:
                params[key] = True
                i += 1
        else:
            i += 1
    print(json.dumps(params, indent=2))
except Exception:
    print('{}')
" 2>/dev/null || echo "{}")
else
    PARAMS_JSON="{}"
    log_warning "Python3 not available, hyperparameters will be empty"
fi

# 获取环境信息 with better error handling
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>/dev/null | awk '{print $2}' || echo "unknown")
else
    PYTHON_VERSION="N/A"
    log_warning "Python3 not available"
fi

if command -v nvcc >/dev/null 2>&1; then
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',' || echo "unknown")
else
    CUDA_VERSION="N/A"
    log_info "nvcc not available"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")
else
    NUM_GPUS=0
    log_info "nvidia-smi not available"
fi

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

echo ""
log_info "========== 实验记录内容 =========="
if command -v python3 >/dev/null 2>&1; then
    cat "$RECORD_FILE" | python3 -m json.tool 2>/dev/null || cat "$RECORD_FILE"
else
    cat "$RECORD_FILE"
    log_warning "Python3 not available for JSON formatting"
fi
log_info "=================================="
echo ""

################################################################################
# 步骤6: 同步SwanLab数据（如果启用且可用）
################################################################################
sync_swanlab_data() {
    log_info "步骤6/7: 检查并同步SwanLab数据..."

    # 检查是否启用了SwanLab
    if [[ "$TRAIN_ARGS" == *"--use_swanlab"* ]] || [[ "$TRAIN_ARGS" == *"use_swanlab=True"* ]]; then
        # 检查swanlab命令是否可用
        if command -v swanlab &> /dev/null; then
            # 查找SwanLab数据目录（通常在输出目录中的swanlog子目录下的run-*目录）
            if [ -n "$HYDRA_OUTPUT_DIR" ] && [ -d "$HYDRA_OUTPUT_DIR/swanlog" ]; then
                log_info "找到SwanLab日志目录: $HYDRA_OUTPUT_DIR/swanlog"

                # 查找具体的run-*目录（SwanLab创建的子目录）
                local run_dir=$(find "$HYDRA_OUTPUT_DIR/swanlog" -name "run-*" -type d | head -n 1)

                if [ -n "$run_dir" ] && [ -d "$run_dir" ]; then
                    log_info "找到SwanLab运行目录: $run_dir"

                    # 尝试同步SwanLab数据
                    log_info "执行SwanLab同步命令..."
                    if swanlab sync "$run_dir"; then
                        log_success "SwanLab数据同步成功"

                        # 获取同步后的URL
                        if [ -f "$SWANLAB_URL_FILE" ]; then
                            SWANLAB_URL=$(cat "$SWANLAB_URL_FILE")
                            log_info "SwanLab URL: $SWANLAB_URL"
                        fi
                    else
                        log_warning "SwanLab同步失败，可能没有网络连接或数据为空"
                    fi
                else
                    log_warning "未找到SwanLab运行目录 (run-*)，跳过同步"
                fi
            else
                log_info "未找到SwanLab日志目录，跳过同步"
            fi
        else
            log_info "SwanLab命令不可用，跳过同步"
        fi
    else
        log_info "SwanLab未启用，跳过同步"
    fi
}

sync_swanlab_data

################################################################################
# 步骤7: Git提交所有变更
################################################################################
log_info "步骤7/7: 提交到Git..."

echo ""
log_info "将要提交的变更："
if git status --short 2>/dev/null; then
    echo ""
else
    log_warning "Git not available or not in repository for status check"
    echo ""
fi

if git add -A 2>/dev/null; then
    if git commit -m "exp: ${EXP_ID} - ${EXP_DESC}" 2>/dev/null; then
        log_success "所有变更已提交到Git"
        log_info "Commit包含："
        log_info "  - 实验记录文件: $RECORD_FILE"
        log_info "  - Checkpoint目录: $CHECKPOINT_DIR (已包含在outputs目录的DVC管理中)"
        if [ -n "$HYDRA_OUTPUT_DIR" ] && [ -f "${HYDRA_OUTPUT_DIR}.dvc" ]; then
            log_info "  - Hydra输出DVC元文件: ${HYDRA_OUTPUT_DIR}.dvc"
        fi
        log_info "  - 其他代码变更 (如有)"
    else
        log_error "Git提交失败"
        exit 1
    fi
else
    log_error "Git add失败"
    exit 1
fi

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
if [ -n "$HYDRA_OUTPUT_DIR" ]; then
    log_info "📋 记录文件 (Hydra): $HYDRA_OUTPUT_DIR/experiment_record_${EXP_ID}.json"
fi
log_info "🔬 SwanLab URL: $SWANLAB_URL"
log_info "💾 Checkpoint: $CHECKPOINT_DIR"
log_info "🏷️  代码版本: ${CODE_COMMIT:0:8}"
log_info "📊 数据版本: ${DATABASE_COMMIT:0:8}"
log_info "📝 Checkpoint已包含在outputs目录的DVC管理中"
if [ -n "$HYDRA_OUTPUT_HASH" ]; then
    log_info "📦 Hydra输出哈希: ${HYDRA_OUTPUT_HASH:0:8}"
fi
echo ""
log_info "复现命令详见: $RECORD_FILE (reproduction字段)"
echo ""
log_success "========================================="
