#!/bin/bash
################################################################################
# ExplicitLM核心实验运行脚本 - Hydra-Zen版
# 用途：被各实验脚本调用的核心逻辑，使用hydra_zen配置
#
# 调用方式：由实验脚本source调用，需要预先定义以下变量：
#   - EXP_ID: 实验ID
#   - EXP_DESC: 实验描述
#   - DATASET_VERSION: 训练数据集版本（Git commit hash，可选）
#   - EMBEDDING_VERSION: 预训练嵌入版本（可选）
#   - DATABASE_VERSION: 知识库初始化版本（可选）
#   - CACHE_VERSION: 缓存数据版本（可选）
#   - TRAIN_ARGS: Hydra-Zen配置覆盖参数 (格式: "param=value param2=value2")
#
# 示例：
#   EXP_ID="exp_001"
#   EXP_DESC="基线实验 Hydra-Zen配置版"
#   DATASET_VERSION=""
#   VAL_DATASET_VERSION=""
#   EMBEDDING_VERSION=""
#   DATABASE_VERSION=""
#   CACHE_VERSION=""
#   TRAIN_ARGS="training.epochs=10 model.knowledge_num=1048576"
#   source experiments/scripts/_run_experiment_core_hydra_zen.sh
################################################################################

set -e  # 遇到错误立即退出
set -o pipefail  # 管道命令中任何一个失败都返回失败

################################################################################
# 颜色定义 - 用于终端输出着色
################################################################################
RED='\033[0;31m'      # 红色 - 错误信息
GREEN='\033[0;32m'    # 绿色 - 成功信息
YELLOW='\033[1;33m'   # 黄色 - 警告信息
BLUE='\033[0;34m'     # 蓝色 - 普通信息
NC='\033[0m'          # 重置颜色

################################################################################
# 日志函数 - 格式化输出不同类型的信息
################################################################################
# 输出普通信息
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# 输出成功信息
log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 输出警告信息
log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 输出错误信息
log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

################################################################################
# 验证必需变量 - 确保调用脚本提供了必要的实验配置
################################################################################
# 检查必需的环境变量是否已设置
if [ -z "$EXP_ID" ] || [ -z "$EXP_DESC" ] || [ -z "$TRAIN_ARGS" ]; then
    log_error "缺少必需变量！"
    echo "需要在调用脚本中定义："
    echo "  EXP_ID=\"实验ID\""
    echo "  EXP_DESC=\"实验描述\""
    echo "  DATASET_VERSION=\"训练数据集版本(可选)\""
    echo "  VAL_DATASET_VERSION=\"验证数据集版本(可选)\""
    echo "  EMBEDDING_VERSION=\"预训练嵌入版本(可选)\""
    echo "  DATABASE_VERSION=\"知识库初始化版本(可选)\""
    echo "  CACHE_VERSION=\"缓存版本(可选)\""
    echo "  TRAIN_ARGS=\"Hydra-Zen配置覆盖参数，格式: 'param=value param2=value2'\""
    exit 1
fi

log_info "========================================="
log_info "实验ID: $EXP_ID"
log_info "实验描述: $EXP_DESC"
log_info "训练数据集版本: ${DATASET_VERSION:-当前版本}"
log_info "Hydra-Zen配置覆盖: $TRAIN_ARGS"
log_info "========================================="

################################################################################
# 目录和文件路径定义 - 设置实验相关的各种路径变量
################################################################################
# 项目根目录路径
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
# Checkpoint目录路径（训练后更新为Hydra输出目录）
CHECKPOINT_DIR=""
# Hydra输出目录中的临时记录文件路径
TEMP_RECORD_FILE=""
# 生成时间戳用于创建唯一的记录文件名
TIMESTAMP_FILENAME=$(date +"%Y%m%d_%H%M%S")
# 永久记录文件路径（带时间戳确保唯一性）
RECORD_FILE="${PROJECT_ROOT}/experiments/records/${EXP_ID}_${TIMESTAMP_FILENAME}.json"
# SwanLab URL文件路径（查找Hydra输出目录后设置）
SWANLAB_URL_FILE=""
# 实验元数据临时文件路径
META_FILE="${PROJECT_ROOT}/.experiment_meta"

# Hydra output directory will be detected after training
HYDRA_OUTPUT_DIR=""

################################################################################
# 前置检查 - 验证实验环境和依赖项
################################################################################
check_prerequisites() {
    log_info "步骤1/11: 前置检查..."

    # 检查是否在Git仓库中
    if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        log_error "当前不在Git仓库中！"
        exit 1
    fi

    # 检查DVC是否初始化
    if [ ! -d "${PROJECT_ROOT}/.dvc" ]; then
        log_error "DVC未初始化！请先运行: dvc init"
        exit 1
    fi

    # 创建实验记录目录
    mkdir -p "${PROJECT_ROOT}/experiments/records"

    # 检查实验ID是否已存在（检查相同EXP_ID前缀的记录）
    existing_records=$(find "${PROJECT_ROOT}/experiments/records" -name "${EXP_ID}_*.json" 2>/dev/null)
    if [ -n "$existing_records" ]; then
        log_warning "实验ID ${EXP_ID} 已有历史记录："
        echo "$existing_records" | while read -r record; do
            log_info "  - $record"
        done
        log_info "将创建新的带时间戳的记录文件: $RECORD_FILE"
    fi

    log_success "前置检查通过"
}

################################################################################
# 记录代码版本（训练前） - 保存实验开始时的代码状态
################################################################################
record_code_version() {
    log_info "步骤2/11: 记录代码版本..."

    # 记录当前HEAD的commit hash（训练前的代码状态）
    CODE_COMMIT=$(git rev-parse HEAD)

    log_success "代码版本已记录: ${CODE_COMMIT:0:8}"

    # 显示当前工作区状态
    if ! git diff --quiet || ! git diff --cached --quiet; then
        log_warning "检测到未提交的变更，将在训练后一起提交"
        git status --short
    fi
}

################################################################################
# 数据版本切换和同步（细粒度） - 精确控制每个数据集的版本
################################################################################
sync_data() {
    log_info "步骤3/11: 数据版本切换和同步（细粒度）..."

    # 保存当前分支，同步完成后需要切回
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

    # 同步函数：切换单个数据集到指定版本
    sync_dataset() {
        local dataset_name=$1
        local target_version=$2
        local dvc_file="data/${dataset_name}.dvc"

        if [ -n "$target_version" ]; then
            log_info "  - ${dataset_name}: 切换到版本 ${target_version:0:8}"

            # 切换到指定commit以获取对应版本的数据集定义
            git checkout "$target_version" --quiet

            # 仅checkout该数据集（使用DVC拉取对应版本的数据）
            dvc checkout "$dvc_file"

            # 记录该数据集版本（用于实验记录）
            eval "${dataset_name^^}_COMMIT=\"$target_version\""

            # 切回当前分支继续执行
            git checkout "$CURRENT_BRANCH" --quiet
        else
            log_info "  - ${dataset_name}: 使用当前版本"

            # 仅checkout该数据集（使用DVC拉取当前版本的数据）
            dvc checkout "$dvc_file" 2>/dev/null || true

            # 获取该数据集对应的Git commit（用于实验记录）
            local commit=$(git log -1 --format="%H" -- "$dvc_file" 2>/dev/null || echo "$CODE_COMMIT")
            eval "${dataset_name^^}_COMMIT=\"$commit\""
        fi
    }

    # 同步训练数据集
    sync_dataset "database" "$DATASET_VERSION"         # data/database.dvc (训练数据集)

    # 同步验证数据集
    sync_dataset "benchmarks" "$VAL_DATASET_VERSION"   # data/benchmarks.dvc (验证数据集)

    # 同步可选数据集（仅在项目使用且指定了版本时同步）
    [ -n "$EMBEDDING_VERSION" ] && [ -f "data/embeddings.dvc" ] && sync_dataset "embeddings" "$EMBEDDING_VERSION"      # data/embeddings.dvc (如果存在)
    [ -n "$DATABASE_VERSION" ] && [ -f "data/database_init.dvc" ] && sync_dataset "database_init" "$DATABASE_VERSION"    # data/database_init.dvc (如果存在)
    [ -n "$CACHE_VERSION" ] && [ -f "${PROJECT_ROOT}/cache.dvc" ] && sync_dataset "cache" "$CACHE_VERSION"                   # cache.dvc (如果存在)

    log_success "数据集同步完成"
    log_info "  - Database (训练数据): ${DATABASE_COMMIT:0:8}"
    # [ -n "$EMBEDDING_VERSION" ] && log_info "  - Embeddings (预训练权重): ${EMBEDDINGS_COMMIT:0:8}"
    # [ -n "$DATABASE_VERSION" ] && log_info "  - Database Init (知识库): ${DATABASE_INIT_COMMIT:0:8}"
    # [ -n "$CACHE_VERSION" ] && log_info "  - Cache: ${CACHE_COMMIT:0:8}"
}

################################################################################
# 记录实验元数据（训练前） - 保存实验配置和环境信息
################################################################################
record_pre_training_meta() {
    log_info "步骤4/11: 记录训练前元数据..."

    # 生成UTC时间戳（用于实验记录）
    TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # 记录到临时文件（JSON格式便于后续处理）
    cat > "$META_FILE" <<EOF
{
  "experiment": {
    "id": "$EXP_ID",
    "description": "$EXP_DESC",
    "timestamp": "$TIMESTAMP",
    "script": "run_experiment_hydra_zen.sh"
  },
  "versions": {
    "code_commit": "$CODE_COMMIT",
    "data": {
      "dataset_commit": "$DATABASE_COMMIT",
      "val_dataset_commit": "${BENCHMARKS_COMMIT:-N/A}",
      "embedding_commit": "${EMBEDDINGS_COMMIT:-N/A}",
      "database_init_commit": "${DATABASE_INIT_COMMIT:-N/A}",
      "cache_commit": "${CACHE_COMMIT:-N/A}"
    }
  },
  "command": "python 1_pretrain.py $TRAIN_ARGS"
}
EOF

    log_success "元数据已记录到临时文件: $META_FILE"
}

################################################################################
# 运行训练 - 执行模型训练过程
################################################################################
run_training() {
    log_info "步骤5/11: 开始训练..."

    # 构建训练命令 - 使用hydra_zen格式的参数
    TRAIN_CMD="accelerate launch 1_pretrain.py $TRAIN_ARGS"

    log_info "执行命令: $TRAIN_CMD"
    echo ""

    # 运行训练（直接显示输出，便于监控训练过程）
    eval $TRAIN_CMD

    # 检查训练是否成功
    if [ $? -ne 0 ]; then
        log_error "训练失败！"
        exit 1
    fi

    log_success "训练完成"
}

################################################################################
# 读取SwanLab URL - 获取训练过程中的实验监控URL
################################################################################
get_swanlab_url() {
    log_info "步骤7/11: 获取SwanLab实验URL..."

    # 从临时文件读取（需要1_pretrain.py配合写入）
    if [ -f "$SWANLAB_URL_FILE" ]; then
        SWANLAB_URL=$(cat "$SWANLAB_URL_FILE")
        log_success "SwanLab URL: $SWANLAB_URL"
    else
        SWANLAB_URL="N/A"
        log_warning "未找到SwanLab URL文件，可能未启用SwanLab或训练脚本未写入"
    fi
}

################################################################################
# 追踪模型权重 - 检查和验证生成的模型权重文件
################################################################################
track_checkpoint() {
    log_info "步骤8/11: 列出生成的模型权重文件..."

    # 使用find_hydra_output_dir函数设置的CHECKPOINT_DIR
    if [ -n "$CHECKPOINT_DIR" ] && [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A "$CHECKPOINT_DIR" 2>/dev/null)" ]; then
        TARGET_CHECKPOINT_DIR="$CHECKPOINT_DIR"
        log_info "使用Checkpoint目录: $TARGET_CHECKPOINT_DIR"
    else
        log_error "Checkpoint目录不存在或为空: $CHECKPOINT_DIR"
        exit 1
    fi

    # 检查checkpoint目录
    if [ ! -d "$TARGET_CHECKPOINT_DIR" ]; then
        log_error "Checkpoint目录不存在: $TARGET_CHECKPOINT_DIR"
        exit 1
    fi

    # 列出生成的文件
    log_info "生成的checkpoint文件:"
    ls -lh "$TARGET_CHECKPOINT_DIR"

    log_success "模型权重文件检查完成"
}

################################################################################
# 生成实验记录文件 - 创建完整的实验记录用于复现和分析
################################################################################
generate_record() {
    log_info "步骤9/11: 生成实验记录文件..."

    # 从训练参数中提取超参数（将hydra_zen格式转换为JSON）
    PARAMS_JSON=$(python3 -c "
import sys, json
import re

# 解析hydra_zen风格的参数（key=value格式）
args_str = '$TRAIN_ARGS'
pairs = args_str.split()

params = {}
for pair in pairs:
    if '=' in pair:
        key, value = pair.split('=', 1)
        # 尝试转换为适当的类型
        try:
            # 首先检查是否为数值类型
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # 尝试布尔值转换
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            # 否则保持为字符串
            else:
                pass
        params[key] = value

print(json.dumps(params, indent=2))
" 2>/dev/null || echo "{}")

    # 获取环境信息
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ',' || echo "N/A")
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")

    # 生成记录文件
    log_info "生成实验记录文件..."
    # 记录文件名（用于复现）
    RECORD_FILENAME=$(basename "$RECORD_FILE")

    cat > "$TEMP_RECORD_FILE" <<EOF
{
  "experiment": {
    "id": "$EXP_ID",
    "description": "$EXP_DESC",
    "timestamp": "$(date -u +\"%Y-%m-%dT%H:%M:%SZ\")",
    "record_filename": "$RECORD_FILENAME",
    "script": "run_experiment_hydra_zen.sh",
    "command": "python 1_pretrain.py $TRAIN_ARGS"
  },
  "versions": {
    "code_commit": "$CODE_COMMIT",
    "code_commit_short": "${CODE_COMMIT:0:8}",
    "data": {
      "dataset_commit": "$DATABASE_COMMIT",
      "dataset_commit_short": "${DATABASE_COMMIT:0:8}",
      "val_dataset_commit": "${BENCHMARKS_COMMIT:-N/A}",
      "val_dataset_commit_short": "${BENCHMARKS_COMMIT:0:8}",
      "embedding_commit": "${EMBEDDINGS_COMMIT:-N/A}",
      "embedding_commit_short": "${EMBEDDINGS_COMMIT:0:8}",
      "database_init_commit": "${DATABASE_INIT_COMMIT:-N/A}",
      "database_init_commit_short": "${DATABASE_INIT_COMMIT:0:8}",
      "cache_commit": "${CACHE_COMMIT:-N/A}",
      "cache_commit_short": "${CACHE_COMMIT:0:8}"
    }
  },
  "hyperparameters": $PARAMS_JSON,
  "results": {
    "swanlab_url": "$SWANLAB_URL",
    "checkpoint_dir": "$CHECKPOINT_DIR"
  },
  "artifacts": {
    "experiment_output": {
      "path": "$HYDRA_OUTPUT_DIR",
      "dvc_file": "${HYDRA_OUTPUT_DIR}.dvc"
    }
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
    "experiment_output_pull": "dvc pull ${HYDRA_OUTPUT_DIR}.dvc",
    "full_command": "# 1. 恢复代码版本\\\\ngit checkout $CODE_COMMIT\\\\n\\\\n# 2. 恢复数据集版本\\\\ngit checkout $DATABASE_COMMIT && dvc checkout data/database.dvc && git checkout -\\\\n\\\\n# 3. 恢复验证数据集版本\\\\ngit checkout $BENCHMARKS_COMMIT && dvc checkout data/benchmarks.dvc && git checkout -\\\\n\\\\n# 4. 拉取实验输出\\\\ndvc pull ${HYDRA_OUTPUT_DIR}.dvc\\\\n\\\\n# 5. 运行训练\\\\npython 1_pretrain.py $TRAIN_ARGS"
  }
}
EOF

    log_success "实验记录已生成: $TEMP_RECORD_FILE"

    # 复制记录到experiments/records目录（永久保存）
    if [ -n "$TEMP_RECORD_FILE" ] && [ -f "$TEMP_RECORD_FILE" ]; then
        cp "$TEMP_RECORD_FILE" "$RECORD_FILE"
        log_success "实验记录已复制到: $RECORD_FILE"
    else
        log_warning "临时记录文件不存在，无法复制到records目录"
    fi

    # 显示记录文件内容
    echo ""
    log_info "========== 实验记录内容 =========="
    cat "$TEMP_RECORD_FILE" | python3 -m json.tool 2>/dev/null || cat "$TEMP_RECORD_FILE"
    log_info "=================================="
    echo ""
}

################################################################################
# 查找Hydra输出目录 - 训练完成后定位Hydra生成的输出目录
################################################################################
find_hydra_output_dir() {
    log_info "步骤6/11: 查找Hydra输出目录..."

    # 查找包含.hydra文件夹的最新输出目录
    # 在outputs目录中搜索包含.hydra子目录的文件夹
    local hydra_dirs=$(find "${PROJECT_ROOT}/outputs" -name ".hydra" -type d -printf "%h\n" 2>/dev/null | sort -r | head -n 1)

    if [ -n "$hydra_dirs" ] && [ -d "$hydra_dirs" ]; then
        HYDRA_OUTPUT_DIR="$hydra_dirs"
        log_success "找到Hydra输出目录: $HYDRA_OUTPUT_DIR"

        # 设置SWANLAB_URL_FILE为Hydra输出目录中的文件
        SWANLAB_URL_FILE="${HYDRA_OUTPUT_DIR}/.swanlab_url"

        # 更新CHECKPOINT_DIR指向Hydra输出目录的out子目录
        if [ -d "$HYDRA_OUTPUT_DIR/out" ]; then
            CHECKPOINT_DIR="$HYDRA_OUTPUT_DIR/out"
            log_info "更新CHECKPOINT_DIR到Hydra输出目录: $CHECKPOINT_DIR"
        fi

        # 设置TEMP_RECORD_FILE为Hydra输出目录中的文件
        TEMP_RECORD_FILE="${HYDRA_OUTPUT_DIR}/experiment_record_${EXP_ID}.json"
    else
        log_warning "未找到Hydra输出目录，使用默认目录"
        HYDRA_OUTPUT_DIR=""
        # 如果没有找到Hydra目录，回退到项目根目录
        SWANLAB_URL_FILE="${PROJECT_ROOT}/.swanlab_url"
        TEMP_RECORD_FILE=""
    fi
}


################################################################################
# 追踪实验输出目录到DVC - 使用DVC管理实验输出以支持版本控制
################################################################################
track_experiment_output() {
    if [ -n "$HYDRA_OUTPUT_DIR" ] && [ -d "$HYDRA_OUTPUT_DIR" ]; then
        log_info "步骤10/11: 追踪实验输出目录到DVC..."
        log_info "实验输出目录内容:"
        ls -la "$HYDRA_OUTPUT_DIR"

        # 为整个实验目录创建DVC追踪
        log_info "开始DVC追踪实验目录: $HYDRA_OUTPUT_DIR"
        if dvc add "$HYDRA_OUTPUT_DIR" 2>/dev/null; then
            local exp_dvc_file="${HYDRA_OUTPUT_DIR}.dvc"
            if [ -f "$exp_dvc_file" ]; then
                local exp_hash=$(grep "md5:" "$exp_dvc_file" | awk '{print $3}')
                log_success "实验目录DVC追踪完成 (Hash: ${exp_hash:0:8})"
                log_info "DVC文件: $exp_dvc_file"
            else
                log_warning "DVC追踪完成但未找到元文件: $exp_dvc_file"
            fi
        else
            log_warning "DVC追踪失败，可能目录已被管理或无变更: $HYDRA_OUTPUT_DIR"
            # 检查是否已经有.dvc文件
            local exp_dvc_file="${HYDRA_OUTPUT_DIR}.dvc"
            if [ -f "$exp_dvc_file" ]; then
                local exp_hash=$(grep "md5:" "$exp_dvc_file" | awk '{print $3}')
                log_info "使用现有DVC追踪 (Hash: ${exp_hash:0:8})"
            fi
        fi
    else
        log_info "实验输出目录不存在，跳过DVC追踪"
    fi
}

################################################################################
# Git提交所有变更（一次性提交） - 将实验相关变更统一提交到版本控制
################################################################################
commit_all_changes() {
    log_info "步骤11/11: 提交所有变更到Git..."

    # 显示将要提交的变更
    echo ""
    log_info "将要提交的变更："
    git status --short
    echo ""

    # 添加所有变更到暂存区
    git add -A

    # 提交变更（使用实验ID和描述作为提交信息）
    git commit -m "exp: ${EXP_ID} - ${EXP_DESC}"

    log_success "所有变更已提交到Git"
    log_info "Commit包含："
    log_info "  - 实验脚本 (如有新增/修改)"
    log_info "  - 记录文件 (Hydra): $TEMP_RECORD_FILE"
    log_info "  - 记录文件 (Records): $RECORD_FILE"
    if [ -n "$HYDRA_OUTPUT_DIR" ]; then
        log_info "  - 实验输出DVC元文件: ${HYDRA_OUTPUT_DIR}.dvc"
    fi
    log_info "  - 其他代码变更 (如有)"
}

################################################################################
# 清理临时文件 - 删除实验过程中生成的临时文件
################################################################################
cleanup() {
    log_info "清理临时文件..."
    [ -n "$SWANLAB_URL_FILE" ] && rm -f "$SWANLAB_URL_FILE"
    rm -f "$META_FILE"
    log_success "清理完成"
}

################################################################################
# 实验总结 - 显示实验执行结果和关键信息
################################################################################
print_summary() {
    echo ""
    log_success "========================================="
    log_success "   实验 ${EXP_ID} 执行完成！"
    log_success "========================================="
    echo ""
    log_info "📋 记录文件 (Records): $RECORD_FILE"
    log_info "📋 记录文件 (Hydra): $TEMP_RECORD_FILE"
    if [ -n "$HYDRA_OUTPUT_DIR" ]; then
        log_info "📋 记录文件 (Hydra副本): $HYDRA_OUTPUT_DIR/experiment_record_${EXP_ID}.json"
    fi
    log_info "🔬 SwanLab URL: $SWANLAB_URL"
    log_info "💾 Checkpoint: $CHECKPOINT_DIR"
    log_info "🏷️  代码版本: ${CODE_COMMIT:0:8}"
    log_info "📊 训练数据集版本: ${DATABASE_COMMIT:0:8}"
    echo ""
    log_info "复现命令（详见记录文件的reproduction字段）:"
    echo "  1. 恢复代码: git checkout $CODE_COMMIT"
    echo "  2. 恢复数据: 使用记录文件中的data_checkout_steps"
    if [ -n "$HYDRA_OUTPUT_DIR" ]; then
        echo "  3. 拉取实验输出: dvc pull ${HYDRA_OUTPUT_DIR}.dvc"
    fi
    echo ""
    log_success "========================================="
}

################################################################################
# 主流程 - 按顺序执行所有实验步骤
################################################################################
main() {
    check_prerequisites
    record_code_version
    sync_data
    record_pre_training_meta
    run_training
    find_hydra_output_dir
    get_swanlab_url
    track_checkpoint
    generate_record
    track_experiment_output
    commit_all_changes
    cleanup
    print_summary
}

# 执行主流程
main