#!/bin/bash

# ========== 配置区域 ==========
# 进入项目目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 优先使用 uv
if command -v uv &> /dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ 使用 uv 运行转换"
    PYTHON_CMD="uv run python"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ 已激活虚拟环境: $(which python)"
    PYTHON_CMD="python"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ 已激活venv虚拟环境: $(which python)"
    PYTHON_CMD="python"
else
    echo "⚠️  未找到虚拟环境，使用系统Python: $(which python)"
    PYTHON_CMD="python"
fi

# ========== 参数解析 ==========
# 默认值
CHECKPOINT_PATH=""
QWEN3_PATH="Qwen_hg/Qwen3-4b"
OUTPUT_PATH="hf_explicitlm_model"
MEMORY_BANK_PATH=""
KEYS_PATH=""
KNOWLEDGE_NUM=1048576
KNOWLEDGE_LENGTH=32
NUM_CANDIDATES=16
NUM_SELECTED=1
GUMBEL_TEMPERATURE=1.0
DIAGNOSE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --qwen3_path)
            QWEN3_PATH="$2"
            shift 2
            ;;
        --output_path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --memory_bank_path)
            MEMORY_BANK_PATH="$2"
            shift 2
            ;;
        --keys_path)
            KEYS_PATH="$2"
            shift 2
            ;;
        --knowledge_num)
            KNOWLEDGE_NUM="$2"
            shift 2
            ;;
        --knowledge_length)
            KNOWLEDGE_LENGTH="$2"
            shift 2
            ;;
        --num_candidates)
            NUM_CANDIDATES="$2"
            shift 2
            ;;
        --num_selected)
            NUM_SELECTED="$2"
            shift 2
            ;;
        --gumbel_temperature)
            GUMBEL_TEMPERATURE="$2"
            shift 2
            ;;
        --diagnose)
            DIAGNOSE=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "必需参数:"
            echo "  --checkpoint_path PATH    checkpoint文件或目录路径"
            echo ""
            echo "可选参数:"
            echo "  --qwen3_path PATH        Qwen3基础模型路径 (默认: Qwen_hg/Qwen3-4b)"
            echo "  --output_path PATH       输出HF模型路径 (默认: hf_explicitlm_model)"
            echo "  --memory_bank_path PATH  Memory Bank文件路径"
            echo "  --keys_path PATH         Keys文件路径"
            echo "  --knowledge_num NUM      记忆库大小 (默认: 1048576)"
            echo "  --knowledge_length NUM   记忆条目长度 (默认: 32)"
            echo "  --num_candidates NUM     候选记忆数 (默认: 16)"
            echo "  --num_selected NUM       选中记忆数 (默认: 1)"
            echo "  --gumbel_temperature NUM Gumbel-Softmax温度 (默认: 1.0)"
            echo "  --diagnose               仅诊断checkpoint，不进行转换"
            echo ""
            echo "示例:"
            echo "  # 诊断checkpoint"
            echo "  $0 --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_7500 --diagnose"
            echo ""
            echo "  # 转换checkpoint"
            echo "  $0 --checkpoint_path checkpoints/fusion_pretrain/checkpoint_step_7500 \\"
            echo "     --qwen3_path Qwen_hg/Qwen3-4b \\"
            echo "     --output_path hf_explicitlm_model"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查必需参数
if [ -z "$CHECKPOINT_PATH" ] && [ "$DIAGNOSE" = false ]; then
    echo "❌ 错误: 必须指定 --checkpoint_path"
    echo "使用 --help 查看帮助信息"
    exit 1
fi

# ========== 显示配置 ==========
echo "=========================================="
echo "🔄 ExplicitLM Checkpoint 转换为 HuggingFace 格式"
echo "=========================================="
echo ""
echo "配置:"
if [ -n "$CHECKPOINT_PATH" ]; then
    echo "  - Checkpoint路径: $CHECKPOINT_PATH"
fi
echo "  - Qwen3模型路径: $QWEN3_PATH"
echo "  - 输出路径: $OUTPUT_PATH"
if [ -n "$MEMORY_BANK_PATH" ]; then
    echo "  - Memory Bank路径: $MEMORY_BANK_PATH"
fi
if [ -n "$KEYS_PATH" ]; then
    echo "  - Keys路径: $KEYS_PATH"
fi
echo "  - 记忆库配置: num=$KNOWLEDGE_NUM, length=$KNOWLEDGE_LENGTH"
echo "  - 候选记忆数: $NUM_CANDIDATES"
echo "  - 选中记忆数: $NUM_SELECTED"
if [ "$DIAGNOSE" = true ]; then
    echo "  - 模式: 诊断模式"
fi
echo ""

# ========== 检查文件 ==========
if [ -n "$CHECKPOINT_PATH" ] && [ ! -e "$CHECKPOINT_PATH" ]; then
    echo "❌ 错误: Checkpoint路径不存在: $CHECKPOINT_PATH"
    exit 1
fi

if [ ! -d "$QWEN3_PATH" ]; then
    echo "❌ 错误: Qwen3模型路径不存在: $QWEN3_PATH"
    exit 1
fi

# ========== 构建命令 ==========
CMD_ARGS=(
    "convert_checkpoint_to_hf.py"
)

if [ -n "$CHECKPOINT_PATH" ]; then
    CMD_ARGS+=("--checkpoint_path" "$CHECKPOINT_PATH")
fi

CMD_ARGS+=(
    "--qwen3_path" "$QWEN3_PATH"
    "--output_path" "$OUTPUT_PATH"
)

if [ -n "$MEMORY_BANK_PATH" ]; then
    CMD_ARGS+=("--memory_bank_path" "$MEMORY_BANK_PATH")
fi

if [ -n "$KEYS_PATH" ]; then
    CMD_ARGS+=("--keys_path" "$KEYS_PATH")
fi

CMD_ARGS+=(
    "--knowledge_num" "$KNOWLEDGE_NUM"
    "--knowledge_length" "$KNOWLEDGE_LENGTH"
    "--num_candidates" "$NUM_CANDIDATES"
    "--num_selected" "$NUM_SELECTED"
    "--gumbel_temperature" "$GUMBEL_TEMPERATURE"
)

if [ "$DIAGNOSE" = true ]; then
    CMD_ARGS+=("--diagnose")
fi

# ========== 执行转换 ==========
echo "=========================================="
echo "开始转换..."
echo "=========================================="
echo ""

$PYTHON_CMD "${CMD_ARGS[@]}"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 转换完成！"
    echo "=========================================="
    if [ "$DIAGNOSE" = false ]; then
        echo ""
        echo "💡 使用方法:"
        echo "  from transformers import AutoTokenizer, AutoModelForCausalLM"
        echo "  tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_PATH', trust_remote_code=True)"
        echo "  model = AutoModelForCausalLM.from_pretrained('$OUTPUT_PATH', trust_remote_code=True)"
    fi
else
    echo ""
    echo "=========================================="
    echo "❌ 转换失败 (退出码: $EXIT_CODE)"
    echo "=========================================="
    exit $EXIT_CODE
fi

