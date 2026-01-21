#!/bin/bash

# ========== 配置区域 ==========
# 设置GPU可见设备（平衡显存）
export CUDA_VISIBLE_DEVICES=0,4,5

# 设置PyTorch内存分配配置（保持即可）
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 设置NCCL超时时间（避免分布式训练超时）
export NCCL_TIMEOUT=1800  # 30分钟
export TORCH_NCCL_BLOCKING_WAIT=1  # 启用阻塞等待以便调试

# 调试相关环境变量（默认关闭，有需要排查问题时再手动打开）
# 如需排查 NCCL / 分布式问题，可以暂时取消注释以下几行：
# export NCCL_DEBUG=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export CUDA_LAUNCH_BLOCKING=1

# 设置SwanLab API Key
export SWANLAB_API_KEY=GtiI1qjU5lco6MKKSrRmN

# 进入项目目录（脚本在 scripts/train/ 下，项目根目录是 ../..）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# 设置进程显示名称（在 nvidia-smi 中显示的名称）
export PYTHON_PROCESS_NAME="llama-env"

# 优先使用 uv
if command -v uv &> /dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
    echo "✅ 使用 uv 运行训练"
    ACCELERATE_CMD="uv run accelerate launch --config_file accelerate_config.yaml"
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "✅ 已激活虚拟环境: $(which python)"
    ACCELERATE_CMD="accelerate launch --config_file accelerate_config.yaml"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ 已激活venv虚拟环境: $(which python)"
    ACCELERATE_CMD="accelerate launch --config_file accelerate_config.yaml"
else
    echo "⚠️  未找到虚拟环境，使用系统Python: $(which python)"
    ACCELERATE_CMD="accelerate launch --config_file accelerate_config.yaml"
fi

# 显示GPU信息
echo "=========================================="
echo "使用GPU"
echo "=========================================="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | grep -E "^[0123],"

# ========== 训练配置 ==========

# Qwen3 模型路径
QWEN3_MODEL_PATH="Qwen_hg/Qwen3-4b"
# 记忆库 cache 路径（.pt 文件）
CACHE_PATH="data/cache/parquet_extract/memory_bank_batches/kb_parquet.pt"
# Keys 文件路径（可选，如果提供则从文件加载 keys 进行初始化）
KEYS_PATH="data/keys_parquet_extract_v2_qwen.pt"
PRETRAINED_MEMORY_GATE_PATH=""
# 预训练格式数据路径（Parquet 目录，支持多文件）
PRETRAIN_DATASET_PATH="data/parquet_data/sample_256"
# 验证数据配置（从训练数据中分割，不使用独立验证数据）
VAL_SPLIT_RATIO=0.05  

# 模型配置（参考 run_sft.sh）
KNOWLEDGE_NUM=10000  # 记忆库条目数
KNOWLEDGE_LENGTH=32   # 每个记忆条目的 token 数
NUM_CANDIDATES=16      # 候选记忆数
NUM_SELECTED=1         # 选中的记忆数
GUMBEL_TEMPERATURE=1.0 # Gumbel-Softmax 温度

# 数据配置
MAX_LENGTH=256         # 最大序列长度（参考 run_sft.sh）

# 训练超参数
LEARNING_RATE=1e-4     # Fusion 训练推荐 1e-4
BATCH_SIZE=2          # 参考 run_sft.sh
ACCUMULATION_STEPS=16 # 参考 run_sft.sh
EPOCHS=1               # 参考 run_sft.sh
WARMUP_STEPS=100

# Loss 配置
SIMILARITY_LOSS_COEF=1.0  # Similarity Loss 基础系数（用于自适应平衡，默认1.0）

# Memory 更新配置
# 注意：Memory 更新配置现在统一在 config/memory_update.py 中管理
# 如需修改配置，请编辑 config/memory_update.py 文件
# 如需通过命令行覆盖，可以添加 --enable_memory_update 等参数

# 输出配置
OUTPUT_DIR="checkpoints/fusion_pretrain"
SAVE_INTERVAL=500
SWANLAB_PROJECT="explicitlm-fusion-pretrain"
SWANLAB_ONLINE=false  # 设置为 true 启用在线模式

echo ""
echo "=========================================="
echo "🚀 启动 Fusion 组件预训练（Pretrain Fusion）"
echo "=========================================="
echo ""
echo "配置："
echo "  - Qwen3 模型: $QWEN3_MODEL_PATH"
if [ -n "$CACHE_PATH" ]; then
    echo "  - 记忆库 Cache: $CACHE_PATH"
else
    echo "  - 记忆库 Cache: 未设置（将使用随机初始化）"
fi
if [ -n "$KEYS_PATH" ]; then
    echo "  - Keys 文件: $KEYS_PATH"
else
    echo "  - Keys 文件: 未设置（将使用随机初始化或自动聚类）"
fi
echo "  - 预训练 MemoryGate: ${PRETRAINED_MEMORY_GATE_PATH:-未设置（将使用随机初始化）}"
echo "  - 训练数据: $PRETRAIN_DATASET_PATH (Parquet 格式，预训练类型)"
if [ -n "$VAL_SPLIT_SIZE" ]; then
    echo "  - 验证数据: 从训练数据中分割 ${VAL_SPLIT_SIZE} 个样本"
elif [ "$VAL_SPLIT_RATIO" != "0.0" ]; then
    echo "  - 验证数据: 从训练数据中分割 ${VAL_SPLIT_RATIO} (${VAL_SPLIT_RATIO}%)"
else
    echo "  - 验证数据: 未设置"
fi
echo "  - 学习率: $LEARNING_RATE"
echo "  - 批次大小: $BATCH_SIZE"
    echo "  - 梯度累积: $ACCUMULATION_STEPS"
    echo "  - 训练轮数: $EPOCHS"
    echo "  - 最大序列长度: $MAX_LENGTH"
    echo "  - 记忆库配置: num=$KNOWLEDGE_NUM, length=$KNOWLEDGE_LENGTH, dim=2560 (Qwen3 hidden_size)"
    echo "  - 候选记忆数: $NUM_CANDIDATES"
    echo "  - 输出目录: $OUTPUT_DIR"
    echo "  - DeepSpeed Stage: 2 (ZeRO-2)"
    echo "  - Memory 更新: 配置来自 config/memory_update.py (如需修改请编辑该文件)"
    echo ""

# 检查必要文件
if [ ! -d "$PRETRAIN_DATASET_PATH" ] && [ ! -f "$PRETRAIN_DATASET_PATH" ]; then
    echo "❌ 错误: 预训练数据路径不存在: $PRETRAIN_DATASET_PATH"
    echo "请确认数据路径是否正确（支持 Parquet 文件或目录）"
    exit 1
fi

if [ -n "$CACHE_PATH" ] && [ ! -f "$CACHE_PATH" ]; then
    echo "❌ 错误: Cache 文件不存在: $CACHE_PATH"
    echo "请先运行数据转换脚本生成 cache 文件"
    exit 1
fi

# 检查 Keys 文件（可选，如果为空或不存在则使用随机初始化或自动聚类）
if [ -n "$KEYS_PATH" ] && [ ! -f "$KEYS_PATH" ]; then
    echo "⚠️  警告: Keys 文件不存在: $KEYS_PATH"
    echo "将使用随机初始化或等待自动聚类生成 keys"
fi

# 检查 MemoryGate 权重（可选，如果为空或不存在则使用随机初始化）
if [ -n "$PRETRAINED_MEMORY_GATE_PATH" ] && [ ! -f "$PRETRAINED_MEMORY_GATE_PATH" ]; then
    echo "⚠️  警告: 预训练 MemoryGate 权重不存在: $PRETRAINED_MEMORY_GATE_PATH"
    echo "将使用随机初始化"
fi

# 确保输出目录存在
mkdir -p "$OUTPUT_DIR"

# 启动训练
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

# 使用 nohup 运行，防止 SSH 断开导致训练中断
# 输出重定向到 train_fusion_pretrain.log 和 nohup.out
LOG_FILE="train_fusion_pretrain.log"
NOHUP_FILE="nohup.out"

echo "训练日志将保存到: $LOG_FILE"
echo "nohup 输出将保存到: $NOHUP_FILE"
echo ""

# 构建训练命令参数
TRAIN_ARGS=(
    --qwen3_model_path "$QWEN3_MODEL_PATH"
    --knowledge_num "$KNOWLEDGE_NUM"
)

# 添加 cache 路径（如果提供）
if [ -n "$CACHE_PATH" ]; then
    TRAIN_ARGS+=(--cache_path "$CACHE_PATH")
fi

# 添加 keys 路径（如果提供）
if [ -n "$KEYS_PATH" ]; then
    TRAIN_ARGS+=(--keys_path "$KEYS_PATH")
fi

# 添加 MemoryGate 权重路径（如果提供）
if [ -n "$PRETRAINED_MEMORY_GATE_PATH" ]; then
    TRAIN_ARGS+=(--pretrained_memory_gate_path "$PRETRAINED_MEMORY_GATE_PATH")
fi

# 继续添加其他参数
TRAIN_ARGS+=(
    --knowledge_length "$KNOWLEDGE_LENGTH"
    --num_candidates "$NUM_CANDIDATES"
    --num_selected "$NUM_SELECTED"
    --gumbel_temperature "$GUMBEL_TEMPERATURE"
    --dataset_path "$PRETRAIN_DATASET_PATH"
    --max_length "$MAX_LENGTH"
    --batch_size "$BATCH_SIZE"
    --accumulation_steps "$ACCUMULATION_STEPS"
    --lr "$LEARNING_RATE"
    --epochs "$EPOCHS"
    --warmup_steps "$WARMUP_STEPS"
    --similarity_loss_coef "$SIMILARITY_LOSS_COEF"
    --output_dir "$OUTPUT_DIR"
    --save_interval "$SAVE_INTERVAL"
    --swanlab_project "$SWANLAB_PROJECT"
)

# 添加验证数据分割配置
if [ -n "$VAL_SPLIT_SIZE" ]; then
    TRAIN_ARGS+=(--val_split_size "$VAL_SPLIT_SIZE")
elif [ "$VAL_SPLIT_RATIO" != "0.0" ]; then
    TRAIN_ARGS+=(--val_split_ratio "$VAL_SPLIT_RATIO")
fi

# 添加 SwanLab 在线模式（如果启用）
if [ "$SWANLAB_ONLINE" = true ]; then
    TRAIN_ARGS+=(--swanlab_online)
fi

# Memory 更新配置
# 启用 Memory Bank 动态更新功能
ENABLE_MEMORY_UPDATE=true
MEMORY_UPDATE_FREQUENCY=500        # 每50步更新一次
MEMORY_UPDATE_STRATEGY="lru"      # 更新策略：fifo, lru, random, importance
MEMORY_COMPRESSION_RATE=0.4       # 事实压缩率（0-1，越小保留信息越多）
LLMLINGUA_MODEL_PATH="llmlingua-2-bert"  # LLMLingua 模型路径

if [ "$ENABLE_MEMORY_UPDATE" = true ]; then
    echo "  - Memory 更新: 已启用"
    echo "    - 更新频率: 每 $MEMORY_UPDATE_FREQUENCY 步"
    echo "    - 更新策略: $MEMORY_UPDATE_STRATEGY"
    echo "    - 压缩率: $MEMORY_COMPRESSION_RATE"
    TRAIN_ARGS+=(--enable_memory_update)
    TRAIN_ARGS+=(--memory_update_frequency "$MEMORY_UPDATE_FREQUENCY")
    TRAIN_ARGS+=(--memory_update_strategy "$MEMORY_UPDATE_STRATEGY")
    TRAIN_ARGS+=(--memory_compression_rate "$MEMORY_COMPRESSION_RATE")
    TRAIN_ARGS+=(--llmlingua_model_path "$LLMLINGUA_MODEL_PATH")
else
    echo "  - Memory 更新: 已禁用"
fi

# 执行训练
# 注意：train_pretrain.py 使用 argparse，直接运行即可（accelerate launch 会自动处理分布式）
nohup $ACCELERATE_CMD scripts/train/train_pretrain.py "${TRAIN_ARGS[@]}" \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!

echo "训练已在后台启动"
echo "进程 PID: $TRAIN_PID"
echo "查看实时日志: tail -f $LOG_FILE"
echo "查看进程状态: ps -p $TRAIN_PID"
echo "停止训练: kill $TRAIN_PID"
echo ""
echo "=========================================="
echo "✅ 训练已启动（后台运行）"
echo "=========================================="

