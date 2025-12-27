#!/bin/bash
# 记忆组件训练启动脚本（Memory Components Training）
# 只训练 MemoryGate、Fusion、MemoryNorm，Qwen3 backbone 完全冻结

# ========== 配置区域 ==========
# 设置GPU可见设备（平衡显存）
export CUDA_VISIBLE_DEVICES=0

# 设置PyTorch内存分配配置
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 启用详细调试信息（可选）
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_LAUNCH_BLOCKING=1

# 设置SwanLab API Key
export SWANLAB_API_KEY=GtiI1qjU5lco6MKKSrRmN

# 进入项目目录
cd /data2/zengzheni/lvchangwei/new_repo/ExplicitLM

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

QWEN3_MODEL_PATH="/data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b" 
# 使用新的 extract 数据（需要先运行转换脚本）
CACHE_PATH="data/cache/kb_parquet.pt"
# CACHE_PATH="data/cache/knowledge_cache.pt"  # 旧数据
PRETRAINED_ROUTER_PATH=""  # Router 预训练权重路径（可选）
PRETRAINED_FUSION_PATH=""  # Fusion 预训练权重路径（可选）
SFT_DATASET_PATH="data/parquet_data/256"
SFT_VAL_DATASET_PATH="sft_data/train_data_with_extract_sft_val.jsonl"


# 训练超参数（优化内存使用）
LEARNING_RATE=5e-5
BATCH_SIZE=2  # 
ACCUMULATION_STEPS=128 
EPOCHS=3
MAX_SEQ_LEN=256  # 保持256，进一步减小可能影响训练效果
SIMILARITY_LOSS_COEF=1.0  # 自适应 loss 平衡的基础系数（1.0=完全平衡，0.5=一半，2.0=两倍）

echo ""
echo "=========================================="
echo "🚀 启动记忆组件训练（Memory Components Training）"
echo "=========================================="
echo ""
echo "配置："
echo "  - Qwen3 模型: $QWEN3_MODEL_PATH"
echo "  - Cache 路径: $CACHE_PATH"
echo "  - Router 权重: $PRETRAINED_ROUTER_PATH"
echo "  - Fusion 权重: $PRETRAINED_FUSION_PATH"
echo "  - 训练数据: $SFT_DATASET_PATH"
echo "  - 验证数据: $SFT_VAL_DATASET_PATH"
echo "  - 学习率: $LEARNING_RATE"
echo "  - 批次大小: $BATCH_SIZE"
    echo "  - 梯度累积: $ACCUMULATION_STEPS"
    echo "  - 训练轮数: $EPOCHS"
    echo "  - 最大序列长度: $MAX_SEQ_LEN"
    echo "  - Similarity Loss 系数: $SIMILARITY_LOSS_COEF"
    echo "  - DeepSpeed Stage: 2 (ZeRO-2)"
    echo "  - Checkpoint 保存目录: out/"
    echo ""

# 检查必要文件
# SFT训练支持JSONL文件或Parquet目录
if [ ! -f "$SFT_DATASET_PATH" ] && [ ! -d "$SFT_DATASET_PATH" ]; then
    echo "❌ 错误: SFT 训练数据不存在: $SFT_DATASET_PATH"
    echo ""
    echo "SFT训练支持以下格式:"
    echo "  - JSONL文件: 包含conversations字段的JSONL文件"
    echo "  - Parquet目录: 包含.parquet文件的目录（每个文件需包含text字段）"
    echo ""
    echo "如果使用Parquet目录，请确保目录中包含.parquet文件"
    echo "如果使用JSONL文件，请先运行数据转换脚本:"
    echo "  cd /data2/zengzheni/lvchangwei/new_repo"
    echo "  python3 util_py/convert_extract_data_to_sft.py \\"
    echo "    --input ExplicitLM/data/train_data_with_extract.json \\"
    echo "    --qwen-model-path $QWEN3_MODEL_PATH \\"
    echo "    --output-dir ExplicitLM/sft_data \\"
    echo "    --cache-path ExplicitLM/$CACHE_PATH \\"
    echo "    --knowledge-num 1048576 \\"
    echo "    --knowledge-length 32 \\"
    echo "    --train-ratio 0.8"
    exit 1
fi

# 如果是Parquet目录，检查是否包含.parquet文件
if [ -d "$SFT_DATASET_PATH" ]; then
    PARQUET_COUNT=$(find "$SFT_DATASET_PATH" -name "*.parquet" -type f | wc -l)
    if [ "$PARQUET_COUNT" -eq 0 ]; then
        echo "❌ 错误: Parquet目录中没有找到.parquet文件: $SFT_DATASET_PATH"
        exit 1
    fi
    echo "✓ 找到 $PARQUET_COUNT 个Parquet文件"
fi

if [ ! -f "$CACHE_PATH" ]; then
    echo "❌ 错误: Cache 文件不存在: $CACHE_PATH"
    echo "请先运行数据转换脚本生成 cache 文件（见上方提示）"
    exit 1
fi

# 确保输出目录存在
mkdir -p out

# 启动训练
echo "=========================================="
echo "开始训练..."
echo "=========================================="
echo ""

LOG_FILE="train.log"
NOHUP_FILE="nohup.out"

echo "训练日志将保存到: $LOG_FILE"
echo "nohup 输出将保存到: $NOHUP_FILE"
echo ""

nohup $ACCELERATE_CMD train_sft.py \
    model.qwen3_model_path="$QWEN3_MODEL_PATH" \
    model.cache_path="$CACHE_PATH" \
    model.recompute_cache=False \
    model.knowledge_num=1048576 \
    model.knowledge_length=32 \
    model.max_seq_len="$MAX_SEQ_LEN" \
    model.num_candidates=16 \
    dataset.sft_dataset_path="$SFT_DATASET_PATH" \
    dataset.pretrained_router_path="$PRETRAINED_ROUTER_PATH" \
    dataset.pretrained_fusion_path="$PRETRAINED_FUSION_PATH" \
    dataset.sft_val_dataset_path="$SFT_VAL_DATASET_PATH" \
    training.learning_rate="$LEARNING_RATE" \
    training.batch_size="$BATCH_SIZE" \
    training.accumulation_steps="$ACCUMULATION_STEPS" \
    training.epochs="$EPOCHS" \
    training.similarity_loss_coef="$SIMILARITY_LOSS_COEF" \
    training.zero_stage=2 \
    model.keys_path="data/keys_parquet_extract_v2_qwen.pt" \
    dataset.pretrained_router_path="gate_ckpt/router_only_lora.pt" \
    logging.out_dir="out" \
    logging.save_dir="out" \
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

