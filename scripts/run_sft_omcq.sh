#!/bin/bash
# SFT 训练启动脚本：使用 OMCQ 数据训练知识融合模块

# ========== 配置区域 ==========
# 请根据实际情况修改以下路径和参数

QWEN3_MODEL_PATH="/path/to/Qwen3-4b"              # Qwen3 模型路径
CACHE_PATH="data/cache/knowledge_cache.pt"        # 预训练知识库 cache
PRETRAINED_MODEL_PATH="out/pretrain_latest.pth"  # 预训练模型权重
SFT_DATASET_PATH="sft_data/omcq_trex_sft.jsonl"  # SFT 训练数据
SFT_VAL_DATASET_PATH="data/benchmarks/eval_data.json"  # SFT 验证数据

# 训练超参数
LEARNING_RATE=5e-5
BATCH_SIZE=4
ACCUMULATION_STEPS=32
EPOCHS=3
MAX_SEQ_LEN=512

# ========== 执行训练 ==========

# 设置SwanLab API Key
export SWANLAB_API_KEY=GtiI1qjU5lco6MKKSrRmN

cd "$(dirname "$0")/.."

echo "=========================================="
echo "🚀 启动 SFT 训练（OMCQ 数据）"
echo "=========================================="
echo ""
echo "配置："
echo "  - Qwen3 模型: $QWEN3_MODEL_PATH"
echo "  - Cache 路径: $CACHE_PATH"
echo "  - 预训练模型: $PRETRAINED_MODEL_PATH"
echo "  - 训练数据: $SFT_DATASET_PATH"
echo "  - 验证数据: $SFT_VAL_DATASET_PATH"
echo "  - 学习率: $LEARNING_RATE"
echo "  - 批次大小: $BATCH_SIZE"
echo "  - 梯度累积: $ACCUMULATION_STEPS"
echo "  - 训练轮数: $EPOCHS"
echo ""

# 检查必要文件
if [ ! -f "$SFT_DATASET_PATH" ]; then
    echo "❌ 错误: SFT 训练数据不存在: $SFT_DATASET_PATH"
    echo "请先运行数据转换脚本:"
    echo "  python3 scripts/convert_omcq_to_sft.py --input sft_data/omcq_trex_data.json --output $SFT_DATASET_PATH"
    exit 1
fi

if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
    echo "⚠️  警告: 预训练模型不存在: $PRETRAINED_MODEL_PATH"
    echo "将从头开始训练（不推荐）"
fi

# 启动训练
python3 2_sft.py \
    model.qwen3_model_path="$QWEN3_MODEL_PATH" \
    model.cache_path="$CACHE_PATH" \
    model.recompute_cache=False \
    dataset.sft_dataset_path="$SFT_DATASET_PATH" \
    dataset.pretrained_sft_model_path="$PRETRAINED_MODEL_PATH" \
    dataset.sft_val_dataset_path="$SFT_VAL_DATASET_PATH" \
    training.learning_rate="$LEARNING_RATE" \
    training.batch_size="$BATCH_SIZE" \
    training.accumulation_steps="$ACCUMULATION_STEPS" \
    training.epochs="$EPOCHS" \
    model.max_seq_len="$MAX_SEQ_LEN"

echo ""
echo "=========================================="
echo "✅ 训练完成"
echo "=========================================="

