#!/bin/bash
################################################################################
# 实验001 - 集群模式统一脚本
#
# 用法：
#   ./exp_001_cluster.sh pre    # 登陆节点：前置工作（数据同步）
#   ./exp_001_cluster.sh train  # 计算节点：训练
#   ./exp_001_cluster.sh post   # 登陆节点：后续工作（Git提交）
################################################################################

# ============================================================================
# 实验配置（只需在这里修改一次）
# ============================================================================
EXP_ID="exp_002_16_layer"
EXP_DESC="模型层数从8层增加到16层实验"

# 数据版本
DATASET_VERSION="" # 训练数据集版本 (对应 dataset_path: data/database/merged_pretrain.jsonl)
VAL_DATASET_VERSION="" # 验证数据集版本 (对应 val_dataset_path: data/benchmarks/eval_data.json)
EMBEDDING_VERSION="" # 预训练嵌入权重版本 (对应 pretrained_embedding_path，可选)
DATABASE_VERSION="" # 知识库初始化数据版本 (对应 database_init_path，可选)
CACHE_VERSION="" # 缓存数据版本 (对应 cache_path: cache/knowledge_cache.pt，可选)

# 训练参数
TRAIN_ARGS="training.epochs=3 model.dim=768 model.n_layers=8 training.batch_size=16 logging.swanlab_online=False"

# ============================================================================
# 执行阶段选择
# ============================================================================
STAGE="${1:-train}"  # 默认为train阶段

case "$STAGE" in
    pre)
        echo "执行前置阶段（登陆节点）..."
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        source "${SCRIPT_DIR}/_run_experiment_cluster_pre.sh"
        ;;
    train)
        echo "执行训练阶段（计算节点）..."
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        source "${SCRIPT_DIR}/_run_experiment_cluster_train.sh"
        ;;
    post)
        echo "执行后续阶段（登陆节点）..."
        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        source "${SCRIPT_DIR}/_run_experiment_cluster_post.sh"
        ;;
    *)
        echo "错误：未知阶段 '$STAGE'"
        echo ""
        echo "用法："
        echo "  $0 pre    - 登陆节点：数据同步和准备"
        echo "  $0 train  - 计算节点：执行训练"
        echo "  $0 post   - 登陆节点：DVC追踪和Git提交"
        exit 1
        ;;
esac
