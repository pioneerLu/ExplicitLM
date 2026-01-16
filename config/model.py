# 模型配置（普通字典）
ModelConf = {
    # 模型类型（基于 Qwen3-4B，架构参数从 qwen3_model_path 加载）
    "model_type": "ExplicitLM",
    "model_variant": "model_memory",
    
    # Qwen3 模型路径（必需，用于加载预训练权重和配置）
    "qwen3_model_path": "",  # 必须通过命令行或配置文件指定
    
    # 序列长度
    "max_seq_len": 256,  # 最大序列长度
    
    # Memory / Knowledge 配置
    "use_token_memory": True,
    "knowledge_length": 32,  # 每个记忆条目的 token 数
    "knowledge_num": 100*100,  # 10000 个记忆条目
    "cache_path": "data/cache/knowledge_cache.pt",  # 记忆库路径（支持 .pt 文件直接加载或 .json 文件处理后保存）
    "recompute_cache": False,  # 是否重新计算缓存（仅对 .json 文件有效）
    "disable_db": False,  # 是否禁用数据库功能
    
    # MemoryGate 配置（新版本，所有层共享）
    "query_dim": 1024,  # Query 投影维度
    "key_proj_dim": 512,  # Key 投影维度（用于 dot product）
    "temperature": 0.1,  # Temperature for softmax in loss computation
    
    # MoE 配置（当前未使用）
    "use_moe": False,
    "n_routed_experts": 4,
    "n_shared_experts": True,
    "num_experts_per_tok": 2,
    "aux_loss_alpha": 0.1,
    "gumbel_temperature": 1.0,
    "norm_topk_prob": True,
    "scoring_func": "softmax",
    
    "contrastive_temperature": 0.07,  # InfoNCE 损失的温度参数，控制正负样本的区分度
                                      # 较小的值（如0.07）会使模型更关注高相似度的记忆
                                      # 较大的值会使分布更平滑

    # 记忆残差缩放（对 memory_output 的额外0-1权重，默认关闭记忆）
    "memory_residual_scale_init": 0.0,         # 有效权重初值（0-1），默认近似0
    "memory_residual_scale_trainable": True,   # 是否训练该缩放参数
}