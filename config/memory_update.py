# Memory Bank 更新配置（普通字典）
MemoryUpdateConf = {
    # ---- 知识更新开关 ----
    "enable_memory_update_during_training": False,  # 是否启用训练时知识更新
    
    # ---- 知识更新参数 ----
    "memory_update_frequency": 1,  # 每N个step更新一次
    "memory_update_strategy": "lru",  # 更新策略：fifo, lru, random, importance
    "memory_compression_rate": 0.4,  # 事实压缩率（0-1，越小保留信息越多）
    
    # ---- LLMLingua 配置 ----
    "llmlingua_model_path": "llmlingua-2-bert",  
    
    # ---- Keys 重新聚类参数 ----
    "keys_recluster_update_ratio_threshold": 0.01,  
    "keys_recluster_on_epoch": False,  # 是否在每个epoch结束时重新聚类
    "keys_recluster_batch_size": 32,  # 重新聚类时的批量大小（embeddings转换）
    
    # ---- 性能优化参数 ----
    "keys_recluster_async": True,  # 是否异步进行keys重新聚类（后台线程，不阻塞训练，推荐True）
    "keys_recluster_sample_ratio": 0.01, 
    
    # ---- 分布式训练参数 ----
    "keys_recluster_on_main_process_only": True,  # 是否只在主进程进行keys重新聚类（推荐True，然后同步到其他进程）
}

