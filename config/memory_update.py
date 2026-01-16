# Memory Bank 更新配置（普通字典）
MemoryUpdateConf = {
    # ---- 知识更新开关 ----
    "enable_memory_update_during_training": True,  # 是否启用训练时知识更新
    
    # ---- 知识更新参数 ----
    "memory_update_frequency": 1,  # 每N个step更新一次
    "memory_update_strategy": "lru",  # 更新策略：fifo, lru, random, importance
    "memory_compression_rate": 0.4,  # 事实压缩率（0-1，越小保留信息越多）
    
    # ---- LLMLingua 配置 ----
    "llmlingua_model_path": "llmlingua-2-bert",  
    
}

