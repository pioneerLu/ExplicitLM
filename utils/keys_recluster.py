"""
Keys 重新聚类工具函数

用于在 memory_bank 更新后重新聚类 keys，以反映新的数据分布。
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from utils.clustering import perform_clustering
from utils.logger import Logger


def get_embedding_layer(model: nn.Module) -> nn.Module:
    """
    获取模型的 embedding 层
    
    Args:
        model: ExplicitLM 模型实例
    
    Returns:
        embedding 层
    """
    # 方法1: 通过 get_input_embeddings()
    if hasattr(model, 'get_input_embeddings'):
        embedding_layer = model.get_input_embeddings()
        if embedding_layer is not None:
            return embedding_layer
    
    # 方法2: 直接访问 embed_tokens
    if hasattr(model, 'embed_tokens'):
        return model.embed_tokens
    
    # 方法3: 从 Qwen3Model 中获取
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens
    
    raise ValueError("无法找到模型的 embedding 层")


def recluster_keys(
    model: nn.Module,
    memory_bank: torch.Tensor,
    valid_mask: torch.Tensor,
    num_keys: int,
    device: str = "cuda",
    batch_size: int = 32,
    sample_ratio: float = 1.0,
    accelerator=None,
) -> Dict[str, any]:
    """
    重新聚类 keys（同步版本，会阻塞训练）
    
    流程：
    1. 获取所有有效条目（valid_mask=True）
    2. 可选：采样部分条目（如果 sample_ratio < 1.0）
    3. 使用 Qwen embedding 层将 token IDs 转换为 embeddings
    4. 对每个序列进行 mean pooling（考虑 attention_mask）
    5. 使用 Product Key Memory 聚类（row + col）
    6. 更新 MemoryGate 的 keys
    
    Args:
        model: ExplicitLM 模型实例
        memory_bank: memory_bank tensor [knowledge_num, seq_len]
        valid_mask: valid_mask tensor [knowledge_num]
        num_keys: 聚类中心数量（√knowledge_num）
        device: 运行设备
        batch_size: 批量大小（embeddings转换）
        sample_ratio: 采样比例（0-1），只对部分条目聚类以加速（1.0=全部）
        accelerator: Accelerator 实例（用于日志）
    
    Returns:
        聚类统计信息
    """
    # 1. 获取有效条目
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    if len(valid_indices) == 0:
        Logger("警告: 没有有效条目，跳过 keys 重新聚类", accelerator)
        return {"error": "no_valid_entries"}
    
    num_valid = len(valid_indices)
    Logger(f"开始重新聚类 keys: {num_valid} 个有效条目，目标聚类数: {num_keys}", accelerator)
    
    # 2. 采样（如果 sample_ratio < 1.0）
    if sample_ratio < 1.0:
        num_samples = int(num_valid * sample_ratio)
        sampled_indices = torch.randperm(num_valid, generator=None)[:num_samples]
        sampled_valid_indices = valid_indices[sampled_indices]
        Logger(f"采样 {num_samples}/{num_valid} 个条目进行聚类 (采样比例: {sample_ratio})", accelerator)
    else:
        sampled_valid_indices = valid_indices
        num_samples = num_valid
    
    valid_memory_bank = memory_bank[sampled_valid_indices]  # [num_samples, seq_len]
    
    # 3. 获取 embedding 层
    embedding_layer = get_embedding_layer(model)
    embedding_layer = embedding_layer.to(device)
    embedding_layer.eval()
    
    # 4. 批量处理，将 token IDs 转换为 embeddings
    embeddings_list = []
    pad_token_id = 0  # 假设 pad_token_id 为 0
    
    Logger(f"开始转换 token IDs 为 embeddings (批量大小: {batch_size})...", accelerator)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_token_ids = valid_memory_bank[i:end_idx].to(device)  # [batch, seq_len]
            
            # 创建 attention_mask（非 pad token 的位置为 1）
            attention_mask = (batch_token_ids != pad_token_id).long().to(device)
            
            # 获取 token embeddings
            token_embeddings = embedding_layer(batch_token_ids)  # [batch, seq_len, hidden_size]
            
            # Mean pooling（考虑 attention_mask）
            mask = attention_mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            sum_hidden = (token_embeddings * mask).sum(dim=1)  # [batch, hidden_size]
            len_hidden = mask.sum(dim=1).clamp(min=1e-6)  # [batch, 1]
            sentence_embeddings = sum_hidden / len_hidden  # [batch, hidden_size]
            
            embeddings_list.append(sentence_embeddings.cpu())
    
    # 合并所有批次的 embeddings
    all_embeddings = torch.cat(embeddings_list, dim=0)  # [num_samples, hidden_size]
    all_embeddings = all_embeddings.to(device)
    
    Logger(f"嵌入向量准备完成: {all_embeddings.shape}", accelerator)
    
    # 5. 聚类（Product Key Memory）
    Logger("开始 Product Key Memory 聚类...", accelerator)
    row_keys, col_keys, grid_indices = perform_clustering(
        all_embeddings,
        num_clusters=num_keys
    )
    
    Logger(f"聚类完成: row_keys={row_keys.shape}, col_keys={col_keys.shape}", accelerator)
    
    # 6. 更新 MemoryGate 的 keys
    shared_memory_gate = model.shared_memory_gate
    shared_memory_gate.update_keys(row_keys, col_keys)
    
    # 清理
    del all_embeddings, embeddings_list
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "num_valid_entries": num_valid,
        "num_sampled_entries": num_samples,
        "sample_ratio": sample_ratio,
        "row_keys_shape": row_keys.shape,
        "col_keys_shape": col_keys.shape,
        "num_clusters": num_keys,
    }

