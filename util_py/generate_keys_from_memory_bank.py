#!/usr/bin/env python3
"""
从 Memory Bank Batch 文件生成 Keys

功能：
1. 从 memory_bank_batch_*.pt 文件加载 memory_bank 和 valid_mask
2. 直接使用 Qwen 的 embedding 层编码 token IDs（无需 detokenize）
3. 生成 Product Key Memory 的 keys（维度对齐到 Qwen 的 hidden_size）
"""

import os
import argparse
from pathlib import Path
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import sys

# 添加项目根目录到路径，以便导入 utils.clustering
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.clustering import perform_clustering

# 检查依赖
try:
    import torch
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ 缺少依赖包: {e}")
    print("\n请安装以下依赖:")
    print("  pip install torch tqdm numpy transformers")
    exit(1)


def load_memory_bank_batch(batch_path: str):
    """加载 memory bank batch 文件"""
    print(f"📖 加载 Memory Bank Batch: {batch_path}")
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"文件不存在: {batch_path}")
    
    data = torch.load(batch_path, map_location='cpu')
    
    if isinstance(data, dict):
        memory_bank = data.get("memory_bank")
        valid_mask = data.get("valid_mask")
    else:
        # 如果直接是tensor，假设是memory_bank
        memory_bank = data
        valid_mask = None
    
    if memory_bank is None:
        raise ValueError("无法从文件中找到 memory_bank")
    
    print(f"  ✅ Memory Bank 形状: {memory_bank.shape}")
    
    if valid_mask is not None:
        num_valid = valid_mask.sum().item()
        print(f"  ✅ Valid Mask: {num_valid}/{len(valid_mask)} 个有效条目")
    else:
        # 如果没有valid_mask，检查是否有全pad条目
        pad_token_id = 0
        is_all_pad = (memory_bank == pad_token_id).all(dim=-1)
        num_valid = (~is_all_pad).sum().item()
        print(f"  ⚠️  未找到 valid_mask，通过检查全pad条目: {num_valid}/{len(memory_bank)} 个有效条目")
        valid_mask = ~is_all_pad
    
    return memory_bank, valid_mask


def generate_keys_from_token_ids(
    memory_bank: torch.Tensor,
    valid_mask: torch.Tensor,
    qwen_model_path: str,
    output_keys_path: str,
    device: str = None,
    batch_size: int = 32,
    knowledge_num: int = None
):
    """基于 Token IDs 使用 Qwen embedding 层生成 Keys"""
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"📦 加载 Qwen 模型: {qwen_model_path}")
    
    # 加载 Qwen 模型（只加载 embedding 层，不需要完整模型）
    # 使用低精度以节省显存
    model = AutoModelForCausalLM.from_pretrained(
        qwen_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # 使用 float32 确保精度
        device_map="auto" if device == "cuda" else None,
    )
    
    # 获取 embedding 层
    if hasattr(model, 'get_input_embeddings'):
        embedding_layer = model.get_input_embeddings()
    else:
        embedding_layer = getattr(model, 'embed_tokens', None)
        if embedding_layer is None:
            raise ValueError("无法找到模型的 embedding 层")
    
    # 获取 hidden_size（embedding 维度）
    embedding_dim = embedding_layer.embedding_dim
    print(f"  ✅ 嵌入维度: {embedding_dim}")
    
    # 将 embedding 层移到指定设备（确保在正确的设备上）
    embedding_layer = embedding_layer.to(device)
    
    # 只处理有效条目
    valid_indices = torch.where(valid_mask)[0].tolist()
    num_valid = len(valid_indices)
    print(f"  📝 处理 {num_valid} 个有效条目...")
    
    # 编码 token IDs 为嵌入向量（批处理）
    print("🔨 使用 Qwen embedding 层编码 Token IDs...")
    kb_embeddings_list = []
    
    embedding_layer.eval()
    with torch.no_grad():
        for i in tqdm(range(0, num_valid, batch_size), desc="编码"):
            batch_indices = valid_indices[i:i+batch_size]
            batch_token_ids = memory_bank[batch_indices].to(device)  # [batch_size, knowledge_length]
            
            # 创建 attention_mask（非 pad token 的位置为 1）
            pad_token_id = 0
            attention_mask = (batch_token_ids != pad_token_id).long().to(device)
            
            # 获取 token embeddings
            token_embeddings = embedding_layer(batch_token_ids)  # [batch_size, knowledge_length, embedding_dim]
            
            # Mean pooling（考虑 attention_mask）
            mask = attention_mask.unsqueeze(-1).float()  # [batch_size, knowledge_length, 1]
            sum_hidden = (token_embeddings * mask).sum(dim=1)  # [batch_size, embedding_dim]
            len_hidden = mask.sum(dim=1).clamp(min=1e-6)  # [batch_size, 1]
            sentence_embeddings = sum_hidden / len_hidden  # [batch_size, embedding_dim]
            
            kb_embeddings_list.append(sentence_embeddings.cpu())
    
    # 合并所有批次的 embeddings
    kb_embeddings = torch.cat(kb_embeddings_list, dim=0)  # [num_valid, embedding_dim]
    print(f"  ✅ 嵌入向量形状: {kb_embeddings.shape}")
    
    # 移到指定设备
    kb_embeddings = kb_embeddings.to(device)
    
    # Residual Quantization（使用新的 FAISS 聚类方法）
    print("🔨 执行 Residual Quantization（使用 FAISS，支持 GPU）...")
    if knowledge_num is not None:
        num_clusters = int(np.sqrt(knowledge_num))
        print(f"  📊 聚类数: {num_clusters} (√{knowledge_num}，基于总知识库大小)")
        print(f"  📝 注意: 只使用 {len(kb_embeddings)} 个有效条目进行 K-Means，但 keys 数量匹配总知识库大小")
    else:
        num_items = len(kb_embeddings)
        num_clusters = int(np.sqrt(num_items))
        print(f"  📊 聚类数: {num_clusters} (√{num_items}，基于有效条目数)")
    
    # 使用新的聚类方法（FAISS，支持 GPU，内存优化）
    row_keys, col_keys, grid_indices = perform_clustering(
        kb_embeddings,
        num_clusters
    )
    
    print(f"    ✅ Row Keys 形状: {row_keys.shape}")
    print(f"    ✅ Col Keys 形状: {col_keys.shape}")
    print(f"    ✅ Grid Indices 形状: {grid_indices.shape}")
    
    # 保存 Keys（新格式：字典格式，包含 row_keys 和 col_keys）
    print(f"💾 保存 Keys（新格式）: {output_keys_path}")
    os.makedirs(os.path.dirname(output_keys_path), exist_ok=True)
    
    # 保存为字典格式，包含 row_keys 和 col_keys（两个独立的 tensor）
    keys_dict = {
        "row_keys": row_keys.cpu(),
        "col_keys": col_keys.cpu(),
        "format": "v2",  # 标记为新格式
        "num_clusters": num_clusters,
        "embedding_dim": embedding_dim,
        "source": "qwen_embedding",  # 标记使用 Qwen embedding 生成
    }
    torch.save(keys_dict, output_keys_path)
    print(f"  ✅ Keys 已保存（新格式：字典格式，包含 row_keys 和 col_keys）")
    
    # 清理模型以释放显存
    del model
    del embedding_layer
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 返回兼容格式（用于向后兼容）
    keys_tensor = torch.stack([row_keys.cpu(), col_keys.cpu()], dim=0)
    return keys_tensor


def main():
    parser = argparse.ArgumentParser(
        description="从 Memory Bank Batch 文件生成 Keys"
    )
    parser.add_argument(
        "--memory-bank-path",
        type=str,
        required=True,
        help="Memory Bank Batch 文件路径（.pt文件）"
    )
    parser.add_argument(
        "--output-keys-path",
        type=str,
        required=True,
        help="输出的 Keys 文件路径"
    )
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default="/data2/zengzheni/lvchangwei/new_repo/Qwen/models/Qwen3-4b",
        help="Qwen 模型路径（用于 tokenizer）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu，默认自动选择）"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="嵌入编码的批处理大小（使用 Qwen embedding 层）"
    )
    parser.add_argument(
        "--knowledge-num",
        type=int,
        default=None,
        help="知识库总大小（用于确定 keys 数量，默认从 memory_bank 形状推断）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 从 Memory Bank Batch 生成 Keys")
    print("=" * 60)
    print()
    
    # 步骤 1: 加载 Memory Bank
    print("步骤 1: 加载 Memory Bank Batch")
    print("-" * 60)
    memory_bank, valid_mask = load_memory_bank_batch(args.memory_bank_path)
    
    # 推断 knowledge_num（如果未提供）
    if args.knowledge_num is None:
        args.knowledge_num = memory_bank.shape[0]
        print(f"  📝 从 memory_bank 形状推断 knowledge_num: {args.knowledge_num}")
    print()
    
    # 步骤 2: 使用 Qwen embedding 层生成 Keys（直接使用 Token IDs，无需 detokenize）
    print("步骤 2: 使用 Qwen embedding 层生成 Keys")
    print("-" * 60)
    keys_tensor = generate_keys_from_token_ids(
        memory_bank,
        valid_mask,
        args.qwen_model_path,
        args.output_keys_path,
        device=args.device,
        batch_size=args.batch_size,
        knowledge_num=args.knowledge_num
    )
    print()
    
    print("=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print(f"Memory Bank: {args.memory_bank_path}")
    print(f"Keys: {args.output_keys_path} (形状: {keys_tensor.shape})")
    print()
    print("📝 下一步:")
    print(f"  更新 run_sft.sh 中的 keys_path 为: {args.output_keys_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

