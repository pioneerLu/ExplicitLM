#!/usr/bin/env python3
"""
基于 extract_support 数据生成新的 Keys

功能：
1. 从 train_data_with_extract.json 提取所有 extract_support
2. 将其作为知识库，生成新的 Keys
3. 确保 Keys 和 Cache 基于相同的知识库
"""

import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
from pathlib import Path

# 检查依赖
try:
    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import MiniBatchKMeans
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ 缺少依赖包: {e}")
    print("\n请安装以下依赖:")
    print("  pip install torch sentence-transformers scikit-learn tqdm numpy")
    print("\n或者使用 uv:")
    print("  uv pip install torch sentence-transformers scikit-learn tqdm numpy")
    exit(1)


def extract_extract_support_as_kb(input_path: str, output_path: str):
    """
    从 train_data_with_extract.json 提取所有 extract_support 作为知识库
    
    Args:
        input_path: 输入的 JSON 文件路径
        output_path: 输出的知识库 JSON 文件路径
    """
    print(f"📖 读取输入文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 总样本数: {len(data)}")
    
    # 提取所有 extract_support
    kb_sentences = []
    for idx, item in enumerate(data):
        extract_support = item.get("extract_support", "").strip()
        if extract_support:  # 只保留非空的 extract_support
            kb_sentences.append({
                "sentence": extract_support,
                "index": idx,
                "question": item.get("question", ""),
                "correct_answer": item.get("correct_answer", "")
            })
    
    print(f"✅ 提取了 {len(kb_sentences)} 个非空的 extract_support")
    
    # 保存为知识库格式（兼容 convert_conversations_to_labeled.py）
    kb_data = []
    for item in kb_sentences:
        kb_data.append({
            "sentence": item["sentence"],
            "uuid": f"extract_{item['index']}",
            "subject": "",
            "predicate": "",
            "object": ""
        })
    
    # 确保数量是完全平方数（Product Key Memory 的要求）
    sqrt_num = int(np.sqrt(len(kb_data)))
    perfect_num = sqrt_num * sqrt_num
    
    if perfect_num != len(kb_data):
        print(f"📝 调整知识库大小: {len(kb_data)} -> {perfect_num} (完全平方数)")
        kb_data = kb_data[:perfect_num]
    
    print(f"💾 保存知识库: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kb_data, f, ensure_ascii=False, indent=2)
    
    print(f"  ✅ 已保存 {len(kb_data)} 个知识库条目")
    return len(kb_data)


def generate_keys_from_kb(
    kb_path: str,
    output_keys_path: str,
    model_name: str = "BAAI/bge-base-en-v1.5",
    local_model_path: str = None,
    device: str = None,
    batch_size: int = 32,
    cache_path: str = None,
    knowledge_num: int = None
):
    """
    基于知识库生成 Keys（只使用有效条目）
    
    Args:
        kb_path: 知识库 JSON 文件路径
        output_keys_path: 输出的 Keys 文件路径
        model_name: 嵌入模型名称（HuggingFace 名称）
        local_model_path: 本地模型路径（如果指定，优先使用本地模型）
        device: 设备（cuda/cpu）
        batch_size: 批处理大小
        cache_path: (可选) Cache 文件路径，用于加载 valid_mask
        knowledge_num: (可选) 知识库总大小，如果提供且 cache_path 存在，会使用 valid_mask 过滤
    """
    import torch
    
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    # 确定使用的模型路径
    if local_model_path and os.path.exists(local_model_path):
        model_path = local_model_path
        print(f"📦 使用本地嵌入模型: {model_path}")
    else:
        model_path = model_name
        print(f"📦 加载嵌入模型: {model_name}")
        if local_model_path:
            print(f"  ⚠️  本地模型路径不存在，将从 HuggingFace 下载")
    
    model = SentenceTransformer(model_path, device=device)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"  ✅ 嵌入维度: {embedding_dim}")
    
    # 加载知识库
    print(f"📖 加载知识库: {kb_path}")
    with open(kb_path, 'r', encoding='utf-8') as f:
        kb_data = json.load(f)
    
    sentences = [item["sentence"] for item in kb_data]
    print(f"  ✅ 加载了 {len(sentences)} 个句子")
    
    # 如果提供了 cache_path 和 knowledge_num，尝试加载 valid_mask 过滤空条目
    valid_indices = None
    if cache_path and knowledge_num and os.path.exists(cache_path):
        try:
            print(f"📖 从 cache 加载 valid_mask: {cache_path}")
            cache_data = torch.load(cache_path)
            if isinstance(cache_data, dict) and "valid_mask" in cache_data:
                valid_mask = cache_data["valid_mask"]
                # 只使用有效条目（前 num_valid 个）
                num_valid = valid_mask.sum().item()
                if num_valid < len(sentences):
                    valid_indices = list(range(num_valid))
                    sentences = sentences[:num_valid]
                    print(f"  ✅ 使用 valid_mask 过滤: {len(sentences)} 个有效条目（从 {len(kb_data)} 个中）")
                else:
                    print(f"  ⚠️  valid_mask 显示 {num_valid} 个有效条目，但知识库只有 {len(sentences)} 个，使用全部")
            else:
                print(f"  ⚠️  cache 文件格式不支持 valid_mask，使用全部条目")
        except Exception as e:
            print(f"  ⚠️  加载 valid_mask 失败: {e}，使用全部条目")
    
    # 编码知识库（只编码有效条目）
    print("🔨 编码知识库为嵌入向量...")
    kb_embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"  ✅ 嵌入向量形状: {kb_embeddings.shape} (只包含有效条目)")
    
    # Residual Quantization
    # 注意：keys 数量仍为 √knowledge_num（如果提供了），而不是 √有效条目数
    # 这样可以保持 Product Key 结构的一致性
    print("🔨 执行 Residual Quantization...")
    if knowledge_num is not None:
        num_clusters = int(np.sqrt(knowledge_num))
        print(f"  📊 聚类数: {num_clusters} (√{knowledge_num}，基于总知识库大小)")
        print(f"  📝 注意: 只使用 {len(kb_embeddings)} 个有效条目进行 K-Means，但 keys 数量匹配总知识库大小")
    else:
        num_items = len(kb_embeddings)
        num_clusters = int(np.sqrt(num_items))
        print(f"  📊 聚类数: {num_clusters} (√{num_items}，基于有效条目数)")
    
    # 步骤 1: 粗粒度聚类（Row Keys）- 只使用有效条目
    print("  📍 步骤 1: 粗粒度聚类 (Row Keys，仅使用有效条目)...")
    
    kmeans_coarse = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=10000,
        n_init='auto',
        random_state=42
    )
    kmeans_coarse.fit(kb_embeddings)
    row_keys = kmeans_coarse.cluster_centers_
    row_labels = kmeans_coarse.labels_
    print(f"    ✅ Row Keys 形状: {row_keys.shape}")
    
    # 步骤 2: 计算残差
    print("  📍 步骤 2: 计算残差...")
    residuals = kb_embeddings - row_keys[row_labels]
    
    # 步骤 3: 细粒度聚类（Col Keys）- 只使用有效条目
    print("  📍 步骤 3: 细粒度聚类 (Col Keys，仅使用有效条目)...")
    kmeans_fine = MiniBatchKMeans(
        n_clusters=num_clusters,
        batch_size=10000,
        n_init='auto',
        random_state=42
    )
    kmeans_fine.fit(residuals)
    col_keys = kmeans_fine.cluster_centers_
    col_labels = kmeans_fine.labels_
    print(f"    ✅ Col Keys 形状: {col_keys.shape}")
    
    # 步骤 4: 组合 Keys
    print("  📍 步骤 4: 组合 Keys...")
    keys_tensor = torch.stack([
        torch.tensor(row_keys, dtype=torch.float32),
        torch.tensor(col_keys, dtype=torch.float32)
    ], dim=0)
    print(f"    ✅ Keys 形状: {keys_tensor.shape}")
    
    # 步骤 5: 保存 Keys
    print(f"💾 保存 Keys: {output_keys_path}")
    os.makedirs(os.path.dirname(output_keys_path), exist_ok=True)
    torch.save(keys_tensor, output_keys_path)
    print(f"  ✅ Keys 已保存")
    
    return keys_tensor


def main():
    parser = argparse.ArgumentParser(
        description="基于 extract_support 数据生成新的 Keys"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="ExplicitLM/data/train_data_with_extract.json",
        help="输入的 JSON 文件路径（包含 extract_support）"
    )
    parser.add_argument(
        "--kb-output",
        type=str,
        default="ExplicitLM/data/knowledge_base/extract_support_kb.json",
        help="输出的知识库 JSON 文件路径"
    )
    parser.add_argument(
        "--keys-output",
        type=str,
        default="ExplicitLM/data/keys_extract.pt",
        help="输出的 Keys 文件路径"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="嵌入模型名称（HuggingFace 名称）"
    )
    parser.add_argument(
        "--local-model-path",
        type=str,
        default=None,
        help="本地模型路径（如果指定，优先使用本地模型）"
    )
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="下载模型到本地（需要指定 --local-model-path）"
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
        help="批处理大小"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Cache 文件路径（可选，用于加载 valid_mask 过滤空条目）"
    )
    parser.add_argument(
        "--knowledge-num",
        type=int,
        default=None,
        help="知识库总大小（如果提供且 cache_path 存在，会使用 valid_mask 过滤）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 基于 extract_support 生成 Keys")
    print("=" * 60)
    print()
    
    # 步骤 0: 下载模型（如果需要）
    if args.download_model:
        if not args.local_model_path:
            print("❌ 错误: 使用 --download-model 时必须指定 --local-model_path")
            exit(1)
        print("步骤 0: 下载模型到本地")
        print("-" * 60)
        print("  ⚠️  下载模型功能暂未实现，请手动下载模型到指定路径")
        print(f"  模型名称: {args.model_name}")
        print(f"  目标路径: {args.local_model_path}")
        print()
    
    # 步骤 1: 提取 extract_support 作为知识库
    print("步骤 1: 提取 extract_support 作为知识库")
    print("-" * 60)
    kb_size = extract_extract_support_as_kb(args.input, args.kb_output)
    print()
    
    # 步骤 2: 生成 Keys
    print("步骤 2: 基于知识库生成 Keys")
    print("-" * 60)
    keys_tensor = generate_keys_from_kb(
        kb_path=args.kb_output,
        output_keys_path=args.keys_output,
        model_name=args.model_name,
        local_model_path=args.local_model_path,
        device=args.device,
        batch_size=args.batch_size,
        cache_path=args.cache_path,
        knowledge_num=args.knowledge_num
    )
    print()
    
    print("=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print(f"知识库: {args.kb_output} ({kb_size} 个条目)")
    print(f"Keys: {args.keys_output} (形状: {keys_tensor.shape})")
    print()
    print("📝 下一步:")
    print(f"  1. 更新 run_sft.sh 中的 keys_path 为: {args.keys_output}")
    print(f"  2. 确保 Keys 和 Cache 基于相同的知识库")
    print("=" * 60)


if __name__ == "__main__":
    main()
