#!/usr/bin/env python3
"""
将 train_data_with_extract.json 转换为 SFT 训练格式

功能：
1. 读取 JSON 数组格式的数据
2. 为每个样本生成 UUID
3. 转换为对话格式的 JSONL
4. 将 extract_support 转换为 token IDs，生成 knowledge_cache.pt
5. 8:2 划分训练/验证集
6. 保存 UUID 映射文件

输出文件：
- sft_data/train_data_with_extract_sft_train.jsonl (训练集)
- sft_data/train_data_with_extract_sft_val.jsonl (验证集)
- data/cache/train_data_with_extract_cache.pt (记忆库缓存)
- data/cache/train_data_with_extract_cache_mapping.json (UUID映射)
"""

import json
import os
import argparse
import uuid
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
import torch
from transformers import AutoTokenizer
from tqdm import tqdm


def load_qwen_tokenizer(qwen_model_path: str) -> AutoTokenizer:
    """加载 Qwen tokenizer"""
    print(f"📦 加载 Qwen tokenizer: {qwen_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)
    
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("  ⚠️  Qwen tokenizer 没有 pad_token，使用 eos_token 作为 pad_token")
    
    print(f"  ✅ Tokenizer 加载完成")
    print(f"  - vocab_size: {tokenizer.vocab_size}")
    print(f"  - pad_token_id: {tokenizer.pad_token_id}")
    return tokenizer


def convert_to_conversation_format(
    item: Dict[str, Any],
    sample_uuid: str
) -> Dict[str, Any]:
    """
    将单个样本转换为对话格式
    
    Args:
        item: 原始数据项
        sample_uuid: 样本 UUID
        
    Returns:
        对话格式的数据项
    """
    question = item.get("question", "").strip()
    correct_answer = item.get("correct_answer", "").strip()
    extract_support = item.get("extract_support", "").strip()
    support = item.get("support", "").strip()
    
    # 验证必要字段
    if not question or not correct_answer:
        return None
    
    # 构建对话格式
    conversation_item = {
        "conversations": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": correct_answer}
        ],
        "uuid": sample_uuid,
        "extract_support": extract_support,
        "support": support
    }
    
    return conversation_item


def tokenize_extract_support(
    extract_support: str,
    tokenizer: AutoTokenizer,
    knowledge_length: int = 32
) -> List[int]:
    """
    将 extract_support 文本转换为 token IDs
    
    Args:
        extract_support: 提取的支持文本
        tokenizer: Qwen tokenizer
        knowledge_length: 目标长度（token 数）
        
    Returns:
        token IDs 列表
    """
    if not extract_support or not extract_support.strip():
        # 空文本，返回全 padding
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        return [pad_token_id] * knowledge_length
    
    # Tokenize（不添加特殊 token，只编码文本）
    tokens_result = tokenizer(
        extract_support,
        add_special_tokens=False,
        truncation=True,
        max_length=knowledge_length,
        padding=False,
        return_tensors="pt",
    )
    
    tokens = tokens_result["input_ids"].squeeze().tolist()
    if not isinstance(tokens, list):
        tokens = [tokens]
    
    # 填充或截断到 knowledge_length
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    if len(tokens) > knowledge_length:
        tokens = tokens[:knowledge_length]
    elif len(tokens) < knowledge_length:
        tokens.extend([pad_token_id] * (knowledge_length - len(tokens)))
    
    return tokens


def process_data(
    input_path: str,
    tokenizer: AutoTokenizer,
    knowledge_num: int = 1048576,
    knowledge_length: int = 32,
    train_ratio: float = 0.8
) -> Tuple[List[Dict], List[Dict], torch.Tensor, torch.Tensor, List[Dict]]:
    """
    处理数据并生成训练数据、cache 和映射
    
    Args:
        input_path: 输入 JSON 文件路径
        tokenizer: Qwen tokenizer
        knowledge_num: 记忆库条目数
        knowledge_length: 每个条目的 token 数
        train_ratio: 训练集比例
        
    Returns:
        (train_conversations, val_conversations, cache_tensor, valid_mask, mapping_list)
    """
    print(f"📖 读取输入文件: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 总样本数: {len(data)}")
    
    conversations_list = []
    cache_rows = []
    mapping_list = []
    
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    # 处理每个样本
    for idx, item in enumerate(tqdm(data, desc="处理样本")):
        # 生成 UUID
        sample_uuid = str(uuid.uuid4())
        
        # 转换为对话格式
        conversation_item = convert_to_conversation_format(item, sample_uuid)
        if conversation_item is None:
            print(f"  ⚠️  跳过样本 {idx}：缺少必要字段")
            continue
        
        conversations_list.append(conversation_item)
        
        # Tokenize extract_support
        extract_support = item.get("extract_support", "").strip()
        tokens = tokenize_extract_support(extract_support, tokenizer, knowledge_length)
        cache_rows.append(tokens)
        
        # 保存映射信息
        mapping_list.append({
            "database_index": idx,
            "uuid": sample_uuid,
            "question": item.get("question", ""),
            "correct_answer": item.get("correct_answer", ""),
            "extract_support": extract_support,
            "token_count": len([t for t in tokens if t != pad_token_id]),
            "is_truncated": len(extract_support) > 0 and len(tokens) == knowledge_length
        })
    
    print(f"✅ 成功处理 {len(conversations_list)} 个样本")
    
    # 生成 cache tensor
    print(f"🔨 生成记忆库缓存 (knowledge_num={knowledge_num}, knowledge_length={knowledge_length})")
    
    # 转换为 tensor
    if cache_rows:
        cache_tensor = torch.tensor(cache_rows, dtype=torch.long)
    else:
        cache_tensor = torch.zeros((0, knowledge_length), dtype=torch.long)
    
    # 记录有效条目数（填充前）
    num_valid = len(cache_rows)
    
    # 填充到 knowledge_num
    if len(cache_rows) < knowledge_num:
        padding_rows = [[pad_token_id] * knowledge_length] * (knowledge_num - len(cache_rows))
        padding_tensor = torch.tensor(padding_rows, dtype=torch.long)
        cache_tensor = torch.cat([cache_tensor, padding_tensor], dim=0)
        print(f"  📝 添加 {knowledge_num - len(cache_rows)} 个 padding 条目")
    
    # 截断到 knowledge_num（如果超过）
    if len(cache_rows) > knowledge_num:
        cache_tensor = cache_tensor[:knowledge_num]
        num_valid = knowledge_num
        print(f"  ✂️  截断到 {knowledge_num} 个条目")
    
    # 生成 valid_mask：前 num_valid 个条目是有效的
    valid_mask = torch.zeros(knowledge_num, dtype=torch.bool)
    valid_mask[:num_valid] = True
    
    print(f"  ✅ Cache tensor 形状: {cache_tensor.shape}")
    print(f"  ✅ Valid mask 形状: {valid_mask.shape}, 有效条目: {num_valid}/{knowledge_num} ({num_valid/knowledge_num*100:.2f}%)")
    
    # 划分训练/验证集
    print(f"📊 划分数据集 (训练集: {train_ratio*100:.1f}%, 验证集: {(1-train_ratio)*100:.1f}%)")
    split_idx = int(len(conversations_list) * train_ratio)
    train_conversations = conversations_list[:split_idx]
    val_conversations = conversations_list[split_idx:]
    
    print(f"  - 训练集: {len(train_conversations)} 个样本")
    print(f"  - 验证集: {len(val_conversations)} 个样本")
    
    return train_conversations, val_conversations, cache_tensor, valid_mask, mapping_list


def save_outputs(
    train_conversations: List[Dict],
    val_conversations: List[Dict],
    cache_tensor: torch.Tensor,
    valid_mask: torch.Tensor,
    mapping_list: List[Dict],
    output_dir: Path,
    cache_path: str,
    mapping_path: str
):
    """保存输出文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_path).parent
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存训练集
    train_path = output_dir / "train_data_with_extract_sft_train.jsonl"
    print(f"💾 保存训练集: {train_path}")
    with open(train_path, 'w', encoding='utf-8') as f:
        for item in train_conversations:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  ✅ 已保存 {len(train_conversations)} 个训练样本")
    
    # 保存验证集
    val_path = output_dir / "train_data_with_extract_sft_val.jsonl"
    print(f"💾 保存验证集: {val_path}")
    with open(val_path, 'w', encoding='utf-8') as f:
        for item in val_conversations:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"  ✅ 已保存 {len(val_conversations)} 个验证样本")
    
    # 保存 cache（包含 memory_bank 和 valid_mask）
    print(f"💾 保存记忆库缓存: {cache_path}")
    cache_data = {
        "memory_bank": cache_tensor,
        "valid_mask": valid_mask,
    }
    torch.save(cache_data, cache_path)
    print(f"  ✅ Cache 已保存:")
    print(f"     - memory_bank 形状: {cache_tensor.shape}")
    print(f"     - valid_mask 形状: {valid_mask.shape}, 有效条目: {valid_mask.sum().item()}")
    
    # 保存映射文件
    print(f"💾 保存 UUID 映射: {mapping_path}")
    mapping_data = {
        "metadata": {
            "total_samples": len(mapping_list),
            "knowledge_num": cache_tensor.shape[0],
            "knowledge_length": cache_tensor.shape[1],
            "train_samples": len(train_conversations),
            "val_samples": len(val_conversations),
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "mappings": mapping_list,
    }
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ UUID 映射已保存: {len(mapping_list)} 个映射")


def main():
    parser = argparse.ArgumentParser(
        description="将 train_data_with_extract.json 转换为 SFT 训练格式"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="ExplicitLM/data/train_data_with_extract.json",
        help="输入的 JSON 文件路径"
    )
    parser.add_argument(
        "--qwen-model-path",
        type=str,
        default="Qwen_hg/Qwen3-4b",  # 相对于 ExplicitLM 根目录
        help="Qwen3 模型路径（用于加载 tokenizer）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ExplicitLM/sft_data",
        help="输出目录（训练/验证 JSONL）"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="ExplicitLM/data/cache/train_data_with_extract_cache.pt",
        help="记忆库缓存文件路径"
    )
    parser.add_argument(
        "--mapping-path",
        type=str,
        default="ExplicitLM/data/cache/train_data_with_extract_cache_mapping.json",
        help="UUID 映射文件路径"
    )
    parser.add_argument(
        "--knowledge-num",
        type=int,
        default=1048576,
        help="记忆库条目数（默认: 1048576）"
    )
    parser.add_argument(
        "--knowledge-length",
        type=int,
        default=32,
        help="每个条目的 token 数（默认: 32）"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="训练集比例（默认: 0.8）"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 开始转换数据")
    print("=" * 60)
    print(f"输入文件: {args.input}")
    print(f"Qwen 模型路径: {args.qwen_model_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"Cache 路径: {args.cache_path}")
    print(f"知识库配置: {args.knowledge_num} x {args.knowledge_length}")
    print(f"训练/验证划分: {args.train_ratio:.1%} / {1-args.train_ratio:.1%}")
    print("=" * 60)
    print()
    
    # 加载 tokenizer
    tokenizer = load_qwen_tokenizer(args.qwen_model_path)
    print()
    
    # 处理数据
    train_conversations, val_conversations, cache_tensor, valid_mask, mapping_list = process_data(
        input_path=args.input,
        tokenizer=tokenizer,
        knowledge_num=args.knowledge_num,
        knowledge_length=args.knowledge_length,
        train_ratio=args.train_ratio
    )
    print()
    
    # 保存输出
    save_outputs(
        train_conversations=train_conversations,
        val_conversations=val_conversations,
        cache_tensor=cache_tensor,
        valid_mask=valid_mask,
        mapping_list=mapping_list,
        output_dir=Path(args.output_dir),
        cache_path=args.cache_path,
        mapping_path=args.mapping_path
    )
    print()
    
    print("=" * 60)
    print("✅ 转换完成！")
    print("=" * 60)
    print(f"训练集: {Path(args.output_dir) / 'train_data_with_extract_sft_train.jsonl'}")
    print(f"验证集: {Path(args.output_dir) / 'train_data_with_extract_sft_val.jsonl'}")
    print(f"Cache: {args.cache_path}")
    print(f"映射: {args.mapping_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
