#!/usr/bin/env python3
"""
从Parquet文件中提取知识并生成Memory Bank缓存

流程：
1. 并行提取：4进程/4卡，从parquet文件中提取facts
2. 合并去重：合并所有facts，去重，采样到1,048,576条
3. 分批保存：每满1,048,576条保存一次（tokenize在合并时进行）
"""
import os
import sys
import json
import time
import argparse
import hashlib
import random
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import glob

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.fact_extractor import FactExtractor
from utils.logger import Logger

try:
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    print("警告: pyarrow 未安装，请先安装: pip install pyarrow")


# ============================================================================
# 配置
# ============================================================================
DEFAULT_CONFIG = {
    "knowledge_num": 1024 * 1024,  # 1,048,576
    "knowledge_length": 32,
    "compression_rate": 0.4,
    "num_workers": 4,
    "batch_size": 1024 * 1024,  # 每批保存的条数
    "min_fact_length": 10,  # 最小fact长度（字符）
    "deduplication": "text_hash",  # "text_hash" 或 "token_hash"
}


# ============================================================================
# 阶段1：并行提取
# ============================================================================
def extract_facts_from_parquet_file(
    parquet_path: str,
    output_path: str,
    fact_extractor: Optional[FactExtractor],
    compression_rate: float = 0.4,
    min_fact_length: int = 10,
    skip_llmlingua: bool = False,
    max_text_length: int = 2000,
) -> Dict[str, Any]:
    """
    从单个parquet文件中提取facts
    
    Returns:
        统计信息：{"total_texts": int, "total_facts": int, "errors": int}
    """
    import re
    
    stats = {
        "total_texts": 0,
        "total_facts": 0,
        "errors": 0,
        "source_file": os.path.basename(parquet_path),
    }
    
    facts = []
    
    try:
        dataset = ds.dataset(parquet_path, format="parquet")
        
        for batch in dataset.to_batches(columns=["text", "uuid", "source_file"]):
            
            for row_idx in range(batch.num_rows):
                try:
                    text = batch["text"][row_idx].as_py() if hasattr(batch["text"][row_idx], 'as_py') else str(batch["text"][row_idx])
                    uuid_val = batch["uuid"][row_idx].as_py() if hasattr(batch["uuid"][row_idx], 'as_py') else str(batch["uuid"][row_idx])
                    source_file = batch["source_file"][row_idx].as_py() if hasattr(batch["source_file"][row_idx], 'as_py') else str(batch["source_file"][row_idx])
                    
                    if not text or not isinstance(text, str) or not text.strip():
                        continue
                    
                    stats["total_texts"] += 1
                    
                    # 限制文本长度，避免处理过长的文本导致卡住
                    if len(text) > max_text_length:
                        text = text[:max_text_length] + "..."
                    
                    # LLMLingua提取
                    if stats["total_texts"] % 100 == 0:
                        print(f"  已处理 {stats['total_texts']} 个文本...", flush=True)
                    
                    # 如果跳过LLMLingua，直接使用原始文本
                    if skip_llmlingua:
                        compressed_text = text
                        fact_result = {
                            'compressed_text': text,
                            'compression_ratio': 1.0,
                        }
                    else:
                        if fact_extractor is None:
                            raise ValueError("fact_extractor 为 None，但 skip_llmlingua=False")
                        
                        # 直接调用，不捕获异常，让错误直接暴露
                        fact_result = fact_extractor.extract_facts(text, return_annotations=False)
                        compressed_text = fact_result.get('compressed_text', '')
                        
                        # 检查提取结果是否有效
                        if not compressed_text or not compressed_text.strip():
                            raise ValueError(f"LLMLingua提取返回空结果 (row {row_idx}, text #{stats['total_texts']})")
                        
                    
                    if not compressed_text or not compressed_text.strip():
                        continue
                    
                    # 按句子分割
                    sentences = re.split(r'[.!?]\s+', compressed_text)
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if sentence and len(sentence) >= min_fact_length:
                            facts.append({
                                "fact_text": sentence,
                                "source_file": source_file,
                                "source_uuid": uuid_val,
                                "source_index": row_idx,
                                "compression_ratio": fact_result.get('compression_ratio', 0.0),
                            })
                            stats["total_facts"] += 1
                
                except Exception as e:
                    stats["errors"] += 1
                    raise RuntimeError(f"处理row {row_idx}失败: {e}") from e
    
    except Exception as e:
        print(f"错误: 处理文件 {parquet_path} 失败: {e}")
        stats["errors"] += 1
    
    # 保存到jsonl
    if facts:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for fact in facts:
                f.write(json.dumps(fact, ensure_ascii=False) + "\n")
    
    return stats


def extract_facts_worker(
    worker_id: int,
    file_list: List[str],
    output_dir: str,
    compression_rate: float,
    min_fact_length: int,
    device: Optional[str] = None,
    skip_llmlingua: bool = False,
    max_text_length: int = 3000,
):
    """工作进程：处理分配的文件列表"""
    # 设置GPU（如果支持）
    if device and device.startswith("cuda:"):
        gpu_id = int(device.split(":")[1])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        actual_device = f"cuda:0"  # 在子进程中，只有可见的GPU，所以是cuda:0
    else:
        actual_device = device
    
    # 初始化FactExtractor（如果不需要LLMLingua，可以跳过）
    fact_extractor = None
    if not skip_llmlingua:
        fact_extractor = FactExtractor(compression_rate=compression_rate, device=actual_device)
    else:
        pass  # 跳过LLMLingua，直接使用原始文本
    
    all_stats = []
    
    for file_idx, parquet_path in enumerate(file_list):
        filename = os.path.basename(parquet_path).replace(".parquet", "")
        output_path = os.path.join(output_dir, "extracted_facts", f"{filename}_facts.jsonl")
        
        # 如果已存在，跳过
        if os.path.exists(output_path):
            print(f"Worker {worker_id}: 跳过已处理的文件 {filename}", flush=True)
            continue
        
        print(f"Worker {worker_id}: 开始处理文件 {file_idx+1}/{len(file_list)}: {filename}", flush=True)
        start_time = time.time()
        
        try:
            stats = extract_facts_from_parquet_file(
                parquet_path,
                output_path,
                fact_extractor,
                compression_rate,
                min_fact_length,
                skip_llmlingua,
                max_text_length,
            )
            all_stats.append(stats)
            
            elapsed = time.time() - start_time
            if stats["total_facts"] > 0:
                print(f"Worker {worker_id}: {filename} -> {stats['total_facts']} facts (耗时 {elapsed:.1f}s)", flush=True)
            else:
                print(f"Worker {worker_id}: {filename} -> 0 facts (耗时 {elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"Worker {worker_id}: 处理文件 {filename} 时出错: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
    
    return all_stats


def extract_facts_parallel(
    parquet_dir: str,
    output_dir: str,
    num_workers: int = 4,
    compression_rate: float = 0.4,
    min_fact_length: int = 10,
    skip_llmlingua: bool = False,
    max_text_length: int = 2000,
) -> Dict[str, Any]:
    """并行提取facts"""
    print("=" * 60)
    print("阶段1: 并行提取facts")
    print("=" * 60)
    
    # 扫描所有parquet文件
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"找到 {len(parquet_files)} 个parquet文件")
    
    if not parquet_files:
        raise ValueError(f"在 {parquet_dir} 中未找到parquet文件")
    
    # 分配文件到各个worker
    files_per_worker = len(parquet_files) // num_workers
    file_batches = []
    for i in range(num_workers):
        start_idx = i * files_per_worker
        if i == num_workers - 1:
            end_idx = len(parquet_files)
        else:
            end_idx = (i + 1) * files_per_worker
        file_batches.append(parquet_files[start_idx:end_idx])
    
    print(f"分配文件到 {num_workers} 个worker:")
    for i, batch in enumerate(file_batches):
        print(f"  Worker {i}: {len(batch)} 个文件")
    
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, "extracted_facts"), exist_ok=True)
    
    # 尝试使用多进程
    try:
        from multiprocessing import Process
        import multiprocessing
        
        # 使用spawn方式（更兼容）
        if hasattr(multiprocessing, 'set_start_method'):
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
        
        processes = []
        
        for i in range(num_workers):
            # 尝试分配GPU
            device = f"cuda:{i}" if torch.cuda.is_available() and torch.cuda.device_count() > i else None
            p = Process(
                target=extract_facts_worker,
                args=(i, file_batches[i], output_dir, compression_rate, min_fact_length, device, skip_llmlingua, max_text_length)
            )
            p.start()
            processes.append(p)
        
        # 等待所有进程完成
        for p in processes:
            p.join()
        
        print("所有worker完成")
    
    except Exception as e:
        print(f"多进程失败，使用单进程: {e}")
        import traceback
        traceback.print_exc()
        # 回退到单进程
        for i, file_batch in enumerate(file_batches):
            device = f"cuda:{i}" if torch.cuda.is_available() and torch.cuda.device_count() > i else None
            extract_facts_worker(i, file_batch, output_dir, compression_rate, min_fact_length, device, skip_llmlingua, max_text_length)
    
    # 统计
    total_stats = {
        "total_files": len(parquet_files),
        "total_texts": 0,
        "total_facts": 0,
        "total_errors": 0,
    }
    
    # 从输出文件统计
    fact_files = glob.glob(os.path.join(output_dir, "extracted_facts", "*_facts.jsonl"))
    for fact_file in fact_files:
        with open(fact_file, "r", encoding="utf-8") as f:
            total_stats["total_facts"] += sum(1 for _ in f)
    
    print(f"\n阶段1完成:")
    print(f"  - 处理文件数: {total_stats['total_files']}")
    print(f"  - 提取facts数: {total_stats['total_facts']:,}")
    
    return total_stats


# ============================================================================
# 阶段2：合并、去重、采样
# ============================================================================
def merge_and_deduplicate(
    extracted_facts_dir: str,
    knowledge_num: int,
    deduplication: str = "text_hash",
) -> List[Dict[str, Any]]:
    """合并所有facts，去重，采样"""
    print("=" * 60)
    print("阶段2: 合并、去重、采样")
    print("=" * 60)
    
    # 读取所有facts
    fact_files = sorted(glob.glob(os.path.join(extracted_facts_dir, "*_facts.jsonl")))
    print(f"读取 {len(fact_files)} 个fact文件...")
    
    all_facts = []
    seen = set()
    
    for fact_file in tqdm(fact_files, desc="读取facts"):
        with open(fact_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    fact = json.loads(line)
                    fact_text = fact.get("fact_text", "").strip()
                    
                    if not fact_text:
                        continue
                    
                    # 去重
                    if deduplication == "text_hash":
                        text_hash = hashlib.md5(fact_text.lower().encode('utf-8')).hexdigest()
                    else:
                        # 使用文本本身作为hash（简单）
                        text_hash = fact_text.lower()
                    
                    if text_hash not in seen:
                        seen.add(text_hash)
                        all_facts.append(fact)
                
                except Exception as e:
                    continue
    
    print(f"去重前: {len(all_facts):,} 条facts")
    
    # 采样
    if len(all_facts) > knowledge_num:
        print(f"随机采样 {knowledge_num:,} 条...")
        all_facts = random.sample(all_facts, knowledge_num)
    else:
        print(f"facts数量不足，将全部保留（{len(all_facts):,} 条）")
    
    print(f"最终facts数: {len(all_facts):,}")
    
    return all_facts


# ============================================================================
# 阶段3：分批保存（tokenize在此时进行）
# ============================================================================
def tokenize_fact(
    fact_text: str,
    tokenizer: AutoTokenizer,
    knowledge_length: int,
) -> Tuple[List[int], bool]:
    """
    Tokenize一个fact
    
    Returns:
        (token_ids, is_truncated)
    """
    if not fact_text or not fact_text.strip():
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        return [pad_token_id] * knowledge_length, False
    
    try:
        tokens_result = tokenizer(
            fact_text,
            add_special_tokens=True,
            truncation=True,
            max_length=knowledge_length,
            padding=False,
            return_tensors="pt",
        )
        tokens = tokens_result["input_ids"].squeeze().tolist()
        if not isinstance(tokens, list):
            tokens = [tokens]
        
        original_length = len(tokens)
        is_truncated = original_length > knowledge_length
        
        # 截断或填充
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        if len(tokens) > knowledge_length:
            tokens = tokens[:knowledge_length]
        elif len(tokens) < knowledge_length:
            tokens.extend([pad_token_id] * (knowledge_length - len(tokens)))
        
        return tokens, is_truncated
    
    except Exception as e:
        print(f"警告: tokenize失败: {e}")
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        return [pad_token_id] * knowledge_length, False


def save_memory_bank_batch(
    facts: List[Dict[str, Any]],
    batch_idx: int,
    output_dir: str,
    tokenizer: AutoTokenizer,
    knowledge_num: int,
    knowledge_length: int,
) -> Dict[str, Any]:
    """保存一个批次的memory bank"""
    batch_dir = os.path.join(output_dir, "memory_bank_batches")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Tokenize所有facts
    processed_rows = []
    database_mapping = []
    truncated_count = 0
    
    for idx, fact in enumerate(tqdm(facts, desc=f"Tokenize batch {batch_idx}")):
        fact_text = fact.get("fact_text", "")
        tokens, is_truncated = tokenize_fact(fact_text, tokenizer, knowledge_length)
        
        if is_truncated:
            truncated_count += 1
        
        processed_rows.append(tokens)
        database_mapping.append({
            "database_index": idx,
            "uuid": fact.get("source_uuid", ""),
            "sentence": fact_text,
            "source_file": fact.get("source_file", ""),
            "token_count": len([t for t in tokens if t != tokenizer.pad_token_id]),
            "is_truncated": is_truncated,
        })
    
    # 填充到knowledge_num
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    while len(processed_rows) < knowledge_num:
        processed_rows.append([pad_token_id] * knowledge_length)
        database_mapping.append({
            "database_index": len(processed_rows) - 1,
            "uuid": "",
            "sentence": "",
            "source_file": "",
            "token_count": 0,
            "is_truncated": False,
        })
    
    # 转换为tensor
    processed_tensor = torch.tensor(processed_rows, dtype=torch.long)
    valid_mask = torch.tensor([i < len(facts) for i in range(knowledge_num)], dtype=torch.bool)
    
    # 保存.pt文件
    cache_path = os.path.join(batch_dir, f"memory_bank_batch_{batch_idx}.pt")
    cache_data = {
        "memory_bank": processed_tensor,
        "valid_mask": valid_mask,
    }
    torch.save(cache_data, cache_path)
    
    # 保存mapping文件
    mapping_path = os.path.join(batch_dir, f"memory_bank_batch_{batch_idx}_mapping.json")
    mapping_data = {
        "metadata": {
            "batch_index": batch_idx,
            "total_entries": len(facts),
            "knowledge_num": knowledge_num,
            "knowledge_length": knowledge_length,
            "truncated_count": truncated_count,
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "mappings": database_mapping,
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"批次 {batch_idx} 已保存:")
    print(f"  - 文件: {cache_path}")
    print(f"  - 有效条目: {len(facts):,}/{knowledge_num:,}")
    print(f"  - 截断条目: {truncated_count:,}")
    
    return {
        "batch_idx": batch_idx,
        "valid_entries": len(facts),
        "truncated_count": truncated_count,
    }


def save_facts_in_batches(
    facts: List[Dict[str, Any]],
    output_dir: str,
    tokenizer: AutoTokenizer,
    knowledge_num: int,
    knowledge_length: int,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """分批保存facts"""
    print("=" * 60)
    print("阶段3: 分批保存（tokenize）")
    print("=" * 60)
    
    batch_results = []
    batch_idx = 0
    
    # 分批处理
    for i in range(0, len(facts), batch_size):
        batch_facts = facts[i:i + batch_size]
        
        result = save_memory_bank_batch(
            batch_facts,
            batch_idx,
            output_dir,
            tokenizer,
            knowledge_num,
            knowledge_length,
        )
        batch_results.append(result)
        batch_idx += 1
    
    print(f"\n阶段3完成: 共保存 {batch_idx} 个批次")
    
    return batch_results


# ============================================================================
# 主流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="从Parquet文件提取知识并生成Memory Bank")
    parser.add_argument("--parquet-dir", type=str, required=True, help="Parquet文件目录")
    parser.add_argument("--output-dir", type=str, default="data/cache/parquet_extract", help="输出目录")
    parser.add_argument("--num-workers", type=int, default=4, help="工作进程数")
    parser.add_argument("--compression-rate", type=float, default=0.4, help="LLMLingua压缩率")
    parser.add_argument("--knowledge-num", type=int, default=1024*1024, help="每个批次的knowledge数量")
    parser.add_argument("--knowledge-length", type=int, default=32, help="每个knowledge的token长度")
    parser.add_argument("--min-fact-length", type=int, default=10, help="最小fact长度（字符）")
    parser.add_argument("--deduplication", type=str, default="text_hash", choices=["text_hash", "token_hash"], help="去重方法")
    parser.add_argument("--qwen-model-path", type=str, default="Qwen_hg/Qwen3-4b", help="Qwen模型路径（用于tokenizer，相对于 ExplicitLM 根目录）")
    parser.add_argument("--skip-extract", action="store_true", help="跳过提取阶段（直接合并）")
    parser.add_argument("--skip-merge", action="store_true", help="跳过合并阶段（直接保存）")
    parser.add_argument("--test-mode", action="store_true", help="测试模式：只处理前10个文本")
    parser.add_argument("--max-texts-per-file", type=int, default=None, help="每个文件最多处理的文本数（用于调试）")
    parser.add_argument("--skip-llmlingua", action="store_true", help="跳过LLMLingua提取，直接使用原始文本（更快但质量较低）")
    parser.add_argument("--max-text-length", type=int, default=2000, help="每个文本的最大长度（字符数），超过会被截断")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 阶段1：提取
    if not args.skip_extract:
        extract_facts_parallel(
            args.parquet_dir,
            args.output_dir,
            args.num_workers,
            args.compression_rate,
            args.min_fact_length,
            args.skip_llmlingua,
            args.max_text_length,
        )
    else:
        print("跳过提取阶段")
    
    # 阶段2：合并去重
    if not args.skip_merge:
        facts = merge_and_deduplicate(
            os.path.join(args.output_dir, "extracted_facts"),
            args.knowledge_num,
            args.deduplication,
        )
        
        # 保存合并后的facts（用于调试）
        merged_facts_path = os.path.join(args.output_dir, "merged_facts.jsonl")
        with open(merged_facts_path, "w", encoding="utf-8") as f:
            for fact in facts:
                f.write(json.dumps(fact, ensure_ascii=False) + "\n")
        print(f"合并后的facts已保存到: {merged_facts_path}")
    else:
        # 从文件加载
        merged_facts_path = os.path.join(args.output_dir, "merged_facts.jsonl")
        if not os.path.exists(merged_facts_path):
            raise FileNotFoundError(f"合并后的facts文件不存在: {merged_facts_path}")
        facts = []
        with open(merged_facts_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    facts.append(json.loads(line))
        print(f"从文件加载 {len(facts):,} 条facts")
    
    # 阶段3：分批保存
    print(f"\n加载tokenizer: {args.qwen_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    save_facts_in_batches(
        facts,
        args.output_dir,
        tokenizer,
        args.knowledge_num,
        args.knowledge_length,
        args.knowledge_num,  # batch_size = knowledge_num
    )
    
    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

