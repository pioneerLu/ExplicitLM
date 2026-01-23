#!/usr/bin/env python3
"""
为 pretrain 数据构建 Oracle Fact Memory Bank（用于上限测试）。

输入：
  - 预训练数据路径（与 run_fusion_pretrain.sh 中 PRETRAIN_DATASET_PATH 一致）
  - 使用 FactExtractor 从每条 text 中提取一个压缩 fact

输出（全部写在 data/test_data/ 下）：
  - pretrain_facts_memory_bank.pt
      {
          "memory_bank": LongTensor [N, knowledge_length],
          "valid_mask": BoolTensor [N],
      }
  - pretrain_facts_mapping.pt
      {
          "uuid_to_fact_indices": Dict[str, List[int]],  # 每个 uuid 对应的 fact indices（通常是单个元素的列表）
      }

注意：
  - 为简化，我们将每条样本的压缩文本视为一条 fact（一对一），N ~= 样本数
  - uuid 格式：如果 parquet 中有 uuid 字段则使用，否则使用 "sample_{idx}"
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple
from functools import partial

# 添加项目根目录到路径
root = Path(__file__).resolve().parents[2]  # ExplicitLM 根目录
sys.path.insert(0, str(root))

import torch
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from transformers import AutoTokenizer

from utils.fact_extractor import FactExtractor


def load_parquet_texts(parquet_path: str) -> tuple[List[str], List[str]]:
    """
    从 parquet 目录/文件中按顺序加载 text 和 uuid 字段。
    
    注意：要求 parquet 文件必须包含 uuid 列，否则会抛出错误。
    """
    paths: List[Path] = []
    p = Path(parquet_path)
    if p.is_dir():
        paths = sorted(p.glob("*.parquet"))
    else:
        # 支持单文件或通配符
        if "*" in parquet_path or "?" in parquet_path:
            paths = sorted(Path(".").glob(parquet_path))
        else:
            paths = [p]

    if not paths:
        raise ValueError(f"未找到 parquet 文件: {parquet_path}")

    texts: List[str] = []
    uuids: List[str] = []
    
    for file_path in paths:
        table = pq.read_table(str(file_path))
        column_names = table.column_names
        
        # 检查必需的列
        if "text" not in column_names:
            raise ValueError(f"Parquet 文件 {file_path} 中不存在 text 列。可用列: {column_names}")
        
        if "uuid" not in column_names:
            raise ValueError(
                f"Parquet 文件 {file_path} 中不存在 uuid 列。可用列: {column_names}\n"
                f"Oracle Fact Fusion 需要 uuid 字段来建立映射关系，请确保数据包含 uuid 列。"
            )
        
        text_col = table["text"]
        uuid_col = table["uuid"]
        
        for i, v in enumerate(text_col.to_pylist()):
            if v is None:
                texts.append("")
            else:
                texts.append(str(v))
            
            # 直接使用 uuid，如果为 None 则报错
            uuid_val = uuid_col[i]
            if uuid_val is None:
                raise ValueError(
                    f"Parquet 文件 {file_path} 第 {i} 行的 uuid 为 None。"
                    f"Oracle Fact Fusion 要求所有样本都有有效的 uuid。"
                )
            uuids.append(str(uuid_val))
    
    return texts, uuids


def process_single_sample(
    args_tuple: Tuple[int, str, str, str, str, int, int]
) -> Tuple[int, str, str, List[int], bool]:
    """
    处理单个样本的 fact 提取和 tokenize（用于多进程）
    精确控制压缩后的 token 数为 knowledge_length（32）
    
    Returns:
        (idx, uuid_str, fact_text, tokens, success)
    """
    idx, text, uuid_str, llmlingua_model_path, qwen_model_path, knowledge_length, pad_id = args_tuple
    
    try:
        # 初始化 tokenizer（每个进程独立）
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # 在每个进程中初始化 FactExtractor（避免共享模型）
        extractor = FactExtractor(
            model_path=llmlingua_model_path,
            compression_rate=0.4,  # 初始压缩率
        )
        
        # 策略：迭代压缩直到 token 数接近 knowledge_length
        fact_text = text
        max_iterations = 5  # 最多迭代5次
        tolerance = 2  # 允许的误差范围（±2 tokens）
        
        for iteration in range(max_iterations):
            # 提取 fact（使用动态压缩率）
            try:
                # 根据迭代次数调整压缩率
                if iteration == 0:
                    # 第一次：使用默认压缩率
                    compression_rate = 0.4
                else:
                    # 后续迭代：根据当前 token 数调整压缩率
                    current_tokens = tokenizer(
                        fact_text,
                        add_special_tokens=True,
                        truncation=False,
                        padding=False,
                        return_tensors="pt",
                    )["input_ids"].squeeze()
                    current_length = len(current_tokens) if current_tokens.dim() > 0 else 1
                    
                    if current_length <= knowledge_length:
                        # 已经足够短，可以退出
                        break
                    
                    # 计算目标压缩率：希望压缩到 knowledge_length
                    # 使用比例调整，但限制在合理范围 [0.1, 0.6]
                    target_ratio = (knowledge_length * 0.95) / current_length
                    compression_rate = max(0.1, min(0.6, compression_rate * target_ratio))
                
                res = extractor.extract_facts(text, return_annotations=False, compression_rate=compression_rate)
                fact_text = res.get("compressed_text", "") or text
                
                if not fact_text or not fact_text.strip():
                    fact_text = text  # 如果压缩失败，使用原始文本
                    break
                
            except Exception as e:
                # 压缩失败，使用原始文本
                fact_text = text
                break
            
            # 检查当前 token 数
            try:
                tokens_result = tokenizer(
                    fact_text,
                    add_special_tokens=True,
                    truncation=False,
                    padding=False,
                    return_tensors="pt",
                )
                current_tokens = tokens_result["input_ids"].squeeze()
                current_length = len(current_tokens) if current_tokens.dim() > 0 else 1
                
                # 如果长度在允许范围内，退出迭代
                if abs(current_length - knowledge_length) <= tolerance:
                    break
                
                # 如果已经足够短（小于目标长度），也退出（后续会 pad）
                if current_length < knowledge_length:
                    break
                
                # 如果已经压缩到极限（压缩率已经很小），也退出
                if compression_rate <= 0.15:
                    break
                    
            except Exception:
                # tokenize 失败，使用原始文本
                fact_text = text
                break
        
        # 最终 tokenize
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
        except Exception:
            tokens = []
        
        # 精确控制到 knowledge_length
        if len(tokens) > knowledge_length:
            # 如果超过，截断（保留前面的部分，通常信息更重要）
            tokens = tokens[:knowledge_length]
        elif len(tokens) < knowledge_length:
            # 如果不足，用 pad 填充
            tokens.extend([pad_id] * (knowledge_length - len(tokens)))
        
        # 确保最终长度正好是 knowledge_length
        assert len(tokens) == knowledge_length, f"Token length mismatch: {len(tokens)} != {knowledge_length}"
        
        return (idx, uuid_str, fact_text, tokens, True)
    except Exception as e:
        # 返回错误信息，使用全 pad 占位
        tokens = [pad_id] * knowledge_length
        return (idx, uuid_str, "", tokens, False)


def save_facts_to_jsonl(
    facts: List[Dict],
    output_path: str,
    lock=None
):
    """
    保存 facts 到 JSONL 文件（线程安全）
    
    Args:
        facts: List of {idx, uuid, fact_text, tokens, success}
        output_path: 输出文件路径
        lock: 可选的锁对象（用于多进程）
    """
    if lock:
        lock.acquire()
    
    try:
        with open(output_path, 'a', encoding='utf-8') as f:
            for fact in facts:
                json.dump(fact, f, ensure_ascii=False)
                f.write('\n')
    finally:
        if lock:
            lock.release()


def convert_jsonl_to_pt(
    jsonl_dir: str,
    output_bank_path: str,
    output_mapping_path: str,
    knowledge_length: int,
    pad_id: int,
):
    """
    将所有 JSONL 文件转换为 PT 格式
    
    Args:
        jsonl_dir: JSONL 文件目录
        output_bank_path: 输出的 memory bank 路径
        output_mapping_path: 输出的 mapping 路径
        knowledge_length: fact 的 token 长度
        pad_id: pad token id
    """
    print("=" * 60)
    print("将 JSONL 转换为 PT 格式")
    print("=" * 60)
    print(f"📂 JSONL 目录: {jsonl_dir}")
    print()
    
    # 收集所有 JSONL 文件
    jsonl_files = sorted(Path(jsonl_dir).glob("facts_*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"在 {jsonl_dir} 中未找到 JSONL 文件")
    
    print(f"📁 找到 {len(jsonl_files)} 个 JSONL 文件")
    print()
    
    # 读取所有 facts
    all_facts = []
    for jsonl_file in tqdm(jsonl_files, desc="读取 JSONL 文件"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    fact = json.loads(line)
                    all_facts.append(fact)
    
    print(f"✅ 共读取 {len(all_facts)} 条 facts")
    print()
    
    # 按 idx 排序
    all_facts.sort(key=lambda x: x['idx'])
    
    # 构建 memory_bank 和 mapping
    print("🔨 构建 memory_bank 和 mapping...")
    memory_rows = []
    uuid_to_fact_indices = {}
    
    for fact in tqdm(all_facts, desc="处理 facts"):
        idx = fact['idx']
        uuid_str = str(fact['uuid'])
        tokens = fact['tokens']
        success = fact.get('success', True)
        
        # 验证 tokens 长度
        if len(tokens) != knowledge_length:
            # 如果长度不对，修正
            if len(tokens) > knowledge_length:
                tokens = tokens[:knowledge_length]
            else:
                tokens.extend([pad_id] * (knowledge_length - len(tokens)))
        
        fact_idx = len(memory_rows)
        memory_rows.append(tokens)
        
        # 建立映射
        if uuid_str not in uuid_to_fact_indices:
            uuid_to_fact_indices[uuid_str] = []
        uuid_to_fact_indices[uuid_str].append(fact_idx)
    
    # 转换为 tensor
    memory_bank = torch.tensor(memory_rows, dtype=torch.long)
    valid_mask = torch.ones(len(memory_rows), dtype=torch.bool)
    
    # 保存
    print(f"💾 保存 memory bank: {output_bank_path}")
    torch.save(
        {
            "memory_bank": memory_bank,
            "valid_mask": valid_mask,
        },
        output_bank_path,
    )
    
    print(f"💾 保存 mapping: {output_mapping_path}")
    torch.save(
        {
            "uuid_to_fact_indices": uuid_to_fact_indices,
        },
        output_mapping_path,
    )
    
    print()
    print("✅ 转换完成")
    print(f"   📐 memory_bank shape: {tuple(memory_bank.shape)}")
    print(f"   📐 valid_mask shape: {tuple(valid_mask.shape)}")
    print(f"   📊 映射条目数: {len(uuid_to_fact_indices)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="构建 pretrain Oracle Fact Memory Bank")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="预训练数据路径（与 run_fusion_pretrain.sh 中 PRETRAIN_DATASET_PATH 一致，例如 data/parquet_data/sample_256）",
    )
    parser.add_argument(
        "--qwen_model_path",
        type=str,
        default="Qwen_hg/Qwen3-4b",
        help="Qwen 模型路径（用于 tokenizer）",
    )
    parser.add_argument(
        "--llmlingua_model_path",
        type=str,
        default="llmlingua-2-bert",
        help="LLMLingua 模型路径（用于 FactExtractor）",
    )
    parser.add_argument(
        "--knowledge_length",
        type=int,
        default=32,
        help="每条 fact 的 token 长度（需与训练时 knowledge_length 一致）",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="可选：最多处理多少条样本（用于快速测试）",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="并行提取的进程数（默认 4，建议根据 CPU 核心数调整）",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10000,
        help="每处理多少样本保存一次结果文件（默认 10000，边存边运行）",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有结果文件恢复（断点续传，如果输出文件已存在）",
    )

    args = parser.parse_args()

    # root 已在顶部定义
    data_dir = root / "data" / "test_data"
    os.makedirs(data_dir, exist_ok=True)

    # JSONL 中间文件目录
    jsonl_dir = data_dir / "oracle_facts_jsonl"
    os.makedirs(jsonl_dir, exist_ok=True)

    fact_bank_path = data_dir / "pretrain_facts_memory_bank.pt"
    mapping_path = data_dir / "pretrain_facts_mapping.pt"

    print("============================================")
    print(" 构建 pretrain Oracle Fact Memory Bank")
    print("============================================")
    print(f"📂 预训练数据: {args.dataset_path}")
    print(f"🤖 Tokenizer: {args.qwen_model_path}")
    print(f"🧠 LLMLingua: {args.llmlingua_model_path}")
    print(f"🔢 knowledge_length: {args.knowledge_length}")
    if args.max_samples is not None:
        print(f"✂️  最多处理样本数: {args.max_samples}")
    print()

    # 1. 加载文本和 uuids
    texts, uuids = load_parquet_texts(args.dataset_path)
    total = len(texts)
    if args.max_samples is not None:
        texts = texts[: args.max_samples]
        uuids = uuids[: args.max_samples]
    print(f"✅ 加载文本: {len(texts)}/{total} 条")

    # 2. 初始化 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.qwen_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # 3. 提取 facts 并保存为 JSONL（支持并行和 checkpoint）
    print("🔍 开始为每条样本提取 fact 并保存为 JSONL...")
    print(f"   📁 JSONL 目录: {jsonl_dir}")
    print(f"   并行进程数: {args.num_workers}")
    print(f"   保存间隔: 每 {args.checkpoint_interval} 个样本保存一份 JSONL 文件")
    if args.resume:
        print(f"   断点续传: 已启用（从已有 JSONL 文件继续）")
    print()
    
    start_time = time.time()
    
    # 准备参数列表
    process_args = [
        (idx, text, str(uuid), args.llmlingua_model_path, args.qwen_model_path, args.knowledge_length, pad_id)
        for idx, (text, uuid) in enumerate(zip(texts, uuids))
    ]
    
    # 检查已有 JSONL 文件（断点续传）
    processed_indices = set()
    if args.resume:
        print(f"📂 检查已有 JSONL 文件...")
        jsonl_files = sorted(jsonl_dir.glob("facts_*.jsonl"))
        if jsonl_files:
            for jsonl_file in jsonl_files:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            fact = json.loads(line)
                            processed_indices.add(fact['idx'])
            
            start_idx = len(processed_indices)
            print(f"   ✅ 发现 {len(jsonl_files)} 个 JSONL 文件，已处理 {start_idx} 个样本")
            print(f"   📍 从索引 {start_idx} 继续处理（剩余 {len(texts) - start_idx} 个样本）")
            print()
        else:
            start_idx = 0
            print(f"   ℹ️  未发现已有 JSONL 文件，从头开始")
            print()
    else:
        start_idx = 0
        # 如果不续传，清空目录
        if jsonl_dir.exists():
            for f in jsonl_dir.glob("facts_*.jsonl"):
                f.unlink()
    
    # 只处理未完成的样本
    if start_idx > 0:
        process_args = process_args[start_idx:]
        print(f"   剩余待处理: {len(process_args)} 个样本")
        print()
    
    # 用于保存 JSONL 的缓冲区
    jsonl_buffer = []
    jsonl_file_counter = len(list(jsonl_dir.glob("facts_*.jsonl"))) if args.resume else 0
    processed_count = start_idx  # 已处理的样本数
    
    if args.num_workers > 1:
        print(f"🚀 使用 {args.num_workers} 个进程并行提取...")
        from multiprocessing import Manager, Lock
        manager = Manager()
        jsonl_lock = manager.Lock()
        
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(process_single_sample, args): args[0] for args in process_args}
            
            # 使用 tqdm 显示进度
            pbar = tqdm(total=len(process_args), desc="提取 facts", unit="样本", ncols=100)
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    idx_result, uuid_str, fact_text, tokens, success = result
                    
                    # 添加到缓冲区
                    fact_data = {
                        "idx": idx_result,
                        "uuid": uuid_str,
                        "fact_text": fact_text,
                        "tokens": tokens,
                        "success": success,
                    }
                    jsonl_buffer.append(fact_data)
                    
                    if not success:
                        error_count += 1
                except Exception as e:
                    error_count += 1
                    # 使用全 pad 占位
                    uuid_str = str(uuids[idx] if idx < len(uuids) else f"sample_{idx}")
                    fact_data = {
                        "idx": idx,
                        "uuid": uuid_str,
                        "fact_text": "",
                        "tokens": [pad_id] * args.knowledge_length,
                        "success": False,
                    }
                    jsonl_buffer.append(fact_data)
                
                pbar.update(1)
                
                # 每 checkpoint_interval 个保存一次 JSONL
                if len(jsonl_buffer) >= args.checkpoint_interval:
                    jsonl_file_counter += 1
                    jsonl_file_path = jsonl_dir / f"facts_{jsonl_file_counter:06d}.jsonl"
                    
                    # 保存 JSONL（线程安全）
                    with open(jsonl_file_path, 'w', encoding='utf-8') as f:
                        for fact in jsonl_buffer:
                            json.dump(fact, f, ensure_ascii=False)
                            f.write('\n')
                    
                    pbar.write(f"💾 已保存 JSONL: {jsonl_file_path.name} ({len(jsonl_buffer)} 条 facts)")
                    processed_count += len(jsonl_buffer)
                    jsonl_buffer = []
                
                # 更新进度信息
                processed = processed_count + len(jsonl_buffer)
                if processed > 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_sample = elapsed_time / processed
                    remaining_samples = len(process_args) - processed
                    estimated_remaining = avg_time_per_sample * remaining_samples
                    
                    # 格式化时间
                    def format_time(seconds):
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        if hours > 0:
                            return f"{hours}h{minutes}m{secs}s"
                        elif minutes > 0:
                            return f"{minutes}m{secs}s"
                        else:
                            return f"{secs}s"
                    
                    pbar.set_postfix({
                        "已用时间": format_time(elapsed_time),
                        "预计剩余": format_time(estimated_remaining),
                        "速度": f"{processed/elapsed_time:.2f} 样本/秒",
                        "错误": error_count,
                    })
            
            # 保存剩余的缓冲区
            if len(jsonl_buffer) > 0:
                jsonl_file_counter += 1
                jsonl_file_path = jsonl_dir / f"facts_{jsonl_file_counter:06d}.jsonl"
                with open(jsonl_file_path, 'w', encoding='utf-8') as f:
                    for fact in jsonl_buffer:
                        json.dump(fact, f, ensure_ascii=False)
                        f.write('\n')
                processed_count += len(jsonl_buffer)
                pbar.write(f"💾 已保存最后一批 JSONL: {jsonl_file_path.name} ({len(jsonl_buffer)} 条 facts)")
            
            pbar.close()
    else:
        # 单进程模式（用于调试）
        print("🐌 使用单进程模式...")
        pbar = tqdm(total=len(process_args), desc="提取 facts", unit="样本", ncols=100)
        
        for args_tuple in process_args:
            result = process_single_sample(args_tuple)
            idx_result, uuid_str, fact_text, tokens, success = result
            
            # 添加到缓冲区
            fact_data = {
                "idx": idx_result,
                "uuid": uuid_str,
                "fact_text": fact_text,
                "tokens": tokens,
                "success": success,
            }
            jsonl_buffer.append(fact_data)
            
            if not success:
                error_count += 1
            
            pbar.update(1)
            
            # 每 checkpoint_interval 个保存一次 JSONL
            if len(jsonl_buffer) >= args.checkpoint_interval:
                jsonl_file_counter += 1
                jsonl_file_path = jsonl_dir / f"facts_{jsonl_file_counter:06d}.jsonl"
                
                with open(jsonl_file_path, 'w', encoding='utf-8') as f:
                    for fact in jsonl_buffer:
                        json.dump(fact, f, ensure_ascii=False)
                        f.write('\n')
                
                pbar.write(f"💾 已保存 JSONL: {jsonl_file_path.name} ({len(jsonl_buffer)} 条 facts)")
                jsonl_buffer = []
            
            # 更新进度信息
            processed = processed_count + len(jsonl_buffer)
            if processed > 0:
                elapsed_time = time.time() - start_time
                pbar.set_postfix({
                    "速度": f"{processed/elapsed_time:.2f} 样本/秒",
                    "错误": error_count,
                })
        
        # 保存剩余的缓冲区
        if len(jsonl_buffer) > 0:
            jsonl_file_counter += 1
            jsonl_file_path = jsonl_dir / f"facts_{jsonl_file_counter:06d}.jsonl"
            with open(jsonl_file_path, 'w', encoding='utf-8') as f:
                for fact in jsonl_buffer:
                    json.dump(fact, f, ensure_ascii=False)
                    f.write('\n')
            processed_count += len(jsonl_buffer)
            pbar.write(f"💾 已保存最后一批 JSONL: {jsonl_file_path.name} ({len(jsonl_buffer)} 条 facts)")
        
        pbar.close()
    
    total_time = time.time() - start_time
    
    print()
    print(f"✅ Fact 提取完成，总耗时: {total_time/3600:.2f} 小时 ({total_time/60:.2f} 分钟)")
    if error_count > 0:
        print(f"⚠️  处理失败的样本数: {error_count}")
    print()
    
    # 5. 将所有 JSONL 转换为 PT 格式
    print("=" * 60)
    print("开始转换 JSONL 为 PT 格式")
    print("=" * 60)
    convert_jsonl_to_pt(
        jsonl_dir=str(jsonl_dir),
        output_bank_path=str(fact_bank_path),
        output_mapping_path=str(mapping_path),
        knowledge_length=args.knowledge_length,
        pad_id=pad_id,
    )
    
    print("\n✅ Oracle Fact Memory Bank 构建完成")
    print(f"   📁 JSONL 目录: {jsonl_dir}")
    print(f"   📁 Fact bank: {fact_bank_path}")
    print(f"   📁 Mapping:   {mapping_path}")
    print("============================================")


if __name__ == "__main__":
    main()


