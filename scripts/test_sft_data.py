#!/usr/bin/env python3
"""
测试 SFT 数据加载

验证转换后的 omcq 数据能否被 SFTDataset 正确加载
"""

import sys
sys.path.insert(0, '.')

from transformers import AutoTokenizer
from utils.sft_datasets import create_sft_dataloader
from pathlib import Path


def test_sft_data(data_path: str, qwen3_model_path: str):
    """测试 SFT 数据加载"""
    print("=" * 60)
    print("🧪 测试 SFT 数据加载")
    print("=" * 60)
    
    # 加载 tokenizer
    print(f"\n📖 加载 tokenizer: {qwen3_model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(qwen3_model_path, trust_remote_code=True)
        print("✅ Tokenizer 加载成功")
    except Exception as e:
        print(f"❌ Tokenizer 加载失败: {e}")
        return False
    
    # 创建数据加载器
    print(f"\n📊 创建数据加载器: {data_path}")
    try:
        train_loader = create_sft_dataloader(
            data_path=data_path,
            tokenizer=tokenizer,
            batch_size=2,
            max_length=512,
            shuffle=False,
            num_workers=0,  # 测试时使用单进程
        )
        print(f"✅ 数据加载器创建成功，共 {len(train_loader)} 个批次")
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试加载一个批次
    print("\n🔄 测试加载一个批次...")
    try:
        batch = next(iter(train_loader))
        print("✅ 批次加载成功")
        print(f"  - input_ids shape: {batch['input_ids'].shape}")
        print(f"  - attention_mask shape: {batch['attention_mask'].shape}")
        print(f"  - loss_mask shape: {batch['loss_mask'].shape}")
        print(f"  - labels shape: {batch['labels'].shape}")
        
        # 解码第一个样本
        print("\n📝 第一个样本（解码前 100 tokens）:")
        sample_input_ids = batch['input_ids'][0]
        sample_text = tokenizer.decode(sample_input_ids[:100], skip_special_tokens=False)
        print(f"  {sample_text}...")
        
        return True
    except Exception as e:
        print(f"❌ 批次加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 SFT 数据加载")
    parser.add_argument(
        "--data-path",
        type=str,
        default="sft_data/omcq_trex_sft.jsonl",
        help="SFT 数据文件路径（JSONL 格式）"
    )
    parser.add_argument(
        "--qwen3-model-path",
        type=str,
        required=True,
        help="Qwen3 模型路径（用于加载 tokenizer）"
    )
    
    args = parser.parse_args()
    
    success = test_sft_data(args.data_path, args.qwen3_model_path)
    
    if success:
        print("\n" + "=" * 60)
        print("✅ 数据加载测试通过！")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ 数据加载测试失败！")
        print("=" * 60)
        sys.exit(1)

