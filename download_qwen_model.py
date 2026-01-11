#!/usr/bin/env python3
"""
下载 Qwen3-4B-Instruct-2507 模型
使用 HuggingFace 镜像加速下载
"""

import os
import sys
from pathlib import Path
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer, AutoModelForCausalLM

# 设置 HuggingFace 镜像


# 模型名称
model_name = "Qwen/Qwen3-4B-Instruct-2507"

# 目标目录
target_dir = Path(__file__).parent / "Qwen_hg" / "Qwen3-4B-Instruct-2507"
target_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("下载 Qwen3-4B-Instruct-2507 模型")
print("=" * 60)
print(f"模型名称: {model_name}")
print(f"目标目录: {target_dir}")
print(f"HuggingFace 镜像: {os.environ.get('HF_ENDPOINT', '默认')}")
print("=" * 60)
print()

# 下载 tokenizer
print("正在下载 tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(target_dir.parent),
        trust_remote_code=True
    )
    print(f"✓ Tokenizer 下载完成")
    print(f"  - 保存位置: {target_dir.parent}")
except Exception as e:
    print(f"❌ Tokenizer 下载失败: {e}")
    sys.exit(1)

print()

# 下载模型
print("正在下载模型（这可能需要一些时间）...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(target_dir.parent),
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    print(f"✓ 模型下载完成")
    print(f"  - 保存位置: {target_dir.parent}")
except Exception as e:
    print(f"❌ 模型下载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 60)
print("下载完成！")
print("=" * 60)
print(f"模型已保存到: {target_dir.parent}")
print()
print("测试模型加载...")

# 测试模型
try:
    # 重新加载以测试
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=str(target_dir.parent),
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=str(target_dir.parent),
        trust_remote_code=True,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # 测试生成
    messages = [
        {"role": "user", "content": "Who are you?"}
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=40)
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    
    print("✓ 模型测试成功！")
    print(f"  测试问题: Who are you?")
    print(f"  模型回复: {response}")
    
except Exception as e:
    print(f"⚠️  模型测试失败: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("完成！")
print("=" * 60)

