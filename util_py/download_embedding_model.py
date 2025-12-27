#!/usr/bin/env python3
"""
下载嵌入模型到本地

功能：
1. 从 HuggingFace 下载指定的嵌入模型
2. 保存到本地目录，之后可以直接使用本地路径
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ 缺少依赖: sentence-transformers")
    print("请安装: pip install sentence-transformers")
    exit(1)


def download_model(model_name: str, local_path: str):
    """
    下载模型到本地
    
    Args:
        model_name: HuggingFace 模型名称（如 "BAAI/bge-base-en-v1.5"）
        local_path: 本地保存路径
    """
    local_path = Path(local_path)
    
    # 检查模型是否已存在
    if local_path.exists() and (local_path / "config.json").exists():
        print(f"✅ 模型已存在于本地: {local_path}")
        print(f"   可以直接使用: --local-model-path {local_path}")
        return
    
    print(f"📥 下载模型: {model_name}")
    print(f"📁 保存到: {local_path}")
    print()
    
    # 创建目录
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 下载模型
    print("正在下载...（这可能需要几分钟）")
    model = SentenceTransformer(model_name)
    
    # 保存到本地
    model.save(str(local_path))
    
    print()
    print("=" * 60)
    print("✅ 模型下载完成！")
    print("=" * 60)
    print(f"本地路径: {local_path}")
    print()
    print("📝 使用方法:")
    print(f"  --local-model-path {local_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="下载嵌入模型到本地"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="BAAI/bge-base-en-v1.5",
        help="HuggingFace 模型名称"
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default="ExplicitLM/models/embedding_models/bge-base-en-v1.5",
        help="本地保存路径"
    )
    
    args = parser.parse_args()
    
    download_model(args.model_name, args.local_path)


if __name__ == "__main__":
    main()
