#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单对话脚本

功能：
- 加载 Qwen3-ExplicitLM 模型结构
- 从 out/example.pth 中加载训练好的权重
- 使用第 6 号 GPU（CUDA_VISIBLE_DEVICES=6）
- 通过命令行与模型多轮对话

使用方式（在 ExplicitLM 目录下）：
    CUDA_VISIBLE_DEVICES=6 python3 chat_example.py
或直接运行（脚本内部也会设置使用卡 6）：
    python3 chat_example.py
"""

import os
from pathlib import Path
from typing import Any, Tuple

import torch

from config import get_default_config, merge_config
from utils.model_initializer import init_model


def setup_device():
    """
    优先将当前进程绑定到物理 GPU 6。
    - 如果外部已经设置了 CUDA_VISIBLE_DEVICES，则尊重外部设置；
    - 否则在脚本内部设置为 6。
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # 显式使用第 6 号 GPU（物理编号）
        os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def load_model_and_tokenizer(
    ckpt_path: Path,
    qwen_model_path: Path,
) -> Tuple[torch.nn.Module, Any, torch.device]:
    """
    加载 ExplicitLM 模型 + Tokenizer，并从 checkpoint 恢复权重。
    """
    device = setup_device()

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到模型权重文件: {ckpt_path}")

    if not qwen_model_path.exists():
        raise FileNotFoundError(f"找不到 Qwen3 预训练模型目录: {qwen_model_path}")

    # 1. 构建配置（基于默认配置 + 覆盖关键字段）
    cfg = get_default_config()

    overrides = {
        "model.qwen3_model_path": str(qwen_model_path),
        # 下面这些值建议与你训练时保持一致，如果后续有变化可以在这里修改
        "model.cache_path": "data/cache/train_data_with_extract_cache.pt",
        "model.database_init_path": "",
        "model.recompute_cache": False,
        "model.knowledge_num": 10404,
        "model.knowledge_length": 32,
        "model.knowledge_dim": 1536,
        "model.num_candidates": 16,
        "model.num_selected": 1,
        "model.keys_path": "data/keys_extract.pt",
        "model.gate_rank": 128,
        "model.fusion_rank": 128,
    }
    cfg = merge_config(cfg, overrides)

    m_cfg = cfg["model"]

    # 2. 初始化模型结构 & tokenizer（会从 Qwen3-4B 加载 backbone 权重）
    model, tokenizer = init_model(m_cfg)

    # 3. 加载 SFT / 记忆组件训练后的权重
    print(f"从 {ckpt_path} 加载模型权重...")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"⚠️ 有 {len(missing_keys)} 个权重在 checkpoint 中缺失（通常可忽略，示例：{missing_keys[:5]}）")
    if unexpected_keys:
        print(f"⚠️ 有 {len(unexpected_keys)} 个权重在模型中不存在（通常可忽略，示例：{unexpected_keys[:5]}）")

    model.to(device)
    model.eval()

    # 确保 tokenizer 的 pad/eos 设置正确（训练脚本里也有类似逻辑）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, device


def chat_loop(model, tokenizer, device):
    """
    简单命令行多轮对话：
    - 输入 q / quit / exit 退出
    """
    print("====== ExplicitLM 对话开始（输入 q / quit / exit 退出）======")

    history = []
    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出对话。")
            break

        if user_input.lower() in {"q", "quit", "exit"}:
            print("退出对话。")
            break
        if not user_input:
            continue

        # 这里使用最简单的拼接方式，你可以根据自己的数据格式改成聊天模板
        history.append(f"用户：{user_input}")
        prompt = "\n".join(history) + "\n助手："

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
            )

        # 只取新生成的部分
        gen_ids = outputs[0, input_ids.size(1) :]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # 有些模型会在一次生成里把后续的“用户：”问题也一起编出来，
        # 这里强制只保留本轮助手的第一段回答，在出现下一个“用户：”之前截断。
        cut_points = []
        for marker in ["\n用户：", "\nUser:", "\nuser:"]:
            idx = text.find(marker)
            if idx != -1:
                cut_points.append(idx)
        if cut_points:
            text = text[: min(cut_points)].strip()

        # 简单清洗：如果包含 “助手：” 等前缀，可以裁掉
        for prefix in ["助手：", "回答：", "Assistant:", "assistant:"]:
            if text.startswith(prefix):
                text = text[len(prefix) :].lstrip()
                break

        print(f"助手：{text}")
        history.append(f"助手：{text}")


def main():
    project_root = Path(__file__).parent.resolve()
    ckpt_path = project_root / "out" / "example.pth"

    # 与训练脚本保持一致的 Qwen3-4B 路径（相对于 ExplicitLM 根目录）
    project_root = Path(__file__).parent.resolve()
    qwen_model_path = project_root / "Qwen_hg" / "Qwen3-4b"

    model, tokenizer, device = load_model_and_tokenizer(
        ckpt_path=ckpt_path,
        qwen_model_path=qwen_model_path,
    )

    chat_loop(model, tokenizer, device)


if __name__ == "__main__":
    main()


