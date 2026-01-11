#!/usr/bin/env python3
"""
对比测试：原始checkpoint推理 vs HuggingFace格式推理
使用正确的chat template格式和合理的生成参数
"""
import os
import sys
import torch
from pathlib import Path

# 设置环境变量
os.environ["EXPLICITLM_ROOT"] = str(Path(__file__).parent.absolute())

from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from models.core.ExplicitLM import ExplicitLM


def format_chat_prompt(text: str, system_message: str = None) -> str:
    """
    将文本格式化为Qwen对话格式（与训练时一致）
    """
    if system_message is None:
        system_message = "You are a helpful assistant."
    
    return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"


def load_original_checkpoint(checkpoint_path: str, qwen3_path: str, device: str = "cuda"):
    """直接加载原始checkpoint进行推理"""
    print("=" * 60)
    print("📦 测试1: 直接加载原始checkpoint")
    print("=" * 60)
    
    # 加载配置
    qwen3_config = Qwen3Config.from_pretrained(qwen3_path)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        qwen3_path,
        trust_remote_code=True,
        fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 初始化模型（与训练时配置保持一致）
    keys_path = "data/keys_extract.pt"
    if not os.path.exists(keys_path):
        print(f"⚠️  keys_path不存在: {keys_path}，将使用随机初始化")
        keys_path = None
    
    memory_cfg = {
        "knowledge_num": 1024 * 1024,  # 1048576
        "knowledge_length": 32,
        "knowledge_dim": 1536,
        "num_candidates": 16,
        "num_selected": 1,
        "gumbel_temperature": 1.0,
        "use_moe": False,
        "dropout": 0.0,
        "gate_rank": 128,
        "fusion_rank": 128,
        "trainable_keys": False,  # 推理时冻结keys
    }
    
    if keys_path:
        memory_cfg["keys_path"] = keys_path
    
    # 创建模型
    model = ExplicitLM(qwen3_config=qwen3_config, memory_cfg=memory_cfg)
    model = model.to(device)
    
    # 先加载Qwen3基础权重（包括embed_tokens和lm_head）
    print(f"\n加载Qwen3基础权重...")
    try:
        from transformers import Qwen3ForCausalLM
        qwen3_model = Qwen3ForCausalLM.from_pretrained(
            qwen3_path,
            torch_dtype=torch.float32,
            device_map=device if device == "cpu" else None,
        )
        if device != "cpu":
            qwen3_model = qwen3_model.to(device)
        
        # 映射Qwen3权重到ExplicitLM
        qwen3_state_dict = qwen3_model.state_dict()
        model_state_dict = model.state_dict()
        
        def map_qwen3_key(qwen3_key: str) -> str:
            """将Qwen3权重名称映射到ExplicitLM格式"""
            # 移除 "model." 前缀
            if qwen3_key.startswith("model."):
                key = qwen3_key[6:]
            else:
                key = qwen3_key
            
            # 映射layers到qwen3_decoder
            if key.startswith("layers."):
                parts = key.split(".", 2)
                if len(parts) >= 3:
                    layer_idx = parts[1]
                    rest = parts[2]
                    return f"layers.{layer_idx}.qwen3_decoder.{rest}"
                else:
                    return key
            
            # embed_tokens, norm, lm_head, rotary_emb 直接使用
            return key
        
        qwen3_loaded = 0
        for qwen3_key, weight in qwen3_state_dict.items():
            model_key = map_qwen3_key(qwen3_key)
            if model_key in model_state_dict:
                if model_state_dict[model_key].shape == weight.shape:
                    model_state_dict[model_key].copy_(weight.to(device))
                    qwen3_loaded += 1
        
        model.load_state_dict(model_state_dict, strict=False)
        print(f"✅ 加载Qwen3基础权重: {qwen3_loaded} 个")
        del qwen3_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"⚠️  加载Qwen3基础权重失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 加载checkpoint（只包含trainable components）
    print(f"\n加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 处理state_dict
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 处理可能的键名前缀
    processed_state_dict = {}
    for k, v in state_dict.items():
        # 移除可能的 'module.' 前缀
        new_key = k[7:] if k.startswith('module.') else k
        processed_state_dict[new_key] = v
    
    # 加载权重
    missing_keys, unexpected_keys = model.load_state_dict(processed_state_dict, strict=False)
    if missing_keys:
        print(f"⚠️  缺失的键: {len(missing_keys)} 个（前5个: {missing_keys[:5]}）")
    if unexpected_keys:
        print(f"⚠️  意外的键: {len(unexpected_keys)} 个（前5个: {unexpected_keys[:5]}）")
    
    model.eval()
    print("✅ 原始checkpoint加载完成")
    
    return model, tokenizer


def inference_with_original_model(model, tokenizer, prompt: str, device: str = "cuda"):
    """使用原始模型进行推理（使用chat template格式和合理的生成参数）"""
    print(f"\n输入prompt（使用chat template）: {prompt[:100]}...")
    
    # 使用chat template格式
    formatted_prompt = format_chat_prompt(prompt)
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # 使用ExplicitLM的generate方法（如果可用）
        if hasattr(model, 'generate'):
            try:
                # 使用模型的generate方法
                generated_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
                generated_token_ids = generated_ids[0][inputs["input_ids"].shape[1]:].tolist()
            except Exception as e:
                print(f"⚠️  使用模型generate方法失败，改用手动生成: {e}")
                generated_token_ids = _manual_generate(model, tokenizer, inputs, device)
        else:
            generated_token_ids = _manual_generate(model, tokenizer, inputs, device)
        
        # 解码生成的token
        response = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        print(f"回复: {response}")
        return response


def _manual_generate(model, tokenizer, inputs, device, max_new_tokens=50):
    """手动生成token（简单greedy decoding）"""
    generated_tokens = []
    input_ids = inputs["input_ids"]
    current_ids = input_ids
    
    for step in range(max_new_tokens):
        # 使用ExplicitLM的forward方法
        outputs = model(**{"input_ids": current_ids})
        
        # 获取logits
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif isinstance(outputs, tuple):
            logits = outputs[0] if len(outputs) > 0 else None
        else:
            logits = outputs
        
        if logits is None:
            break
        
        # 获取最后一个token的logits
        next_token_logits = logits[0, -1, :]
        next_token_id = next_token_logits.argmax().item()
        
        # 检查是否到达EOS
        if next_token_id == tokenizer.eos_token_id:
            break
        
        generated_tokens.append(next_token_id)
        
        # 准备下一轮输入
        next_token_tensor = torch.tensor([[next_token_id]], device=device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)
        
        # 扩展attention_mask
        if "attention_mask" in inputs:
            next_attention_mask = torch.cat([
                inputs["attention_mask"],
                torch.ones((1, 1), device=device, dtype=inputs["attention_mask"].dtype)
            ], dim=1)
            inputs["attention_mask"] = next_attention_mask
    
    return generated_tokens


def inference_with_hf_model(hf_model_path: str, prompt: str):
    """使用HuggingFace格式模型进行推理（使用chat template格式和合理的生成参数）"""
    print("\n" + "=" * 60)
    print("📦 测试2: 使用HuggingFace格式模型")
    print("=" * 60)
    
    from transformers import AutoModelForCausalLM
    
    print(f"\n加载HF模型: {hf_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_model_path,
        trust_remote_code=True,
        fix_mistral_regex=True
    )
    model = AutoModelForCausalLM.from_pretrained(hf_model_path, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    print("✅ HF模型加载完成")
    
    # 使用chat template格式
    formatted_prompt = format_chat_prompt(prompt)
    print(f"\n输入prompt（使用chat template）: {formatted_prompt[:100]}...")
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # 使用greedy decoding，与原始模型保持一致
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,  # 添加重复惩罚，避免重复生成
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"回复: {response}")
    return response


def main():
    """主测试函数"""
    # 配置
    checkpoint_path = "checkpoints/fusion_pretrain/checkpoint_step_14500/trainable_components.pth"
    qwen3_path = "Qwen_hg/Qwen3-4b"
    hf_model_path = "hf_explicitlm_model_step_14500_fixed"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 测试prompt（裸文本，会在函数内部格式化为chat template）
    user_prompt = "你好，请介绍一下人工智能"
    
    print("\n" + "=" * 60)
    print("🔬 Checkpoint对比测试（使用Chat Template格式）")
    print("=" * 60)
    print(f"Checkpoint路径: {checkpoint_path}")
    print(f"Qwen3路径: {qwen3_path}")
    print(f"HF模型路径: {hf_model_path}")
    print(f"设备: {device}")
    print(f"用户输入: {user_prompt}")
    print("=" * 60)
    
    # 测试1: 原始checkpoint推理
    try:
        model, tokenizer = load_original_checkpoint(checkpoint_path, qwen3_path, device)
        original_response = inference_with_original_model(model, tokenizer, user_prompt, device)
    except Exception as e:
        print(f"❌ 原始checkpoint推理失败: {e}")
        import traceback
        traceback.print_exc()
        original_response = None
    
    # 测试2: HF格式推理（需要先转换）
    print("\n" + "=" * 60)
    print("📝 步骤: 先转换checkpoint到HF格式")
    print("=" * 60)
    
    # 检查是否已转换
    if not os.path.exists(hf_model_path):
        print(f"⚠️  HF模型不存在，开始转换...")
        import subprocess
        cmd = [
            "uv", "run", "python", "convert_checkpoint_to_hf.py",
            "--checkpoint_path", checkpoint_path,
            "--qwen3_path", qwen3_path,
            "--output_path", hf_model_path
        ]
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ 转换失败: {result.stderr}")
            return
        print("✅ 转换完成")
    else:
        print(f"✅ HF模型已存在: {hf_model_path}")
    
    # 使用HF格式推理
    try:
        hf_response = inference_with_hf_model(hf_model_path, user_prompt)
    except Exception as e:
        print(f"❌ HF格式推理失败: {e}")
        import traceback
        traceback.print_exc()
        hf_response = None
    
    # 对比结果
    print("\n" + "=" * 60)
    print("📊 结果对比")
    print("=" * 60)
    print(f"原始checkpoint回复: {original_response}")
    print(f"HF格式回复: {hf_response}")
    
    if original_response and hf_response:
        if original_response == hf_response:
            print("✅ 两种方式生成的回复完全一致！")
        else:
            print("⚠️  两种方式生成的回复不一致")
            print(f"   差异长度: {abs(len(original_response) - len(hf_response))}")
            # 显示前50个字符的差异
            min_len = min(len(original_response), len(hf_response))
            if min_len > 0:
                for i in range(min(min_len, 50)):
                    if original_response[i] != hf_response[i]:
                        print(f"   第一个差异位置: {i}")
                        print(f"   原始: '{original_response[max(0,i-10):i+10]}'")
                        print(f"   HF:   '{hf_response[max(0,i-10):i+10]}'")
                        break
    else:
        print("⚠️  无法对比：至少有一种方式推理失败")


if __name__ == "__main__":
    main()
