#!/usr/bin/env python3
"""
独立的评估测试脚本

用于调试和测试评估流程，使用指定的权重文件进行测试
"""

import os
import sys
import torch
from pathlib import Path
from accelerate import Accelerator
from transformers import AutoTokenizer

# 添加项目路径
proj_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(proj_root))

from config import get_default_config, merge_config
from utils.model_initializer import init_model
from utils.sft_datasets import create_sft_eval_dataloader
from utils.train_loop_sft import eval_model_sft, extract_answer_from_generation, judger
from utils.logger import Logger


def main():
    # 配置（相对于 ExplicitLM 根目录）
    checkpoint_path = "out/sft_2560.pth"
    eval_data_path = "sft_data/train_data_with_extract_sft_val.jsonl" 
    
    # 初始化 Accelerator（单卡模式用于测试）
    accelerator = Accelerator()
    
    Logger("=" * 60, accelerator)
    Logger("评估测试脚本", accelerator)
    Logger("=" * 60, accelerator)
    
    # 加载配置
    cfg = get_default_config()
    
    # 配置是字典格式
    m_cfg = cfg.get("model", {})
    d_cfg = cfg.get("dataset", {})
    tr_cfg = cfg.get("training", {})
    
    # 设置默认模型路径（如果未配置）
    if not m_cfg.get("qwen3_model_path"):
        # 尝试从环境变量或使用默认路径
        default_model_path = os.environ.get("QWEN3_MODEL_PATH", "Qwen_hg/Qwen3-4b")
        m_cfg["qwen3_model_path"] = default_model_path
        Logger(f"使用默认模型路径: {default_model_path}", accelerator)
    
    # 检查权重文件中的 memory_bank 形状，以确定正确的 knowledge_length
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict):
                if "memory_bank" in ckpt:
                    mb_shape = ckpt["memory_bank"].shape
                    if len(mb_shape) == 2:
                        knowledge_length = mb_shape[1]
                        Logger(f"从权重文件检测到 knowledge_length: {knowledge_length}", accelerator)
                        m_cfg["knowledge_length"] = knowledge_length
                # 也检查 state_dict 中的
                elif "state_dict" in ckpt and "memory_bank" in ckpt["state_dict"]:
                    mb_shape = ckpt["state_dict"]["memory_bank"].shape
                    if len(mb_shape) == 2:
                        knowledge_length = mb_shape[1]
                        Logger(f"从权重文件 state_dict 检测到 knowledge_length: {knowledge_length}", accelerator)
                        m_cfg["knowledge_length"] = knowledge_length
        except Exception as e:
            Logger(f"检查权重文件时出错: {e}，使用默认值", accelerator)
    
    # 如果权重文件中有 memory_bank，就不使用 cache_path（避免尺寸不匹配）
    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "memory_bank" in ckpt:
                Logger("权重文件中包含 memory_bank，将禁用 cache_path 以避免冲突", accelerator)
                m_cfg["cache_path"] = None
        except:
            pass
    
    # 检查权重文件
    if not os.path.exists(checkpoint_path):
        Logger(f"❌ 错误: 权重文件不存在: {checkpoint_path}", accelerator)
        return
    
    Logger(f"✓ 权重文件: {checkpoint_path}", accelerator)
    
    # 初始化模型
    Logger("初始化模型...", accelerator)
    model, tokenizer = init_model(m_cfg, accelerator)
    Logger("模型初始化完成", accelerator)
    
    # 加载权重
    Logger(f"加载权重: {checkpoint_path}", accelerator)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 处理不同的权重格式
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 移除 "module." 前缀（如果存在）
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # 加载权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            Logger(f"⚠️  缺失的键: {len(missing_keys)} 个", accelerator)
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    Logger(f"  - {key}", accelerator)
            else:
                Logger(f"  - {missing_keys[0]} ... (共{len(missing_keys)}个)", accelerator)
        
        if unexpected_keys:
            Logger(f"⚠️  意外的键: {len(unexpected_keys)} 个", accelerator)
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    Logger(f"  - {key}", accelerator)
            else:
                Logger(f"  - {unexpected_keys[0]} ... (共{len(unexpected_keys)}个)", accelerator)
        
        Logger("✓ 权重加载完成", accelerator)
    except Exception as e:
        Logger(f"❌ 加载权重失败: {e}", accelerator)
        import traceback
        Logger(traceback.format_exc(), accelerator)
        return
    
    # 准备模型
    model = accelerator.prepare(model)
    
    # 检查评估数据文件
    eval_data_path_full = proj_root / eval_data_path
    if not eval_data_path_full.exists():
        Logger(f"⚠️  评估数据文件不存在: {eval_data_path_full}", accelerator)
        Logger("使用示例数据进行测试...", accelerator)
        
        # 创建示例数据（使用正确的对话格式）
        # 注意：评估时应该使用完整的对话格式，包括 system 和 assistant 开始标记
        system_msg = "You are a helpful artificial intelligence assistant."
        example_data = [
            (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nWhat is cancer?<|im_end|>\n<|im_start|>assistant\n",
                "cancer cells"
            ),
            (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\nWhat is momentum?<|im_end|>\n<|im_start|>assistant\n",
                "momentum"
            ),
        ]
        
        # 手动测试几个样本
        Logger("\n" + "=" * 60, accelerator)
        Logger("开始手动测试评估...", accelerator)
        Logger("=" * 60, accelerator)
        
        model.eval()
        with torch.no_grad():
            for idx, (prompt, std_output) in enumerate(example_data):
                Logger(f"\n{'=' * 60}", accelerator)
                Logger(f"测试样本 {idx}", accelerator)
                Logger(f"{'=' * 60}", accelerator)
                Logger(f"\nPrompt（完整）:", accelerator)
                Logger(f"{prompt}", accelerator)
                Logger(f"\n标准答案（完整）:", accelerator)
                Logger(f"{std_output}", accelerator)
                
                try:
                    # Tokenize
                    input_ids = tokenizer(prompt)['input_ids']
                    Logger(f"输入 token 数量: {len(input_ids)}", accelerator)
                    Logger(f"最后10个 token: {input_ids[-10:]}", accelerator)
                    Logger(f"最后10个 token 解码: {repr(tokenizer.decode(input_ids[-10:]))}", accelerator)
                    
                    x = torch.tensor(
                        input_ids,
                        device=accelerator.device
                    ).unsqueeze(0)
                    
                    # 生成
                    unwrapped_model = accelerator.unwrap_model(model)
                    torch.cuda.empty_cache()  # 清理显存
                    
                    # 检查模型状态
                    Logger(f"模型训练模式: {unwrapped_model.training}", accelerator)
                    
                    # 检查输入的前几个 token 和后几个 token
                    Logger(f"输入的前5个 token: {x[0, :5].tolist()} -> {tokenizer.decode(x[0, :5].tolist())}", accelerator)
                    Logger(f"输入的后5个 token: {x[0, -5:].tolist()} -> {tokenizer.decode(x[0, -5:].tolist())}", accelerator)
                    
                    generated = unwrapped_model.generate(
                        x,
                        max_new_tokens=min(tr_cfg.get("max_new_tokens", 50), 30),  # 限制生成长度
                        max_length=m_cfg.get("max_seq_len", 256) + min(tr_cfg.get("max_new_tokens", 50), 30),
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    torch.cuda.empty_cache()  # 清理显存
                    
                    Logger(f"生成 token 数量: {generated.shape[1]}", accelerator)
                    Logger(f"输入长度: {x.shape[1]}, 生成长度: {generated.shape[1]}", accelerator)
                    Logger(f"生成的前10个 token: {generated[0, x.shape[1]:x.shape[1]+10].tolist()} -> {tokenizer.decode(generated[0, x.shape[1]:x.shape[1]+10].tolist())}", accelerator)
                    
                    # 解码原始生成文本
                    raw_generated = tokenizer.decode(
                        generated.squeeze()[x.shape[1]:].tolist(),
                        skip_special_tokens=False
                    )
                    Logger(f"\n原始生成文本（完整）:", accelerator)
                    Logger(f"{raw_generated}", accelerator)
                    
                    # 解码（跳过特殊token）
                    generated_text = tokenizer.decode(
                        generated.squeeze()[x.shape[1]:].tolist(),
                        skip_special_tokens=True
                    )
                    Logger(f"\n解码后文本（完整）:", accelerator)
                    Logger(f"{generated_text}", accelerator)
                    
                    # 清理
                    generated_text_before_extraction = generated_text.strip()
                    Logger(f"\n清理前文本（完整）:", accelerator)
                    Logger(f"{generated_text_before_extraction}", accelerator)
                    
                    # 提取答案
                    extracted = extract_answer_from_generation(generated_text_before_extraction)
                    Logger(f"\n提取后文本（完整）:", accelerator)
                    Logger(f"{extracted}", accelerator)
                    
                    # 判断
                    flag = judger(extracted, std_output, tr_cfg.get("judger_mode", "fuzzy_contains"))
                    Logger(f"\n匹配结果: {flag}", accelerator)
                    Logger(f"标准答案: {std_output}", accelerator)
                    
                    # 详细分析
                    Logger(f"\n详细分析:", accelerator)
                    Logger(f"  - 原始生成长度: {len(raw_generated)} 字符", accelerator)
                    Logger(f"  - 解码后长度: {len(generated_text)} 字符", accelerator)
                    Logger(f"  - 提取后长度: {len(extracted)} 字符", accelerator)
                    Logger(f"  - 标准答案长度: {len(std_output)} 字符", accelerator)
                    Logger(f"  - 匹配模式: {tr_cfg.get('judger_mode', 'fuzzy_contains')}", accelerator)
                    
                    if len(extracted) == 0:
                        Logger("  ⚠️  警告: 提取后的文本为空！", accelerator)
                        Logger(f"  - 原始文本是否为空: {len(generated_text) == 0}", accelerator)
                        Logger(f"  - 是否包含特殊字符: {bool(set(generated_text) & set(['<', '>', '\n', '\t']))}", accelerator)
                    
                    Logger("\n" + "-" * 60, accelerator)
                    
                except Exception as e:
                    Logger(f"❌ 测试失败: {e}", accelerator)
                    import traceback
                    Logger(traceback.format_exc(), accelerator)
        
        return
    
    # 创建评估数据加载器
    Logger(f"创建评估数据加载器: {eval_data_path_full}", accelerator)
    try:
        eval_loader = create_sft_eval_dataloader(
            eval_data_path=str(eval_data_path_full),
            system_message="You are MiniMind, a helpful artificial intelligence assistant.",
            batch_size=1,
            max_samples=50  # 测试前50个样本
        )
        eval_loader = accelerator.prepare(eval_loader)
        Logger(f"✓ 评估数据加载器创建完成，样本数: {len(eval_loader)}", accelerator)
    except Exception as e:
        Logger(f"❌ 创建评估数据加载器失败: {e}", accelerator)
        import traceback
        Logger(traceback.format_exc(), accelerator)
        return
    
    # 创建配置对象（模拟 args）
    class Args:
        def __init__(self):
            self.model = type('obj', (object,), m_cfg)()
            self.training = type('obj', (object,), tr_cfg)()
            self.dataset = type('obj', (object,), d_cfg)()
    
    args = Args()
    
    # 运行评估
    Logger("\n" + "=" * 60, accelerator)
    Logger("开始评估...", accelerator)
    Logger("=" * 60, accelerator)
    
    try:
        performance = eval_model_sft(
            model=model,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
            accelerator=accelerator,
            args=args,
            judger_mode=tr_cfg.get("judger_mode", "startswith")
        )
        
        Logger("\n" + "=" * 60, accelerator)
        Logger("评估完成", accelerator)
        Logger("=" * 60, accelerator)
        
        # 打印详细结果
        if 'overall' in performance:
            overall = performance['overall']
            Logger(f"总样本数: {overall['total_steps']}", accelerator)
            Logger(f"正确数: {overall['total_correct']}", accelerator)
            Logger(f"准确率: {overall['accuracy']:.4f}", accelerator)
        
        # 打印所有样本的详细信息
        Logger("\n" + "=" * 80, accelerator)
        Logger("所有样本的详细信息", accelerator)
        Logger("=" * 80, accelerator)
        
        sample_keys = [k for k in performance.keys() if isinstance(k, int)]
        sample_keys.sort()
        
        for i in sample_keys:
            sample = performance[i]
            Logger(f"\n{'=' * 80}", accelerator)
            Logger(f"样本 {i}", accelerator)
            Logger(f"{'=' * 80}", accelerator)
            
            Logger(f"\n【Prompt（完整）】", accelerator)
            Logger(f"{sample.get('prompt', 'N/A')}", accelerator)
            
            Logger(f"\n【标准答案（完整）】", accelerator)
            Logger(f"{sample.get('std_output', 'N/A')}", accelerator)
            
            Logger(f"\n【原始生成文本（完整，提取前）】", accelerator)
            Logger(f"{sample.get('generated_text_raw', sample.get('generated_text', 'N/A'))}", accelerator)
            
            Logger(f"\n【提取后文本（完整）】", accelerator)
            Logger(f"{sample.get('generated_text', 'N/A')}", accelerator)
            
            Logger(f"\n【匹配结果】", accelerator)
            Logger(f"  结果: {sample.get('result', 'N/A')}", accelerator)
            Logger(f"  模式: {sample.get('judger_mode', 'N/A')}", accelerator)
            
            # 显示长度信息
            std_len = len(sample.get('std_output', ''))
            gen_raw_len = len(sample.get('generated_text_raw', sample.get('generated_text', '')))
            gen_extracted_len = len(sample.get('generated_text', ''))
            Logger(f"\n【长度统计】", accelerator)
            Logger(f"  标准答案: {std_len} 字符", accelerator)
            Logger(f"  原始生成: {gen_raw_len} 字符", accelerator)
            Logger(f"  提取后: {gen_extracted_len} 字符", accelerator)
        
    except Exception as e:
        Logger(f"❌ 评估失败: {e}", accelerator)
        import traceback
        Logger(traceback.format_exc(), accelerator)


if __name__ == "__main__":
    main()

