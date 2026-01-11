"""
基线对比测试：使用Qwen3-4B原生模型（不带记忆库）

功能：
1. 加载Qwen3-4B原生模型
2. 使用相同的测试用例进行推理
3. 与ExplicitLM版本进行对比
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 为了确保日志能正常输出，直接使用print
def log_print(msg: str):
    """直接打印日志，确保能看到输出"""
    print(msg)
    import sys
    sys.stdout.flush()  # 立即刷新输出缓冲区


def format_chat_prompt(text: str, system_message: str = None, is_question: bool = False) -> str:
    """
    将文本格式化为Qwen对话格式
    
    Args:
        text: 用户输入文本
        system_message: 系统提示消息（如果为None，使用默认的知识问答助手提示）
        is_question: 是否为问题（如果是问题，会在prompt中要求使用特殊符号包裹答案）
    
    Returns:
        格式化的对话prompt
    """
    if system_message is None:
        if is_question:
            # 对于问题，更强调不要thinking，直接回答
            system_message = "你是一个知识问答助手。你的任务是直接回答问题，不要进行推理思考，不要使用任何thinking标签，不要重复问题。直接给出简洁的答案，并用 <answer>答案</answer> 格式包裹。"
        else:
            # 对于知识输入，只需要接收
            system_message = "你是一个知识接收助手。你的任务是接收并理解用户提供的知识信息。"
    
    # 如果是问题，在用户输入中添加要求使用特殊符号包裹答案的指令
    if is_question:
        text_with_instruction = f"{text}\n\n请用 <answer>答案</answer> 的格式包裹你的答案。"
    else:
        text_with_instruction = text
    
    return f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{text_with_instruction}<|im_end|>\n<|im_start|>assistant\n"


def load_baseline_model_and_tokenizer(
    qwen3_model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """加载Qwen3-4B原生模型和tokenizer"""
    log_print(f"加载Qwen3-4B原生模型: {qwen3_model_path}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        qwen3_model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载原生Qwen3模型
    model = AutoModelForCausalLM.from_pretrained(
        qwen3_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # 使用bfloat16以节省显存
        device_map="auto" if device == "cuda" else None,
    )
    
    if device == "cuda" and model.device.type != "cuda":
        model = model.to(device)
    
    model.eval()
    
    log_print(f"模型加载完成，设备: {device}")
    return model, tokenizer


def extract_answer(raw_answer: str) -> str:
    """
    从生成的文本中提取答案
    
    1. 移除thinking标签和内容
    2. 提取 <answer>...</answer> 格式
    3. 如果未找到，使用清理后的回答
    """
    import re
    
    # 首先移除thinking标签和内容
    thinking_markers = ['<think>', '<reasoning>', '<thinking>', '<thought>']
    cleaned_answer = raw_answer
    for marker in thinking_markers:
        if marker in cleaned_answer:
            # 找到thinking标签的位置
            marker_idx = cleaned_answer.find(marker)
            # 移除thinking标签及其后的内容，直到遇到换行后的实际内容
            after_marker = cleaned_answer[marker_idx + len(marker):].strip()
            lines = after_marker.split('\n')
            # 跳过明显的thinking内容
            actual_content = ''
            thinking_prefixes = ('好的', 'okay', 'the user', 'let me', 'i need', 'i should', 'i think', 'this is', 'well', 'so', 'now', '首先', '接下来', '我需要', '我应该')
            for line in lines:
                line = line.strip()
                # 跳过thinking内容和空行
                if line and not any(line.lower().startswith(prefix) for prefix in thinking_prefixes):
                    # 检查是否包含问号（可能是问题而不是答案）
                    if '？' not in line and '?' not in line:
                        actual_content = line
                        break
            if actual_content:
                cleaned_answer = actual_content
            else:
                # 如果没有找到实际内容，移除thinking标签部分
                cleaned_answer = cleaned_answer[:marker_idx].strip()
            break
    
    # 从清理后的文本中提取被 <answer>...</answer> 包裹的答案
    answer = None
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, cleaned_answer, re.DOTALL)
    if matches:
        # 如果找到多个，取第一个
        answer = matches[0].strip()
        log_print(f"  ✅ 找到 <answer> 格式，提取的答案: {answer}")
    else:
        # 如果没有找到特殊符号，使用清理后的回答
        answer = cleaned_answer
        # 移除 <|im_end|> 标记
        if "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        # 移除可能的"助手："等前缀
        for prefix in ["助手：", "助手:", "Assistant:", "assistant:"]:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()
        # 如果答案仍然很长，尝试截取第一句话
        if len(answer) > 100:
            for sep in ["。", ".", "\n", "，"]:
                if sep in answer:
                    answer = answer.split(sep)[0].strip()
                    break
        log_print(f"  ⚠️  未找到 <answer> 格式，使用清理后的回答: {answer}...")
    
    return answer


def test_baseline_inference():
    """测试Qwen3-4B原生模型的推理（基线对比）"""
    
    qwen3_model_path = "Qwen_hg/Qwen3-4b"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model, tokenizer = load_baseline_model_and_tokenizer(
        qwen3_model_path=qwen3_model_path,
        device=device,
    )
    
    # "大海捞针"测试用例：使用完全虚构的新奇知识（模型训练时未见过）
    # 这样可以验证知识库更新是否真正起作用，而不是依赖预训练知识
    test_cases = [
        {
            "name": "测试1: 虚构人物信息",
            "setup": "张星辰是一位量子生物学家，在时间遗传学研究所工作。他的研究重点是时序形态DNA序列。他于2047年3月17日出生在新光市。",
            "question": "张星辰的研究重点是什么？",
            "expected_answer": "时序形态DNA序列",
        },
        {
            "name": "测试2: 虚构数字事实",
            "setup": "位于翡翠城的晶体尖塔纪念碑于2156年9月23日竣工。它高847.3米，包含12,456块独立的水晶板，这些水晶板可以产生太阳能。",
            "question": "晶体尖塔包含多少块水晶板？",
            "expected_answer": "12,456",
        },
        {
            "name": "测试3: 虚构科学发现",
            "setup": "艾莉亚·维斯珀博士发现，元素木素-92在暴露于量子波动时，其半衰期恰好为47.3天。这一发现于2093年11月8日发表在《合成化学期刊》上。",
            "question": "木素-92的半衰期是多少？",
            "expected_answer": "47.3天",
        },
        {
            "name": "测试4: 虚构历史事件",
            "setup": "大同步事件发生在2089年2月14日，当时所有全球网络精确对齐了3.7秒。这一事件由量子联盟策划，由指挥官莱拉·内克斯和她的127名专家团队领导。",
            "question": "大同步事件期间，谁领导了量子联盟？",
            "expected_answer": "指挥官莱拉·内克斯",
        },
        {
            "name": "测试5: 虚构地理知识",
            "setup": "星流河流经三个国家：风之国、光之国和以太国。它发源于海拔9,847米的天际山，具有独特的特性：由于生物发光微生物，其河水在夜晚会发出蓝光。",
            "question": "星流河流经哪三个国家？",
            "expected_answer": "风之国",
        },
        {
            "name": "测试6: 虚构技术规格",
            "setup": "神经链接X7设备的处理速度为847万亿次浮点运算，使用名为量子同步的专有算法。它由科技漩涡工业公司开发，于2112年6月30日发布。该设备重量恰好为234.7克。",
            "question": "神经链接X7设备的处理速度是多少？",
            "expected_answer": "847万亿次",
        },
        {
            "name": "测试7: 虚构生物特征",
            "setup": "闪光翼蝴蝶物种的翼展为23.4厘米，仅生活在迷雾森林中。它的独特特征是每47秒可以改变一次颜色。该物种由奥里昂·星织博士于2098年4月5日首次记录。",
            "question": "闪光翼蝴蝶多久可以改变一次颜色？",
            "expected_answer": "47秒",
        },
    ]
    
    log_print("=" * 60)
    log_print("基线测试：Qwen3-4B原生模型（无记忆库）")
    log_print("=" * 60)
    log_print("测试说明：每个测试包含两步：")
    log_print("  1. Setup: 输入包含特定知识的事实")
    log_print("  2. Question: 询问该知识，验证模型是否能回答")
    log_print("  注意：原生模型没有记忆库，只能依赖预训练知识或上下文")
    log_print("=" * 60)
    
    correct_count = 0
    total_count = len(test_cases)
    
    for test_idx, test_case in enumerate(test_cases):
        log_print(f"\n{'='*60}")
        log_print(f"测试 {test_idx + 1}/{total_count}: {test_case['name']}")
        log_print(f"{'='*60}")
        
        # 步骤1: Setup - 输入包含知识的事实
        log_print(f"\n[步骤1] 输入知识事实:")
        log_print(f"  {test_case['setup']}")
        
        # 使用对话格式（作为用户输入，不是问题，所以is_question=False）
        setup_prompt = format_chat_prompt(test_case['setup'], is_question=False)
        setup_input_ids = tokenizer(
            setup_prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=256,
        )["input_ids"].to(device)
        
        with torch.no_grad():
            setup_outputs = model.generate(
                setup_input_ids,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 解码生成的文本（只解码新生成的部分）
        input_length = setup_input_ids.shape[1]
        setup_generated_text = tokenizer.decode(
            setup_outputs[0, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        log_print(f"  生成: {setup_generated_text[:80]}...")
        
        # 步骤2: Question - 询问之前输入的知识
        log_print(f"\n[步骤2] 询问知识:")
        log_print(f"  问题: {test_case['question']}")
        log_print(f"  期望答案包含: {test_case['expected_answer']}")
        
        # 使用对话格式（作为用户问题，is_question=True会添加答案格式要求）
        question_prompt = format_chat_prompt(test_case['question'], is_question=True)
        question_input_ids = tokenizer(
            question_prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=256,
        )["input_ids"].to(device)
        
        with torch.no_grad():
            question_outputs = model.generate(
                question_input_ids,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # 解码生成的文本（只解码新生成的部分）
        input_length = question_input_ids.shape[1]
        raw_answer = tokenizer.decode(
            question_outputs[0, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()
        
        log_print(f"  原始生成: {raw_answer[:150]}...")
        
        # 提取答案
        answer = extract_answer(raw_answer)
        log_print(f"  最终答案: {answer}")
        
        # 检查答案是否正确（包含期望的关键词）
        expected_lower = test_case['expected_answer'].lower()
        answer_lower = answer.lower()
        is_correct = expected_lower in answer_lower
        
        if is_correct:
            correct_count += 1
            log_print(f"  ✅ 答案正确！找到了期望的关键词: '{test_case['expected_answer']}'")
        else:
            log_print(f"  ❌ 答案未包含期望的关键词: '{test_case['expected_answer']}'")
            log_print(f"     模型可能无法记住之前输入的知识（原生模型没有记忆库）")
    
    # 总结
    log_print(f"\n{'='*60}")
    log_print(f"基线测试总结")
    log_print(f"{'='*60}")
    log_print(f"总测试数: {total_count}")
    log_print(f"正确答案: {correct_count}")
    log_print(f"准确率: {correct_count/total_count*100:.1f}%")
    log_print(f"{'='*60}")
    log_print(f"\n注意：原生模型没有记忆库机制，无法在推理时动态更新知识。")
    log_print(f"如果准确率较低，说明模型无法记住之前输入的知识，")
    log_print(f"这可以证明ExplicitLM的记忆库机制的有效性。")
    log_print(f"{'='*60}")
    
    log_print("\n" + "=" * 60)
    log_print("基线测试完成")
    log_print("=" * 60)


if __name__ == "__main__":
    test_baseline_inference()

