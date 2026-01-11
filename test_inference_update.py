"""
测试推理时的知识库更新功能

功能：
1. 加载训练好的模型 (sft_2560.pth)
2. 使用 DualPathInference 进行推理
3. 验证知识库更新是否正常工作
4. 验证更新后的知识是否影响后续推理
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from utils.dual_path_inference import DualPathInference

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


def load_model_and_tokenizer(
    model_path: str,
    qwen3_model_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """加载模型和tokenizer"""
    log_print(f"加载模型: {model_path}")
    log_print(f"Qwen3模型路径: {qwen3_model_path}")
    
    # 加载配置
    qwen3_config = Qwen3Config.from_pretrained(qwen3_model_path)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        qwen3_model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 初始化模型（与训练时配置保持一致）
    # 如果keys_path存在则加载，否则随机初始化（和训练时一致）
    keys_path = "data/keys_extract.pt"  # 与训练时使用的keys_path一致
    if not os.path.exists(keys_path):
        log_print(f"⚠️  keys_path不存在: {keys_path}，将使用随机初始化（与训练时一致）")
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
    
    # 如果keys_path存在，添加到配置中
    if keys_path:
        memory_cfg["keys_path"] = keys_path
    
    from models.core.ExplicitLM import ExplicitLM
    model = ExplicitLM(qwen3_config=qwen3_config, memory_cfg=memory_cfg)
    model = model.to(device)
    
    # 加载训练好的权重
    checkpoint = torch.load(model_path, map_location=device)
    # 处理可能的键名前缀
    state_dict = {}
    for k, v in checkpoint.items():
        # 移除可能的 'module.' 前缀
        new_key = k[7:] if k.startswith('module.') else k
        state_dict[new_key] = v
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    log_print(f"模型加载完成，设备: {device}")
    return model, tokenizer


def test_inference_with_update():
    """测试推理时的知识库更新"""
    
    # 配置（相对于 ExplicitLM 根目录）
    model_path = "out/sft_2560.pth"
    qwen3_model_path = "Qwen_hg/Qwen3-4b"
    llmlingua_model_path = "llmlingua-2-bert"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        qwen3_model_path=qwen3_model_path,
        device=device,
    )
    
    # 初始化双路推理包装器
    # 注意：为了直接注入答案，我们禁用事实提取，但仍需要 memory_bank_updater 来手动注入
    # 所以需要先创建 memory_bank_updater，然后传入
    from utils.fact_extractor import FactExtractor
    from utils.memory_bank_updater import MemoryBankUpdater
    
    # 创建 fact_extractor（虽然不使用，但 memory_bank_updater 需要它）
    fact_extractor = FactExtractor(
        model_path=llmlingua_model_path,
        compression_rate=0.4
    )
    
    # 创建 memory_bank_updater
    memory_bank_updater = MemoryBankUpdater(
        model=model,
        tokenizer=tokenizer,
        fact_extractor=fact_extractor,
        update_strategy="lru",
    )
    
    # 初始化双路推理包装器（禁用自动事实提取，但保留 memory_bank_updater）
    dual_path = DualPathInference(
        model=model,
        tokenizer=tokenizer,
        fact_extractor=fact_extractor,
        memory_bank_updater=memory_bank_updater,
        enable_fact_extraction=False,  # 禁用自动事实提取，改为手动注入答案
        fact_update_frequency=1,
        update_strategy="lru",  # 最不常用替换
        compression_rate=0.4,
        llmlingua_model_path=llmlingua_model_path,
    )
    
    # 检查初始状态
    initial_stats = dual_path.get_statistics()
    log_print(f"初始统计: {initial_stats}")
    
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
    log_print("开始测试推理时的知识库更新（大海捞针测试）")
    log_print("=" * 60)
    log_print("测试说明：每个测试包含两步：")
    log_print("  1. Setup: 输入包含特定知识的事实")
    log_print("  2. Question: 询问该知识，验证是否能从记忆库中检索")
    log_print("")
    log_print("⚠️  特殊模式：直接注入答案")
    log_print("  - 在 Setup 阶段，直接将期望答案注入到 memory bank")
    log_print("  - 跳过 LLMLingua 事实提取步骤")
    log_print("  - 测试 memory bank 是否能正确检索到直接注入的答案")
    log_print("=" * 60)
    
    correct_count = 0
    total_count = len(test_cases)
    index_match_count = 0  # 记录成功索引到注入事实的测试数量
    
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
            setup_result = dual_path.generate(
                input_ids=setup_input_ids,
                input_text=test_case['setup'],
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        log_print(f"  生成: {setup_result['generated_text'][:80]}...")
        
        # 显示知识使用情况
        if 'accessed_knowledge_indices' in setup_result and setup_result['accessed_knowledge_indices']:
            log_print(f"  📚 使用了 {len(setup_result['accessed_knowledge_indices'])} 个知识条目")
        
        # 步骤2: Question - 询问之前输入的知识
        log_print(f"\n[步骤2] 询问知识:")
        log_print(f"  问题: {test_case['question']}")
        log_print(f"  期望答案包含: {test_case['expected_answer']}")
        
        # 直接注入"问题+答案"组合到 memory bank（不通过 LLMLingua 提取）
        # 格式：问题 + 答案，例如："神经链接X7设备的处理速度是847万亿次"
        question = test_case['question']
        expected_answer = test_case['expected_answer']
        
        # 组合问题和答案：将问题转换为陈述句形式
        # 按照用户示例："神经链接X7设备的处理速度是多少？" -> "神经链接X7设备的处理速度是847万亿次"
        # 去掉问号
        question_clean = question.rstrip('？?').strip()
        
        # 处理不同类型的疑问句，转换为陈述句
        # 1. "是多少" -> "是"
        if '是多少' in question_clean:
            question_clean = question_clean.replace('是多少', '是')
        # 2. "是什么" -> "是"
        elif '是什么' in question_clean:
            question_clean = question_clean.replace('是什么', '是')
        # 3. "多少" (单独出现，不在"是多少"中) -> 保留"多少"前的部分，替换"多少"及其后的内容为"是"
        elif '多少' in question_clean and '是多少' not in question_clean:
            # 例如："晶体尖塔包含多少块水晶板？" -> "晶体尖塔包含是12,456"
            # 找到"多少"的位置，保留前面的部分，加上"是"
            idx = question_clean.find('多少')
            question_clean = question_clean[:idx] + '是'
        # 4. "多久" -> "是"
        elif '多久' in question_clean:
            question_clean = question_clean.replace('多久', '是')
        # 5. "谁" -> "是"
        elif '谁' in question_clean:
            question_clean = question_clean.replace('谁', '是')
        # 6. "哪" -> 去掉"哪"，保留后面的内容，然后加上"是"
        elif '哪' in question_clean:
            # 例如："星流河流经哪三个国家？" -> "星流河流经三个国家是风之国"
            question_clean = question_clean.replace('哪', '')
            if not question_clean.endswith('是'):
                question_clean = question_clean + '是'
        # 7. 其他情况，直接加上"是"
        else:
            if not question_clean.endswith('是'):
                question_clean = question_clean + '是'
        
        # 组合成完整的知识条目：问题 + 答案
        knowledge_entry = f"{question_clean}{expected_answer}"
        
        log_print(f"\n[直接注入] 将问题+答案组合直接注入到 memory bank:")
        log_print(f"  问题: {question}")
        log_print(f"  答案: {expected_answer}")
        log_print(f"  组合知识: {knowledge_entry}")
        
        # 直接调用 memory_bank_updater 的 update_with_facts 方法
        injected_indices = []  # 记录注入的索引，用于后续检查
        if dual_path.memory_bank_updater is not None:
            # 将"问题+答案"组合作为事实列表注入
            facts_to_inject = [knowledge_entry]
            update_info = dual_path.memory_bank_updater.update_with_facts(facts_to_inject)
            injected_indices = update_info.get('update_indices', [])
            log_print(f"  ✅ 知识库已直接注入: {update_info.get('updated_count', 0)} 条事实")
            log_print(f"     - 新槽位: {update_info.get('new_slots', 0)}")
            log_print(f"     - 替换槽位: {update_info.get('replaced_slots', 0)}")
            log_print(f"     - 有效条目: {update_info.get('valid_entries', 0)}")
            log_print(f"     - 注入索引: {injected_indices}")
        else:
            log_print(f"  ❌ memory_bank_updater 不可用")
        
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
            question_result = dual_path.generate(
                input_ids=question_input_ids,
                input_text=test_case['question'],
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        raw_answer = question_result['generated_text'].strip()
        log_print(f"  原始生成: {raw_answer[:150]}...")
        
        # 首先移除thinking标签和内容
        import re
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
        
        # 显示知识使用情况并检查是否索引到了注入的事实
        if 'accessed_knowledge_indices' in question_result and question_result['accessed_knowledge_indices']:
            accessed_indices = question_result['accessed_knowledge_indices']
            log_print(f"  📚 使用了 {len(accessed_indices)} 个知识条目")
            log_print(f"     访问的索引: {accessed_indices[:20]}{'...' if len(accessed_indices) > 20 else ''}")
            
            # 检查是否索引到了注入的事实
            if injected_indices:
                injected_set = set(injected_indices)
                accessed_set = set(accessed_indices)
                matched_indices = injected_set.intersection(accessed_set)
                
                if matched_indices:
                    index_match_count += 1
                    log_print(f"  ✅ 成功索引到注入的事实！")
                    log_print(f"     - 注入的索引: {list(injected_set)}")
                    log_print(f"     - 匹配的索引: {list(matched_indices)}")
                    log_print(f"     - 匹配率: {len(matched_indices)}/{len(injected_set)} = {len(matched_indices)/len(injected_set)*100:.1f}%")
                else:
                    log_print(f"  ❌ 未索引到注入的事实！")
                    log_print(f"     - 注入的索引: {list(injected_set)}")
                    log_print(f"     - 访问的索引: {list(accessed_set)[:20]}{'...' if len(accessed_set) > 20 else ''}")
                    log_print(f"     - 交集为空，说明模型未检索到直接注入的答案")
            else:
                log_print(f"  ⚠️  无法检查：注入索引为空")
        else:
            log_print(f"  ⚠️  未记录到知识使用情况")
            if injected_indices:
                log_print(f"     - 注入的索引: {injected_indices}")
                log_print(f"     - 但问题阶段未访问任何知识条目")
        
        # 显示统计信息
        stats = dual_path.get_statistics()
        if 'memory_bank_stats' in stats:
            mb_stats = stats['memory_bank_stats']
            log_print(f"  📊 记忆库统计: 总条目={mb_stats.get('total_entries', 'N/A')}, "
                     f"有效条目={mb_stats.get('valid_entries', 'N/A') if 'valid_entries' in mb_stats else 'N/A'}")
    
    # 总结
    log_print(f"\n{'='*60}")
    log_print(f"测试总结")
    log_print(f"{'='*60}")
    log_print(f"总测试数: {total_count}")
    log_print(f"正确答案: {correct_count}")
    log_print(f"答案准确率: {correct_count/total_count*100:.1f}%")
    log_print(f"")
    log_print(f"索引匹配: {index_match_count}/{total_count} 个测试成功索引到注入的事实")
    log_print(f"索引匹配率: {index_match_count/total_count*100:.1f}%")
    log_print(f"{'='*60}")
    
    log_print("\n" + "=" * 60)
    log_print("测试完成")
    log_print("=" * 60)


if __name__ == "__main__":
    test_inference_with_update()

