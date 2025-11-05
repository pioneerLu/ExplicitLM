"""
谓词多选题生成模块

本模块用于将三元组数据（主语-谓词-宾语）转换为谓词识别的多选题格式。
主要功能：
1. 从三元组数据中提取所有谓词作为干扰项库
2. 为每个三元组生成5选1的多选题（固定模板："What is the relationship between X and Y?"）
3. 随机分配正确答案位置，并从谓词库中选择4个干扰项
4. 支持大规模数据的分批处理以避免内存溢出

数据格式要求：
输入JSON格式：
[
    {
        "target": [
            {
                "subject": "主语",
                "predicate": "谓词",
                "object": "宾语",
                "uuid": "唯一标识符"
            }
        ]
    }
]

输出JSON格式：
[
    {
        "target": [
            {
                "question": "What is the relationship between X and Y?",
                "options": "A:选项1,B:选项2,C:选项3,D:选项4,E:选项5",
                "correct_answer": "C:正确答案",
                "uuid": "唯一标识符"
            }
        ]
    }
]
"""

import json
import os
import random
from typing import List, Dict, Set, Any
from collections import defaultdict

# 选项标签常量
OPTION_KEYS: List[str] = ["A", "B", "C", "D", "E"]

# 批量写入大小（条目数）
BATCH_SIZE: int = 100000


def collect_all_predicates(input_file: str) -> List[str]:
    """
    收集输入文件中的所有唯一谓词

    遍历输入JSON文件中的所有三元组，提取其中的谓词字段，
    用于后续生成错误选项时的干扰项池。

    参数:
        input_file: 输入JSON文件路径，包含三元组数据

    返回:
        所有唯一谓词组成的列表

    异常:
        FileNotFoundError: 输入文件不存在
        json.JSONDecodeError: 输入文件格式错误
    """
    print("第一阶段：收集所有可能的谓词...")
    all_predicates: Set[str] = set()

    with open(input_file, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)
        total: int = len(data)
        print(f"文件中共 {total} 条数据")

        # 遍历所有条目收集谓词
        for i, entry in enumerate(data):
            # 进度提示（每10000条打印一次）
            if i % 10000 == 0:
                print(f"收集谓词进度: {i}/{total} ({i/total*100:.1f}%)")

            if 'target' not in entry:
                continue

            # 提取每个目标中的谓词
            for target in entry['target']:
                if isinstance(target, dict) and 'predicate' in target:
                    all_predicates.add(target['predicate'])

    result: List[str] = list(all_predicates)
    print(f"共收集 {len(result)} 个不同的谓词")
    return result


def generate_mcq_from_triple(
    subject: str,
    predicate: str,
    object_value: str,
    uuid: str,
    all_predicates: List[str],
    correct_answer_counts: Dict[str, int]
) -> Dict[str, str]:
    """
    从单个三元组生成多选题

    基于给定的三元组（主语-谓词-宾语）生成一道5选1的多选题。
    问题格式固定为："What is the relationship between {subject} and {object}?"
    正确答案随机分配到A-E中的某一个位置，其余4个位置填充从谓词库中随机选择的干扰项。

    参数:
        subject: 三元组的主语
        predicate: 三元组的谓词（正确答案）
        object_value: 三元组的宾语
        uuid: 三元组的唯一标识符
        all_predicates: 所有可用谓词列表，用于生成干扰项
        correct_answer_counts: 记录每个选项位置被选为正确答案的次数，用于平衡分布

    返回:
        包含问题、选项、正确答案和uuid的字典
    """
    # 第一阶段：生成问题文本（固定模板）
    question: str = f"What is the relationship between {subject} and {object_value}?"

    # 第二阶段：随机选择正确答案位置
    correct_option: str = random.choice(OPTION_KEYS)
    options: Dict[str, str] = {k: "" for k in OPTION_KEYS}
    options[correct_option] = predicate
    correct_answer_counts[correct_option] += 1

    # 第三阶段：生成4个错误答案（干扰项）
    wrong_options: Set[str] = set()
    while len(wrong_options) < 4:
        if len(all_predicates) > 5:
            # 从谓词库中随机选择
            wrong_predicate: str = random.choice(all_predicates)
            if wrong_predicate != predicate:
                wrong_options.add(wrong_predicate)
        else:
            # 谓词库不足时使用生成方式（避免重复）
            wrong_options.add(f"not {predicate}_{len(wrong_options)}")

    wrong_options_list: List[str] = list(wrong_options)

    # 第四阶段：将错误答案填充到剩余选项位置
    wrong_idx: int = 0
    for opt in OPTION_KEYS:
        if opt != correct_option:
            options[opt] = wrong_options_list[wrong_idx]
            wrong_idx += 1

    # 第五阶段：格式化选项字符串（A:选项1,B:选项2,...）
    options_str: str = ",".join([f"{k}:{options[k]}" for k in OPTION_KEYS])

    return {
        "question": question,
        "options": options_str,
        "correct_answer": f"{correct_option}:{options[correct_option]}",
        "uuid": uuid
    }


def write_batch_to_file(
    data: List[Dict[str, Any]],
    output_file: str,
    is_final_batch: bool
) -> None:
    """
    分批写入数据到输出文件

    采用追加写入模式，避免一次性加载所有数据导致内存溢出。
    维护正确的JSON数组格式（首次写入时添加开始括号，最后一批添加结束括号）。

    参数:
        data: 待写入的数据批次
        output_file: 输出文件路径
        is_final_batch: 是否为最后一批数据，用于控制JSON数组闭合
    """
    # 第一阶段：确定写入模式（首次写入用'w'，后续用'a'）
    mode: str = 'w' if not os.path.exists(output_file) else 'a'

    with open(output_file, mode, encoding='utf-8') as f:
        # 第二阶段：首次写入时添加JSON数组开始标记
        if mode == 'w':
            f.write('[\n')

        # 第三阶段：写入当前批次的所有数据
        for i, entry in enumerate(data):
            json_str: str = json.dumps(entry, ensure_ascii=False, indent=2)
            # 非最后一个元素添加逗号
            if i < len(data) - 1 or not is_final_batch:
                f.write(json_str + ',\n')
            else:
                f.write(json_str + '\n')

        # 第四阶段：最后一批时添加JSON数组结束标记
        if is_final_batch:
            f.write(']\n')

    print(f"已写入 {len(data)} 条数据到 {output_file}")


def convert_triples_to_predicate_mcqs(input_file: str, output_file: str) -> None:
    """
    将三元组数据转换为谓词多选题格式

    整体流程：
    1. 收集所有谓词构建干扰项池
    2. 遍历所有三元组，为每个三元组生成多选题
    3. 分批写入输出文件，避免内存溢出
    4. 统计并输出正确答案分布情况

    参数:
        input_file: 输入JSON文件路径，包含三元组数据
        output_file: 输出JSON文件路径，保存生成的多选题

    异常:
        Exception: 捕获所有异常并打印详细错误信息和堆栈跟踪
    """
    try:
        # 第一阶段：收集所有谓词
        all_predicates: List[str] = collect_all_predicates(input_file)

        print(f"第二阶段：开始转换为谓词选择题...")
        mcq_count: int = 0
        correct_answer_counts: Dict[str, int] = {k: 0 for k in OPTION_KEYS}

        with open(input_file, 'r', encoding='utf-8') as f_in:
            data: List[Dict[str, Any]] = json.load(f_in)
            total: int = len(data)

            result: List[Dict[str, Any]] = []

            # 第二阶段：遍历所有条目生成多选题
            for i, entry in enumerate(data):
                # 进度提示（每5000条打印一次）
                if i % 5000 == 0:
                    print(f"处理进度: {i}/{total} ({i/total*100:.1f}%)")

                if 'target' not in entry:
                    continue

                new_targets: List[Dict[str, str]] = []

                # 处理当前条目中的所有三元组
                for target in entry['target']:
                    # 验证数据完整性
                    if not isinstance(target, dict):
                        continue

                    required_fields: List[str] = ['subject', 'predicate', 'object', 'uuid']
                    if not all(field in target for field in required_fields):
                        continue

                    # 提取三元组字段
                    subject: str = target['subject']
                    predicate: str = target['predicate']
                    object_value: str = target['object']
                    uuid: str = target['uuid']

                    # 生成单个多选题
                    mcq: Dict[str, str] = generate_mcq_from_triple(
                        subject, predicate, object_value, uuid,
                        all_predicates, correct_answer_counts
                    )

                    new_targets.append(mcq)
                    mcq_count += 1

                # 将生成的多选题添加到结果中
                if new_targets:
                    result.append({"target": new_targets})

                # 第三阶段：达到批量大小时写入文件
                if len(result) >= BATCH_SIZE:
                    write_batch_to_file(result, output_file, i == total - 1)
                    result = []

            # 第四阶段：写入剩余数据
            if result:
                write_batch_to_file(result, output_file, True)

        # 第五阶段：输出统计信息
        print(f"\n转换完成！共生成 {mcq_count} 道谓词选择题")
        print("正确答案分布:", correct_answer_counts)

        # 验证正确答案分布的均衡性
        if mcq_count > 0:
            expected_percentage: float = 100.0 / len(OPTION_KEYS)
            print("\n正确答案分布详情:")
            for option, count in correct_answer_counts.items():
                percentage: float = count / mcq_count * 100
                print(f"  {option}: {count} 次 ({percentage:.2f}%, 期望: {expected_percentage:.2f}%)")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """
    主函数：执行谓词多选题生成流程

    配置输入输出文件路径并调用转换函数。
    """
    # 配置文件路径（相对于当前脚本的位置）
    input_file: str = "../../dataset/sentence_trex_data.json"
    output_file: str = "../../dataset/pmcq_trex_data.json"

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("-" * 60)

    # 验证输入文件存在性
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    convert_triples_to_predicate_mcqs(input_file, output_file)


if __name__ == "__main__":
    main()