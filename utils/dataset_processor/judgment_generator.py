"""
判断题生成模块

本模块用于将三元组数据（主语-谓词-宾语）转换为是非判断题格式。
主要功能：
1. 加载谓词对应的一般疑问句模板（支持自定义和两种默认模板）
2. 从三元组数据中提取所有宾语作为错误答案库
3. 为每个三元组生成一对判断题：
   - 正确判断题（使用原始宾语，答案为"yes"）
   - 错误判断题（使用随机错误宾语，答案为"no"）
4. 支持大规模数据的分批处理以避免内存溢出
5. 统计并输出yes/no答案分布

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
                "subject": "主语",
                "predicate": "谓词",
                "object": "宾语（或错误宾语）",
                "question": "Is [object] the [predicate] of [subject]?",
                "answer": "yes/no",
                "uuid": "唯一标识符"
            }
        ]
    }
]

模板文件格式：
// 注释行（以//开头）
// 通用一般疑问句模板：
// Is {object} the {predicate} of {subject}?
// Does {subject} {predicate} {object}?
predicate1 - "一般疑问句模板1，使用{subject}和{object}占位符"
predicate2 - "一般疑问句模板2，使用{subject}和{object}占位符"
"""

import json
import os
import random
from typing import Dict, List, Set, Tuple, Any

# 批量写入大小（条目数）
BATCH_SIZE: int = 100000

# 默认问题模板
DEFAULT_TEMPLATE: str = "Is {object} the {predicate} of {subject}?"
ALT_DEFAULT_TEMPLATE: str = "Does {subject} {predicate} {object}?"

# 需要使用备用模板的谓词前缀列表
VERB_PREFIXES: Tuple[str, ...] = (
    'is', 'was', 'are', 'were', 'has', 'have', 'does', 'do', 'did'
)


def load_question_templates(
    template_file: str
) -> Tuple[Dict[str, str], str, str]:
    """
    加载谓词对应的一般疑问句模板

    从模板文件中读取谓词到问题模板的映射关系。
    文件格式：
    - 注释行以 "//" 开头
    - 模板定义格式：predicate - "问题模板"
    - 通用模板定义在注释中：
      * Is {object} the {predicate} of {subject}?
      * Does {subject} {predicate} {object}?

    参数:
        template_file: 模板文件路径

    返回:
        (templates, default_template, alt_default_template) 三元组:
        - templates: 谓词到问题模板的字典映射
        - default_template: 未在映射中找到时使用的第一种默认模板
        - alt_default_template: 未在映射中找到时使用的第二种默认模板（动词类谓词）

    异常:
        FileNotFoundError: 模板文件不存在时发出警告但不中断执行
    """
    templates: Dict[str, str] = {}
    default_template: str = DEFAULT_TEMPLATE
    alt_default_template: str = ALT_DEFAULT_TEMPLATE

    try:
        # 第一阶段：打开并读取模板文件
        with open(template_file, 'r', encoding='utf-8') as f:
            lines: List[str] = f.readlines()

            # 第二阶段：解析每一行
            for i, line in enumerate(lines):
                line = line.strip()

                # 跳过空行和注释行（但从注释中提取默认模板）
                if not line or line.startswith("//"):
                    # 提取通用模板定义
                    if "通用一般疑问句模板" in line:
                        # 查找后续行中的模板定义
                        for j in range(i + 1, min(i + 10, len(lines))):
                            next_line: str = lines[j].strip()
                            if "Is {object} the {predicate} of {subject}?" in next_line:
                                # 提取模板（去除前导//和空格）
                                default_template = next_line.lstrip("//").strip()
                            if "Does {subject} {predicate} {object}?" in next_line:
                                alt_default_template = next_line.lstrip("//").strip()
                    continue

                # 第三阶段：解析谓词模板映射（格式：predicate - "template"）
                if " - " in line:
                    parts: List[str] = line.split(" - ", 1)
                    if len(parts) == 2:
                        predicate: str = parts[0].strip()
                        question: str = parts[1].strip().strip('"')
                        templates[predicate] = question

    except FileNotFoundError:
        print(f"警告：模板文件 '{template_file}' 未找到，使用默认模板")

    print(f"加载了 {len(templates)} 个谓词的一般疑问句模板")
    return templates, default_template, alt_default_template


def collect_all_objects(input_file: str) -> List[str]:
    """
    收集输入文件中的所有唯一宾语

    遍历输入JSON文件中的所有三元组，提取其中的宾语字段，
    用于后续生成错误判断题时的错误答案池。

    参数:
        input_file: 输入JSON文件路径，包含三元组数据

    返回:
        所有唯一宾语组成的列表

    异常:
        FileNotFoundError: 输入文件不存在
        json.JSONDecodeError: 输入文件格式错误
    """
    print("第一阶段：收集所有可能的宾语...")
    all_objects: Set[str] = set()

    # 第二阶段：流式处理减少内存占用
    with open(input_file, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)
        total: int = len(data)
        print(f"文件中共 {total} 条数据")

        # 遍历所有条目收集宾语
        for i, entry in enumerate(data):
            # 进度提示（每10000条打印一次）
            if i % 10000 == 0:
                print(f"收集宾语进度: {i}/{total} ({i/total*100:.1f}%)")

            if 'target' not in entry:
                continue

            # 提取每个目标中的宾语
            for target in entry['target']:
                if isinstance(target, dict) and 'object' in target:
                    all_objects.add(target['object'])

    # 第三阶段：转换为列表方便随机选择
    result: List[str] = list(all_objects)
    print(f"共收集 {len(result)} 个不同的宾语")

    return result


def select_appropriate_template(
    predicate: str,
    templates: Dict[str, str],
    default_template: str,
    alt_default_template: str
) -> str:
    """
    为给定谓词选择合适的问题模板

    优先使用谓词特定模板，如果不存在则根据谓词特征选择默认模板：
    - 动词类谓词（以is/was/are等开头）使用备用默认模板
    - 其他谓词使用标准默认模板

    参数:
        predicate: 三元组的谓词
        templates: 谓词到问题模板的映射
        default_template: 标准默认模板
        alt_default_template: 备用默认模板（动词类谓词）

    返回:
        选定的问题模板字符串
    """
    # 第一阶段：检查是否有谓词特定模板
    if predicate in templates:
        return templates[predicate]

    # 第二阶段：根据谓词特征选择默认模板
    # 检查谓词是否以动词前缀开头
    if predicate.lower().startswith(VERB_PREFIXES):
        return alt_default_template
    else:
        return default_template


def generate_judgment_pair(
    subject: str,
    predicate: str,
    object_value: str,
    uuid: str,
    all_objects: List[str],
    question_template: str
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    从单个三元组生成一对判断题（正确+错误）

    基于给定的三元组生成两道判断题：
    1. 正确判断题：使用原始宾语，答案为"yes"
    2. 错误判断题：使用从宾语库中随机选择的错误宾语，答案为"no"

    参数:
        subject: 三元组的主语
        predicate: 三元组的谓词
        object_value: 三元组的宾语（正确答案）
        uuid: 三元组的唯一标识符
        all_objects: 所有可用宾语列表，用于生成错误答案
        question_template: 问题模板字符串

    返回:
        (yes_judgment, no_judgment) 元组，分别为正确和错误判断题
    """
    # 第一阶段：生成正确判断题（答案为yes）
    yes_question: str = (
        question_template
        .replace("{subject}", subject)
        .replace("{object}", object_value)
        .replace("{predicate}", predicate)
    )

    yes_judgment: Dict[str, str] = {
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "question": yes_question,
        "answer": "yes",
        "uuid": uuid
    }

    # 第二阶段：选择错误宾语
    wrong_object: str = object_value
    # 确保选择的宾语不同于正确答案
    while wrong_object == object_value:
        if len(all_objects) > 1:
            # 从宾语库中随机选择
            wrong_object = random.choice(all_objects)
        else:
            # 宾语库不足时使用生成方式（避免过长）
            if len(object_value) < 20:
                wrong_object = f"not {object_value}"
            else:
                wrong_object = "something else"
            break

    # 第三阶段：生成错误判断题（答案为no）
    no_question: str = (
        question_template
        .replace("{subject}", subject)
        .replace("{object}", wrong_object)
        .replace("{predicate}", predicate)
    )

    no_judgment: Dict[str, str] = {
        "subject": subject,
        "predicate": predicate,
        "object": wrong_object,
        "question": no_question,
        "answer": "no",
        "uuid": uuid
    }

    return yes_judgment, no_judgment


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


def convert_triples_to_judgments(
    input_file: str,
    output_file: str,
    template_file: str
) -> None:
    """
    将三元组数据转换为判断题格式

    整体流程：
    1. 加载问题模板（谓词特定模板和两种默认模板）
    2. 收集所有宾语构建错误答案池
    3. 遍历所有三元组，为每个三元组生成一对判断题（yes+no）
    4. 分批写入输出文件，避免内存溢出
    5. 统计并输出yes/no答案分布

    参数:
        input_file: 输入JSON文件路径，包含三元组数据
        output_file: 输出JSON文件路径，保存生成的判断题
        template_file: 问题模板文件路径

    异常:
        Exception: 捕获所有异常并打印详细错误信息和堆栈跟踪
    """
    try:
        # 第一阶段：加载问题模板
        templates, default_template, alt_default_template = load_question_templates(
            template_file
        )

        # 第二阶段：收集所有宾语
        all_objects: List[str] = collect_all_objects(input_file)

        print(f"第三阶段：开始转换为判断题...")
        judgment_count: int = 0
        yes_count: int = 0
        no_count: int = 0

        # 第三阶段：读取输入文件并处理
        with open(input_file, 'r', encoding='utf-8') as f_in:
            data: List[Dict[str, Any]] = json.load(f_in)
            total: int = len(data)

            result: List[Dict[str, Any]] = []

            # 遍历所有条目生成判断题
            for i, entry in enumerate(data):
                # 进度提示（每5000条打印一次）
                if i % 5000 == 0:
                    print(f"处理进度: {i}/{total} ({i/total*100:.1f}%)")

                if 'target' not in entry:
                    continue

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

                    # 选择合适的问题模板
                    question_template: str = select_appropriate_template(
                        predicate, templates, default_template, alt_default_template
                    )

                    # 生成一对判断题（yes+no）
                    yes_judgment, no_judgment = generate_judgment_pair(
                        subject, predicate, object_value, uuid,
                        all_objects, question_template
                    )

                    # 添加到结果中（每个判断题作为独立条目）
                    result.append({"target": [yes_judgment]})
                    result.append({"target": [no_judgment]})

                    judgment_count += 2
                    yes_count += 1
                    no_count += 1

                # 第四阶段：达到批量大小时写入文件
                if len(result) >= BATCH_SIZE:
                    write_batch_to_file(result, output_file, i == total - 1)
                    result = []

            # 第五阶段：写入剩余数据
            if result:
                write_batch_to_file(result, output_file, True)

        # 第六阶段：输出统计信息
        print(f"\n转换完成！共生成 {judgment_count} 道判断题")
        print(f"答案分布: Yes: {yes_count} ({yes_count/judgment_count*100:.1f}%), "
              f"No: {no_count} ({no_count/judgment_count*100:.1f}%)")

        # 验证yes/no分布的平衡性
        if abs(yes_count - no_count) > judgment_count * 0.01:
            print(f"警告: Yes/No答案分布不均衡（差异 > 1%）")

    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


def main() -> None:
    """
    主函数：执行判断题生成流程

    配置输入输出文件路径并调用转换函数。
    """
    # 配置文件路径（相对于当前脚本的位置）
    input_file: str = "../../dataset/sentence_trex_data.json"
    output_file: str = "../../dataset/judgment_trex_data.json"
    template_file: str = "../../dataset/General_question.txt"

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"模板文件: {template_file}")
    print("-" * 60)

    # 验证输入文件存在性
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    convert_triples_to_judgments(input_file, output_file, template_file)


if __name__ == "__main__":
    main()
