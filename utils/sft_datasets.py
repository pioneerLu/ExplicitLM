"""
SFT（监督微调）数据集处理模块

本模块提供用于监督微调的数据集类，支持对话格式的数据加载和处理。
主要包含两个核心类：
1. SFTDataset: 用于训练的数据集，支持对话格式和损失掩码生成
2. SFTEvalDataset: 用于评估的数据集，只返回原始文本对
"""

import json
import os
from typing import Dict, List, Tuple, Any, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# 设置tokenizers并行化为True以提升性能
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class SFTDataset(Dataset):
    """
    监督微调训练数据集

    用于加载和处理对话格式的训练数据，支持：
    - 对话格式的prompt构建
    - 自动生成损失掩码（只对assistant回复计算损失）
    - 序列截断和padding

    数据格式要求：
    每行为一个JSON对象，包含'conversations'字段，该字段为对话列表：
    {
        "conversations": [
            {"role": "user", "content": "用户问题"},
            {"role": "assistant", "content": "助手回答"}
        ]
    }

    注意：role字段可选，如果缺失则按索引推断（偶数=user，奇数=assistant）
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        max_length: int = 512,
        system_message: str = "You are MiniMind, a helpful artificial intelligence assistant."
    ) -> None:
        """
        初始化SFT数据集

        参数:
            jsonl_path: JSONL格式数据文件路径
            tokenizer: 用于编码文本的tokenizer
            max_length: 最大序列长度，超过部分会被截断
            system_message: 系统提示消息，可自定义
        """
        super().__init__()

        # 文件存在性检查
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"数据文件不存在: {jsonl_path}")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_message = system_message
        self.samples = self._load_data(jsonl_path)

        # 编码特殊标记
        self.bos_id = self._encode_special_token('<|im_start|>assistant')
        self.eos_id = self._encode_special_token('<|im_end|>')

        # 验证tokenizer配置
        assert tokenizer.bos_token is not None, "tokenizer必须定义bos_token"
        assert tokenizer.eos_token is not None, "tokenizer必须定义eos_token"

    def _encode_special_token(self, token: str) -> List[int]:
        """
        编码特殊标记为token ID列表

        参数:
            token: 特殊标记字符串

        返回:
            token ID列表
        """
        encoded = self.tokenizer(
            token,
            add_special_tokens=False,
            return_tensors='pt'
        )
        token_ids = encoded.input_ids.squeeze().tolist()
        # 确保返回列表格式
        return token_ids if isinstance(token_ids, list) else [token_ids]

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载数据并进行验证

        参数:
            path: 数据文件路径

        返回:
            验证通过的样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())

                    # 验证数据格式
                    if not isinstance(data, dict):
                        raise ValueError("样本必须是字典类型")

                    if 'conversations' not in data:
                        raise ValueError("缺少conversations字段")

                    if not isinstance(data['conversations'], list):
                        raise ValueError("conversations必须是列表类型")

                    # 检查空对话列表
                    if len(data['conversations']) == 0:
                        raise ValueError("conversations不能为空列表")

                    # 验证每轮对话
                    for turn_idx, turn in enumerate(data['conversations']):
                        if not isinstance(turn, dict):
                            raise ValueError(f"第{turn_idx}轮对话必须是字典类型")
                        if 'content' not in turn:
                            raise ValueError(f"第{turn_idx}轮对话缺少content字段")

                        # 验证role字段（如果存在）
                        if 'role' in turn and turn['role'] not in ['user', 'assistant', 'system']:
                            raise ValueError(f"第{turn_idx}轮对话role必须是user/assistant/system")

                    samples.append(data)

                except Exception as e:
                    print(f"[警告] 跳过第{line_num}行: {e}")
                    continue

        print(f"成功加载 {len(samples)} 个训练样本")
        return samples

    def _create_chat_prompt(self, conversations: List[Dict[str, str]]) -> str:
        """
        根据对话列表构建符合模型格式的prompt

        对话格式：
        <|im_start|>system
        系统提示
        <|im_end|>
        <|im_start|>user
        用户问题
        <|im_end|>
        <|im_start|>assistant
        助手回答
        <|im_end|>

        参数:
            conversations: 对话列表，每个元素包含content字段和可选的role字段

        返回:
            格式化的完整prompt字符串
        """
        messages = []

        # 添加系统提示
        messages.append(f"<|im_start|>system\n{self.system_message}<|im_end|>\n")

        # 处理对话轮次（支持role字段或索引推断）
        for idx, turn in enumerate(conversations):
            # 优先使用role字段，否则按索引推断
            role = turn.get('role', 'user' if idx % 2 == 0 else 'assistant')
            content = turn['content']

            if role == 'user':
                messages.append(
                    f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
                )
            elif role == 'assistant':
                messages.append(f"{content}<|im_end|>\n")
            # system role在循环中不处理，因为已经在开头添加

        return ''.join(messages)

    def _generate_loss_mask(self, input_ids: List[int]) -> List[int]:
        """
        生成损失掩码，只对assistant的回复部分计算损失

        掩码规则：
        - 0: 不计算损失（system和user部分）
        - 1: 计算损失（assistant回复部分，包括<|im_end|>标记）

        参数:
            input_ids: token ID列表

        返回:
            与input_ids长度相同的掩码列表
        """
        loss_mask = [0] * len(input_ids)
        idx = 0

        while idx < len(input_ids):
            # 检查是否匹配 <|im_start|>assistant 标记
            if (idx + len(self.bos_id) <= len(input_ids) and
                input_ids[idx:idx + len(self.bos_id)] == self.bos_id):

                # assistant内容从标记后开始
                start = idx + len(self.bos_id)

                # 查找对应的 <|im_end|> 标记
                eos_found = False
                end = start

                while end < len(input_ids):
                    if (end + len(self.eos_id) <= len(input_ids) and
                        input_ids[end:end + len(self.eos_id)] == self.eos_id):
                        # 找到结束标记
                        eos_found = True
                        break
                    end += 1

                # 标记损失掩码
                if eos_found:
                    # 标记范围：assistant内容 + <|im_end|>
                    for j in range(start, end + len(self.eos_id)):
                        loss_mask[j] = 1
                    idx = end + len(self.eos_id)
                else:
                    # 未找到结束标记（可能被截断），标记到序列末尾
                    for j in range(start, len(loss_mask)):
                        loss_mask[j] = 1
                    break
            else:
                idx += 1

        return loss_mask

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息

        返回:
            包含以下字段的字典:
            - total_samples: 样本总数
            - avg_conversation_turns: 平均对话轮数
            - avg_prompt_length: 平均prompt长度(token)
            - max_prompt_length: 最大prompt长度
            - min_prompt_length: 最小prompt长度
            - truncation_rate: 被截断样本比例
            - empty_response_count: 空回复样本数量
        """
        stats = {
            'total_samples': len(self.samples),
            'conversation_turns': [],
            'prompt_lengths': [],
            'truncated_count': 0,
            'empty_response_count': 0
        }

        for sample in self.samples:
            turns = len(sample['conversations'])
            stats['conversation_turns'].append(turns)

            # 计算实际token长度
            prompt = self._create_chat_prompt(sample['conversations'])
            input_ids = self.tokenizer(prompt).input_ids
            original_length = len(input_ids)
            stats['prompt_lengths'].append(original_length)

            if original_length > self.max_length:
                stats['truncated_count'] += 1

            # 检测空回复
            for idx, turn in enumerate(sample['conversations']):
                role = turn.get('role', 'user' if idx % 2 == 0 else 'assistant')
                if role == 'assistant' and len(turn['content'].strip()) == 0:
                    stats['empty_response_count'] += 1
                    break

        return {
            'total_samples': stats['total_samples'],
            'avg_conversation_turns': sum(stats['conversation_turns']) / len(stats['conversation_turns']) if stats['conversation_turns'] else 0,
            'avg_prompt_length': sum(stats['prompt_lengths']) / len(stats['prompt_lengths']) if stats['prompt_lengths'] else 0,
            'max_prompt_length': max(stats['prompt_lengths']) if stats['prompt_lengths'] else 0,
            'min_prompt_length': min(stats['prompt_lengths']) if stats['prompt_lengths'] else 0,
            'truncation_rate': stats['truncated_count'] / stats['total_samples'] if stats['total_samples'] > 0 else 0,
            'empty_response_count': stats['empty_response_count']
        }

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个训练样本

        参数:
            index: 样本索引

        返回:
            (X, Y, loss_mask)元组:
            - X: 输入序列 (max_length-1,)
            - Y: 目标序列 (max_length-1,)，相对于X右移一位
            - loss_mask: 损失掩码 (max_length-1,)

        异常:
            在处理失败时会打印错误信息并重新抛出异常
        """
        try:
            sample = self.samples[index]

            # 第一阶段：构建prompt
            prompt = self._create_chat_prompt(sample['conversations'])

            # 第二阶段：编码和截断
            input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

            # 第三阶段：padding到固定长度
            padding_length = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length

            # 第四阶段：生成损失掩码
            loss_mask = self._generate_loss_mask(input_ids)

            # 第五阶段：构建训练对（input右移得到target）
            X = torch.tensor(input_ids[:-1], dtype=torch.long)
            Y = torch.tensor(input_ids[1:], dtype=torch.long)
            loss_mask_tensor = torch.tensor(loss_mask[1:], dtype=torch.long)

            return X, Y, loss_mask_tensor

        except Exception as e:
            print(f"[错误] 处理样本 {index} 失败: {e}")
            raise


class SFTEvalDataset(Dataset):
    """
    监督微调评估数据集

    用于模型评估，返回原始的问题-答案对，不进行tokenization。
    格式与SFTDataset相同，但返回的是文本而非token。
    支持多轮对话评估。
    """

    def __init__(
        self,
        jsonl_path: str,
        system_message: str = "You are MiniMind, a helpful artificial intelligence assistant."
    ) -> None:
        """
        初始化评估数据集

        参数:
            jsonl_path: JSONL格式数据文件路径
            system_message: 系统提示消息，可自定义
        """
        super().__init__()

        # 文件存在性检查
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"数据文件不存在: {jsonl_path}")

        self.system_message = system_message
        self.samples = self._load_data(jsonl_path)

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载评估数据

        参数:
            path: 数据文件路径

        返回:
            样本列表
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    samples.append(data)
                except Exception as e:
                    print(f"[警告] 跳过第{line_num}行: {e}")
                    continue

        print(f"成功加载 {len(samples)} 个评估样本")
        return samples

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        """
        获取单个评估样本（支持多轮对话）

        参数:
            index: 样本索引

        返回:
            (prompt, target)元组:
            - prompt: 包含system和历史对话的完整prompt
            - target: 期望的最后一个assistant回复
        """
        sample = self.samples[index]
        conversations = sample['conversations']

        # 构建完整上下文（除了最后一轮assistant回复）
        messages = [f"<|im_start|>system\n{self.system_message}<|im_end|>\n"]

        # 遍历对话历史，保留除最后一个assistant回复外的所有内容
        for idx, turn in enumerate(conversations[:-1]):
            role = turn.get('role', 'user' if idx % 2 == 0 else 'assistant')
            content = turn['content']

            if role == 'user':
                messages.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == 'assistant':
                messages.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")

        # 添加最后一个user输入（如果最后一轮是assistant，则添加倒数第二个user）
        last_turn = conversations[-1]
        last_role = last_turn.get('role', 'assistant' if (len(conversations) - 1) % 2 == 1 else 'user')

        if last_role == 'assistant':
            # 最后一轮是assistant，确保前面有user输入
            if len(conversations) > 1:
                second_last = conversations[-2]
                second_last_role = second_last.get('role', 'user' if (len(conversations) - 2) % 2 == 0 else 'assistant')
                if second_last_role == 'user' and not any('user' in msg for msg in messages[-2:]):
                    messages.append(f"<|im_start|>user\n{second_last['content']}<|im_end|>\n")

        # 添加assistant开始标记
        messages.append("<|im_start|>assistant\n")
        prompt = ''.join(messages)

        # 目标是最后一个assistant回复
        target = conversations[-1]['content']

        return prompt, target