"""
SFT（监督微调）数据集处理模块

本模块提供用于监督微调的数据集类和工厂函数，支持对话格式的数据加载和处理。

主要组件：
1. SFTDataset: 用于训练的数据集，支持对话格式和损失掩码生成
2. SFTEvalDataset: 用于评估的数据集，只返回原始文本对
3. create_sft_dataloader: SFT训练数据加载器工厂函数
4. create_sft_validation_dataloader: SFT验证数据加载器工厂函数
5. create_sft_eval_dataloader: SFT生成式评估数据加载器工厂函数
"""

import json
import os
import hashlib
from typing import Dict, List, Tuple, Any, Union

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

# 设置tokenizers并行化为True以提升性能
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Tokenizer 缓存目录
TOKENIZER_CACHE_DIR = "/data2/zengzheni/lvchangwei/new_repo/ExplicitLM/tokenizer_cache"


def _get_cache_path_sft(data_path: str, max_length: int, system_message: str) -> str:
    """
    生成SFT数据集的缓存文件路径

    Args:
        data_path: 数据路径
        max_length: 最大长度参数
        system_message: 系统消息（影响tokenize结果）

    Returns:
        缓存文件路径
    """
    # 确保缓存目录存在
    os.makedirs(TOKENIZER_CACHE_DIR, exist_ok=True)

    # 规范化路径
    abs_path = os.path.abspath(data_path)

    # 创建包含所有参数的hash
    safe_name = abs_path.replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in safe_name)

    # 包含system_message的hash
    content_hash = hashlib.md5(f"{abs_path}_{max_length}_{system_message}".encode()).hexdigest()

    if len(safe_name) > 100:
        safe_name = content_hash[:16]

    cache_filename = f"sft_{safe_name}_{content_hash[:8]}_maxlen{max_length}.pt"
    return os.path.join(TOKENIZER_CACHE_DIR, cache_filename)


def _normalize_to_conversations(data: Dict[str, Any], is_eval: bool = False) -> Dict[str, Any]:
    """
    将输入样本规范化为包含 conversations 的字典，并做基础校验。

    支持：
    - 已有 conversations
    - 仅有 text（转为双轮对话，沿用 TREX 逻辑）
    - conversations 若为 JSON 字符串也会尝试解析
    
    参数:
        data: 输入数据字典
        is_eval: 是否为评估数据。如果是评估数据且是 TREX 格式，会生成更合理的问题
    """
    if not isinstance(data, dict):
        raise ValueError("样本必须是字典类型")

    # TREX 格式：只有 text，没有 conversations
    if "text" in data and "conversations" not in data:
        text = str(data["text"]).replace("<|im_start|>", "").replace("<|im_end|>", "").strip()
        
        if is_eval:
            # 评估数据：从文档中提取关键信息作为问题
            # 对于 TREX 格式的评估数据，我们需要生成一个合理的问题
            # 策略：提取文档的第一句话或关键实体，生成 "What is X?" 格式的问题
            
            # 提取第一句话（到第一个句号）
            first_sentence = text.split('.')[0].strip() if '.' in text else text[:200].strip()
            
            # 提取关键主题词（通常是第一个名词短语）
            words = first_sentence.split()
            
            # 移除常见的停用词和标点
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
            meaningful_words = [w.strip('.,!?;:()[]{}"\'') for w in words if w.lower() not in stop_words and len(w) > 2]
            
            if len(meaningful_words) > 0:
                # 提取前2-4个有意义的词作为主题
                topic = ' '.join(meaningful_words[:min(4, len(meaningful_words))])
                # 生成问题："What is [topic]?"
                question = f"What is {topic}?"
            else:
                # 如果无法提取主题，使用通用问题
                # 但将文档的第一句话作为上下文
                if len(first_sentence) > 10:
                    question = f"Please explain: {first_sentence[:150]}"
                else:
                    question = "Please summarize the following content."
            
            data = {
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": text},
                ]
            }
        else:
            # 训练数据：保持原有逻辑
            data = {
                "conversations": [
                    {"role": "user", "content": "请根据以下内容回答问题。"},
                    {"role": "assistant", "content": text},
                ]
            }

    if "conversations" not in data:
        raise ValueError("缺少conversations字段")

    conversations = data["conversations"]
    # 允许字符串形式（如存成 JSON 字符串）
    if isinstance(conversations, str):
        conversations = json.loads(conversations)

    if not isinstance(conversations, list):
        raise ValueError("conversations必须是列表类型")
    if len(conversations) == 0:
        raise ValueError("conversations不能为空列表")

    for turn_idx, turn in enumerate(conversations):
        if not isinstance(turn, dict):
            raise ValueError(f"第{turn_idx}轮对话必须是字典类型")
        if "content" not in turn:
            raise ValueError(f"第{turn_idx}轮对话缺少content字段")
        if "role" in turn and turn["role"] not in ["user", "assistant", "system"]:
            raise ValueError("role必须是user/assistant/system")

    return {"conversations": conversations}


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

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_message = system_message
        self.samples = self._load_data(jsonl_path)

        # 编码特殊标记
        self.bos_id = self._encode_special_token('<|im_start|>assistant')
        self.eos_id = self._encode_special_token('<|im_end|>')

        # 验证tokenizer配置（Qwen tokenizer可能没有bos_token，使用pad_token作为fallback）
        if tokenizer.bos_token is None:
            if tokenizer.pad_token is not None:
                tokenizer.bos_token = tokenizer.pad_token
            else:
                raise ValueError("tokenizer必须定义bos_token或pad_token")
        if tokenizer.eos_token is None:
            raise ValueError("tokenizer必须定义eos_token")

        # 检查缓存
        cache_path = _get_cache_path_sft(jsonl_path, max_length, system_message)

        if os.path.exists(cache_path):
            # 从缓存加载
            print(f"从缓存加载 SFT tokenized 数据: {cache_path}")
            try:
                cached_data = torch.load(cache_path, map_location='cpu')
                self.input_id_seqs = cached_data['input_id_seqs']
                self.target_seqs = cached_data['target_seqs']
                self.loss_masks = cached_data['loss_masks']
                self.prompt_texts = cached_data['prompt_texts']
                print(f"成功加载 {len(self.input_id_seqs)} 个 SFT tokenized 样本")
                return
            except Exception as e:
                print(f"警告: 加载SFT缓存失败 ({e})，将重新 tokenize")

        # 预tokenize所有样本为 token id 序列
        print(f"开始 tokenize {len(self.samples)} 个 SFT 样本...")
        self.input_id_seqs: List[torch.Tensor] = []  # X: input_ids[:-1]
        self.target_seqs: List[torch.Tensor] = []    # Y: input_ids[1:]
        self.loss_masks: List[torch.Tensor] = []
        self.prompt_texts: List[str] = []

        for sample in self.samples:
            # 构建prompt
            prompt = self._create_chat_prompt(sample['conversations'])

            # 编码和截断
            input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

            # padding到固定长度
            padding_length = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length

            # 生成损失掩码
            loss_mask = self._generate_loss_mask(input_ids)

            # 构建训练对
            X = torch.tensor(input_ids[:-1], dtype=torch.long)  # 输入序列
            Y = torch.tensor(input_ids[1:], dtype=torch.long)   # 目标序列（右移一位）
            loss_mask_tensor = torch.tensor(loss_mask[1:], dtype=torch.long)

            self.input_id_seqs.append(X)
            self.target_seqs.append(Y)
            self.loss_masks.append(loss_mask_tensor)
            self.prompt_texts.append(prompt)

        # 保存缓存
        print(f"SFT Tokenize 完成，保存缓存到: {cache_path}")
        try:
            cache_data = {
                'input_id_seqs': self.input_id_seqs,
                'target_seqs': self.target_seqs,
                'loss_masks': self.loss_masks,
                'prompt_texts': self.prompt_texts
            }
            torch.save(cache_data, cache_path)
            print(f"SFT缓存保存成功: {len(self.input_id_seqs)} 个样本")
        except Exception as e:
            print(f"警告: 保存SFT缓存失败 ({e})")

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

    def _detect_format(self, path: str) -> str:
        lower = path.lower()
        if lower.endswith(".parquet") or "*" in path or "," in path or os.path.isdir(path):
            return "parquet"
        return "jsonl"

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        自动检测 JSONL / Parquet 读取：
        - JSONL：原有行读取逻辑
        - Parquet：支持单/多文件（逗号、通配符、目录），列自适应映射
        """
        if self._detect_format(path) == "parquet":
            return self._load_parquet(path)
        return self._load_jsonl(path)

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """JSONL 按行读取并校验。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")
        samples = []
        skipped_count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    normalized = _normalize_to_conversations(data)
                    samples.append(normalized)

                except Exception as e:
                    skipped_count += 1
                    if skipped_count <= 20:  # 只显示前20个警告
                        print(f"[警告] 跳过第{line_num}行: {e}")
                    continue

        print(f"成功加载 {len(samples)} 个训练样本")
        if skipped_count > 20:
            print(f"[警告] 还有 {skipped_count - 20} 个样本被跳过")
        return samples

    def _load_parquet(self, path: str) -> List[Dict[str, Any]]:
        """Parquet 读取（支持逗号/通配符/目录，多文件自动聚合）。"""
        try:
            import pyarrow.dataset as ds  # type: ignore
        except ImportError as e:
            raise ImportError("需要安装 pyarrow 才能读取 Parquet：pip install pyarrow") from e

        import os
        import glob
        
        # 逗号分隔的多路径处理
        path_list = [p.strip() for p in path.split(",") if p.strip()]
        if not path_list:
            raise FileNotFoundError(f"无效的数据路径: {path}")
        
        # 展开所有路径：如果是目录，列出其中的所有 parquet 文件
        expanded_paths = []
        for p in path_list:
            if os.path.isdir(p):
                # 目录：列出所有 .parquet 文件
                parquet_files = glob.glob(os.path.join(p, "*.parquet"))
                if not parquet_files:
                    raise FileNotFoundError(f"目录中没有找到 Parquet 文件: {p}")
                expanded_paths.extend(sorted(parquet_files))
            elif os.path.isfile(p):
                # 文件：直接添加
                expanded_paths.append(p)
            else:
                # 可能是通配符模式
                matched = glob.glob(p)
                if not matched:
                    raise FileNotFoundError(f"路径不存在或没有匹配的文件: {p}")
                expanded_paths.extend(sorted(matched))
        
        if not expanded_paths:
            raise FileNotFoundError(f"没有找到任何 Parquet 文件: {path}")

        dataset = ds.dataset(expanded_paths, format="parquet")
        schema_names = set(dataset.schema.names)

        has_conv = "conversations" in schema_names
        has_text = "text" in schema_names
        has_pair = "user" in schema_names and "assistant" in schema_names

        if has_conv:
            columns = ["conversations"]
        elif has_text:
            columns = ["text"]
        elif has_pair:
            columns = ["user", "assistant"]
        else:
            raise ValueError("Parquet 缺少 conversations/text 或 user+assistant 列，无法构造对话")

        table = dataset.to_table(columns=columns)

        samples: List[Dict[str, Any]] = []
        skipped_count = 0

        for idx, row in enumerate(table.to_pylist(), 1):
            try:
                if has_conv:
                    conv = row.get("conversations")
                    normalized = _normalize_to_conversations({"conversations": conv})
                elif has_text:
                    text = row.get("text", "")
                    normalized = _normalize_to_conversations({"text": text})
                else:  # user + assistant
                    user = row.get("user", "")
                    assistant = row.get("assistant", "")
                    normalized = _normalize_to_conversations(
                        {
                            "conversations": [
                                {"role": "user", "content": str(user)},
                                {"role": "assistant", "content": str(assistant)},
                            ]
                        }
                    )
                samples.append(normalized)
            except Exception as e:
                skipped_count += 1
                if skipped_count <= 20:
                    print(f"[警告] 跳过第{idx}行: {e}")
                continue

        print(f"成功加载 {len(samples)} 个训练样本（Parquet）")
        if skipped_count > 20:
            print(f"[警告] 还有 {skipped_count - 20} 个样本被跳过")
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

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        获取单个训练样本（从预缓存数据中返回）

        参数:
            index: 样本索引

        返回:
            (X, Y, loss_mask, prompt_text)元组:
            - X: 输入序列 (max_length-1,)
            - Y: 目标序列 (max_length-1,)，相对于X右移一位
            - loss_mask: 损失掩码 (max_length-1,)
            - prompt_text: 原始prompt文本（用于知识提取，无需解码）
        """
        try:
            # 从缓存的数据中直接返回
            X = self.input_id_seqs[index]
            Y = self.target_seqs[index]
            loss_mask = self.loss_masks[index]
            prompt_text = self.prompt_texts[index]

            return X, Y, loss_mask, prompt_text

        except Exception as e:
            print(f"[错误] 获取缓存样本 {index} 失败: {e}")
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
        system_message: str = "You are a helpful artificial intelligence assistant."
    ) -> None:
        """
        初始化评估数据集

        参数:
            jsonl_path: JSONL格式数据文件路径
            system_message: 系统提示消息，可自定义
        """
        super().__init__()

        self.system_message = system_message
        self.samples = self._load_data(jsonl_path)

    def _detect_format(self, path: str) -> str:
        lower = path.lower()
        if lower.endswith(".parquet") or "*" in path or "," in path or os.path.isdir(path):
            return "parquet"
        return "jsonl"

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        """自动检测 JSONL / Parquet 进行评估集加载。"""
        if self._detect_format(path) == "parquet":
            return self._load_parquet(path)
        return self._load_jsonl(path)

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}")

        samples = []
        skipped_count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    normalized = _normalize_to_conversations(data, is_eval=True)
                    samples.append(normalized)
                except Exception as e:
                    skipped_count += 1
                    if skipped_count <= 20:
                        print(f"[警告] 跳过第{line_num}行: {e}")
                    continue

        print(f"成功加载 {len(samples)} 个评估样本")
        if skipped_count > 20:
            print(f"[警告] 还有 {skipped_count - 20} 个样本被跳过")
        return samples

    def _load_parquet(self, path: str) -> List[Dict[str, Any]]:
        try:
            import pyarrow.dataset as ds  # type: ignore
        except ImportError as e:
            raise ImportError("需要安装 pyarrow 才能读取 Parquet：pip install pyarrow") from e

        path_list = [p.strip() for p in path.split(",") if p.strip()]
        if not path_list:
            raise FileNotFoundError(f"无效的数据路径: {path}")

        dataset = ds.dataset(path_list, format="parquet")
        schema_names = set(dataset.schema.names)

        has_conv = "conversations" in schema_names
        has_text = "text" in schema_names
        has_pair = "user" in schema_names and "assistant" in schema_names

        if has_conv:
            columns = ["conversations"]
        elif has_text:
            columns = ["text"]
        elif has_pair:
            columns = ["user", "assistant"]
        else:
            raise ValueError("Parquet 缺少 conversations/text 或 user+assistant 列，无法构造对话")

        table = dataset.to_table(columns=columns)

        samples: List[Dict[str, Any]] = []
        skipped_count = 0

        for idx, row in enumerate(table.to_pylist(), 1):
            try:
                if has_conv:
                    conv = row.get("conversations")
                    normalized = _normalize_to_conversations({"conversations": conv}, is_eval=True)
                elif has_text:
                    text = row.get("text", "")
                    normalized = _normalize_to_conversations({"text": text}, is_eval=True)
                else:
                    user = row.get("user", "")
                    assistant = row.get("assistant", "")
                    normalized = _normalize_to_conversations(
                        {
                            "conversations": [
                                {"role": "user", "content": str(user)},
                                {"role": "assistant", "content": str(assistant)},
                            ]
                        },
                        is_eval=True
                    )
                samples.append(normalized)
            except Exception as e:
                skipped_count += 1
                if skipped_count <= 20:
                    print(f"[警告] 跳过第{idx}行: {e}")
                continue

        print(f"成功加载 {len(samples)} 个评估样本（Parquet）")
        if skipped_count > 20:
            print(f"[警告] 还有 {skipped_count - 20} 个样本被跳过")
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


#########################################################
# 工厂函数：数据加载器创建
#########################################################


def _sft_collate_fn(batch):
    """
    自定义 collate 函数，处理混合类型（tensor + string）
    
    输入:
        batch: List[Tuple[X, Y, loss_mask, prompt_text]]
    
    返回:
        (X_batch, Y_batch, loss_mask_batch, prompt_texts)
        - X_batch, Y_batch, loss_mask_batch: [batch_size, max_length-1]
        - prompt_texts: List[str] (batch_size,)
    """
    # 分离各个组件
    X_list, Y_list, loss_mask_list, prompt_texts = zip(*batch)
    
    # 堆叠 tensor
    X_batch = torch.stack(X_list, dim=0)
    Y_batch = torch.stack(Y_list, dim=0)
    loss_mask_batch = torch.stack(loss_mask_list, dim=0)
    
    # prompt_texts 保持为列表
    return X_batch, Y_batch, loss_mask_batch, list(prompt_texts)


def create_sft_dataloader(
    data_path: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    batch_size: int,
    max_length: int = 512,
    system_message: str = "You are MiniMind, a helpful artificial intelligence assistant.",
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    创建SFT训练数据加载器的工厂函数

    参数:
        data_path: JSONL格式数据文件路径，每行包含conversations字段
        tokenizer: Tokenizer实例，支持PreTrainedTokenizer或PreTrainedTokenizerFast
        batch_size: 批次大小
        max_length: 最大序列长度，默认512
        system_message: 系统提示消息，可自定义
        shuffle: 是否打乱数据，默认True
        num_workers: 数据加载进程数，默认4
        pin_memory: 是否使用pin_memory加速GPU传输，默认True

    返回:
        DataLoader实例

    使用示例:
        ```python
        from utils.sft_datasets import create_sft_dataloader

        train_loader = create_sft_dataloader(
            data_path='dataset/splits/judgement-100k/train.jsonl',
            tokenizer=tokenizer,
            batch_size=32,
            max_length=512,
            system_message="You are a helpful assistant."
        )

        for batch_idx, (X, Y, loss_mask, prompt_texts) in enumerate(train_loader):
            # X, Y, loss_mask: [batch_size, max_length-1]
            # loss_mask中只有assistant回复部分为1
            # prompt_texts: List[str] (batch_size,)，原始prompt文本
            ...
        ```
    """
    # 第一阶段：创建SFT数据集实例
    dataset = SFTDataset(
        jsonl_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        system_message=system_message
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_sft_collate_fn  # 使用自定义 collate_fn 处理混合类型
    )

    return dataloader


def create_sft_validation_dataloader(
    val_data_path: str,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    batch_size: int,
    max_length: int = 512,
    system_message: str = "You are MiniMind, a helpful artificial intelligence assistant.",
    num_samples: int = 200,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Union[DataLoader, None]:
    """
    创建SFT验证数据加载器的工厂函数

    参数:
        val_data_path: 验证数据文件路径
        tokenizer: Tokenizer实例，支持PreTrainedTokenizer或PreTrainedTokenizerFast
        batch_size: 批次大小
        max_length: 最大序列长度，默认512
        system_message: 系统提示消息，可自定义
        num_samples: 验证样本数量限制（用于加快验证），默认200
        num_workers: 数据加载进程数，默认4
        pin_memory: 是否使用pin_memory加速GPU传输，默认True

    返回:
        DataLoader实例，如果文件不存在返回None

    使用示例:
        ```python
        from utils.sft_datasets import create_sft_validation_dataloader

        val_loader = create_sft_validation_dataloader(
            val_data_path='dataset/splits/judgement-100k/valid.jsonl',
            tokenizer=tokenizer,
            batch_size=32,
            max_length=512,
            num_samples=100
        )

        if val_loader is not None:
            for batch_idx, (X, Y, loss_mask, prompt_texts) in enumerate(val_loader):
                # X, Y, loss_mask: [batch_size, max_length-1]
                # prompt_texts: List[str] (batch_size,)，原始prompt文本
                ...
        ```
    """
    # 第一阶段：检查文件是否存在
    if not os.path.exists(val_data_path):
        print(f"[警告] 验证数据文件不存在: {val_data_path}")
        return None

    # 第二阶段：创建验证数据集
    dataset = SFTDataset(
        jsonl_path=val_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        system_message=system_message
    )

    if len(dataset) > num_samples:
        indices = list(range(num_samples))
        dataset = Subset(dataset, indices)
        print(f"[信息] 验证集采样: {num_samples}/{len(dataset)} 样本")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_sft_collate_fn  # 使用自定义 collate_fn 处理混合类型
    )

    return dataloader


def create_sft_eval_dataloader(
    eval_data_path: str,
    system_message: str = "You are MiniMind, a helpful artificial intelligence assistant.",
    batch_size: int = 1,
    max_samples: int = 100
) -> Union[DataLoader, None]:
    """
    创建SFT生成式评估数据加载器的工厂函数

    用于评估模型的实际生成质量，返回原始的问题-答案对。
    支持多轮对话评估。

    参数:
        eval_data_path: 评估数据文件路径
        system_message: 系统提示消息，可自定义
        batch_size: 批次大小，默认1（生成式评估通常单样本处理）
        max_samples: 最大评估样本数，默认100

    返回:
        DataLoader实例，如果文件不存在返回None

    使用示例:
        ```python
        from utils.sft_datasets import create_sft_eval_dataloader

        eval_loader = create_sft_eval_dataloader(
            eval_data_path='dataset/splits/judgement-100k/test.jsonl',
            system_message="You are a helpful assistant.",
            batch_size=1,
            max_samples=50
        )

        if eval_loader is not None:
            for prompt, target in eval_loader:
                # prompt: 包含system和user的完整prompt（支持多轮对话）
                # target: 期望的assistant回复
                ...
        ```
    """
    # 第一阶段：检查文件是否存在
    if not os.path.exists(eval_data_path):
        print(f"[警告] 评估数据文件不存在: {eval_data_path}")
        return None

    # 第二阶段：创建评估数据集
    dataset = SFTEvalDataset(
        jsonl_path=eval_data_path,
        system_message=system_message
    )

    if len(dataset) > max_samples:
        indices = list(range(max_samples))
        dataset = Subset(dataset, indices)
        print(f"[信息] 评估集采样: {max_samples}/{len(dataset)} 样本")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return dataloader