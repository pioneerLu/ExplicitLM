"""
预训练数据集模块（用于纯文本数据，非对话格式）

功能：
- PretrainDataset: 基础预训练数据集类（支持JSONL和Parquet）
- create_pretrain_dataloader: 数据加载器工厂函数
- 支持数据过滤、验证和批处理

注意：
- 预训练数据集用于纯文本数据（如 data/parquet_data），不是对话格式
- SFT数据集用于对话格式数据（如 train.jsonl），使用 utils/sft_datasets.py
"""

import json
import os
import hashlib
from typing import Dict, Any, List, Tuple, Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# 启用tokenizer并行化
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Tokenizer 缓存目录（相对于 ExplicitLM 根目录）
from pathlib import Path
_EXPLICITLM_ROOT = Path(__file__).parent.parent
TOKENIZER_CACHE_DIR = str(_EXPLICITLM_ROOT / "tokenizer_cache")


def _get_cache_path(data_path: str, max_length: int, num_samples: Optional[int] = None) -> str:
    """
    生成缓存文件路径
    
    Args:
        data_path: 数据路径（可能是文件、目录、通配符等）
        max_length: 最大长度参数
        num_samples: 可选，样本数量（用于 ValidationDataset）
        
    Returns:
        缓存文件路径
    """
    # 确保缓存目录存在
    os.makedirs(TOKENIZER_CACHE_DIR, exist_ok=True)
    
    # 规范化路径（转换为绝对路径）
    abs_path = os.path.abspath(data_path)
    
    # 将路径中的特殊字符替换为下划线，生成安全的文件名
    # 例如：/data2/zengzheni/.../sample_256 -> data2_zengzheni_..._sample_256
    safe_name = abs_path.replace("/", "_").replace("\\", "_").replace(" ", "_")
    safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in safe_name)
    
    # 如果路径太长，使用 hash
    if len(safe_name) > 200:
        safe_name = hashlib.md5(abs_path.encode()).hexdigest()
    
    # 生成缓存文件名
    if num_samples is not None:
        cache_filename = f"{safe_name}_maxlen{max_length}_nsamples{num_samples}.pt"
    else:
        cache_filename = f"{safe_name}_maxlen{max_length}.pt"
    
    return os.path.join(TOKENIZER_CACHE_DIR, cache_filename)


def build_pretrain_collate_fn(pad_token_id: int, return_uuids: bool = False):
    """
    返回用于 PretrainDataset / ValidationDataset 的 collate_fn：
    
    - 输入：List[(1D LongTensor, uuid)]，每个为一条未 padding 的 token 序列和对应的 uuid
    - 输出：(X, Y, loss_mask) 或 (X, Y, loss_mask, batch_uuids)，形状为:
        X: [batch_size, max_seq_len-1]
        Y: [batch_size, max_seq_len-1]
        loss_mask: [batch_size, max_seq_len-1] (bool)
        batch_uuids: List[str] (可选)
    - 在 batch 内按最长序列做 dynamic padding
    """
    
    def collate_fn(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        # 分离 sequences 和 uuids
        sequences = [item[0] for item in batch if item[0] is not None and item[0].numel() >= 2]
        uuids = [item[1] for item in batch if item[0] is not None and item[0].numel() >= 2]
        
        if len(sequences) == 0:
            raise ValueError("collate_fn 收到空 batch 或所有样本长度不足 2。")
    
        batch_sizes = [seq.size(0) for seq in sequences]
        max_len = max(batch_sizes)
    
        padded = torch.full(
            (len(sequences), max_len),
            fill_value=pad_token_id,
            dtype=torch.long,
        )
    
        for i, seq in enumerate(sequences):
            cur_len = seq.size(0)
            padded[i, :cur_len] = seq
    
        X = padded[:, :-1]
        Y = padded[:, 1:]
        loss_mask = (Y != pad_token_id)
    
        if return_uuids:
            return X, Y, loss_mask, uuids
        else:
            return X, Y, loss_mask
    
    return collate_fn


class PretrainDataset(Dataset):
    """
    预训练数据集类（用于纯文本数据，非对话格式）
    
    - __getitem__ 只返回一段 token id 序列（1D LongTensor，不做 padding）
    - Qwen 特殊 token 能力只在 __init__ 中探测一次
    - tokenizer 只在 __init__ 中调用一次，对所有样本完成预编码
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 512
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 加载原始样本（包含 text 和 uuid）
        self.samples: List[Dict[str, Any]] = self._load_data(data_path)

        # 只探测一次 Qwen 对 <|im_start|> / <|im_end|> 的支持
        self.im_start_token = "<|im_start|>"
        self.im_end_token = "<|im_end|>"
        has_im_start = False
        has_im_end = False

        if hasattr(self.tokenizer, 'added_tokens_decoder'):
            added_tokens = self.tokenizer.added_tokens_decoder
            token_contents = [
                t.content if hasattr(t, 'content') else str(t)
                for t in added_tokens.values()
            ]
            has_im_start = self.im_start_token in token_contents
            has_im_end = self.im_end_token in token_contents

        if not (has_im_start and has_im_end):
            try:
                im_start_ids = self.tokenizer.encode(self.im_start_token, add_special_tokens=False)
                im_end_ids = self.tokenizer.encode(self.im_end_token, add_special_tokens=False)
                has_im_start = len(im_start_ids) > 0
                has_im_end = len(im_end_ids) > 0
            except Exception:
                pass

        self.use_im_tokens = bool(has_im_start and has_im_end)

        # 检查缓存（注意：缓存不包含 uuid，需要重新 tokenize 时保留 uuid）
        cache_path = _get_cache_path(data_path, max_length)
        cache_uuid_path = cache_path.replace('.pt', '_uuids.pt')
        
        if os.path.exists(cache_path) and os.path.exists(cache_uuid_path):
            # 从缓存加载
            print(f"从缓存加载 tokenized 数据: {cache_path}")
            try:
                self.input_id_seqs = torch.load(cache_path, map_location='cpu')
                self.sample_uuids = torch.load(cache_uuid_path, map_location='cpu')
                print(f"成功加载 {len(self.input_id_seqs)} 个 tokenized 样本")
                return
            except Exception as e:
                print(f"警告: 加载缓存失败 ({e})，将重新 tokenize")

        # 预编码所有样本为 token id 序列（不做 padding）
        print(f"开始 tokenize {len(self.samples)} 个样本...")
        self.input_id_seqs: List[torch.Tensor] = []
        self.sample_uuids: List[str] = []
        for sample in self.samples:
            text = str(sample['text'])

            if self.use_im_tokens:
                formatted = f"{self.im_start_token}user\n{text}{self.im_end_token}"
            else:
                if getattr(self.tokenizer, 'bos_token', None) is not None and not text.startswith(self.tokenizer.bos_token):
                    text = f"{self.tokenizer.bos_token}{text}"
                if getattr(self.tokenizer, 'eos_token', None) is not None and not text.endswith(self.tokenizer.eos_token):
                    text = f"{text}{self.tokenizer.eos_token}"
                formatted = text

            encoded = self.tokenizer(
                formatted,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=False,
                return_attention_mask=False,
            )

            ids = encoded["input_ids"]
            input_ids = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
            input_ids = input_ids.to(dtype=torch.long)

            # 至少保留 2 个 token 才能构造 (X, Y)
            if input_ids.numel() < 2:
                continue

            self.input_id_seqs.append(input_ids)
            # 保存对应的 uuid（必须存在）
            if 'uuid' not in sample:
                raise ValueError(f"样本缺少 uuid 字段。样本内容: {sample.get('text', '')[:100]}...")
            uuid = sample['uuid']
            if not uuid or uuid == '':
                raise ValueError(f"样本的 uuid 为空。样本内容: {sample.get('text', '')[:100]}...")
            self.sample_uuids.append(str(uuid))
        
        # 保存缓存
        print(f"Tokenize 完成，保存缓存到: {cache_path}")
        try:
            torch.save(self.input_id_seqs, cache_path)
            torch.save(self.sample_uuids, cache_uuid_path)
            print(f"缓存保存成功: {len(self.input_id_seqs)} 个样本")
        except Exception as e:
            print(f"警告: 保存缓存失败 ({e})")

    def _load_data(self, path: str) -> List[Dict[str, Any]]:
        """
        从JSONL或Parquet文件加载数据（自动检测格式）

        Args:
            path: JSONL或Parquet文件路径（支持目录、通配符、逗号分隔）

        Returns:
            样本列表，每个样本为包含'text'字段的字典
        """
        # 检测文件格式
        if self._detect_format(path) == "parquet":
            return self._load_parquet(path)
        return self._load_jsonl(path)
    
    def _detect_format(self, path: str) -> str:
        """检测数据格式"""
        lower = path.lower()
        if lower.endswith(".parquet") or "*" in path or "," in path or os.path.isdir(path):
            return "parquet"
        return "jsonl"
    
    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        """
        从JSONL文件加载数据

        Args:
            path: JSONL文件路径

        Returns:
            样本列表，每个样本为包含'text'和'uuid'字段的字典
        """
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    data = json.loads(line)
                    # 检查必需的字段
                    if 'text' not in data:
                        raise ValueError(f"JSONL 文件 {path} 第 {line_num} 行缺少 'text' 字段")
                    if 'uuid' not in data:
                        raise ValueError(
                            f"JSONL 文件 {path} 第 {line_num} 行缺少 'uuid' 字段。"
                            f"Oracle Fact Fusion 需要 uuid 字段来建立映射关系。"
                        )
                    if data.get('uuid') is None or data.get('uuid') == '':
                        raise ValueError(
                            f"JSONL 文件 {path} 第 {line_num} 行的 uuid 为空。"
                            f"Oracle Fact Fusion 要求所有样本都有有效的 uuid。"
                        )
                    samples.append(data)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSONL 文件 {path} 第 {line_num} 行 JSON 解析失败: {e}")
        return samples
    
    def _load_parquet(self, path: str) -> List[Dict[str, Any]]:
        """
        从Parquet文件加载数据（支持多文件、目录、通配符）

        Args:
            path: Parquet文件路径（支持逗号分隔、通配符、目录）

        Returns:
            样本列表，每个样本为包含'text'字段的字典
        """
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
                matched = glob.glob(p)
                if not matched:
                    raise FileNotFoundError(f"路径不存在或没有匹配的文件: {p}")
                expanded_paths.extend(sorted(matched))
        
        if not expanded_paths:
            raise FileNotFoundError(f"没有找到任何 Parquet 文件: {path}")
        
        dataset = ds.dataset(expanded_paths, format="parquet")
        schema_names = set(dataset.schema.names)
        
        # 预训练数据集需要 'text' 和 'uuid' 列
        if "text" not in schema_names:
            raise ValueError(f"Parquet 缺少 'text' 列，无法用于预训练。可用列: {schema_names}")
        
        if "uuid" not in schema_names:
            raise ValueError(
                f"Parquet 缺少 'uuid' 列，无法用于预训练。可用列: {schema_names}\n"
                f"Oracle Fact Fusion 需要 uuid 字段来建立映射关系，请确保数据包含 uuid 列。"
            )
        
        # 加载 text 和 uuid 列
        columns_to_load = ["text", "uuid"]
        table = dataset.to_table(columns=columns_to_load)
        
        samples: List[Dict[str, Any]] = []
        skipped_count = 0
        
        for idx, row in enumerate(table.to_pylist(), 1):
            try:
                text = row.get("text", "")
                if not text or not str(text).strip():
                    skipped_count += 1
                    continue
                
                # 检查 uuid 是否存在且不为 None
                if "uuid" not in row:
                    raise ValueError(f"Parquet 文件第 {idx} 行缺少 uuid 字段")
                
                uuid_val = row.get("uuid")
                if uuid_val is None:
                    raise ValueError(
                        f"Parquet 文件第 {idx} 行的 uuid 为 None。"
                        f"Oracle Fact Fusion 要求所有样本都有有效的 uuid。"
                    )
                
                sample = {
                    "text": str(text).strip(),
                    "uuid": str(uuid_val)
                }
                
                samples.append(sample)
            except Exception as e:
                skipped_count += 1
                if skipped_count <= 20:
                    print(f"[警告] 跳过第{idx}行: {e}")
                continue
        
        print(f"成功加载 {len(samples)} 个预训练样本（Parquet）")
        if skipped_count > 20:
            print(f"[警告] 还有 {skipped_count - 20} 个样本被跳过")
        return samples

    def __len__(self) -> int:
        return len(self.input_id_seqs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        返回 (token id 序列, uuid)
        """
        return self.input_id_seqs[index], self.sample_uuids[index]


def create_pretrain_dataloader(
    data_path: str,
    tokenizer: Any,
    batch_size: int,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    val_split_ratio: float = 0.0,
    val_split_size: Optional[int] = None,
    return_uuids: bool = False,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    创建预训练数据加载器（tokenizer 仅在 Dataset 初始化阶段调用一次）
    
    Args:
        return_uuids: 是否在 collate_fn 中返回 uuids
    """
    dataset = PretrainDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    collate_fn = build_pretrain_collate_fn(pad_token_id=tokenizer.pad_token_id, return_uuids=return_uuids)

    val_loader = None
    if val_split_ratio > 0.0 or val_split_size is not None:
        total_size = len(dataset)

        if val_split_size is not None:
            val_size = min(val_split_size, total_size - 1)
        else:
            val_size = int(total_size * val_split_ratio)

        train_size = total_size - val_size

        if train_size <= 0 or val_size <= 0:
            raise ValueError(f"数据集分割失败: 总样本数={total_size}, 训练集={train_size}, 验证集={val_size}")

        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate_fn,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate_fn,
        )

        return train_dataloader, val_loader
    else:
        train_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=collate_fn,
        )

        return train_dataloader, None


def validate_dataset(
    data_path: str,
    tokenizer: Any,
    max_samples: int = 10
) -> Dict[str, Any]:
    """
    验证数据集完整性和统计信息

    Args:
        data_path: JSONL数据文件路径
        tokenizer: Tokenizer实例
        max_samples: 打印的最大样本数，默认10

    Returns:
        包含统计信息的字典：
        - total_samples: 总样本数
        - avg_text_length: 平均文本长度
        - sample_examples: 示例样本列表
    """
    dataset = PretrainDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=512
    )

    total_samples = len(dataset)
    text_lengths = []
    sample_examples = []

    for i in range(min(max_samples, total_samples)):
        seq = dataset[i]
        # 动态构造统计信息（无 padding）
        x_len = max(seq.size(0) - 1, 0)
        sample_info = {
            'index': i,
            'seq_len': seq.size(0),
            'X_shape': (x_len,),
            'Y_shape': (x_len,),
            'loss_mask_shape': (x_len,),
            'num_valid_tokens': x_len,
            'text_preview': dataset.samples[i]['text'][:100]  # 前100个字符
        }
        sample_examples.append(sample_info)
        text_lengths.append(len(dataset.samples[i]['text']))

    stats = {
        'total_samples': total_samples,
        'avg_text_length': sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        'sample_examples': sample_examples
    }

    return stats


class ValidationDataset(Dataset):
    """
    验证数据集类
    
    - __getitem__ 只返回一段 token id 序列（1D LongTensor，不做 padding）
    - Qwen 特殊 token 能力只在 __init__ 中探测一次
    - tokenizer 只在 __init__ 中调用一次，对所有样本完成预编码
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 512,
        num_samples: int = 200
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_samples = num_samples
        self.samples = self._load_validation_data(data_path)

        # 只探测一次 Qwen 对 <|im_start|> / <|im_end|> 的支持
        self.im_start_token = "<|im_start|>"
        self.im_end_token = "<|im_end|>"
        has_im_start = False
        has_im_end = False

        if hasattr(self.tokenizer, 'added_tokens_decoder'):
            added_tokens = self.tokenizer.added_tokens_decoder
            token_contents = [
                t.content if hasattr(t, 'content') else str(t)
                for t in added_tokens.values()
            ]
            has_im_start = self.im_start_token in token_contents
            has_im_end = self.im_end_token in token_contents

        if not (has_im_start and has_im_end):
            try:
                im_start_ids = self.tokenizer.encode(self.im_start_token, add_special_tokens=False)
                im_end_ids = self.tokenizer.encode(self.im_end_token, add_special_tokens=False)
                has_im_start = len(im_start_ids) > 0
                has_im_end = len(im_end_ids) > 0
            except Exception:
                pass

        self.use_im_tokens = bool(has_im_start and has_im_end)

        # 检查缓存
        cache_path = _get_cache_path(data_path, max_length, num_samples=num_samples)
        
        if os.path.exists(cache_path):
            # 从缓存加载
            print(f"从缓存加载 tokenized 数据: {cache_path}")
            try:
                self.input_id_seqs = torch.load(cache_path, map_location='cpu')
                print(f"成功加载 {len(self.input_id_seqs)} 个 tokenized 样本")
                return
            except Exception as e:
                print(f"警告: 加载缓存失败 ({e})，将重新 tokenize")

        # 预编码所有样本为 token id 序列（不做 padding）
        print(f"开始 tokenize {len(self.samples)} 个验证样本...")
        self.input_id_seqs: List[torch.Tensor] = []
        for sample in self.samples:
            text = str(sample['text'])

            if self.use_im_tokens:
                formatted = f"{self.im_start_token}user\n{text}{self.im_end_token}"
            else:
                if getattr(self.tokenizer, 'bos_token', None) is not None and not text.startswith(self.tokenizer.bos_token):
                    text = f"{self.tokenizer.bos_token}{text}"
                if getattr(self.tokenizer, 'eos_token', None) is not None and not text.endswith(self.tokenizer.eos_token):
                    text = f"{text}{self.tokenizer.eos_token}"
                formatted = text

            encoded = self.tokenizer(
                formatted,
                max_length=self.max_length,
                truncation=True,
                padding=False,
                add_special_tokens=False,
                return_attention_mask=False,
            )

            ids = encoded["input_ids"]
            input_ids = ids if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
            input_ids = input_ids.to(dtype=torch.long)

            if input_ids.numel() < 2:
                continue

            self.input_id_seqs.append(input_ids)
        
        # 保存缓存
        print(f"Tokenize 完成，保存缓存到: {cache_path}")
        try:
            torch.save(self.input_id_seqs, cache_path)
            print(f"缓存保存成功: {len(self.input_id_seqs)} 个样本")
        except Exception as e:
            print(f"警告: 保存缓存失败 ({e})")

    def _load_validation_data(self, path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"验证数据文件不存在: {path}")

        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.num_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if 'text' in data:
                        samples.append({'text': data['text']})
                except json.JSONDecodeError:
                    continue

        return samples

    def __len__(self) -> int:
        return len(self.input_id_seqs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        返回 (token id 序列, uuid)
        """
        return self.input_id_seqs[index], self.sample_uuids[index]


def create_validation_dataloader(
    val_data_path: str,
    tokenizer: Any,
    batch_size: int,
    max_length: int = 512,
    num_samples: int = 200,
    num_workers: int = 0,  # 分布式训练中设置为 0 避免死锁
    pin_memory: bool = True
) -> DataLoader:
    """
    创建验证数据加载器（tokenizer 仅在 Dataset 初始化阶段调用一次）
    """
    if not os.path.exists(val_data_path):
        return None

    dataset = ValidationDataset(
        data_path=val_data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        num_samples=num_samples
    )

    collate_fn = build_pretrain_collate_fn(pad_token_id=tokenizer.pad_token_id)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )

    return dataloader
