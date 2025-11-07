"""
模型初始化工具模块

提供统一的模型初始化接口，支持多种模型类型和初始化策略。
包含权重初始化、预训练嵌入加载、知识数据库处理等功能。

兼容新版 dict 配置系统。
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from hydra.utils import get_original_cwd
from pathlib import Path
from models.configs.LMConfig import LMConfig
from utils.logger import Logger


# ---------- 以下代码完全不变，仅把 args 当成 dict 使用 ----------
class ModelTypeConfig:
    """模型类型配置映射"""
    SUPPORTED_TYPES = {
        "model": {
            "module_path": "model.core.ExplicitLM",
            "class_name": "ExplicitLM",
            "requires_weight_init": True,
            "database_attribute": "knowledge_dataset.knowledge_dataset",
        },
        "model_original": {
            "module_path": "model.model_original",
            "class_name": "ExplicitLM",
            "requires_weight_init": False,
            "database_attribute": None,
        },
        "model_no_feed": {
            "module_path": "model.model_no_feed",
            "class_name": "ExplicitLM",
            "requires_weight_init": True,
            "database_attribute": "knowledge_dataset.knowledge_dataset",
        },
        "model_memory": {
            "module_path": "models.core.ExplicitLM",
            "class_name": "ExplicitLM",
            "requires_weight_init": True,
            "database_attribute": "memory_bank",
            "memory_optimization": True,
        },
    }

    @classmethod
    def get_config(cls, model_type: str) -> Dict[str, Any]:
        if model_type not in cls.SUPPORTED_TYPES:
            raise ValueError(
                f"不支持的模型类型: {model_type}。"
                f"支持的类型: {list(cls.SUPPORTED_TYPES.keys())}"
            )
        return cls.SUPPORTED_TYPES[model_type]


class WeightInitializer:
    """权重初始化器"""

    @staticmethod
    def initialize_model_weights(model: nn.Module, model_type: str, accelerator=None) -> None:
        Logger("执行模型权重初始化...", accelerator)
        RMSNorm = WeightInitializer._import_rmsnorm(model_type, accelerator)
        WeightInitializer._init_embeddings(model, accelerator)
        WeightInitializer._init_layers(model, RMSNorm, accelerator)
        WeightInitializer._init_knowledge_components(model, accelerator)
        Logger("模型权重初始化完成", accelerator)

    @staticmethod
    def _import_rmsnorm(model_type: str, accelerator=None):
        try:
            config = ModelTypeConfig.get_config(model_type)
            module = __import__(config["module_path"], fromlist=["RMSNorm"])
            return module.RMSNorm
        except (ImportError, AttributeError):
            Logger("警告: 无法导入RMSNorm，跳过RMSNorm初始化", accelerator)
            return None

    @staticmethod
    def _init_embeddings(model: nn.Module, accelerator=None) -> None:
        if hasattr(model, "tok_embeddings"):
            nn.init.normal_(model.tok_embeddings.weight, mean=0.0, std=0.02)
        if hasattr(model, "output"):
            is_shared = (
                hasattr(model, "tok_embeddings")
                and hasattr(model.tok_embeddings, "weight")
                and model.output.weight is model.tok_embeddings.weight
            )
            if not is_shared:
                nn.init.normal_(model.output.weight, mean=0.0, std=0.02)

    @staticmethod
    def _init_layers(model: nn.Module, RMSNorm, accelerator=None) -> None:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif RMSNorm and isinstance(module, RMSNorm):
                if hasattr(module, "weight"):
                    nn.init.ones_(module.weight)

    @staticmethod
    def _init_knowledge_components(model: nn.Module, accelerator=None) -> None:
        if hasattr(model, "knowledge_dataset") and hasattr(model.knowledge_dataset, "keys"):
            nn.init.normal_(model.knowledge_dataset.keys, mean=0.0, std=0.02)


class EmbeddingLoader:
    @staticmethod
    def load_pretrained_embeddings(model: nn.Module, embedding_path: str, accelerator=None) -> None:
        Logger(f"加载预训练嵌入权重: {embedding_path}", accelerator)
        pretrained_embeddings = torch.load(embedding_path)
        if hasattr(model, "tok_embeddings"):
            model.tok_embeddings.weight.data.copy_(pretrained_embeddings)
        if hasattr(model, "output"):
            model.output.weight.data.copy_(pretrained_embeddings)
        Logger("预训练嵌入权重加载完成", accelerator)


class DatabaseProcessor:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 以下所有方法完全不变，仅把 args 当 dict 使用
    def load_or_process_database(
        self,
        database_path: str,
        cache_path: str,
        knowledge_num: int,
        knowledge_length: int,
        recompute: bool = False,
    ) -> torch.Tensor:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        processed_tensor = self._try_load_cache(
            cache_path, knowledge_num, knowledge_length, recompute
        )
        if processed_tensor is None:
            processed_tensor = self._process_database(
                database_path, cache_path, knowledge_num, knowledge_length
            )
        return processed_tensor

    def _try_load_cache(
        self,
        cache_path: str,
        knowledge_num: int,
        knowledge_length: int,
        recompute: bool,
    ) -> Optional[torch.Tensor]:
        if recompute or not os.path.exists(cache_path):
            return None
        try:
            Logger(f"加载缓存文件: {cache_path}")
            processed_tensor = torch.load(cache_path)
            cached_num, cached_length = processed_tensor.shape
            if cached_length != knowledge_length:
                Logger("缓存 knowledge_length 不匹配，重新计算...")
                return None
            if cached_num < knowledge_num:
                Logger("缓存 knowledge_num 不足，重新计算...")
                return None
            if cached_num > knowledge_num:
                processed_tensor = processed_tensor[:knowledge_num, :]
            Logger(f"成功加载缓存数据，形状: {processed_tensor.shape}")
            return processed_tensor
        except Exception as e:
            Logger(f"加载缓存失败: {e}，重新计算...")
            return None

    def _process_database(
        self,
        database_path: str,
        cache_path: str,
        knowledge_num: int,
        knowledge_length: int,
    ) -> torch.Tensor:
        Logger(f"加载数据库文件: {database_path}")
        with open(database_path, "r", encoding="utf-8") as f:
            database_data = json.load(f)
        sentences_data = self._extract_sentences(database_data)
        Logger(f"从数据库加载了 {len(sentences_data)} 条句子")
        processed_tensor, database_mapping = self._process_sentences(
            sentences_data, knowledge_num, knowledge_length
        )
        self._save_cache_and_mapping(
            processed_tensor, database_mapping, cache_path, database_path
        )
        return processed_tensor

    def _extract_sentences(self, database_data: List[Dict]) -> List[Dict[str, str]]:
        sentences_data = []
        for data in database_data:
            if "target" in data and len(data["target"]) > 0:
                target = data["target"][0]
                sentences_data.append(
                    {
                        "sentence": target.get("sentence", ""),
                        "uuid": target.get("uuid", ""),
                        "subject": target.get("subject", ""),
                        "predicate": target.get("predicate", ""),
                        "object": target.get("object", ""),
                    }
                )
        return sentences_data

    def _process_sentences(
        self,
        sentences_data: List[Dict[str, str]],
        knowledge_num: int,
        knowledge_length: int,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        Logger("处理句子数据...")
        processed_rows = []
        database_mapping = []
        num_to_process = min(knowledge_num, len(sentences_data))
        total_sentences = 0
        truncated_sentences = 0
        for i in range(num_to_process):
            sentence_data = sentences_data[i]
            sentence = sentence_data["sentence"]
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            original_length = len(sentence_tokens)
            total_sentences += 1
            if len(sentence_tokens) > knowledge_length:
                truncated_sentences += 1
                sentence_tokens = sentence_tokens[:knowledge_length]
            elif len(sentence_tokens) < knowledge_length:
                sentence_tokens.extend([self.pad_token_id] * (knowledge_length - len(sentence_tokens)))
            processed_rows.append(sentence_tokens)
            database_mapping.append(
                {
                    "database_index": i,
                    "uuid": sentence_data["uuid"],
                    "sentence": sentence,
                    "subject": sentence_data.get("subject", ""),
                    "predicate": sentence_data.get("predicate", ""),
                    "object": sentence_data.get("object", ""),
                    "token_count": len(sentence_tokens),
                    "is_truncated": original_length > knowledge_length,
                }
            )
        while len(processed_rows) < knowledge_num:
            processed_rows.append([self.pad_token_id] * knowledge_length)
        processed_tensor = torch.tensor(processed_rows, dtype=torch.long)
        self._log_statistics(
            total_sentences, truncated_sentences, num_to_process,
            knowledge_num, knowledge_length, processed_tensor.shape
        )
        return processed_tensor, database_mapping

    def _log_statistics(
        self,
        total_sentences: int,
        truncated_sentences: int,
        num_processed: int,
        knowledge_num: int,
        knowledge_length: int,
        final_shape: torch.Size,
    ) -> None:
        truncation_ratio = truncated_sentences / total_sentences if total_sentences > 0 else 0.0
        Logger(f"截断句子统计:")
        Logger(f"  - 总句子数: {total_sentences}")
        Logger(f"  - 截断句子数: {truncated_sentences}")
        Logger(f"  - 截断占比: {truncation_ratio:.4f} ({truncation_ratio*100:.2f}%)")
        Logger(f"数据处理完成:")
        Logger(f"  - 处理句子数: {num_processed}")
        Logger(f"  - 添加空条目数: {knowledge_num - num_processed}")
        Logger(f"  - 最终形状: {final_shape}")
        Logger(f"  - 期望形状: ({knowledge_num}, {knowledge_length})")

    def _save_cache_and_mapping(
        self,
        processed_tensor: torch.Tensor,
        database_mapping: List[Dict],
        cache_path: str,
        database_path: str,
    ) -> None:
        try:
            torch.save(processed_tensor, cache_path)
            Logger(f"处理结果已保存到: {cache_path}")
        except Exception as e:
            Logger(f"保存处理结果失败: {e}")
        try:
            mapping_file_path = cache_path.replace(".pt", "_mapping.json")
            mapping_data = {
                "metadata": {
                    "total_entries": len(database_mapping),
                    "knowledge_num": processed_tensor.shape[0],
                    "knowledge_length": processed_tensor.shape[1],
                    "source_file": database_path,
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "mappings": database_mapping,
            }
            with open(mapping_file_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
            Logger(f"数据库映射已保存到: {mapping_file_path}")
        except Exception as e:
            Logger(f"保存数据库映射失败: {e}")


class MemoryBankProcessor:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    def process_memory_bank(
        self,
        database_path: str,
        cache_path: str,
        knowledge_num: int,
        knowledge_length: int,
    ) -> torch.Tensor:
        if os.path.exists(cache_path):
            Logger(f"从缓存加载memory_bank初始化数据: {cache_path}")
            processed_tensor = torch.load(cache_path)
            Logger(f"加载的memory_bank数据形状: {processed_tensor.shape}")
            return processed_tensor
        Logger(f"处理文本数据用于memory_bank初始化: {database_path}")
        with open(database_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        Logger(f"从 {database_path} 加载了 {len(data)} 条句子")
        processed_tensor, database_mapping = self._process_memory_sentences(
            data, knowledge_num, knowledge_length
        )
        self._save_memory_cache(
            processed_tensor, database_mapping, cache_path, database_path
        )
        return processed_tensor

    def _process_memory_sentences(
        self,
        data: List,
        knowledge_num: int,
        knowledge_length: int,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        processed_rows = []
        database_mapping = []
        total_sentences = len(data)
        truncated_sentences = 0
        num_to_process = min(len(data), knowledge_num)
        Logger(f"处理 {num_to_process}/{total_sentences} 条句子")
        for idx, item in enumerate(data[:num_to_process]):
            if idx % 1000 == 0:
                Logger(f"处理句子 {idx+1}/{num_to_process}")
            sentence_info = self._extract_sentence_info(item)
            sentence = sentence_info["sentence"]
            try:
                tokens_result = self.tokenizer(
                    sentence,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=len(sentence),
                    padding=False,
                    return_tensors="pt",
                )
                tokens = tokens_result["input_ids"].squeeze().tolist()
                if not isinstance(tokens, list):
                    tokens = [tokens]
                original_length = len(tokens)
                if len(tokens) > knowledge_length:
                    tokens = tokens[:knowledge_length]
                    truncated_sentences += 1
                elif len(tokens) < knowledge_length:
                    tokens.extend([self.pad_token_id] * (knowledge_length - len(tokens)))
                processed_rows.append(tokens)
                database_mapping.append(
                    {
                        "database_index": idx,
                        "uuid": sentence_info["uuid"],
                        "sentence": sentence,
                        "subject": sentence_info["subject"],
                        "predicate": sentence_info["predicate"],
                        "object": sentence_info["object"],
                        "token_count": len(tokens),
                        "is_truncated": original_length > knowledge_length,
                    }
                )
            except Exception as e:
                Logger(f"处理句子 {idx} 时出错: {e}")
                empty_tokens = [self.pad_token_id] * knowledge_length
                processed_rows.append(empty_tokens)
                database_mapping.append(
                    {
                        "database_index": idx,
                        "uuid": sentence_info["uuid"],
                        "sentence": sentence,
                        "subject": sentence_info["subject"],
                        "predicate": sentence_info["predicate"],
                        "object": sentence_info["object"],
                        "token_count": knowledge_length,
                        "is_truncated": False,
                        "processing_error": str(e),
                    }
                )
        while len(processed_rows) < knowledge_num:
            processed_rows.append([self.pad_token_id] * knowledge_length)
        processed_tensor = torch.tensor(processed_rows, dtype=torch.long)
        self._log_memory_statistics(
            total_sentences, truncated_sentences, num_to_process,
            knowledge_num, knowledge_length, processed_tensor.shape
        )
        return processed_tensor, database_mapping

    def _extract_sentence_info(self, item: Any) -> Dict[str, str]:
        if isinstance(item, dict):
            if "target" in item and len(item["target"]) > 0:
                target = item["target"][0]
                return {
                    "sentence": target.get("sentence", ""),
                    "uuid": target.get("uuid", ""),
                    "subject": target.get("subject", ""),
                    "predicate": target.get("predicate", ""),
                    "object": target.get("object", ""),
                }
            else:
                return {
                    "sentence": item.get("sentence", "") or item.get("text", "") or str(item),
                    "uuid": item.get("uuid", ""),
                    "subject": item.get("subject", ""),
                    "predicate": item.get("predicate", ""),
                    "object": item.get("object", ""),
                }
        else:
            return {
                "sentence": str(item),
                "uuid": "",
                "subject": "",
                "predicate": "",
                "object": "",
            }

    def _log_memory_statistics(
        self,
        total_sentences: int,
        truncated_sentences: int,
        num_processed: int,
        knowledge_num: int,
        knowledge_length: int,
        final_shape: torch.Size,
    ) -> None:
        truncation_ratio = truncated_sentences / total_sentences if total_sentences > 0 else 0.0
        Logger(f"截断句子统计:")
        Logger(f"  - 总句子数: {total_sentences}")
        Logger(f"  - 截断句子数: {truncated_sentences}")
        Logger(f"  - 截断占比: {truncation_ratio:.4f} ({truncation_ratio*100:.2f}%)")
        Logger(f"Memory_bank数据处理完成:")
        Logger(f"  - 处理句子数: {num_processed}")
        Logger(f"  - 添加空条目数: {knowledge_num - num_processed}")
        Logger(f"  - 最终形状: {final_shape}")
        Logger(f"  - 期望形状: ({knowledge_num}, {knowledge_length})")

    def _save_memory_cache(
        self,
        processed_tensor: torch.Tensor,
        database_mapping: List[Dict],
        cache_path: str,
        database_path: str,
    ) -> None:
        try:
            torch.save(processed_tensor, cache_path)
            Logger(f"处理结果已保存到: {cache_path}")
        except Exception as e:
            Logger(f"保存处理结果失败: {e}")
        try:
            mapping_file_path = cache_path.replace(".pt", "_mapping.json")
            mapping_data = {
                "metadata": {
                    "total_entries": len(database_mapping),
                    "knowledge_num": processed_tensor.shape[0],
                    "knowledge_length": processed_tensor.shape[1],
                    "source_file": database_path,
                    "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "mappings": database_mapping,
            }
            with open(mapping_file_path, "w", encoding="utf-8") as f:
                json.dump(mapping_data, f, ensure_ascii=False, indent=2)
            Logger(f"数据库映射已保存到: {mapping_file_path}")
        except Exception as e:
            Logger(f"保存数据库映射失败: {e}")


# ------------------------------------------------------------------
# 统一入口函数：只改参数读取方式，其余不动
# ------------------------------------------------------------------
def init_model(args: dict, accelerator=None):
    """
    统一的模型初始化接口（直接使用 dict 配置）

    Args:
        args: 配置字典，含全部超参
        accelerator: Accelerator 对象，用于分布式训练时的日志输出

    Returns:
        (model, tokenizer) tuple
    """
    model_type = args.get("model_variant", "model_memory")
    pretrained_embedding_path = args.get("pretrained_embedding_path", None)
    database_init_path = args.get("database_init_path", None)
    cache_path = args.get("cache_path", "cache/knowledge_cache.pt")
    recompute_cache = args.get("recompute_cache", False)

    Logger("=" * 60, accelerator)
    Logger("🚀 开始模型初始化流程", accelerator)
    Logger("=" * 60, accelerator)
    Logger(f"📋 模型配置信息:", accelerator)
    Logger(f"  - 模型类型: {model_type}", accelerator)
    Logger(f"  - 预训练嵌入路径: {pretrained_embedding_path if pretrained_embedding_path else '未指定'}", accelerator)
    Logger(f"  - 数据库初始化路径: {database_init_path if database_init_path else '未指定'}", accelerator)
    Logger(f"  - 缓存路径: {cache_path}", accelerator)
    Logger(f"  - 重新计算缓存: {recompute_cache}", accelerator)
    
    type_config = ModelTypeConfig.get_config(model_type)
    Logger(f"  - 数据库属性: {type_config.get('database_attribute', '无')}", accelerator)
    Logger(f"  - 需要权重初始化: {type_config.get('requires_weight_init', False)}", accelerator)
    Logger(f"  - 内存优化: {type_config.get('memory_optimization', False)}", accelerator)

    # 动态导入模型类
    Logger(f"📦 导入模型模块: {type_config['module_path']}", accelerator)
    module = __import__(type_config["module_path"], fromlist=[type_config["class_name"]])
    ExplicitLM = getattr(module, type_config["class_name"])
    
    # 输出当前目录
    Logger(f"📍 当前工作目录: {os.getcwd()}", accelerator)
    
    # 加载 tokenizer
    Logger("🔤 加载tokenizer...", accelerator)
    tokenizer_dir = Path(get_original_cwd()) / "models" / "ExplicitLM_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    Logger(f"✅ Tokenizer加载完成，词汇表大小: {len(tokenizer)}", accelerator)

    # 构造 LMConfig 对象（仅用于满足旧构造函数签名）
    # lm_config = LMConfig(
    #     dim=args["dim"],
    #     n_layers=args["n_layers"],
    #     n_heads=args["n_heads"],
    #     n_kv_heads=args["n_kv_heads"],
    #     vocab_size=args["vocab_size"],
    #     max_seq_len=args["max_seq_len"],
    #     knowledge_num=args["knowledge_num"],
    #     knowledge_length=args["knowledge_length"],
    #     knowledge_dim=args["knowledge_dim"],
    #     model_variant=model_type,
    #     pretrained_embedding_path=pretrained_embedding_path,
    #     database_init_path=database_init_path,
    #     cache_path=cache_path,
    #     recompute_cache=recompute_cache,
    #     use_moe=args.get("use_moe", False),
    #     flash_attn=args.get("flash_attn", True),
    #     dropout=args.get("dropout", 0.0),
    # )

    # 创建模型
    Logger("🏗️  创建模型实例...", accelerator)
    model = ExplicitLM(args)
    Logger("✅ 模型实例创建完成", accelerator)

    # 权重初始化
    if type_config["requires_weight_init"]:
        Logger("⚖️  执行模型权重初始化...", accelerator)
        WeightInitializer.initialize_model_weights(model, model_type, accelerator)
        Logger("✅ 模型权重初始化完成", accelerator)

    if type_config.get("memory_optimization"):
        Logger("✅ 显存优化策略：候选项减少(32→16) + DeepSpeed参数offload", accelerator)

    # 预训练嵌入
    if pretrained_embedding_path:
        Logger("🎯 加载预训练嵌入权重...", accelerator)
        EmbeddingLoader.load_pretrained_embeddings(model, pretrained_embedding_path, accelerator)
        Logger("✅ 预训练嵌入权重加载完成", accelerator)

    # 数据库 / 记忆库初始化
    if database_init_path and type_config["database_attribute"]:
        Logger("🗄️  开始数据库/记忆库初始化...", accelerator)
        Logger(f"  - 数据库路径: {database_init_path}", accelerator)
        Logger(f"  - 缓存路径: {cache_path}", accelerator)
        Logger(f"  - 知识库大小: {args.knowledge_num}", accelerator)
        Logger(f"  - 知识条目长度: {args.knowledge_length}", accelerator)
        Logger(f"  - 目标属性: {type_config['database_attribute']}", accelerator)
        
        _initialize_database(
            model=model,
            tokenizer=tokenizer,
            database_path=database_init_path,
            cache_path=cache_path,
            knowledge_num=args.knowledge_num,
            knowledge_length=args.knowledge_length,
            recompute=recompute_cache,
            model_type=model_type,
            database_attribute=type_config["database_attribute"],
            accelerator=accelerator,
        )
        Logger("✅ 数据库/记忆库初始化完成", accelerator)
    else:
        if not database_init_path:
            Logger("⚠️  未指定数据库初始化路径，跳过数据库初始化", accelerator)
        if not type_config["database_attribute"]:
            Logger("⚠️  当前模型类型不支持数据库初始化，跳过", accelerator)

    # 参数统计
    Logger("📊 计算模型参数统计...", accelerator)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f"📈 LLM总参数量：{total_params:.3f} 百万", accelerator)
    
    Logger("=" * 60, accelerator)
    Logger("🎉 模型初始化流程完成", accelerator)
    Logger("=" * 60, accelerator)

    return model, tokenizer


# ---------- 以下辅助函数完全不变 ----------
def _initialize_database(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    database_path: str,
    cache_path: str,
    knowledge_num: int,
    knowledge_length: int,
    recompute: bool,
    model_type: str,
    database_attribute: str,
    accelerator=None,
) -> None:
    Logger("🔍 开始数据库/记忆库初始化详细流程", accelerator)
    Logger(f"📊 初始化参数:", accelerator)
    Logger(f"  - 模型类型: {model_type}", accelerator)
    Logger(f"  - 数据库路径: {database_path}", accelerator)
    Logger(f"  - 缓存路径: {cache_path}", accelerator)
    Logger(f"  - 知识库大小: {knowledge_num}", accelerator)
    Logger(f"  - 知识条目长度: {knowledge_length}", accelerator)
    Logger(f"  - 重新计算缓存: {recompute}", accelerator)
    Logger(f"  - 目标属性路径: {database_attribute}", accelerator)
    
    # 检查数据库文件是否存在
    if not os.path.exists(database_path):
        Logger(f"❌ 错误: 数据库文件不存在: {database_path}", accelerator)
        raise FileNotFoundError(f"数据库文件不存在: {database_path}")
    
    # 获取数据库文件信息
    try:
        file_size = os.path.getsize(database_path)
        file_size_mb = file_size / (1024 * 1024)
        Logger(f"📁 数据库文件信息:", accelerator)
        Logger(f"  - 文件大小: {file_size_mb:.2f} MB ({file_size:,} bytes)", accelerator)
        Logger(f"  - 文件路径: {os.path.abspath(database_path)}", accelerator)
    except Exception as e:
        Logger(f"⚠️  无法获取数据库文件信息: {e}", accelerator)
    
    # 确保缓存目录存在
    cache_dir = os.path.dirname(cache_path)
    if cache_dir and not os.path.exists(cache_dir):
        Logger(f"📁 创建缓存目录: {cache_dir}", accelerator)
        os.makedirs(cache_dir, exist_ok=True)
    
    # 根据模型类型选择处理器
    if model_type == "model_memory":
        Logger("🧠 使用MemoryBankProcessor处理记忆库数据", accelerator)
        processor = MemoryBankProcessor(tokenizer)
        if not cache_path or cache_path == "cache/knowledge_cache.pt":
            cache_path = f"cache/memory_bank_init_{knowledge_num}_{knowledge_length}.pt"
            Logger(f"🔄 自动调整缓存路径为: {cache_path}", accelerator)
        
        Logger("🚀 开始处理记忆库数据...", accelerator)
        start_time = time.time()
        processed_tensor = processor.process_memory_bank(
            database_path=database_path,
            cache_path=cache_path,
            knowledge_num=knowledge_num,
            knowledge_length=knowledge_length,
        )
        processing_time = time.time() - start_time
        Logger(f"⏱️  记忆库数据处理完成，耗时: {processing_time:.2f} 秒", accelerator)
        
    else:
        Logger("💾 使用DatabaseProcessor处理知识库数据", accelerator)
        processor = DatabaseProcessor(tokenizer)
        
        Logger("🚀 开始处理知识库数据...", accelerator)
        start_time = time.time()
        processed_tensor = processor.load_or_process_database(
            database_path=database_path,
            cache_path=cache_path,
            knowledge_num=knowledge_num,
            knowledge_length=knowledge_length,
            recompute=recompute,
        )
        processing_time = time.time() - start_time
        Logger(f"⏱️  知识库数据处理完成，耗时: {processing_time:.2f} 秒", accelerator)
    
    # 验证处理后的张量
    Logger("🔍 验证处理后的数据张量...", accelerator)
    if processed_tensor is not None:
        Logger(f"✅ 张量验证通过:", accelerator)
        Logger(f"  - 张量形状: {processed_tensor.shape}", accelerator)
        Logger(f"  - 张量类型: {processed_tensor.dtype}", accelerator)
        Logger(f"  - 张量设备: {processed_tensor.device}", accelerator)
        Logger(f"  - 内存占用: {processed_tensor.numel() * processed_tensor.element_size() / (1024*1024):.2f} MB", accelerator)
        
        # 检查数据范围
        if processed_tensor.numel() > 0:
            min_val = processed_tensor.min().item()
            max_val = processed_tensor.max().item()
            Logger(f"  - 数据范围: [{min_val}, {max_val}]", accelerator)
            
            # 检查是否有异常值
            if min_val < 0:
                Logger(f"⚠️  警告: 发现负值token ID，最小值: {min_val}", accelerator)
            if hasattr(tokenizer, 'vocab_size') and max_val >= tokenizer.vocab_size:
                Logger(f"⚠️  警告: 发现超出词汇表范围的token ID，最大值: {max_val}, 词汇表大小: {tokenizer.vocab_size}", accelerator)
    else:
        Logger("❌ 错误: 处理后的张量为None", accelerator)
        raise ValueError("数据处理失败，返回的张量为None")
    
    # 设置模型属性
    Logger("🔧 设置模型数据库属性...", accelerator)
    start_time = time.time()
    _set_database_attribute(model, database_attribute, processed_tensor, accelerator)
    attribute_setting_time = time.time() - start_time
    Logger(f"⏱️  模型属性设置完成，耗时: {attribute_setting_time:.4f} 秒", accelerator)
    
    # 验证模型属性是否正确设置
    Logger("🔍 验证模型属性设置...", accelerator)
    attributes = database_attribute.split(".")
    target = model
    for attr in attributes[:-1]:
        if hasattr(target, attr):
            target = getattr(target, attr)
        else:
            Logger(f"❌ 错误: 无法找到中间属性 {attr}", accelerator)
            return
    
    final_attr = attributes[-1]
    if hasattr(target, final_attr):
        stored_tensor = getattr(target, final_attr)
        if torch.equal(stored_tensor, processed_tensor):
            Logger(f"✅ 模型属性验证成功: model.{database_attribute}", accelerator)
            Logger(f"  - 存储张量形状: {stored_tensor.shape}", accelerator)
            Logger(f"  - 存储张量类型: {stored_tensor.dtype}", accelerator)
            Logger(f"  - 存储张量设备: {stored_tensor.device}", accelerator)
        else:
            Logger(f"❌ 错误: 模型属性验证失败，存储的张量与原始张量不匹配", accelerator)
    else:
        Logger(f"❌ 错误: 无法找到目标属性 {final_attr}", accelerator)
    
    # 总体统计
    total_time = time.time() - start_time
    Logger("📊 数据库/记忆库初始化统计:", accelerator)
    Logger(f"  - 总处理时间: {total_time:.2f} 秒", accelerator)
    Logger(f"  - 数据条目数: {processed_tensor.shape[0]}", accelerator)
    Logger(f"  - 每条目长度: {processed_tensor.shape[1]}", accelerator)
    Logger(f"  - 总token数: {processed_tensor.numel()}", accelerator)
    Logger(f"  - 处理速度: {processed_tensor.numel() / total_time:.0f} tokens/秒", accelerator)
    
    Logger("✅ 数据库嵌入和句子已成功存储到模型", accelerator)


def _set_database_attribute(model: nn.Module, attribute_path: str, data: torch.Tensor, accelerator=None) -> None:
    attributes = attribute_path.split(".")
    target = model
    for attr in attributes[:-1]:
        if not hasattr(target, attr):
            Logger(f"警告: 找不到属性 {attr}，无法初始化数据库", accelerator)
            return
        target = getattr(target, attr)
    final_attr = attributes[-1]
    if hasattr(target, final_attr):
        getattr(target, final_attr).data.copy_(data)
        Logger(f"成功初始化 model.{attribute_path} 使用处理后的数据", accelerator)
    else:
        Logger(f"警告: 找不到 model.{attribute_path} 进行初始化", accelerator)
        globals()["processed_database"] = data
