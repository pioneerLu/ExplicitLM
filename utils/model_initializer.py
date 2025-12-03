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
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from hydra.utils import get_original_cwd
from pathlib import Path
from utils.logger import Logger


# ---------- 以下代码用于处理记忆库数据 ----------
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
        recompute: bool = False,
    ) -> torch.Tensor:
        if not recompute and os.path.exists(cache_path):
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
    统一的模型初始化接口（仅支持Qwen3架构）

    Args:
        args: 配置字典，需包含：
            - qwen3_model_path: Qwen3预训练模型路径（必需）
            - 记忆库相关配置（knowledge_num, knowledge_dim等）
        accelerator: Accelerator 对象，用于分布式训练时的日志输出

    Returns:
        (model, tokenizer) tuple
    """
    # 只支持 Qwen3 架构
    qwen3_model_path = args.get("qwen3_model_path", None)
    if qwen3_model_path is None:
        raise ValueError("必须指定qwen3_model_path参数，指向Qwen3-4B预训练模型路径")
    
    return _init_qwen3_model(args, accelerator)


def _init_qwen3_model(args: dict, accelerator=None):
    """Qwen3 架构模型初始化"""
    qwen3_model_path = args.get("qwen3_model_path", None)
    if qwen3_model_path is None:
        raise ValueError("必须指定qwen3_model_path参数，指向Qwen3-4B预训练模型路径")
    
    database_init_path = args.get("database_init_path", None)
    cache_path = args.get("cache_path", "cache/knowledge_cache.pt")
    recompute_cache = args.get("recompute_cache", False)

    Logger("=" * 60, accelerator)
    Logger("🚀 开始模型初始化流程（Qwen3架构）", accelerator)
    Logger("=" * 60, accelerator)
    Logger(f"📋 模型配置信息:", accelerator)
    Logger(f"  - Qwen3模型路径: {qwen3_model_path}", accelerator)
    Logger(f"  - 数据库初始化路径: {database_init_path if database_init_path else '未指定'}", accelerator)
    Logger(f"  - 缓存路径: {cache_path}", accelerator)
    Logger(f"  - 重新计算缓存: {recompute_cache}", accelerator)
    
    # 加载Qwen3配置
    Logger("📦 加载Qwen3配置...", accelerator)
    qwen3_config = Qwen3Config.from_pretrained(qwen3_model_path)
    Logger(f"✅ Qwen3配置加载完成:", accelerator)
    Logger(f"  - hidden_size: {qwen3_config.hidden_size}", accelerator)
    Logger(f"  - num_hidden_layers: {qwen3_config.num_hidden_layers}", accelerator)
    Logger(f"  - num_attention_heads: {qwen3_config.num_attention_heads}", accelerator)
    Logger(f"  - vocab_size: {qwen3_config.vocab_size}", accelerator)
    
    # 提取记忆库配置
    memory_cfg = {
        "knowledge_num": args.get("knowledge_num", 1024 * 1024),
        "knowledge_length": args.get("knowledge_length", 16),
        "knowledge_dim": args.get("knowledge_dim", 128),
        "use_ema_update": args.get("use_ema_update", True),
        "ema_decay": args.get("ema_decay", 0.9),
        "ema_update_freq": args.get("ema_update_freq", 5),
        "freeze_ratio": args.get("freeze_ratio", 0.2),
        "num_candidates": args.get("num_candidates", 16),
        "num_selected": args.get("num_selected", 1),
        "gumbel_temperature": args.get("gumbel_temperature", 1.0),
        "use_moe": args.get("use_moe", False),
        "dropout": args.get("dropout", 0.0),
    }
    Logger(f"📋 记忆库配置:", accelerator)
    Logger(f"  - knowledge_num: {memory_cfg['knowledge_num']}", accelerator)
    Logger(f"  - knowledge_length: {memory_cfg['knowledge_length']}", accelerator)
    Logger(f"  - knowledge_dim: {memory_cfg['knowledge_dim']}", accelerator)
    
    # 导入模型类
    Logger("📦 导入模型模块...", accelerator)
    from models.core.ExplicitLM import ExplicitLM
    
    # 输出当前目录
    Logger(f"📍 当前工作目录: {os.getcwd()}", accelerator)
    
    # 加载 Qwen3 tokenizer
    # 优先使用本地tokenizer路径，如果不存在则从Qwen模型路径加载
    Logger("🔤 加载Qwen3 tokenizer...", accelerator)
    try:
        original_cwd = get_original_cwd()
    except ValueError:
        # 非 Hydra 环境，使用当前工作目录
        original_cwd = os.getcwd()
    local_tokenizer_path = Path(original_cwd) / "models" / "qwen_tokenizer"
    if local_tokenizer_path.exists() and (local_tokenizer_path / "tokenizer.json").exists():
        Logger(f"  - 使用本地tokenizer: {local_tokenizer_path}", accelerator)
        tokenizer = AutoTokenizer.from_pretrained(str(local_tokenizer_path), trust_remote_code=True)
    else:
        Logger(f"  - 从Qwen模型路径加载: {qwen3_model_path}", accelerator)
        tokenizer = AutoTokenizer.from_pretrained(qwen3_model_path, trust_remote_code=True)
    
    # 确保Qwen tokenizer的特殊token配置正确
    if tokenizer.pad_token is None:
        # Qwen tokenizer可能没有pad_token，使用eos_token作为pad_token
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        Logger("⚠️  Qwen tokenizer没有pad_token，使用eos_token作为pad_token", accelerator)
    
    Logger(f"✅ Tokenizer加载完成，词汇表大小: {len(tokenizer)}", accelerator)
    Logger(f"  - pad_token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})", accelerator)
    Logger(f"  - bos_token: {tokenizer.bos_token} (id: {tokenizer.bos_token_id})", accelerator)
    Logger(f"  - eos_token: {tokenizer.eos_token} (id: {tokenizer.eos_token_id})", accelerator)

    # 创建模型
    Logger("🏗️  创建模型实例...", accelerator)
    model = ExplicitLM(qwen3_config=qwen3_config, memory_cfg=memory_cfg)
    Logger("✅ 模型实例创建完成", accelerator)
    
    # 从Qwen3预训练模型加载权重
    Logger("🎯 从Qwen3预训练模型加载权重...", accelerator)
    try:
        from transformers import Qwen3ForCausalLM
        pretrained_model = Qwen3ForCausalLM.from_pretrained(
            qwen3_model_path,
            torch_dtype=torch.float32,  # 先加载为float32，后续可以转换
            device_map="cpu",
        )
        
        # 加载匹配的权重
        model_state_dict = model.state_dict()
        pretrained_state_dict = pretrained_model.state_dict()
        
        def map_pretrained_key(pretrained_key: str) -> str:
            """将预训练模型的层名称映射到我们的模型层名称"""
            # 移除 "model." 前缀
            if pretrained_key.startswith("model."):
                key = pretrained_key[6:]  # 移除 "model." 前缀
            else:
                key = pretrained_key
            
            # 映射层名称
            if key.startswith("layers."):
                # 将 layers.X.xxx 映射为 layers.X.qwen3_decoder.xxx
                parts = key.split(".", 2)
                if len(parts) >= 3:
                    layer_idx = parts[1]
                    rest = parts[2]
                    return f"layers.{layer_idx}.qwen3_decoder.{rest}"
                else:
                    return key
            else:
                # embed_tokens, norm, lm_head 等直接使用
                return key
        
        loaded_keys = []
        missing_keys = []
        shape_mismatches = []
        
        for key in model_state_dict.keys():
            # 跳过新增的参数
            if key.startswith("memory_bank") or key.startswith("tok_embeddings") or \
               "memory_gate" in key or "gated_memory_fusion" in key or "memory_norm" in key:
                continue
            
            # 尝试直接匹配
            if key in pretrained_state_dict:
                if model_state_dict[key].shape == pretrained_state_dict[key].shape:
                    model_state_dict[key] = pretrained_state_dict[key]
                    loaded_keys.append(key)
                else:
                    shape_mismatches.append(f"{key} (shape: {model_state_dict[key].shape} vs {pretrained_state_dict[key].shape})")
            else:
                # 尝试通过映射找到对应的预训练权重
                found = False
                for pretrained_key in pretrained_state_dict.keys():
                    mapped_key = map_pretrained_key(pretrained_key)
                    if mapped_key == key:
                        if model_state_dict[key].shape == pretrained_state_dict[pretrained_key].shape:
                            model_state_dict[key] = pretrained_state_dict[pretrained_key]
                            loaded_keys.append(key)
                            found = True
                            break
                        else:
                            shape_mismatches.append(f"{key} (shape: {model_state_dict[key].shape} vs {pretrained_state_dict[pretrained_key].shape})")
                            found = True
                            break
                
                if not found:
                    missing_keys.append(key)
        
        model.load_state_dict(model_state_dict, strict=False)
        Logger(f"✅ 权重加载完成: {len(loaded_keys)}个参数已加载", accelerator)
        if missing_keys:
            Logger(f"⚠️  以下参数未加载（可能是新增的记忆相关参数）: {len(missing_keys)}个", accelerator)
            if len(missing_keys) <= 10:
                for key in missing_keys[:10]:
                    Logger(f"    - {key}", accelerator)
        if shape_mismatches:
            Logger(f"⚠️  以下参数形状不匹配: {len(shape_mismatches)}个", accelerator)
            if len(shape_mismatches) <= 5:
                for key in shape_mismatches[:5]:
                    Logger(f"    - {key}", accelerator)
        
        # 清理临时模型
        del pretrained_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        Logger(f"⚠️  从预训练模型加载权重失败: {e}", accelerator)
        Logger("将使用随机初始化的权重", accelerator)

    # 数据库 / 记忆库初始化（MOE 模式下跳过）
    use_moe = memory_cfg.get("use_moe", False)
    if use_moe:
        Logger("⚠️  MOE 模式：跳过记忆库初始化（MOE 模式不需要 memory_bank）", accelerator)
    elif database_init_path:
        Logger("🗄️  开始记忆库初始化...", accelerator)
        Logger(f"  - 数据库路径: {database_init_path}", accelerator)
        Logger(f"  - 缓存路径: {cache_path}", accelerator)
        Logger(f"  - 知识库大小: {memory_cfg['knowledge_num']}", accelerator)
        Logger(f"  - 知识条目长度: {memory_cfg['knowledge_length']}", accelerator)
        
        _initialize_database(
            model=model,
            tokenizer=tokenizer,
            database_path=database_init_path,
            cache_path=cache_path,
            knowledge_num=memory_cfg["knowledge_num"],
            knowledge_length=memory_cfg["knowledge_length"],
            recompute=recompute_cache,
            model_type="qwen3_explicitlm",
            database_attribute="memory_bank",
            accelerator=accelerator,
        )
        Logger("✅ 记忆库初始化完成", accelerator)
    else:
        Logger("⚠️  未指定数据库初始化路径，记忆库将使用随机初始化", accelerator)

    # 冻结Qwen主模型参数，只保留记忆库相关参数可训练
    Logger("🔒 冻结Qwen主模型参数...", accelerator)
    frozen_params = 0
    trainable_params = 0
    
    use_ema_update = memory_cfg.get("use_ema_update", False)
    
    # 冻结所有Qwen基础组件
    for name, param in model.named_parameters():
        # 保留可训练的参数：记忆库相关组件
        is_memory_component = any(keyword in name for keyword in [
            "memory_gate",  # 记忆门控模块
            "gated_memory_fusion",  # 记忆融合模块
            "memory_norm",  # 记忆归一化层
        ])
        
        # memory_bank存储的是token IDs（int64），不应该直接通过梯度更新
        # 应该通过EMA机制更新，所以始终设置为不可训练
        is_memory_bank = "memory_bank" in name
        if is_memory_bank:
            # memory_bank始终不可训练，避免DeepSpeed梯度平均时的类型错误
            param.requires_grad = False
            frozen_params += param.numel()
        elif is_memory_component:
            # 其他记忆相关组件始终可训练
            param.requires_grad = True
            trainable_params += param.numel()
        else:
            # 冻结所有其他参数（Qwen主模型）
            param.requires_grad = False
            frozen_params += param.numel()
    
    Logger(f"✅ 参数冻结完成:", accelerator)
    Logger(f"  - 冻结参数: {frozen_params / 1e6:.3f} 百万", accelerator)
    Logger(f"  - 可训练参数: {trainable_params / 1e6:.3f} 百万", accelerator)
    if frozen_params + trainable_params > 0:
        Logger(f"  - 冻结比例: {frozen_params / (frozen_params + trainable_params) * 100:.2f}%", accelerator)
    
    # 参数统计
    Logger("📊 计算模型参数统计...", accelerator)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    Logger(f"📈 可训练参数量：{total_params:.3f} 百万", accelerator)
    
    Logger("=" * 60, accelerator)
    Logger("🎉 模型初始化流程完成", accelerator)
    Logger("=" * 60, accelerator)

    return model, tokenizer


# ---------- 以下辅助函数 ----------
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
    
    # 使用MemoryBankProcessor处理记忆库数据（Qwen3架构只支持memory_bank）
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
        recompute=recompute,
    )
    processing_time = time.time() - start_time
    Logger(f"⏱️  记忆库数据处理完成，耗时: {processing_time:.2f} 秒", accelerator)
    
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
