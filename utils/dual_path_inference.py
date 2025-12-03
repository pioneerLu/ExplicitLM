"""
双路推理包装器：主路（Qwen3ExplicitLM）和辅路（llmlingua事实提取）
"""
import torch
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path

from utils.fact_extractor import FactExtractor
from utils.memory_bank_updater import MemoryBankUpdater
from utils.logger import Logger


class DualPathInference:
    """双路推理包装器"""
    
    def __init__(
        self,
        model,
        tokenizer,
        fact_extractor: Optional[FactExtractor] = None,
        memory_bank_updater: Optional[MemoryBankUpdater] = None,
        enable_fact_extraction: bool = True,
        fact_update_frequency: int = 1,  # 每N次推理更新一次事实
        update_strategy: str = "fifo",
        compression_rate: float = 0.4,
        llmlingua_model_path: Optional[str] = None,  # LLMLingua模型路径
    ):
        """
        初始化双路推理包装器
        
        Args:
            model: ExplicitLM 模型实例
            tokenizer: tokenizer 实例
            fact_extractor: 事实提取器（如果为None且启用事实提取，将自动创建）
            memory_bank_updater: 记忆库更新器（如果为None，将自动创建）
            enable_fact_extraction: 是否启用事实提取
            fact_update_frequency: 事实更新频率（每N次推理更新一次）
            update_strategy: 记忆库更新策略
            compression_rate: 文本压缩率
            llmlingua_model_path: LLMLingua模型路径（如果fact_extractor为None时使用）
        """
        self.model = model
        self.tokenizer = tokenizer
        self.enable_fact_extraction = enable_fact_extraction
        self.fact_update_frequency = fact_update_frequency
        self.inference_counter = 0
        
        # 初始化事实提取器
        if fact_extractor is None and enable_fact_extraction:
            if llmlingua_model_path is None:
                raise ValueError(
                    "启用事实提取时，必须提供 llmlingua_model_path 参数或传入 fact_extractor 实例"
                )
            self.fact_extractor = FactExtractor(
                model_path=llmlingua_model_path,
                compression_rate=compression_rate
            )
        else:
            self.fact_extractor = fact_extractor
        
        # 初始化记忆库更新器
        if memory_bank_updater is None and enable_fact_extraction:
            self.memory_bank_updater = MemoryBankUpdater(
                model=model,
                tokenizer=tokenizer,
                fact_extractor=self.fact_extractor,
                update_strategy=update_strategy,
            )
        else:
            self.memory_bank_updater = memory_bank_updater
        
        Logger(f"✅ 双路推理包装器初始化完成")
        Logger(f"  - 事实提取: {'启用' if enable_fact_extraction else '禁用'}（仅在推理时）")
        Logger(f"  - 更新频率: 每 {fact_update_frequency} 次推理")
        Logger(f"  - 更新策略: {update_strategy}")
        Logger(f"  - 训练模式: 只更新知识融合部分（通过梯度），不更新知识库")
        Logger(f"  - 推理模式: 提取事实并更新知识库")
    
    def generate(
        self,
        input_ids: torch.Tensor,
        input_text: Optional[str] = None,
        **generation_kwargs,
    ) -> Dict[str, any]:
        """
        双路推理生成
        
        推理时进行：
        1. 主路：正常生成文本
        2. 辅路：提取浓缩事实并更新知识库
        
        Args:
            input_ids: 输入token IDs
            input_text: 输入文本（用于事实提取，如果为None将自动解码）
            **generation_kwargs: 传递给model.generate的参数
        
        Returns:
            {
                'generated_ids': torch.Tensor,  # 生成的token IDs
                'generated_text': str,  # 生成的文本
                'fact_extraction': Dict,  # 事实提取结果（如果启用）
                'memory_update': Dict,  # 记忆库更新结果（如果启用）
            }
        """
        # ===== 主路：正常推理 =====
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids, **generation_kwargs)
        
        # 解码生成的文本（只解码新生成的部分，不包括输入）
        input_length = input_ids.shape[1]
        if generated_ids.dim() > 1:
            # 只取新生成的token
            new_tokens = generated_ids[0, input_length:]
            generated_text = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        else:
            new_tokens = generated_ids[input_length:]
            generated_text = self.tokenizer.decode(
                new_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
        
        result = {
            'generated_ids': generated_ids,
            'generated_text': generated_text,
        }
        
        # ===== 辅路：事实提取和记忆库更新（仅在推理时） =====
        if self.enable_fact_extraction:
            self.inference_counter += 1
            
            # 获取输入文本（如果未提供）
            if input_text is None:
                input_text = self.tokenizer.decode(
                    input_ids[0] if input_ids.dim() > 1 else input_ids,
                    skip_special_tokens=True
                )
            
            # 检查是否需要更新（根据频率）
            should_update = (self.inference_counter % self.fact_update_frequency == 0)
            
            if should_update:
                # 提取事实
                fact_result = self.fact_extractor.extract_facts(
                    input_text,
                    return_annotations=False,
                )
                result['fact_extraction'] = fact_result
                
                # 更新记忆库
                if fact_result['compressed_text']:
                    update_result = self.memory_bank_updater.update_from_text(
                        input_text,
                        compression_rate=self.fact_extractor.compression_rate,
                    )
                    result['memory_update'] = update_result
                    Logger(
                        f"📝 记忆库已更新: {update_result.get('updated_count', 0)} 条事实",
                        accelerator=None
                    )
            else:
                result['fact_extraction'] = {"skipped": True, "reason": "frequency_check"}
                result['memory_update'] = {"skipped": True, "reason": "frequency_check"}
        
        return result
    
    def forward(
        self,
        input_ids: torch.Tensor,
        input_text: Optional[str] = None,
        **forward_kwargs,
    ) -> Dict[str, any]:
        """
        双路前向传播（用于训练或评估）
        
        注意：训练时只进行正常的前向传播，不进行事实提取和知识库更新。
        知识融合部分会通过梯度自动更新。
        
        Args:
            input_ids: 输入token IDs
            input_text: 输入文本（训练时不需要，会被忽略）
            **forward_kwargs: 传递给model.forward的参数
        
        Returns:
            {
                'model_output': ModelOutput,  # 模型输出
            }
        """
        # ===== 主路：正常前向传播 =====
        # 训练时只进行正常的前向传播，知识融合部分通过梯度更新
        model_output = self.model(input_ids, **forward_kwargs)
        
        result = {
            'model_output': model_output,
        }
        
        # ===== 训练时不进行事实提取和知识库更新 =====
        # 知识融合部分（MemoryGate, GatedMemoryFusion）会通过梯度自动更新
        # 知识库更新只在推理时进行
        
        return result
    
    def get_statistics(self) -> Dict[str, any]:
        """获取统计信息"""
        stats = {
            "inference_counter": self.inference_counter,
            "enable_fact_extraction": self.enable_fact_extraction,
            "fact_update_frequency": self.fact_update_frequency,
        }
        
        if self.memory_bank_updater:
            stats["memory_bank_stats"] = self.memory_bank_updater.get_statistics()
        
        return stats

