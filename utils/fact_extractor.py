"""
事实提取器：使用 LLMLingua 从输入文本中提取浓缩事实
"""
import os
from typing import List, Dict, Optional, Tuple
import torch
from pathlib import Path

try:
    from llmlingua import PromptCompressor
    LLMLINGUA_AVAILABLE = True
except ImportError:
    LLMLINGUA_AVAILABLE = False
    print("警告: llmlingua 未安装，事实提取功能将不可用")


class FactExtractor:
    """使用 LLMLingua 提取浓缩事实"""
    
    def __init__(
        self,
        model_path: str = "llmlingua-2-bert",  
        compression_rate: float = 0.4,  # 压缩到40%，保留60%的关键信息
        force_tokens: Optional[List[str]] = None,
        chunk_end_tokens: Optional[List[str]] = None,
        device: Optional[str] = None,
    ):
        """
        初始化事实提取器
        
        Args:
            model_path: LLMLingua 模型路径
            compression_rate: 压缩率（0-1），越小保留的信息越多
            force_tokens: 强制保留的token列表（如标点符号）
            chunk_end_tokens: 分块结束token列表
            device: 运行设备
        """
        if not LLMLINGUA_AVAILABLE:
            raise ImportError("llmlingua 未安装，请先安装: pip install llmlingua")
        
        self.model_path = model_path
        self.compression_rate = compression_rate
        self.force_tokens = force_tokens or ['\n', '.', '!', '?', ',', ':', ';']
        self.chunk_end_tokens = chunk_end_tokens or ['.', '\n', '!', '?']
        # 优化：LLMLingua 默认使用 CPU，避免占用 GPU 显存
        # 如果需要使用 GPU，可以在调用时指定 device="cuda"
        self.device = device or "cpu"  # 默认使用 CPU，避免占用训练 GPU 显存
        
        # 初始化 LLMLingua
        self._init_compressor()
    
    def _init_compressor(self):
        """初始化 PromptCompressor"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"LLMLingua 模型路径不存在: {self.model_path}\n"
                f"请运行 bert/get_model.py 下载模型"
            )
        
        self.compressor = PromptCompressor(
            model_name=self.model_path,
            use_llmlingua2=True
        )
        # 修复NotImplementedError: is_begin_of_new_word函数需要识别模型名称
        if hasattr(self.compressor, 'model_name'):
            model_name_str = str(self.compressor.model_name).lower()
            if 'bert-base-multilingual-cased' not in model_name_str and 'xlm-roberta-large' not in model_name_str:
                import json
                config_path = os.path.join(self.model_path, 'config.json')
                if os.path.exists(config_path):
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                            if config.get('model_type', '').lower() == 'bert':
                                self.compressor.model_name = 'bert-base-multilingual-cased'
                    except Exception:
                        pass
    
    def extract_facts(
        self,
        text: str,
        return_annotations: bool = False,
        compression_rate: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        从文本中提取浓缩事实
        
        Args:
            text: 输入文本
            return_annotations: 是否返回标注信息（哪些token被保留/删除）
            compression_rate: 可选，动态指定压缩率（如果为None，使用初始化时的compression_rate）
        
        Returns:
            {
                'compressed_text': str,  # 压缩后的文本（浓缩事实）
                'original_tokens': int,  # 原始token数
                'compressed_tokens': int,  # 压缩后token数
                'compression_ratio': float,  # 压缩比例
                'annotations': List[Tuple[str, str]],  # (word, label) 列表，label为'+'或'-'
            }
        """
        if not text or not text.strip():
            return {
                'compressed_text': '',
                'original_tokens': 0,
                'compressed_tokens': 0,
                'compression_ratio': 0.0,
                'annotations': [],
            }
        
        # 使用动态压缩率或默认压缩率
        actual_compression_rate = compression_rate if compression_rate is not None else self.compression_rate
        
        try:
            # compress_prompt_llmlingua2 需要 context 参数为 List[str]，而不是单个字符串
            # 将文本按句子分割成列表
            import re
            # 按句子分割文本
            sentences = re.split(r'([.!?]\s+)', text)
            # 重新组合句子（保留分隔符）
            context_list = []
            for i in range(0, len(sentences), 2):
                if i < len(sentences):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]
                    if sentence.strip():
                        context_list.append(sentence.strip())
            
            # 如果没有分割出句子，使用整个文本
            if not context_list:
                context_list = [text]
            
            results = self.compressor.compress_prompt_llmlingua2(
                context_list,  # 传入字符串列表
                rate=actual_compression_rate,
                force_tokens=self.force_tokens,
                chunk_end_tokens=self.chunk_end_tokens,
                return_word_label=return_annotations,
                drop_consecutive=True,
            )
            
            # 解析标注信息
            annotations = []
            if return_annotations and "fn_labeled_original_prompt" in results:
                word_sep = "\t\t|\t\t"
                label_sep = " "
                lines = results["fn_labeled_original_prompt"].split(word_sep)
                for line in lines:
                    if label_sep in line:
                        parts = line.split(label_sep, 1)
                        if len(parts) == 2:
                            word, label = parts
                            annotations.append((word, '+' if label == '1' else '-'))
            
            # compress_prompt_llmlingua2 返回的 compressed_prompt 
            compressed_prompt = results.get('compressed_prompt', '')
            if isinstance(compressed_prompt, list):
                compressed_text = ' '.join(compressed_prompt)
            else:
                compressed_text = str(compressed_prompt) if compressed_prompt else ''
            
            return {
                'compressed_text': compressed_text,
                'original_tokens': results.get('origin_tokens', 0),
                'compressed_tokens': results.get('compressed_tokens', 0),
                'compression_ratio': results.get('rate', 0.0),
                'annotations': annotations,
            }
        except Exception as e:
            raise RuntimeError(f"LLMLingua提取失败: {e}") from e
    
    def extract_facts_batch(
        self,
        texts: List[str],
        return_annotations: bool = False,
    ) -> List[Dict[str, any]]:
        """批量提取事实"""
        return [self.extract_facts(text, return_annotations) for text in texts]

