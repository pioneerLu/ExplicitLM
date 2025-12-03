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
    print("⚠️  llmlingua 未安装，事实提取功能将不可用")


class FactExtractor:
    """使用 LLMLingua 提取浓缩事实"""
    
    def __init__(
        self,
        model_path: str = "/data2/zengzheni/lvchangwei/new_repo/bert/llmlingua-2-bert",
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
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化 LLMLingua
        self._init_compressor()
    
    def _init_compressor(self):
        """初始化 PromptCompressor"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"LLMLingua 模型路径不存在: {self.model_path}\n"
                f"请运行 bert/get_model.py 下载模型"
            )
        
        print(f"🔤 加载 LLMLingua 模型: {self.model_path}")
        self.compressor = PromptCompressor(
            model_name=self.model_path,
            use_llmlingua2=True
        )
        print("✅ LLMLingua 模型加载完成")
    
    def extract_facts(
        self,
        text: str,
        return_annotations: bool = False,
    ) -> Dict[str, any]:
        """
        从文本中提取浓缩事实
        
        Args:
            text: 输入文本
            return_annotations: 是否返回标注信息（哪些token被保留/删除）
        
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
        
        try:
            results = self.compressor.compress_prompt_llmlingua2(
                text,
                rate=self.compression_rate,
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
            
            return {
                'compressed_text': results.get('compressed_prompt', ''),
                'original_tokens': results.get('origin_tokens', 0),
                'compressed_tokens': results.get('compressed_tokens', 0),
                'compression_ratio': results.get('rate', 0.0),
                'annotations': annotations,
            }
        except Exception as e:
            print(f"⚠️  事实提取失败: {e}")
            # 失败时返回原始文本
            return {
                'compressed_text': text,
                'original_tokens': len(text.split()),
                'compressed_tokens': len(text.split()),
                'compression_ratio': 1.0,
                'annotations': [],
            }
    
    def extract_facts_batch(
        self,
        texts: List[str],
        return_annotations: bool = False,
    ) -> List[Dict[str, any]]:
        """批量提取事实"""
        return [self.extract_facts(text, return_annotations) for text in texts]

