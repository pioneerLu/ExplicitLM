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
        
        # 在初始化 PromptCompressor 之前，设置 torch 的默认设备
        # 注意：CUDA_VISIBLE_DEVICES 已在进程启动时设置，这里不再修改
        # 如果设置了 CUDA_VISIBLE_DEVICES，在子进程中只有 cuda:0 可见
        if self.device != "cpu":
            try:
                import torch
                if torch.cuda.is_available():
                    # 由于 CUDA_VISIBLE_DEVICES 已设置，只有 cuda:0 可见，所以总是使用 0
                    torch.cuda.set_device(0)
            except Exception:
                pass
        
        # 初始化PromptCompressor，如果支持device参数则传入
        compressor_kwargs = {
            "model_name": self.model_path,
            "use_llmlingua2": True
        }
        # 尝试传入device参数（如果PromptCompressor支持）
        try:
            import inspect
            sig = inspect.signature(PromptCompressor.__init__)
            if 'device' in sig.parameters:
                compressor_kwargs['device'] = self.device
        except Exception:
            pass  # 如果不支持device参数，忽略
        
        self.compressor = PromptCompressor(**compressor_kwargs)
        
        # 如果PromptCompressor不支持device参数，手动设置模型到指定设备
        if self.device != "cpu" and hasattr(self.compressor, 'model'):
            try:
                import torch
                # 强制将模型移动到指定设备
                self.compressor.model = self.compressor.model.to(self.device)
                # 确保模型的所有参数都在指定设备上
                if hasattr(self.compressor.model, 'to'):
                    self.compressor.model = self.compressor.model.to(self.device)
                # 验证模型是否在正确的设备上
                if hasattr(self.compressor.model, 'parameters'):
                    for param in self.compressor.model.parameters():
                        if param.device.type != self.device.split(':')[0] or (self.device.startswith('cuda:') and param.device.index != int(self.device.split(':')[1])):
                            param.data = param.data.to(self.device)
            except Exception as e:
                # 如果无法移动模型，继续使用原设备（不打印警告，避免多进程输出混乱）
                pass
        
        # 如果 compressor 有 tokenizer，也移动到指定设备
        if self.device != "cpu" and hasattr(self.compressor, 'tokenizer'):
            try:
                # tokenizer 通常不需要移动，但检查一下
                pass
            except Exception:
                pass
        
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
        
        # 在调用压缩之前，确保模型在正确的设备上
        if self.device != "cpu" and torch.cuda.is_available():
            try:
                # 确保当前线程使用正确的设备
                if self.device.startswith("cuda:"):
                    device_idx = int(self.device.split(":")[1])
                    if device_idx < torch.cuda.device_count():
                        torch.cuda.set_device(device_idx)
            except Exception:
                pass
        
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
            
            # 在调用压缩之前，确保模型在正确的设备上
            # 注意：CUDA_VISIBLE_DEVICES 已在进程启动时设置，这里只确保模型在 cuda:0 上
            if self.device != "cpu" and hasattr(self.compressor, 'model'):
                try:
                    import torch
                    if torch.cuda.is_available():
                        # 由于 CUDA_VISIBLE_DEVICES 已设置，只有 cuda:0 可见
                        self.compressor.model = self.compressor.model.to("cuda:0")
                except Exception:
                    pass
            
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

