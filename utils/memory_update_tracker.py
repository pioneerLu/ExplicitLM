"""
Memory Update Tracker: 追踪 memory_bank 的更新情况

用于判断何时需要重新聚类 keys（基于10%更新阈值）。
"""
from typing import Set, List, Dict, Optional


class MemoryUpdateTracker:
    """
    追踪 memory_bank 的更新情况
    
    使用选项B：基于实际变化的条目数（去重），更准确反映 memory bank 的变化。
    """
    
    def __init__(
        self,
        total_valid_entries: int,
        update_ratio_threshold: float = 0.1,
    ):
        """
        初始化更新追踪器
        
        Args:
            total_valid_entries: 总有效条目数
            update_ratio_threshold: 更新比例阈值（0-1），超过此阈值需要重新聚类
        """
        self.total_valid_entries = total_valid_entries
        self.update_ratio_threshold = update_ratio_threshold
        self.updated_indices_set: Set[int] = set()  # 记录被更新的条目索引（去重）
    
    def record_update(self, update_result: Dict) -> None:
        """
        记录一次更新
        
        Args:
            update_result: MemoryBankUpdater.update_with_facts() 或 update_from_text() 的返回结果
                应包含 "update_indices" 字段（List[int]）
        """
        update_indices = update_result.get("update_indices", [])
        if isinstance(update_indices, list):
            self.updated_indices_set.update(update_indices)
        else:
            # 兼容旧格式（只返回前10个的情况）
            if isinstance(update_indices, (tuple, set)):
                self.updated_indices_set.update(update_indices)
    
    def should_recluster(self) -> bool:
        """
        判断是否需要重新聚类
        
        Returns:
            True 如果更新比例超过阈值
        """
        if len(self.updated_indices_set) == 0:
            return False
        
        if self.total_valid_entries == 0:
            return False
        
        update_ratio = len(self.updated_indices_set) / self.total_valid_entries
        return update_ratio >= self.update_ratio_threshold
    
    def get_update_ratio(self) -> float:
        """
        获取当前更新比例
        
        Returns:
            更新比例（0-1）
        """
        if self.total_valid_entries == 0:
            return 0.0
        return len(self.updated_indices_set) / self.total_valid_entries
    
    def get_updated_count(self) -> int:
        """
        获取已更新的条目数（去重后）
        
        Returns:
            已更新的条目数
        """
        return len(self.updated_indices_set)
    
    def reset(self) -> None:
        """
        重置计数器（聚类后调用）
        """
        self.updated_indices_set.clear()
    
    def get_stats(self) -> Dict[str, any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_valid_entries": self.total_valid_entries,
            "updated_count": len(self.updated_indices_set),
            "update_ratio": self.get_update_ratio(),
            "update_ratio_threshold": self.update_ratio_threshold,
            "should_recluster": self.should_recluster(),
        }

