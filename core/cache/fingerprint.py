"""
场景指纹定义

定义场景指纹的数据结构，用于唯一标识场景配置。
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional, List, Set, Any
from datetime import datetime

# 常量定义
HASH_TRUNCATE_LENGTH = 16  # 哈希截断长度（字符数）
PARTIAL_HASH_LENGTH = 8    # 部分哈希长度（用于复合键）


@dataclass(frozen=True)
class ComponentHash:
    """场景组成部分的哈希"""
    hash_value: str          # SHA256哈希值(前16位)
    component_type: str      # 类型: satellites/ground_stations/targets/time_range
    item_count: int          # 组件数量
    item_ids: Set[str]       # 组件ID集合

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'hash': self.hash_value,
            'type': self.component_type,
            'count': self.item_count,
            'ids': sorted(list(self.item_ids))
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ComponentHash':
        """从字典创建"""
        return cls(
            hash_value=data['hash'],
            component_type=data['type'],
            item_count=data['count'],
            item_ids=set(data['ids'])
        )


@dataclass(frozen=True)
class ScenarioFingerprint:
    """完整场景指纹"""
    # 完整哈希
    full_hash: str

    # 分层哈希
    satellites: ComponentHash
    ground_stations: ComponentHash
    targets: ComponentHash
    time_range: ComponentHash

    # 元数据
    scenario_name: str
    created_at: datetime

    def get_cache_key(self) -> str:
        """生成缓存键"""
        return self.full_hash[:HASH_TRUNCATE_LENGTH]

    def get_partial_key(self, components: List[str]) -> str:
        """生成部分组件的复合键"""
        hashes = []
        if 'satellites' in components:
            hashes.append(self.satellites.hash_value[:PARTIAL_HASH_LENGTH])
        if 'ground_stations' in components:
            hashes.append(self.ground_stations.hash_value[:PARTIAL_HASH_LENGTH])
        if 'targets' in components:
            hashes.append(self.targets.hash_value[:PARTIAL_HASH_LENGTH])
        if 'time_range' in components:
            hashes.append(self.time_range.hash_value[:PARTIAL_HASH_LENGTH])
        return "_".join(hashes) if hashes else self.full_hash[:HASH_TRUNCATE_LENGTH]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'full_hash': self.full_hash,
            'satellites': self.satellites.to_dict(),
            'ground_stations': self.ground_stations.to_dict(),
            'targets': self.targets.to_dict(),
            'time_range': self.time_range.to_dict(),
            'scenario_name': self.scenario_name,
            'created_at': self.created_at.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScenarioFingerprint':
        """从字典重建指纹"""
        return cls(
            full_hash=data['full_hash'],
            satellites=ComponentHash.from_dict(data['satellites']),
            ground_stations=ComponentHash.from_dict(data['ground_stations']),
            targets=ComponentHash.from_dict(data['targets']),
            time_range=ComponentHash.from_dict(data['time_range']),
            scenario_name=data['scenario_name'],
            created_at=datetime.fromisoformat(data['created_at'])
        )

    def get_component_hash(self, component_type: str) -> Optional[str]:
        """获取指定组件类型的哈希"""
        mapping = {
            'satellites': self.satellites.hash_value,
            'ground_stations': self.ground_stations.hash_value,
            'targets': self.targets.hash_value,
            'time_range': self.time_range.hash_value
        }
        return mapping.get(component_type)
