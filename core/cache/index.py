"""
缓存索引定义

定义缓存索引的数据结构和状态枚举。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class CacheStatus(Enum):
    """缓存状态"""
    VALID = "valid"           # 有效可用
    STALE = "stale"           # 过期但保留
    CORRUPTED = "corrupted"   # 已损坏
    DELETED = "deleted"       # 已标记删除


@dataclass
class CacheIndexEntry:
    """缓存索引条目"""
    # 指纹关联
    full_hash: str

    # 文件路径
    cache_file: str           # 可见性窗口缓存路径
    orbit_file: Optional[str]  # 轨道数据缓存路径

    # 组件哈希（用于部分匹配）
    satellites_hash: str
    ground_stations_hash: str
    targets_hash: str
    time_range_hash: str

    # 元数据
    scenario_name: str
    created_at: datetime
    accessed_at: datetime
    access_count: int
    file_size_mb: float

    # 状态
    status: CacheStatus = field(default=CacheStatus.VALID)

    # 统计信息
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'full_hash': self.full_hash,
            'cache_file': self.cache_file,
            'orbit_file': self.orbit_file,
            'component_hashes': {
                'satellites': self.satellites_hash,
                'ground_stations': self.ground_stations_hash,
                'targets': self.targets_hash,
                'time_range': self.time_range_hash
            },
            'scenario_name': self.scenario_name,
            'created_at': self.created_at.isoformat(),
            'accessed_at': self.accessed_at.isoformat(),
            'access_count': self.access_count,
            'file_size_mb': self.file_size_mb,
            'status': self.status.value,
            'stats': self.stats
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheIndexEntry':
        """从字典创建"""
        component_hashes = data.get('component_hashes', data)
        return cls(
            full_hash=data['full_hash'],
            cache_file=data['cache_file'],
            orbit_file=data.get('orbit_file'),
            satellites_hash=component_hashes['satellites'],
            ground_stations_hash=component_hashes['ground_stations'],
            targets_hash=component_hashes['targets'],
            time_range_hash=component_hashes['time_range'],
            scenario_name=data['scenario_name'],
            created_at=datetime.fromisoformat(data['created_at']),
            accessed_at=datetime.fromisoformat(data['accessed_at']),
            access_count=data['access_count'],
            file_size_mb=data['file_size_mb'],
            status=CacheStatus(data.get('status', 'valid')),
            stats=data.get('stats', {})
        )

    def touch(self) -> None:
        """更新访问时间"""
        self.accessed_at = datetime.now()
        self.access_count += 1

    def is_valid(self) -> bool:
        """检查条目是否有效"""
        return self.status == CacheStatus.VALID
