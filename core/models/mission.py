"""
任务模型 - 定义完整的任务场景

包含卫星、目标、地面站的完整配置
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

from .satellite import Satellite
from .target import Target
from .ground_station import GroundStation


@dataclass
class Mission:
    """
    任务场景模型

    Attributes:
        name: 场景名称
        start_time: 规划周期开始
        end_time: 规划周期结束
        satellites: 卫星列表
        targets: 目标列表
        ground_stations: 地面站列表
        description: 场景描述
    """
    name: str
    start_time: datetime
    end_time: datetime
    satellites: List[Satellite] = field(default_factory=list)
    targets: List[Target] = field(default_factory=list)
    ground_stations: List[GroundStation] = field(default_factory=list)
    description: str = ""

    def __post_init__(self):
        """验证时间范围"""
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")

    def get_duration(self) -> timedelta:
        """获取规划周期时长"""
        return self.end_time - self.start_time

    def add_satellite(self, satellite: Satellite) -> None:
        """添加卫星"""
        self.satellites.append(satellite)

    def add_target(self, target: Target) -> None:
        """添加目标"""
        self.targets.append(target)

    def add_ground_station(self, ground_station: GroundStation) -> None:
        """添加地面站"""
        self.ground_stations.append(ground_station)

    def get_satellite_by_id(self, sat_id: str) -> Optional[Satellite]:
        """根据ID获取卫星"""
        for sat in self.satellites:
            if sat.id == sat_id:
                return sat
        return None

    def get_target_by_id(self, target_id: str) -> Optional[Target]:
        """根据ID获取目标"""
        for target in self.targets:
            if target.id == target_id:
                return target
        return None

    def get_ground_station_by_id(self, gs_id: str) -> Optional[GroundStation]:
        """根据ID获取地面站"""
        for gs in self.ground_stations:
            if gs.id == gs_id:
                return gs
        return None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'satellites': [sat.to_dict() for sat in self.satellites],
            'targets': [target.to_dict() for target in self.targets],
            'ground_stations': [gs.to_dict() for gs in self.ground_stations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Mission':
        """从字典创建"""
        satellites = [Satellite.from_dict(sat_data) for sat_data in data.get('satellites', [])]
        targets = [Target.from_dict(target_data) for target_data in data.get('targets', [])]
        ground_stations = [GroundStation.from_dict(gs_data) for gs_data in data.get('ground_stations', [])]

        return cls(
            name=data['name'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            satellites=satellites,
            targets=targets,
            ground_stations=ground_stations,
            description=data.get('description', ''),
        )

    def save(self, filepath: str) -> None:
        """保存到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'Mission':
        """从JSON文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def summary(self) -> Dict[str, Any]:
        """获取场景摘要"""
        return {
            'name': self.name,
            'description': self.description,
            'duration_hours': self.get_duration().total_seconds() / 3600,
            'satellite_count': len(self.satellites),
            'target_count': len(self.targets),
            'ground_station_count': len(self.ground_stations),
            'satellite_types': {sat.sat_type.value for sat in self.satellites},
            'target_types': {target.target_type.value for target in self.targets},
        }
