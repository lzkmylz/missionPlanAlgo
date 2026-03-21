"""
目标模型 - 定义观测目标的属性和约束

支持两种目标类型：
- 点目标：特定地理位置的点
- 区域目标：多边形区域，需要分解为条带或网格
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import math

from core.constants import EARTH_RADIUS_M


class TargetType(Enum):
    """目标类型枚举"""
    POINT = "point"
    AREA = "area"


@dataclass
class GeoPoint:
    """
    地理坐标点

    Attributes:
        longitude: 经度（度）
        latitude: 纬度（度）
        altitude: 海拔高度（米）
    """
    longitude: float
    latitude: float
    altitude: float = 0.0

    def to_ecef(self) -> Tuple[float, float, float]:
        """转换为地心固定坐标（ECEF，米）"""
        lon_rad = math.radians(self.longitude)
        lat_rad = math.radians(self.latitude)
        r = EARTH_RADIUS_M + self.altitude
        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)
        return (x, y, z)


@dataclass
class Target:
    """
    观测目标模型

    Attributes:
        id: 目标唯一标识
        name: 目标名称
        target_type: 目标类型（点/区域）
        longitude: 经度（点目标）
        latitude: 纬度（点目标）
        area_vertices: 区域顶点列表（区域目标）
        priority: 优先级（1-100，数字越小优先级越高）
        required_observations: 需要观测次数
        time_window_start: 时间窗口开始
        time_window_end: 时间窗口结束
        resolution_required: 所需分辨率（米）
        immediate_downlink: 是否需要立即回传
    """
    id: str
    name: str = ""
    target_type: TargetType = TargetType.POINT

    # 点目标参数
    longitude: Optional[float] = None
    latitude: Optional[float] = None

    # 区域目标参数
    area_vertices: List[Tuple[float, float]] = field(default_factory=list)  # [(lon, lat), ...]

    # 观测要求
    priority: int = 1
    required_observations: int = 1
    resolution_required: float = 10.0  # 米

    # 时间约束
    time_window_start: Optional[datetime] = None
    time_window_end: Optional[datetime] = None

    # 数据回传要求
    immediate_downlink: bool = False

    # 状态（运行时被更新）
    completed_observations: int = 0

    # ========== 成像模式与卫星类型要求 ==========
    # 旧字段（单值，向后兼容保留）
    # 指定成像模式（如 'push_broom', 'forward_pushbroom_pmc', 'spotlight' 等）
    required_imaging_mode: Optional[str] = None

    # 指定卫星类型（如 'optical', 'sar'）
    required_satellite_type: Optional[str] = None

    # 精准需求字段（新增，列表形式，全部可选，空列表表示不限制）
    # 允许的卫星大类，如 ['sar'] 或 ['optical']
    allowed_satellite_types: List[str] = field(default_factory=list)

    # 允许的具体卫星 ID 列表，如 ['SAR-01', 'SAR-02']
    allowed_satellite_ids: List[str] = field(default_factory=list)

    # 允许的成像模式列表（满足其一即可），如 ['spotlight', 'sliding_spotlight']
    required_imaging_modes: List[str] = field(default_factory=list)

    # PMC模式优先级（如果支持PMC模式）
    # 0 = 不使用PMC, 1 = 低优先级, 2 = 中优先级, 3 = 高优先级
    pmc_priority: int = 0

    # 期望降速比范围（仅PMC模式有效）[min, max]
    pmc_speed_reduction_range: Optional[Tuple[float, float]] = None

    # ========== 拼幅覆盖相关字段 ==========
    # 是否启用拼幅覆盖（仅区域目标有效）
    mosaic_required: bool = False

    # 拼幅策略: "strip"条带 / "grid"网格 / "auto"自动
    mosaic_strategy: str = "auto"

    # 最小覆盖率要求 (0-1)，默认95%
    min_coverage_ratio: float = 0.95

    # 最大允许重叠比例 (0-1)，默认15%（10-20%范围）
    max_overlap_ratio: float = 0.15

    # 瓦片优先级模式: "uniform"统一 / "center_first"中心优先 /
    #                 "edge_first"边缘优先 / "corner_first"角落优先
    tile_priority_mode: str = "center_first"

    # 是否启用动态瓦片大小
    dynamic_tile_sizing: bool = True

    # 覆盖策略: "max_coverage"最大覆盖 / "max_profit"最高收益
    coverage_strategy: str = "max_coverage"

    # 区域目标参考分辨率（米），用于动态瓦片计算
    target_resolution_m: float = 10.0

    # ========== 单次多条带拼幅成像（Single-Pass Mosaic）相关字段 ==========
    # 期望条带数量（None表示使用卫星默认配置）
    mosaic_strip_count: Optional[int] = None

    # 期望总幅宽（米，None表示使用条带数量×条带幅宽计算）
    mosaic_total_width_m: Optional[float] = None

    # 拼幅中心滚转偏置（度，0.0表示星下点对称）
    mosaic_center_roll_deg: float = 0.0

    # 条带间重叠比例要求（覆盖单次多条带拼幅模式的全局overlap_ratio）
    required_swath_overlap_ratio: float = 0.10

    def __post_init__(self):
        """验证数据一致性"""
        if self.target_type == TargetType.POINT:
            if self.longitude is None or self.latitude is None:
                raise ValueError("Point target must have longitude and latitude")
        elif self.target_type == TargetType.AREA:
            if len(self.area_vertices) < 3:
                raise ValueError("Area target must have at least 3 vertices")

        # 验证比率字段范围
        if not 0 <= self.min_coverage_ratio <= 1:
            raise ValueError(f"min_coverage_ratio must be between 0 and 1, got {self.min_coverage_ratio}")
        if not 0 <= self.max_overlap_ratio <= 1:
            raise ValueError(f"max_overlap_ratio must be between 0 and 1, got {self.max_overlap_ratio}")

    def get_center(self) -> Tuple[float, float]:
        """获取目标中心坐标"""
        if self.target_type == TargetType.POINT:
            return (self.longitude, self.latitude)
        else:
            # 计算多边形中心
            lons = [v[0] for v in self.area_vertices]
            lats = [v[1] for v in self.area_vertices]
            if not lons:
                return (0.0, 0.0)
            return (sum(lons) / len(lons), sum(lats) / len(lats))

    def get_area(self) -> float:
        """获取区域面积（平方公里）"""
        if self.target_type == TargetType.POINT:
            return 0.0

        # 使用鞋带公式计算多边形面积
        n = len(self.area_vertices)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += self.area_vertices[i][0] * self.area_vertices[j][1]
            area -= self.area_vertices[j][0] * self.area_vertices[i][1]

        # 转换为平方公里（简化，假设在小范围内）
        km_per_deg = 111.0
        return abs(area) * 0.5 * km_per_deg * km_per_deg

    def is_time_valid(self, dt: datetime) -> bool:
        """检查时间是否在允许窗口内"""
        if self.time_window_start and dt < self.time_window_start:
            return False
        if self.time_window_end and dt > self.time_window_end:
            return False
        return True

    def get_ecef_position(self) -> Tuple[float, float, float]:
        """
        获取地心固定坐标系(ECEF)中的位置

        Returns:
            (x, y, z) in meters
        """
        if self.target_type != TargetType.POINT:
            raise ValueError("Only point targets have a single ECEF position")

        lon_rad = math.radians(self.longitude)
        lat_rad = math.radians(self.latitude)

        x = EARTH_RADIUS_M * math.cos(lat_rad) * math.cos(lon_rad)
        y = EARTH_RADIUS_M * math.cos(lat_rad) * math.sin(lon_rad)
        z = EARTH_RADIUS_M * math.sin(lat_rad)

        return (x, y, z)

    def requires_mosaic(self) -> bool:
        """检查是否需要单次多条带拼幅成像模式。

        mosaic_strip_count 只在未显式指定其他模式时才触发 mosaic。
        若 required_imaging_mode 明确指定了非 mosaic 模式（如 'push_broom'），
        则以 required_imaging_mode 为准，忽略 mosaic_strip_count。
        """
        if self.required_imaging_mode == 'single_pass_mosaic':
            return True
        if 'single_pass_mosaic' in self.required_imaging_modes:
            return True
        # 仅当 required_imaging_mode 未指定其他模式时，才以 mosaic_strip_count 作为触发条件
        if (self.mosaic_strip_count is not None
                and self.mosaic_strip_count > 1
                and self.required_imaging_mode is None):
            return True
        return False

    def requires_pmc(self) -> bool:
        """检查是否需要PMC模式（前向或反向）"""
        pmc_modes = {'forward_pushbroom_pmc', 'reverse_pushbroom_pmc'}
        return (
            self.pmc_priority > 0
            or self.required_imaging_mode in pmc_modes
            or bool(set(self.required_imaging_modes) & pmc_modes)
        )

    def requires_forward_pmc(self) -> bool:
        """检查是否需要前向PMC模式"""
        return (self.required_imaging_mode == 'forward_pushbroom_pmc'
                or 'forward_pushbroom_pmc' in self.required_imaging_modes)

    def requires_reverse_pmc(self) -> bool:
        """检查是否需要反向PMC模式"""
        return (self.required_imaging_mode == 'reverse_pushbroom_pmc'
                or 'reverse_pushbroom_pmc' in self.required_imaging_modes)

    def get_preferred_speed_reduction(self) -> Optional[float]:
        """获取首选速度变化比（前向为降速比，反向为增速比）"""
        if self.pmc_speed_reduction_range:
            # 返回范围中值
            return (self.pmc_speed_reduction_range[0] + self.pmc_speed_reduction_range[1]) / 2
        return None

    def get_pmc_direction(self) -> Optional[str]:
        """获取PMC方向（'forward' 或 'reverse'）。

        若 ``required_imaging_modes`` 同时包含 forward 和 reverse，forward 优先。
        """
        if (self.required_imaging_mode == 'forward_pushbroom_pmc'
                or 'forward_pushbroom_pmc' in self.required_imaging_modes):
            return 'forward'
        if (self.required_imaging_mode == 'reverse_pushbroom_pmc'
                or 'reverse_pushbroom_pmc' in self.required_imaging_modes):
            return 'reverse'
        return None

    def check_satellite_compatibility(self, payload_type: str, mode_name: str,
                                       sat_id: Optional[str] = None) -> bool:
        """
        检查卫星是否与目标要求兼容

        Args:
            payload_type: 载荷类型（'optical' 或 'sar'）
            mode_name: 成像模式名称
            sat_id: 卫星 ID（用于精准 ID 约束检查）

        Returns:
            是否兼容
        """
        # 旧字段：单值卫星类型约束
        if self.required_satellite_type:
            if self.required_satellite_type.lower() != payload_type.lower():
                return False

        # 精准需求：允许的卫星 ID 列表
        # sat_id=None 视为不满足 ID 约束（不能因参数缺失而跳过过滤）
        if self.allowed_satellite_ids:
            if sat_id is None or sat_id not in self.allowed_satellite_ids:
                return False

        # 精准需求：允许的卫星类型列表
        if self.allowed_satellite_types:
            if payload_type.lower() not in {t.lower() for t in self.allowed_satellite_types}:
                return False

        # 成像模式约束：将旧字段（单值）与新字段（列表）合并为并集，满足其一即可
        # 与 _check_precise_requirements 语义保持一致（OR/并集）
        effective_modes = {m.lower() for m in self.required_imaging_modes}
        if self.required_imaging_mode:
            effective_modes.add(self.required_imaging_mode.lower())
        if effective_modes:
            if mode_name.lower() not in effective_modes:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'target_type': self.target_type.value,
            'longitude': self.longitude,
            'latitude': self.latitude,
            'area_vertices': self.area_vertices,
            'priority': self.priority,
            'required_observations': self.required_observations,
            'resolution_required': self.resolution_required,
            'time_window_start': self.time_window_start.isoformat() if self.time_window_start else None,
            'time_window_end': self.time_window_end.isoformat() if self.time_window_end else None,
            'immediate_downlink': self.immediate_downlink,
            'required_imaging_mode': self.required_imaging_mode,
            'required_satellite_type': self.required_satellite_type,
            'allowed_satellite_types': self.allowed_satellite_types,
            'allowed_satellite_ids': self.allowed_satellite_ids,
            'required_imaging_modes': self.required_imaging_modes,
            'pmc_priority': self.pmc_priority,
            'pmc_speed_reduction_range': self.pmc_speed_reduction_range,
            # 单次多条带拼幅字段
            'mosaic_strip_count': self.mosaic_strip_count,
            'mosaic_total_width_m': self.mosaic_total_width_m,
            'mosaic_center_roll_deg': self.mosaic_center_roll_deg,
            'required_swath_overlap_ratio': self.required_swath_overlap_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Target':
        """从字典创建，支持多种格式"""
        # 处理 target_type（默认为 POINT）
        target_type_str = data.get('target_type', 'point')
        target_type = TargetType(target_type_str)

        # 支持 location 数组格式 [longitude, latitude]
        location = data.get('location')
        if location and len(location) >= 2:
            longitude = location[0]
            latitude = location[1]
        else:
            longitude = data.get('longitude')
            latitude = data.get('latitude')

        # 解析PMC速度范围
        pmc_range = data.get('pmc_speed_reduction_range')
        if pmc_range and len(pmc_range) == 2:
            pmc_speed_reduction_range = tuple(pmc_range)
        else:
            pmc_speed_reduction_range = None

        return cls(
            id=data['id'],
            name=data.get('name', ''),
            target_type=target_type,
            longitude=longitude,
            latitude=latitude,
            area_vertices=data.get('area_vertices', []),
            priority=data.get('priority', 1),
            required_observations=data.get('required_observations', 1),
            resolution_required=data.get('resolution_required', 10.0),
            time_window_start=datetime.fromisoformat(data['time_window_start']) if data.get('time_window_start') else None,
            time_window_end=datetime.fromisoformat(data['time_window_end']) if data.get('time_window_end') else None,
            immediate_downlink=data.get('immediate_downlink', False),
            required_imaging_mode=data.get('required_imaging_mode'),
            required_satellite_type=data.get('required_satellite_type'),
            allowed_satellite_types=data.get('allowed_satellite_types', []),
            allowed_satellite_ids=data.get('allowed_satellite_ids', []),
            required_imaging_modes=data.get('required_imaging_modes', []),
            pmc_priority=data.get('pmc_priority', 0),
            pmc_speed_reduction_range=pmc_speed_reduction_range,
            # 单次多条带拼幅字段
            mosaic_strip_count=data.get('mosaic_strip_count'),
            mosaic_total_width_m=data.get('mosaic_total_width_m'),
            mosaic_center_roll_deg=float(data.get('mosaic_center_roll_deg', 0.0)),
            required_swath_overlap_ratio=float(data.get('required_swath_overlap_ratio', 0.10)),
        )
