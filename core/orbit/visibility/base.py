"""
可见性计算基类

提供卫星-目标和卫星-地面站可见性计算的抽象接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from datetime import datetime, timedelta
import math

from core.constants import EARTH_RADIUS_M as CONST_EARTH_RADIUS_M
from core.quality.quality_config import QualityTier, QualityThresholds, DEFAULT_QUALITY_CONFIG


@dataclass(frozen=True, slots=True)
class VisibilityWindow:
    """
    可见时间窗口

    Attributes:
        satellite_id: 卫星ID
        target_id: 目标ID（或地面站ID，格式为"GS:{gs_id}"）
        start_time: 窗口开始时间
        end_time: 窗口结束时间
        max_elevation: 最大仰角（度）
        quality_score: 窗口质量评分（0-1），向后兼容，综合评分
        attitude_feasible: 姿态可行性（由Java预计算）
        attitude_samples: 姿态采样数据列表 [(timestamp, roll, pitch), ...]
        # 扩展质量评分字段（可选，用于详细分析）
        quality_details: 详细质量评分信息，包含各维度评分
        quality_tier: 质量等级，用于快速筛选
        detailed_scores: 各维度评分字典
    """
    satellite_id: str
    target_id: str
    start_time: datetime
    end_time: datetime
    max_elevation: float = 90.0
    quality_score: float = 1.0  # 向后兼容：综合评分
    attitude_feasible: bool = True
    attitude_samples: Optional[List[Tuple[float, float, float]]] = None

    # 扩展质量评分字段（可选，用于详细分析）
    quality_details: Optional[Dict[str, Any]] = None
    quality_tier: QualityTier = QualityTier.MEDIUM
    detailed_scores: Optional[Dict[str, float]] = None

    # ========== 区域目标拼幅相关字段 ==========
    # 瓦片相关（用于区域目标拼幅覆盖）
    tile_id: Optional[str] = None  # 瓦片ID，如 "AREA-001-T001"
    is_mosaic_window: bool = False  # 是否为拼幅窗口
    area_coverage_fraction: float = 1.0  # 足迹对瓦片的覆盖比例（点目标为1.0）

    # 预测的足迹信息（用于区域覆盖）
    footprint_at_start: Optional[List[Tuple[float, float]]] = None  # 窗口开始时的足迹角点
    footprint_at_end: Optional[List[Tuple[float, float]]] = None  # 窗口结束时的足迹角点
    footprint_center_at_start: Optional[Tuple[float, float]] = None  # 开始时的足迹中心
    footprint_center_at_end: Optional[Tuple[float, float]] = None  # 结束时的足迹中心

    def duration(self) -> float:
        """窗口持续时间（秒）"""
        return (self.end_time - self.start_time).total_seconds()

    def __lt__(self, other):
        """用于排序"""
        return self.start_time < other.start_time

    def get_effective_quality(self) -> float:
        """获取有效质量评分（向后兼容）"""
        return self.quality_score

    def is_high_quality(self, config: Optional[Any] = None) -> bool:
        """
        是否为高质量窗口

        Args:
            config: 质量评分配置，默认使用DEFAULT_QUALITY_CONFIG
        """
        cfg = config or DEFAULT_QUALITY_CONFIG
        return self.quality_score >= cfg.thresholds.high_quality

    def is_acceptable_quality(self, config: Optional[Any] = None) -> bool:
        """
        是否为可接受质量

        Args:
            config: 质量评分配置，默认使用DEFAULT_QUALITY_CONFIG
        """
        cfg = config or DEFAULT_QUALITY_CONFIG
        return self.quality_score >= cfg.thresholds.low_quality

    def get_quality_tier(self) -> QualityTier:
        """获取质量等级"""
        return self.quality_tier

    def update_quality_scores(self, overall: float, details: Dict[str, float],
                              tier: QualityTier = QualityTier.MEDIUM) -> 'VisibilityWindow':
        """
        创建更新质量评分后的新窗口对象

        由于dataclass是frozen的，需要创建新对象

        Args:
            overall: 综合质量评分
            details: 详细评分字典
            tier: 质量等级 (QualityTier enum)

        Returns:
            更新后的VisibilityWindow
        """
        from dataclasses import replace
        return replace(
            self,
            quality_score=overall,
            detailed_scores=details,
            quality_tier=tier,
            quality_details={
                'dimensions': details,
                'tier': tier.value,
            }
        )


class VisibilityCalculator(ABC):
    """
    可见性计算器基类

    抽象基类，定义可见性计算的接口
    """

    def __init__(self, min_elevation: float = 5.0):
        """
        初始化

        Args:
            min_elevation: 最小仰角（度）
        """
        self.min_elevation = min_elevation
        self.EARTH_RADIUS = CONST_EARTH_RADIUS_M  # 米

    @abstractmethod
    def compute_satellite_target_windows(
        self,
        satellite,
        target,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta = timedelta(seconds=60)
    ) -> List[VisibilityWindow]:
        """
        计算卫星-目标可见窗口

        Args:
            satellite: 卫星模型
            target: 目标模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        pass

    @abstractmethod
    def compute_satellite_ground_station_windows(
        self,
        satellite,
        ground_station,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta = timedelta(seconds=60)
    ) -> List[VisibilityWindow]:
        """
        计算卫星-地面站可见窗口

        Args:
            satellite: 卫星模型
            ground_station: 地面站模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        pass

    def _calculate_elevation(
        self,
        sat_pos: Tuple[float, float, float],
        ground_pos: Tuple[float, float, float]
    ) -> float:
        """
        计算仰角

        Args:
            sat_pos: 卫星位置 (x, y, z) in meters
            ground_pos: 地面位置 (x, y, z) in meters

        Returns:
            仰角（度），如果卫星在地球背面返回负值
        """
        # 计算地面站到卫星的向量
        dx = sat_pos[0] - ground_pos[0]
        dy = sat_pos[1] - ground_pos[1]
        dz = sat_pos[2] - ground_pos[2]

        # 地面站位置向量
        gx, gy, gz = ground_pos

        # 地面站到天顶的单位向量（从地心指向外的方向）
        g_norm = math.sqrt(gx**2 + gy**2 + gz**2)
        up_x = gx / g_norm
        up_y = gy / g_norm
        up_z = gz / g_norm

        # 归一化卫星方向向量
        d_norm = math.sqrt(dx**2 + dy**2 + dz**2)
        dx_norm = dx / d_norm
        dy_norm = dy / d_norm
        dz_norm = dz / d_norm

        # 计算与天顶方向的夹角余弦
        cos_zenith = dx_norm * up_x + dy_norm * up_y + dz_norm * up_z

        # 仰角 = 90° - 天顶角
        elevation = math.degrees(math.asin(max(-1, min(1, cos_zenith))))

        # 地球遮挡检查：只有仰角大于0时才需要检查
        if elevation > 0:
            # 检查视线是否穿过地球
            # 视线参数方程: P = G + t * D, t >= 0
            # 与地球相交: |P|^2 = R^2
            # 解二次方程求交点
            G_dot_G = gx**2 + gy**2 + gz**2
            G_dot_D = gx*dx + gy*dy + gz*dz
            D_dot_D = dx**2 + dy**2 + dz**2

            a = D_dot_D
            b = 2 * G_dot_D
            c = G_dot_G - self.EARTH_RADIUS**2

            discriminant = b*b - 4*a*c

            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2*a)
                t2 = (-b + sqrt_disc) / (2*a)

                # 如果存在 t > 0.001 的解（除地面站本身外），视线穿过地球
                if t1 > 0.001 or t2 > 0.001:
                    elevation = -90.0

        return elevation

    def _find_windows_from_elevations(
        self,
        satellite_id: str,
        target_id: str,
        elevation_data: List[Tuple[datetime, float]],
        min_elevation: float
    ) -> List[VisibilityWindow]:
        """
        从仰角数据中提取可见窗口

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID
            elevation_data: [(timestamp, elevation), ...]
            min_elevation: 最小仰角

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        windows = []
        in_window = False
        window_start = None
        max_elev_in_window = 0.0

        for timestamp, elevation in elevation_data:
            if elevation >= min_elevation:
                if not in_window:
                    # 窗口开始
                    in_window = True
                    window_start = timestamp
                    max_elev_in_window = elevation
                else:
                    # 窗口内，更新最大仰角
                    max_elev_in_window = max(max_elev_in_window, elevation)
            else:
                if in_window:
                    # 窗口结束
                    in_window = False
                    # 计算质量评分（基于最大仰角）
                    quality = min(1.0, max_elev_in_window / 90.0)

                    windows.append(VisibilityWindow(
                        satellite_id=satellite_id,
                        target_id=target_id,
                        start_time=window_start,
                        end_time=timestamp,
                        max_elevation=max_elev_in_window,
                        quality_score=quality
                    ))
                    max_elev_in_window = 0.0

        # 处理最后一个窗口（如果还在窗口内）
        if in_window and elevation_data:
            quality = min(1.0, max_elev_in_window / 90.0)
            windows.append(VisibilityWindow(
                satellite_id=satellite_id,
                target_id=target_id,
                start_time=window_start,
                end_time=elevation_data[-1][0],
                max_elevation=max_elev_in_window,
                quality_score=quality
            ))

        return windows
