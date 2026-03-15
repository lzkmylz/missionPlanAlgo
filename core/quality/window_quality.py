"""
窗口质量评分计算器

实现多维度窗口质量评分计算，包括：
- 仰角评分
- 姿态约束满足度评分
- 窗口持续时间评分
- 光照条件评分（光学卫星）
- 地面站配合度评分
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from enum import Enum

from .quality_config import (
    QualityScoreConfig,
    QualityDimensionWeights,
    SatelliteType,
    QualityTier,
    DEFAULT_QUALITY_CONFIG,
)

if TYPE_CHECKING:
    from core.orbit.visibility.base import VisibilityWindow
    from core.models.satellite import Satellite
    from core.models.mission import Mission
    from core.models.target import Target

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """质量评分维度"""
    ELEVATION = "elevation"
    ATTITUDE = "attitude"
    DURATION = "duration"
    ILLUMINATION = "illumination"
    DOWNLINK = "downlink"


@dataclass
class WindowQualityScore:
    """
    窗口质量评分结果

    Attributes:
        overall_score: 综合质量评分（0-1）
        elevation_score: 仰角评分（0-1）
        attitude_score: 姿态约束满足度评分（0-1）
        duration_score: 持续时间评分（0-1）
        illumination_score: 光照条件评分（0-1）
        downlink_score: 地面站配合度评分（0-1）
        quality_tier: 质量等级 (QualityTier enum)
        details: 详细评分信息
    """
    overall_score: float
    elevation_score: float
    attitude_score: float
    duration_score: float
    illumination_score: float
    downlink_score: float
    quality_tier: QualityTier = field(default=QualityTier.MEDIUM)
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """确保评分在有效范围内"""
        self.overall_score = max(0.0, min(1.0, self.overall_score))
        self.elevation_score = max(0.0, min(1.0, self.elevation_score))
        self.attitude_score = max(0.0, min(1.0, self.attitude_score))
        self.duration_score = max(0.0, min(1.0, self.duration_score))
        self.illumination_score = max(0.0, min(1.0, self.illumination_score))
        self.downlink_score = max(0.0, min(1.0, self.downlink_score))

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'overall_score': round(self.overall_score, 4),
            'elevation_score': round(self.elevation_score, 4),
            'attitude_score': round(self.attitude_score, 4),
            'duration_score': round(self.duration_score, 4),
            'illumination_score': round(self.illumination_score, 4),
            'downlink_score': round(self.downlink_score, 4),
            'quality_tier': self.quality_tier.value,
            'details': self.details,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WindowQualityScore':
        """从字典创建"""
        tier_str = data.get('quality_tier', 'medium')
        # Handle both string and enum values
        if isinstance(tier_str, QualityTier):
            tier = tier_str
        else:
            tier = QualityTier(tier_str)

        return cls(
            overall_score=data.get('overall_score', 0.5),
            elevation_score=data.get('elevation_score', 0.5),
            attitude_score=data.get('attitude_score', 0.5),
            duration_score=data.get('duration_score', 0.5),
            illumination_score=data.get('illumination_score', 0.5),
            downlink_score=data.get('downlink_score', 0.5),
            quality_tier=tier,
            details=data.get('details', {}),
        )

    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """是否为高质量窗口"""
        return self.overall_score >= threshold

    def is_acceptable(self, threshold: float = 0.3) -> bool:
        """是否可接受"""
        return self.overall_score >= threshold

    def get_worst_dimension(self) -> Tuple[str, float]:
        """获取评分最低的维度"""
        dimensions = {
            'elevation': self.elevation_score,
            'attitude': self.attitude_score,
            'duration': self.duration_score,
            'illumination': self.illumination_score,
            'downlink': self.downlink_score,
        }
        worst = min(dimensions.items(), key=lambda x: x[1])
        return worst

    def get_best_dimension(self) -> Tuple[str, float]:
        """获取评分最高的维度"""
        dimensions = {
            'elevation': self.elevation_score,
            'attitude': self.attitude_score,
            'duration': self.duration_score,
            'illumination': self.illumination_score,
            'downlink': self.downlink_score,
        }
        best = max(dimensions.items(), key=lambda x: x[1])
        return best


class WindowQualityCalculator:
    """
    窗口质量评分计算器

    提供多维度窗口质量评分计算功能。
    支持缓存以提高性能。
    """

    # 默认成像时间（秒）
    DEFAULT_IMAGING_DURATION = 10.0

    # 最佳窗口持续时间（秒）
    OPTIMAL_WINDOW_DURATION = 120.0  # 2分钟

    # 最小可接受持续时间（秒）
    MIN_REQUIRED_DURATION = 15.0  # 15秒

    def __init__(self, config: Optional[QualityScoreConfig] = None):
        """
        初始化质量评分计算器

        Args:
            config: 质量评分配置，使用默认配置 if None
        """
        self.config = config or DEFAULT_QUALITY_CONFIG
        self._cache: Dict[str, WindowQualityScore] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_key(self, window: 'VisibilityWindow', satellite_id: str) -> str:
        """生成缓存键"""
        return f"{satellite_id}:{window.start_time.isoformat()}:{window.target_id}"

    def _get_from_cache(self, key: str) -> Optional[WindowQualityScore]:
        """从缓存获取"""
        if not self.config.enable_caching:
            return None
        return self._cache.get(key)

    def _add_to_cache(self, key: str, score: WindowQualityScore) -> None:
        """添加到缓存"""
        if self.config.enable_caching:
            self._cache[key] = score

    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计"""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'total': total,
            'hit_rate': round(hit_rate, 4),
            'cache_size': len(self._cache),
        }

    def clear_cache(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def calculate_quality(
        self,
        window: 'VisibilityWindow',
        satellite: 'Satellite',
        mission: Optional['Mission'] = None,
        config: Optional[QualityScoreConfig] = None,
        ground_station_windows: Optional[List['VisibilityWindow']] = None,
    ) -> WindowQualityScore:
        """
        计算窗口综合质量评分

        Args:
            window: 可见性窗口
            satellite: 卫星
            mission: 任务（可选）
            config: 质量评分配置（可选，覆盖默认配置）
            ground_station_windows: 地面站可见窗口（可选，用于数传配合度评分）

        Returns:
            WindowQualityScore: 包含各维度评分和综合评分
        """
        # 检查缓存
        cache_key = self._get_cache_key(window, satellite.id)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1

        # 使用传入的配置或默认配置
        cfg = config or self.config

        # 确定卫星类型和权重
        sat_type = self._determine_satellite_type(satellite)
        weights = cfg.get_weights_for_satellite(sat_type)

        # 计算各维度评分
        elevation_score = self._calculate_elevation_score(window.max_elevation)

        attitude_score = self._calculate_attitude_score(
            getattr(window, 'attitude_samples', None),
            satellite.capabilities.max_roll_angle if hasattr(satellite, 'capabilities') else 45.0,
            satellite.capabilities.max_pitch_angle if hasattr(satellite, 'capabilities') else 30.0,
        )

        duration_score = self._calculate_duration_score(
            window.duration(),
            self.MIN_REQUIRED_DURATION,
            self.OPTIMAL_WINDOW_DURATION,
        )

        illumination_score = self._calculate_illumination_score(
            window,
            satellite,
            mission,
        )

        downlink_score = self._calculate_downlink_score(
            window,
            ground_station_windows,
        )

        # 计算综合评分
        overall_score = (
            elevation_score * weights.elevation_weight +
            attitude_score * weights.attitude_weight +
            duration_score * weights.duration_weight +
            illumination_score * weights.illumination_weight +
            downlink_score * weights.downlink_weight
        )

        # 确定质量等级
        quality_tier = cfg.thresholds.get_quality_tier(overall_score)

        # 收集详细信息
        details = {
            'satellite_type': sat_type.value,
            'weights': weights.to_dict(),
            'elevation': window.max_elevation,
            'duration_seconds': window.duration(),
            'config_thresholds': {
                'high': cfg.thresholds.high_quality,
                'medium': cfg.thresholds.medium_quality,
                'low': cfg.thresholds.low_quality,
            },
        }

        # 如果有姿态数据，添加姿态详情
        if hasattr(window, 'attitude_samples') and window.attitude_samples:
            details['attitude_samples_count'] = len(window.attitude_samples)

        score = WindowQualityScore(
            overall_score=overall_score,
            elevation_score=elevation_score,
            attitude_score=attitude_score,
            duration_score=duration_score,
            illumination_score=illumination_score,
            downlink_score=downlink_score,
            quality_tier=quality_tier,
            details=details,
        )

        # 添加到缓存
        self._add_to_cache(cache_key, score)

        return score

    def _determine_satellite_type(self, satellite: 'Satellite') -> SatelliteType:
        """确定卫星类型"""
        sat_id = getattr(satellite, 'id', '').lower()
        sat_type = getattr(satellite, 'satellite_type', '').lower()

        if 'optical' in sat_id or 'optical' in sat_type or 'opt' in sat_type:
            return SatelliteType.OPTICAL
        elif 'sar' in sat_id or 'sar' in sat_type:
            return SatelliteType.SAR
        else:
            # 尝试从其他属性推断
            if hasattr(satellite, 'payloads'):
                for payload in satellite.payloads:
                    payload_type = getattr(payload, 'payload_type', '').lower()
                    if 'sar' in payload_type:
                        return SatelliteType.SAR
                    elif 'optical' in payload_type or 'camera' in payload_type:
                        return SatelliteType.OPTICAL

            return SatelliteType.UNKNOWN

    def _calculate_elevation_score(self, max_elevation: float) -> float:
        """
        计算仰角评分

        使用非线性映射，高仰角获得更高评分：
        - 90° -> 1.0
        - 45° -> 0.5
        - 0° -> 0.0

        采用平方根映射以强调高仰角的优势
        """
        # 限制在有效范围内
        elevation = max(0.0, min(90.0, max_elevation))

        # 非线性映射（平方根）
        score = math.sqrt(elevation / 90.0)

        return score

    def _calculate_attitude_score(
        self,
        attitude_samples: Optional[List[Tuple[float, float, float]]],
        max_roll: float,
        max_pitch: float,
    ) -> float:
        """
        计算姿态约束满足度评分

        评估窗口内姿态偏离中心的程度：
        - 中心位置（0,0）= 1.0
        - 50%边界内 = 0.75-1.0
        - 边界附近 = 0.5
        - 超出约束 = 0.0

        Args:
            attitude_samples: 姿态采样数据 [(timestamp, roll, pitch), ...]
            max_roll: 最大滚转角（度）
            max_pitch: 最大俯仰角（度）

        Returns:
            姿态约束满足度评分（0-1）
        """
        if not attitude_samples or len(attitude_samples) == 0:
            # 没有姿态数据，假设中等质量
            return 0.7

        scores = []
        for sample in attitude_samples:
            if len(sample) >= 3:
                _, roll, pitch = sample[0], sample[1], sample[2]
            elif len(sample) == 2:
                roll, pitch = sample[0], sample[1]
            else:
                continue

            # 计算相对于最大角度的归一化位置
            roll_ratio = abs(roll) / max_roll if max_roll > 0 else 0
            pitch_ratio = abs(pitch) / max_pitch if max_pitch > 0 else 0

            # 检查是否超出约束
            if roll_ratio > 1.0 or pitch_ratio > 1.0:
                return 0.0  # 一旦超出约束，整个窗口不合格

            # 计算距离中心的程度（使用欧氏距离）
            distance = math.sqrt(roll_ratio**2 + pitch_ratio**2)

            # 距离中心越近，评分越高
            # 线性映射：中心=1.0，边界=0.5
            sample_score = 1.0 - 0.5 * distance
            scores.append(sample_score)

        if not scores:
            return 0.7

        # 使用平均分，但惩罚低分样本
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)

        # 综合评分：70%平均分 + 30%最低分
        return 0.7 * avg_score + 0.3 * min_score

    def _calculate_duration_score(
        self,
        duration: float,
        min_required: float,
        optimal: float,
    ) -> float:
        """
        计算窗口持续时间评分

        评分规则：
        - 低于最小要求：0.0
        - 达到最佳：1.0
        - 超出最佳：保持1.0（上限）

        Args:
            duration: 实际持续时间（秒）
            min_required: 最小要求持续时间（秒）
            optimal: 最佳持续时间（秒）

        Returns:
            持续时间评分（0-1）
        """
        if duration < min_required:
            return 0.0

        if duration >= optimal:
            return 1.0

        # 线性插值
        return (duration - min_required) / (optimal - min_required)

    def _calculate_illumination_score(
        self,
        window: 'VisibilityWindow',
        satellite: 'Satellite',
        mission: Optional['Mission'],
    ) -> float:
        """
        计算光照条件评分（光学卫星）

        评估成像时的光照条件：
        - 太阳高度角（越高越好）
        - 避免阴影区

        对于SAR卫星，此评分始终返回1.0（不依赖光照）

        Args:
            window: 可见性窗口
            satellite: 卫星
            mission: 任务（用于获取太阳位置）

        Returns:
            光照条件评分（0-1）
        """
        # 如果是SAR卫星，不依赖光照
        sat_type = self._determine_satellite_type(satellite)
        if sat_type == SatelliteType.SAR:
            return 1.0

        # 尝试获取目标位置和太阳位置
        try:
            if mission is None:
                return 0.8  # 默认中等光照

            # 获取目标
            target = None
            if hasattr(mission, 'get_target_by_id'):
                target = mission.get_target_by_id(window.target_id)

            if target is None:
                return 0.8

            # 计算窗口中间时刻
            mid_time = window.start_time + timedelta(
                seconds=(window.end_time - window.start_time).total_seconds() / 2
            )

            # 尝试使用太阳位置计算器
            try:
                from core.dynamics.sun_position_calculator import SunPositionCalculator

                sun_calc = SunPositionCalculator(use_orekit=False)
                sun_pos = sun_calc.get_sun_position(mid_time)

                # 计算太阳高度角
                target_lat = getattr(target, 'latitude', 0)
                target_lon = getattr(target, 'longitude', 0)

                # 简化计算：使用太阳赤纬和时角
                # 这里使用简化模型，实际应该使用完整的太阳位置计算
                sun_elevation = self._estimate_sun_elevation(
                    mid_time, target_lat, target_lon, sun_pos
                )

                # 太阳高度角 > 30° 得满分
                if sun_elevation >= 30:
                    return 1.0
                elif sun_elevation > 0:
                    # 线性插值
                    return sun_elevation / 30.0
                else:
                    # 太阳在地平线下，低分
                    return max(0.0, 0.2 + sun_elevation / 100.0)

            except ImportError:
                # 无法导入太阳位置计算器，使用简化估算
                return self._estimate_illumination_simple(mid_time, target)

        except Exception as e:
            logger.warning(f"Failed to calculate illumination score: {e}")
            return 0.8  # 默认中等光照

    def _estimate_sun_elevation(
        self,
        dt: datetime,
        lat: float,
        lon: float,
        sun_pos: Tuple[float, float, float],
    ) -> float:
        """
        估算太阳高度角

        简化模型：基于太阳ECEF位置和目标位置计算
        """
        try:
            import math

            # 目标ECEF位置（简化计算，假设地球半径6371km）
            R = 6371000  # 米
            lat_rad = math.radians(lat)
            lon_rad = math.radians(lon)

            target_x = R * math.cos(lat_rad) * math.cos(lon_rad)
            target_y = R * math.cos(lat_rad) * math.sin(lon_rad)
            target_z = R * math.sin(lat_rad)

            # 从目标指向太阳的向量
            dx = sun_pos[0] - target_x
            dy = sun_pos[1] - target_y
            dz = sun_pos[2] - target_z

            # 目标位置向量（从地心指向目标）
            tx, ty, tz = target_x, target_y, target_z

            # 计算两个向量的夹角
            dot = dx * tx + dy * ty + dz * tz
            mag_sun = math.sqrt(dx**2 + dy**2 + dz**2)
            mag_target = math.sqrt(tx**2 + ty**2 + tz**2)

            if mag_sun == 0 or mag_target == 0:
                return 45.0  # 默认值

            cos_angle = dot / (mag_sun * mag_target)
            cos_angle = max(-1.0, min(1.0, cos_angle))

            # 高度角 = 90° - 与天顶方向的夹角
            angle = math.degrees(math.acos(cos_angle))
            elevation = 90.0 - angle

            return elevation

        except Exception:
            return 45.0  # 默认值

    def _estimate_illumination_simple(
        self,
        dt: datetime,
        target: 'Target',
    ) -> float:
        """
        简化光照估算（基于时间和纬度）

        这是一个非常简化的模型，仅用于近似估计
        """
        try:
            # 获取当地小时（近似）
            hour = dt.hour + dt.minute / 60.0

            # 简化的太阳高度角估计
            # 正午12点最高，6点和18点为0°
            lat = abs(getattr(target, 'latitude', 0))

            # 基础太阳高度角（正午）
            max_elevation = 90 - lat

            # 根据时间调整
            if 6 <= hour <= 18:
                # 白天
                hour_from_noon = abs(hour - 12)
                elevation = max_elevation * (1 - hour_from_noon / 6)
                elevation = max(0, elevation)
            else:
                # 夜间
                elevation = -10

            # 转换为评分
            if elevation >= 30:
                return 1.0
            elif elevation > 0:
                return elevation / 30.0
            else:
                return 0.1

        except Exception:
            return 0.8

    def _calculate_downlink_score(
        self,
        window: 'VisibilityWindow',
        ground_station_windows: Optional[List['VisibilityWindow']],
    ) -> float:
        """
        计算地面站配合度评分

        评估窗口结束后进行数传的便利性：
        - 窗口结束后很快有地面站可见窗口 = 高分
        - 窗口结束后长时间无地面站可见 = 低分
        - 存储容量充足时可接受较长的数传延迟

        Args:
            window: 可见性窗口
            ground_station_windows: 地面站可见窗口列表

        Returns:
            地面站配合度评分（0-1）
        """
        if not ground_station_windows or len(ground_station_windows) == 0:
            # 没有地面站信息，假设中等配合度
            return 0.6

        window_end = window.end_time

        # 寻找窗口结束后最近的地面站窗口
        best_delay = None

        for gs_window in ground_station_windows:
            # 只考虑同一卫星的地面站窗口
            if gs_window.satellite_id != window.satellite_id:
                continue

            # 计算从成像结束到数传开始的时间差
            delay = (gs_window.start_time - window_end).total_seconds()

            # 只考虑成像结束后的地面站窗口
            if delay >= 0:
                if best_delay is None or delay < best_delay:
                    best_delay = delay

        if best_delay is None:
            # 窗口结束后没有可用的地面站窗口
            return 0.2

        # 评分规则：
        # - 立即数传（0-5分钟）：1.0
        # - 5-30分钟：线性下降到0.7
        # - 30-120分钟：线性下降到0.3
        # - >120分钟：0.2

        if best_delay <= 300:  # 5分钟
            return 1.0
        elif best_delay <= 1800:  # 30分钟
            return 1.0 - 0.3 * (best_delay - 300) / 1500
        elif best_delay <= 7200:  # 120分钟
            return 0.7 - 0.4 * (best_delay - 1800) / 5400
        else:
            return 0.2

    def calculate_quality_batch(
        self,
        windows: List['VisibilityWindow'],
        satellite: 'Satellite',
        mission: Optional['Mission'] = None,
        config: Optional[QualityScoreConfig] = None,
        ground_station_windows: Optional[List['VisibilityWindow']] = None,
    ) -> List[WindowQualityScore]:
        """
        批量计算窗口质量评分

        Args:
            windows: 可见性窗口列表
            satellite: 卫星
            mission: 任务（可选）
            config: 质量评分配置（可选）
            ground_station_windows: 地面站可见窗口（可选）

        Returns:
            WindowQualityScore列表
        """
        return [
            self.calculate_quality(
                window=window,
                satellite=satellite,
                mission=mission,
                config=config,
                ground_station_windows=ground_station_windows,
            )
            for window in windows
        ]

    def filter_windows_by_quality(
        self,
        windows: List['VisibilityWindow'],
        satellite: 'Satellite',
        min_quality: float,
        mission: Optional['Mission'] = None,
    ) -> List['VisibilityWindow']:
        """
        按质量筛选窗口

        Args:
            windows: 可见性窗口列表
            satellite: 卫星
            min_quality: 最低质量阈值
            mission: 任务（可选）

        Returns:
            质量合格的窗口列表
        """
        result = []
        for window in windows:
            score = self.calculate_quality(window, satellite, mission)
            if score.overall_score >= min_quality:
                result.append(window)
        return result

    def sort_windows_by_quality(
        self,
        windows: List['VisibilityWindow'],
        satellite: 'Satellite',
        mission: Optional['Mission'] = None,
        reverse: bool = True,
    ) -> List[Tuple['VisibilityWindow', WindowQualityScore]]:
        """
        按质量排序窗口

        Args:
            windows: 可见性窗口列表
            satellite: 卫星
            mission: 任务（可选）
            reverse: 是否降序（高质量在前）

        Returns:
            (窗口, 评分)元组列表
        """
        scored_windows = [
            (window, self.calculate_quality(window, satellite, mission))
            for window in windows
        ]

        scored_windows.sort(
            key=lambda x: x[1].overall_score,
            reverse=reverse
        )

        return scored_windows
