"""
质量评分配置

提供可配置的质量评分权重系统，支持不同卫星类型的默认配置。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


class SatelliteType(Enum):
    """卫星类型"""
    OPTICAL = "optical"
    SAR = "sar"
    UNKNOWN = "unknown"


class QualityTier(str, Enum):
    """质量等级枚举"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNACCEPTABLE = "unacceptable"


@dataclass
class QualityDimensionWeights:
    """
    质量评分维度权重

    所有权重之和应等于1.0

    Attributes:
        elevation_weight: 仰角权重（仰角越高，质量越好）
        attitude_weight: 姿态约束满足度权重（姿态越接近中心，质量越好）
        duration_weight: 窗口持续时间权重（持续时间越长，质量越好）
        illumination_weight: 光照条件权重（光学卫星适用，光照越好，质量越好）
        downlink_weight: 地面站配合度权重（数传越便利，质量越好）
    """
    elevation_weight: float = 0.25      # 仰角权重
    attitude_weight: float = 0.25       # 姿态约束满足度权重
    duration_weight: float = 0.15       # 窗口持续时间权重
    illumination_weight: float = 0.20   # 光照条件权重
    downlink_weight: float = 0.15       # 地面站配合度权重

    def __post_init__(self):
        """验证权重之和为1.0"""
        total = (self.elevation_weight + self.attitude_weight +
                 self.duration_weight + self.illumination_weight +
                 self.downlink_weight)
        if abs(total - 1.0) > 0.001:
            # 自动归一化
            factor = 1.0 / total
            self.elevation_weight *= factor
            self.attitude_weight *= factor
            self.duration_weight *= factor
            self.illumination_weight *= factor
            self.downlink_weight *= factor

    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'elevation': self.elevation_weight,
            'attitude': self.attitude_weight,
            'duration': self.duration_weight,
            'illumination': self.illumination_weight,
            'downlink': self.downlink_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'QualityDimensionWeights':
        """从字典创建"""
        return cls(
            elevation_weight=data.get('elevation', 0.25),
            attitude_weight=data.get('attitude', 0.25),
            duration_weight=data.get('duration', 0.15),
            illumination_weight=data.get('illumination', 0.20),
            downlink_weight=data.get('downlink', 0.15),
        )


@dataclass
class QualityThresholds:
    """
    质量评分阈值

    用于快速分类窗口质量等级
    """
    high_quality: float = 0.7    # 高质量阈值
    medium_quality: float = 0.4  # 中等质量阈值
    low_quality: float = 0.3     # 低质量阈值（最低可接受）

    def get_quality_tier(self, score: float) -> QualityTier:
        """
        根据评分获取质量等级

        Returns:
            QualityTier enum value: HIGH, MEDIUM, LOW, or UNACCEPTABLE
        """
        if score >= self.high_quality:
            return QualityTier.HIGH
        elif score >= self.medium_quality:
            return QualityTier.MEDIUM
        elif score >= self.low_quality:
            return QualityTier.LOW
        else:
            return QualityTier.UNACCEPTABLE


@dataclass
class QualityScoreConfig:
    """
    质量评分配置

    包含完整的质量评分参数配置

    Attributes:
        weights: 通用权重配置
        optical_weights: 光学卫星专用权重
        sar_weights: SAR卫星专用权重
        thresholds: 质量等级阈值
        min_quality_threshold: 最低可接受质量阈值
        enable_caching: 是否启用缓存
        cache_ttl_seconds: 缓存过期时间（秒）
    """
    # 通用权重
    weights: QualityDimensionWeights = field(default_factory=QualityDimensionWeights)

    # 光学卫星权重（更注重光照条件）
    optical_weights: QualityDimensionWeights = field(default_factory=lambda:
        QualityDimensionWeights(
            elevation_weight=0.20,
            attitude_weight=0.20,
            duration_weight=0.15,
            illumination_weight=0.30,  # 光学卫星更看重光照
            downlink_weight=0.15
        )
    )

    # SAR卫星权重（不依赖光照，更看重仰角和姿态）
    sar_weights: QualityDimensionWeights = field(default_factory=lambda:
        QualityDimensionWeights(
            elevation_weight=0.30,       # SAR更看重仰角
            attitude_weight=0.30,        # SAR更看重姿态
            duration_weight=0.15,
            illumination_weight=0.05,    # SAR不依赖光照
            downlink_weight=0.20
        )
    )

    # 质量等级阈值
    thresholds: QualityThresholds = field(default_factory=QualityThresholds)

    # 最低可接受质量
    min_quality_threshold: float = 0.3

    # 缓存配置
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1小时

    def get_weights_for_satellite(self, satellite_type: SatelliteType) -> QualityDimensionWeights:
        """
        获取指定卫星类型的权重配置

        Args:
            satellite_type: 卫星类型

        Returns:
            对应的权重配置
        """
        if satellite_type == SatelliteType.OPTICAL:
            return self.optical_weights
        elif satellite_type == SatelliteType.SAR:
            return self.sar_weights
        else:
            return self.weights

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'weights': self.weights.to_dict(),
            'optical_weights': self.optical_weights.to_dict(),
            'sar_weights': self.sar_weights.to_dict(),
            'thresholds': {
                'high_quality': self.thresholds.high_quality,
                'medium_quality': self.thresholds.medium_quality,
                'low_quality': self.thresholds.low_quality,
            },
            'min_quality_threshold': self.min_quality_threshold,
            'enable_caching': self.enable_caching,
            'cache_ttl_seconds': self.cache_ttl_seconds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QualityScoreConfig':
        """从字典创建"""
        config = cls()

        if 'weights' in data:
            config.weights = QualityDimensionWeights.from_dict(data['weights'])
        if 'optical_weights' in data:
            config.optical_weights = QualityDimensionWeights.from_dict(data['optical_weights'])
        if 'sar_weights' in data:
            config.sar_weights = QualityDimensionWeights.from_dict(data['sar_weights'])
        if 'thresholds' in data:
            t = data['thresholds']
            config.thresholds = QualityThresholds(
                high_quality=t.get('high_quality', 0.7),
                medium_quality=t.get('medium_quality', 0.4),
                low_quality=t.get('low_quality', 0.3),
            )
        if 'min_quality_threshold' in data:
            config.min_quality_threshold = data['min_quality_threshold']
        if 'enable_caching' in data:
            config.enable_caching = data['enable_caching']
        if 'cache_ttl_seconds' in data:
            config.cache_ttl_seconds = data['cache_ttl_seconds']

        return config


@dataclass
class SatelliteTypeWeights:
    """
    卫星类型与权重的映射

    用于为不同卫星指定不同的质量评分策略
    """
    default_weights: QualityDimensionWeights = field(default_factory=QualityDimensionWeights)
    optical_weights: Optional[QualityDimensionWeights] = None
    sar_weights: Optional[QualityDimensionWeights] = None

    def get_weights(self, sat_type: str) -> QualityDimensionWeights:
        """
        获取指定卫星类型的权重

        Args:
            sat_type: 卫星类型标识符（如 'optical', 'sar', 'SAT-OPTICAL-01'）

        Returns:
            对应的质量维度权重配置
        """
        sat_type_lower = sat_type.lower()

        # 使用精确匹配或包含检查，优先检查更具体的模式
        if 'optical' in sat_type_lower:
            return self.optical_weights or self.default_weights
        elif 'sar' in sat_type_lower:
            return self.sar_weights or self.default_weights
        else:
            return self.default_weights


# 默认配置实例
DEFAULT_QUALITY_CONFIG = QualityScoreConfig()

# 严格质量配置（用于高价值目标）
STRICT_QUALITY_CONFIG = QualityScoreConfig(
    thresholds=QualityThresholds(
        high_quality=0.8,
        medium_quality=0.6,
        low_quality=0.5,
    ),
    min_quality_threshold=0.5,
)

# 宽松质量配置（用于快速调度）
LENIENT_QUALITY_CONFIG = QualityScoreConfig(
    thresholds=QualityThresholds(
        high_quality=0.6,
        medium_quality=0.3,
        low_quality=0.2,
    ),
    min_quality_threshold=0.2,
)
