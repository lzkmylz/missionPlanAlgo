"""核心数据模型"""

from .satellite import Satellite, SatelliteType, SatelliteCapabilities, ImagingMode, Orbit, OrbitType, OrbitSource
from .target import Target, TargetType, GeoPoint
from .ground_station import GroundStation, Antenna
from .mission import Mission
from .mosaic_tile import MosaicTile, TileStatus, TilePriorityMode, TileVisibilityInfo
from .area_coverage_plan import AreaCoveragePlan, MultiAreaCoveragePlan, CoverageStrategy, OverlapHandling, CoverageStatistics
from .imaging_mode import (
    ImagingModeConfig,
    ImagingModeType,
    OPTICAL_PUSH_BROOM_HIGH_RES,
    OPTICAL_PUSH_BROOM_MEDIUM_RES,
    SAR_STRIPMAP_MODE,
    SAR_SPOTLIGHT_MODE,
    SAR_SCAN_MODE,
    SAR_SLIDING_SPOTLIGHT_MODE,
    MODE_TEMPLATES,
    get_mode_template,
)
from .payload_config import (
    PayloadConfiguration,
    create_optical_payload_config,
    create_sar_payload_config,
)

__all__ = [
    # 卫星相关
    'Satellite', 'SatelliteType', 'SatelliteCapabilities', 'ImagingMode', 'Orbit', 'OrbitType', 'OrbitSource',
    # 目标相关
    'Target', 'TargetType', 'GeoPoint',
    # 地面站相关
    'GroundStation', 'Antenna',
    # 任务相关
    'Mission',
    # 覆盖规划相关
    'MosaicTile', 'TileStatus', 'TilePriorityMode', 'TileVisibilityInfo',
    'AreaCoveragePlan', 'MultiAreaCoveragePlan', 'CoverageStrategy', 'OverlapHandling', 'CoverageStatistics',
    # 成像模式相关 (新增)
    'ImagingModeConfig',
    'ImagingModeType',
    'OPTICAL_PUSH_BROOM_HIGH_RES',
    'OPTICAL_PUSH_BROOM_MEDIUM_RES',
    'SAR_STRIPMAP_MODE',
    'SAR_SPOTLIGHT_MODE',
    'SAR_SCAN_MODE',
    'SAR_SLIDING_SPOTLIGHT_MODE',
    'MODE_TEMPLATES',
    'get_mode_template',
    # 载荷配置相关 (新增)
    'PayloadConfiguration',
    'create_optical_payload_config',
    'create_sar_payload_config',
]
