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
    ImagingMode,
    OPTICAL_PUSH_BROOM_HIGH_RES,
    OPTICAL_PUSH_BROOM_MEDIUM_RES,
    OPTICAL_PMC_25PERCENT,
    OPTICAL_PMC_50PERCENT,
    OPTICAL_REVERSE_PMC_25PERCENT,
    OPTICAL_REVERSE_PMC_50PERCENT,
    SAR_STRIPMAP_MODE,
    SAR_SPOTLIGHT_MODE,
    SAR_SCAN_MODE,
    SAR_SLIDING_SPOTLIGHT_MODE,
    SAR_TOPSAR_MODE,
    SAR_SCANSAR_MODE,
    SAR_PMC_25PERCENT,
    MODE_TEMPLATES,
    get_mode_template,
    create_pmc_mode_config,
)
from .sar_scansar_config import (
    ScanSARSubSwathPosition,
    SARScanSARConfig,
    SARScanSARResult,
)
from .payload_config import (
    PayloadConfiguration,
    create_optical_payload_config,
    create_sar_payload_config,
)
from .multi_strip_mosaic_config import (
    MultiStripMosaicConfig,
    MOSAIC_CONFIG_3STRIP,
    MOSAIC_CONFIG_5STRIP,
    MOSAIC_CONFIG_TEMPLATES,
    get_mosaic_config_template,
)
from .pmc_config import (
    PitchMotionCompensationConfig,
    PMC_CONFIG_10PERCENT,
    PMC_CONFIG_25PERCENT,
    PMC_CONFIG_50PERCENT,
    PMC_CONFIG_75PERCENT,
    PMC_REVERSE_CONFIG_10PERCENT,
    PMC_REVERSE_CONFIG_25PERCENT,
    PMC_REVERSE_CONFIG_50PERCENT,
    PMC_CONFIG_TEMPLATES,
    get_pmc_config_template,
    create_pmc_config_for_altitude,
    create_reverse_pmc_config,
)
from .isl_config import (
    ISLLinkType,
    ISLLinkSelectionStrategy,
    LaserISLConfig,
    MicrowaveISLConfig,
    ISLPeerConfig,
    ISLCapabilityConfig,
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
    # 成像模式相关
    'ImagingModeConfig',
    'ImagingModeType',
    'ImagingMode',
    'OPTICAL_PUSH_BROOM_HIGH_RES',
    'OPTICAL_PUSH_BROOM_MEDIUM_RES',
    'OPTICAL_PMC_25PERCENT',
    'OPTICAL_PMC_50PERCENT',
    'OPTICAL_REVERSE_PMC_25PERCENT',
    'OPTICAL_REVERSE_PMC_50PERCENT',
    'SAR_STRIPMAP_MODE',
    'SAR_SPOTLIGHT_MODE',
    'SAR_SCAN_MODE',
    'SAR_SLIDING_SPOTLIGHT_MODE',
    'SAR_TOPSAR_MODE',
    'SAR_SCANSAR_MODE',
    'SAR_PMC_25PERCENT',
    'MODE_TEMPLATES',
    'get_mode_template',
    'create_pmc_mode_config',
    # ScanSAR物理配置
    'ScanSARSubSwathPosition',
    'SARScanSARConfig',
    'SARScanSARResult',
    # 载荷配置相关
    'PayloadConfiguration',
    'create_optical_payload_config',
    'create_sar_payload_config',
    # 多条带拼幅配置相关
    'MultiStripMosaicConfig',
    'MOSAIC_CONFIG_3STRIP',
    'MOSAIC_CONFIG_5STRIP',
    'MOSAIC_CONFIG_TEMPLATES',
    'get_mosaic_config_template',
    # PMC配置相关 (新增)
    'PitchMotionCompensationConfig',
    'PMC_CONFIG_10PERCENT',
    'PMC_CONFIG_25PERCENT',
    'PMC_CONFIG_50PERCENT',
    'PMC_CONFIG_75PERCENT',
    'PMC_REVERSE_CONFIG_10PERCENT',
    'PMC_REVERSE_CONFIG_25PERCENT',
    'PMC_REVERSE_CONFIG_50PERCENT',
    'PMC_CONFIG_TEMPLATES',
    'get_pmc_config_template',
    'create_pmc_config_for_altitude',
    'create_reverse_pmc_config',
    # ISL配置相关 (新增)
    'ISLLinkType',
    'ISLLinkSelectionStrategy',
    'LaserISLConfig',
    'MicrowaveISLConfig',
    'ISLPeerConfig',
    'ISLCapabilityConfig',
]
