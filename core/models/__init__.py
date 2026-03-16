"""核心数据模型"""

from .satellite import Satellite, SatelliteType, SatelliteCapabilities, ImagingMode, Orbit, OrbitType, OrbitSource
from .target import Target, TargetType, GeoPoint
from .ground_station import GroundStation, Antenna
from .mission import Mission
from .mosaic_tile import MosaicTile, TileStatus, TilePriorityMode, TileVisibilityInfo
from .area_coverage_plan import AreaCoveragePlan, MultiAreaCoveragePlan, CoverageStrategy, OverlapHandling, CoverageStatistics

__all__ = [
    'Satellite', 'SatelliteType', 'SatelliteCapabilities', 'ImagingMode', 'Orbit', 'OrbitType', 'OrbitSource',
    'Target', 'TargetType', 'GeoPoint',
    'GroundStation', 'Antenna',
    'Mission',
    'MosaicTile', 'TileStatus', 'TilePriorityMode', 'TileVisibilityInfo',
    'AreaCoveragePlan', 'MultiAreaCoveragePlan', 'CoverageStrategy', 'OverlapHandling', 'CoverageStatistics',
]
