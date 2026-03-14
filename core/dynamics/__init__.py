"""
动力学模块 - 卫星姿态机动和轨道动力学计算

包含:
- precise: 精确姿态机动计算（基于刚体动力学）
- attitude_calculator: 卫星成像姿态角计算
- orbit_batch_propagator: Orekit批量轨道传播器

注意: 简化模型已移除，统一使用精确模型
"""

from .attitude_calculator import (
    AttitudeCalculator,
    AttitudeAngles,
    PropagatorType,
)
from .orbit_batch_propagator import (
    OrekitBatchPropagator,
    SatelliteOrbitCache,
    get_batch_propagator,
)

__all__ = [
    'AttitudeCalculator',
    'AttitudeAngles',
    'PropagatorType',
    'OrekitBatchPropagator',
    'SatelliteOrbitCache',
    'get_batch_propagator',
]
