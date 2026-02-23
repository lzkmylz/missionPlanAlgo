"""
在轨处理模块

第20章：星载边缘计算
"""

from .onboard_processing_manager import (
    AIAcceleratorType,
    AIAcceleratorSpec,
    ProcessingTaskType,
    ProcessingTaskSpec,
    ProcessingDecision,
    SatelliteResourceState,
    DecisionContext,
    OnboardProcessingManager,
)
from .pareto_optimizer import (
    ObjectiveFunction,
    ParetoOptimizer,
)

__all__ = [
    'AIAcceleratorType',
    'AIAcceleratorSpec',
    'ProcessingTaskType',
    'ProcessingTaskSpec',
    'ProcessingDecision',
    'SatelliteResourceState',
    'DecisionContext',
    'OnboardProcessingManager',
    'ObjectiveFunction',
    'ParetoOptimizer',
]
