"""
目标分解模块

将大区域目标分解为可观测的子任务
"""

from .base_decomposer import BaseDecomposer, DecompositionStrategy
from .grid_decomposer import GridDecomposer
from .strip_decomposer import StripDecomposer
from .decomposer_factory import DecomposerFactory

__all__ = [
    'BaseDecomposer',
    'DecompositionStrategy',
    'GridDecomposer',
    'StripDecomposer',
    'DecomposerFactory',
]
