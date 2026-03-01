"""
可见性计算工厂

根据设计文档第3.2章实现双后端可见性计算工厂
自动检测STK并切换至Orekit
"""

import importlib.util
from typing import Dict, Any, Optional

from .base import VisibilityCalculator

# Import calculator classes for factory use
# These are imported at module level to enable proper mocking in tests
try:
    from .stk_visibility import STKVisibilityCalculator
except ImportError:
    STKVisibilityCalculator = None

try:
    from .orekit_visibility import OrekitVisibilityCalculator
except ImportError:
    OrekitVisibilityCalculator = None


class VisibilityCalculatorFactory:
    """
    可见性计算工厂

    根据可用性自动选择STK或Orekit后端
    auto模式：优先STK，fallback到Orekit
    """

    _backends = {
        'stk': 'STKVisibilityCalculator',
        'orekit': 'OrekitVisibilityCalculator'
    }

    @classmethod
    def create(cls, preferred: str = 'auto', config: Optional[Dict[str, Any]] = None):
        """
        创建可见性计算器

        Args:
            preferred: 首选后端 ('auto', 'stk', 'orekit')
            config: 配置参数
                - time_step: 计算时间步长（秒，默认1）
                - use_batch_propagator: 是否使用OrekitBatchPropagator（默认True）
                - min_elevation: 最小仰角（度，默认5.0）

        Returns:
            VisibilityCalculator: 可见性计算器实例

        Raises:
            ValueError: 如果指定了无效的后端类型
        """
        # 默认配置：1秒步长，启用BatchPropagator
        default_config = {
            'time_step': 1,  # 1秒步长（HPOP高精度）
            'use_batch_propagator': True,  # 默认启用OrekitBatchPropagator
            'min_elevation': 5.0,
        }

        # 合并用户配置
        if config:
            default_config.update(config)

        config = default_config

        if preferred == 'auto':
            if cls._check_stk_available() and STKVisibilityCalculator is not None:
                return STKVisibilityCalculator(config)
            else:
                return OrekitVisibilityCalculator(config)
        elif preferred == 'stk':
            if STKVisibilityCalculator is None:
                raise ImportError("STKVisibilityCalculator is not available")
            return STKVisibilityCalculator(config)
        elif preferred == 'orekit':
            if OrekitVisibilityCalculator is None:
                raise ImportError("OrekitVisibilityCalculator is not available")
            return OrekitVisibilityCalculator(config)
        else:
            raise ValueError(f"Unknown backend: {preferred}. Use 'auto', 'stk', or 'orekit'")

    @classmethod
    def _check_stk_available(cls) -> bool:
        """
        检查STK是否可用

        Returns:
            bool: STK是否可用
        """
        # 检查STK Python API是否可导入
        spec = importlib.util.find_spec("stk")
        return spec is not None

    @classmethod
    def get_available_backends(cls) -> Dict[str, bool]:
        """
        获取可用后端列表

        Returns:
            Dict[str, bool]: 后端名称到可用性的映射
        """
        return {
            'stk': cls._check_stk_available(),
            'orekit': True  # Orekit总是可用（纯Python实现）
        }
