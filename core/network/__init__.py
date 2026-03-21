"""
ISL (Inter-Satellite Link) and network routing module.

Provides satellite network communication capabilities:
- ISL visibility calculation between satellites
- Multi-hop routing with Dijkstra algorithm
- Relay satellite support (e.g., Tianlian)
- Command uplink scheduling
"""

from .isl_visibility import (
    ISLLink,
    ISLWindowCache,
    load_isl_windows_from_cache,
    ISLVisibilityCalculatorLegacy,
)
from .network_router import NetworkRouter, RoutePath
from .relay_satellite import RelaySatellite, RelayNetwork

# 向后兼容别名（已废弃，请使用 load_isl_windows_from_cache）
ISLVisibilityCalculator = ISLVisibilityCalculatorLegacy

__all__ = [
    'ISLLink',
    'ISLWindowCache',
    'load_isl_windows_from_cache',
    'ISLVisibilityCalculatorLegacy',
    'ISLVisibilityCalculator',  # deprecated alias
    'NetworkRouter',
    'RoutePath',
    'RelaySatellite',
    'RelayNetwork',
]
