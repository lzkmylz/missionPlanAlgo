"""
missionplanalgo - 卫星星座任务规划算法研究平台

顶层API设计参考numpy风格，提供简洁的函数式接口。

Examples
--------
>>> import missionplanalgo as mpa

>>> # 执行调度
>>> result = mpa.schedule("scenarios/test.json", algorithm="greedy")

>>> # 计算可见性
>>> windows = mpa.compute_visibility("scenarios/test.json")

>>> # 评估结果
>>> metrics = mpa.evaluate_schedule(result)
"""

# 首先处理路径兼容性
from ._compat import ensure_project_in_path

# 版本信息
__version__ = "1.0.0"
__author__ = "赵林"
__email__ = "zhaolin@hit.edu.cn"

# 核心数据模型
from core.models import (
    Satellite,
    SatelliteType,
    SatelliteCapabilities,
    Target,
    TargetType,
    GroundStation,
    Mission,
)

# 调度器
from scheduler.greedy.greedy_scheduler import GreedyScheduler
from scheduler.greedy.edd_scheduler import EDDScheduler
from scheduler.unified_scheduler import UnifiedScheduler

# 评估
from evaluation.metrics import MetricsCalculator

# 便捷API
from .api import (
    schedule,
    compute_visibility,
    load_scenario,
    evaluate_schedule,
)

__all__ = [
    # 元信息
    "__version__",
    "__author__",
    "__email__",

    # 核心模型
    "Satellite",
    "SatelliteType",
    "SatelliteCapabilities",
    "Target",
    "TargetType",
    "GroundStation",
    "Mission",

    # 调度器
    "GreedyScheduler",
    "EDDScheduler",
    "UnifiedScheduler",

    # 评估
    "MetricsCalculator",

    # 便捷函数
    "schedule",
    "compute_visibility",
    "load_scenario",
    "evaluate_schedule",
]
