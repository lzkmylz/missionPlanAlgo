"""
实验管理模块

实现第4章设计：
- 实验配置管理
- 批量对比实验
- 参数敏感性分析
- 结果记录和导出
"""

from .runner import ExperimentRunner, ExperimentConfig, ExperimentResult

__all__ = [
    'ExperimentRunner',
    'ExperimentConfig',
    'ExperimentResult',
]
