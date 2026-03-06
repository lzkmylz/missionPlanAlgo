"""
优化版调度器

使用轨道预计算缓存和Java并行传播进行可见性计算。
相比基础调度器，可大幅提升大规模场景的计算性能。
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

from .base_scheduler import BaseScheduler, ScheduledTask, TaskFailure, TaskFailureReason
from core.orbit.visibility.window_cache import VisibilityWindowCache

logger = logging.getLogger(__name__)


class OptimizedSchedulerMixin:
    """
    优化版可见性计算混入类

    使用Java端的轨道预计算缓存和并行计算，
    将可见性计算时间从小时级降至分钟级。
    """

    def __init__(self, *args, use_optimized_visibility: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_optimized_visibility = use_optimized_visibility
        self._java_bridge = None
        self._orbit_positions = None  # 缓存的轨道位置数据

    def _initialize_visibility_optimized(self) -> None:
        """
        使用优化版Java接口初始化可见性窗口

        单次JNI调用完成所有计算，包括：
        1. 并行传播所有卫星轨道
        2. 缓存轨道状态
        3. 批量计算所有卫星-目标对的可见性
        """
        if not self.use_optimized_visibility:
            # 回退到基础实现
            return self._initialize_visibility_base()

        try:
            from core.orbit.visibility.orekit_java_bridge import OrekitJavaBridge

            self._java_bridge = OrekitJavaBridge()

            logger.info("使用优化版可见性计算（轨道预计算缓存 + Java并行）...")
            logger.info(f"场景规模: {len(self.satellites)}颗卫星 x {len(self.targets)}个目标")

            start_time = time.time()

            # 准备卫星配置
            sat_configs = []
            for sat in self.satellites:
                sat_config = {
                    'id': sat.sat_id if hasattr(sat, 'sat_id') else str(sat),
                    'tle_line1': getattr(sat, 'tle_line1', ''),
                    'tle_line2': getattr(sat, 'tle_line2', ''),
                    'min_elevation': getattr(sat, 'min_elevation', 5.0),
                }
                sat_configs.append(sat_config)

            # 准备目标配置
            target_configs = []
            for target in self.targets:
                target_config = {
                    'id': target.target_id if hasattr(target, 'target_id') else str(target),
                    'longitude': getattr(target, 'longitude', 0.0),
                    'latitude': getattr(target, 'latitude', 0.0),
                    'altitude': getattr(target, 'altitude', 0.0),
                    'min_observation_duration': getattr(target, 'min_observation_duration', 60),
                }
                target_configs.append(target_config)

            # 获取计算步长配置
            coarse_step = getattr(self, 'visibility_config', {}).get('coarse_step', 5.0)
            fine_step = getattr(self, 'visibility_config', {}).get('fine_step', 1.0)

            logger.info(f"HPOP轨道传播: 粗扫步长={coarse_step}s, 精扫步长={fine_step}s")

            # 调用优化版Java接口（单次JNI调用）
            result = self._java_bridge.compute_all_windows_optimized(
                satellites=sat_configs,
                targets=target_configs,
                start_time=self.start_time,
                end_time=self.end_time,
                coarse_step=coarse_step,
                fine_step=fine_step
            )

            elapsed = time.time() - start_time

            # 解析结果并缓存到VisibilityWindowCache
            self.window_cache = VisibilityWindowCache()

            for window_data in result.get('windows', []):
                from core.orbit.visibility.base import VisibilityWindow

                window = VisibilityWindow(
                    satellite_id=window_data['satellite_id'],
                    target_id=window_data['target_id'],
                    start_time=window_data['start_time'],
                    end_time=window_data['end_time'],
                    duration=timedelta(seconds=window_data['duration_seconds']),
                    max_elevation=window_data['max_elevation']
                )
                self.window_cache.add_window(window)

            # 记录统计信息
            stats = result.get('stats', {})
            logger.info(f"预计算完成:")
            logger.info(f"  - 计算时间: {elapsed:.2f}秒")
            logger.info(f"  - 可见窗口数: {len(result.get('windows', []))}")
            logger.info(f"  - Java统计: {stats}")

        except Exception as e:
            logger.error(f"优化版可见性计算失败: {e}")
            logger.info("回退到基础实现...")
            self.use_optimized_visibility = False
            return self._initialize_visibility_base()

    def _initialize_visibility_base(self) -> None:
        """基础可见性计算实现（回退用）"""
        # 调用父类的初始化方法
        if hasattr(super(), '_initialize_visibility'):
            super()._initialize_visibility()
        elif hasattr(super(), 'initialize'):
            super().initialize()

    def get_orbit_position_at_time(self, sat_id: str, query_time: datetime) -> Optional[Dict[str, float]]:
        """
        获取指定时刻的卫星位置

        使用缓存的轨道状态进行线性插值。

        Args:
            sat_id: 卫星ID
            query_time: 查询时间

        Returns:
            Dict with keys: x, y, z, vx, vy, vz, latitude, longitude, altitude
        """
        if self._orbit_positions is None:
            return None

        # 计算相对于起始时间的秒数
        seconds = (query_time - self.start_time).total_seconds()

        # 从缓存获取插值后的状态
        # TODO: 实现从Java缓存获取位置
        # 这需要扩展Java端接口返回轨道位置数据

        return None


def create_optimized_scheduler(base_scheduler_class: type) -> type:
    """
    为现有调度器类添加优化版可见性计算能力

    Usage:
        from scheduler.genetic_scheduler import GeneticScheduler
        OptimizedGeneticScheduler = create_optimized_scheduler(GeneticScheduler)

        scheduler = OptimizedGeneticScheduler(scenario, use_optimized_visibility=True)
    """

    class OptimizedScheduler(OptimizedSchedulerMixin, base_scheduler_class):
        """
        带优化可见性计算的调度器
        """

        def __init__(self, *args, use_optimized_visibility: bool = True, **kwargs):
            # 先调用OptimizedSchedulerMixin的初始化
            OptimizedSchedulerMixin.__init__(self, *args, use_optimized_visibility=use_optimized_visibility, **kwargs)
            # 再调用父类的初始化
            base_scheduler_class.__init__(self, *args, **kwargs)

        def initialize(self) -> None:
            """初始化：使用优化版可见性计算"""
            # 先调用优化版可见性计算
            if self.use_optimized_visibility:
                self._initialize_visibility_optimized()
            else:
                # 调用父类的初始化
                if hasattr(base_scheduler_class, 'initialize'):
                    base_scheduler_class.initialize(self)

        def solve(self, *args, **kwargs):
            """确保初始化完成后求解"""
            if not hasattr(self, 'window_cache') or self.window_cache is None:
                self.initialize()
            return base_scheduler_class.solve(self, *args, **kwargs)

    # 复制类名和文档
    OptimizedScheduler.__name__ = f"Optimized{base_scheduler_class.__name__}"
    OptimizedScheduler.__doc__ = f"""
    优化版{base_scheduler_class.__name__}

    使用轨道预计算缓存和Java并行传播进行可见性计算。
    {base_scheduler_class.__doc__ or ''}
    """

    return OptimizedScheduler


# 预创建的常用优化版调度器
# 这些会在对应调度器模块导入后动态创建

# 使用示例：
# from scheduler.optimized_scheduler import create_optimized_scheduler
# from scheduler.genetic_scheduler import GeneticScheduler
#
# OptimizedGeneticScheduler = create_optimized_scheduler(GeneticScheduler)
# scheduler = OptimizedGeneticScheduler(scenario)
