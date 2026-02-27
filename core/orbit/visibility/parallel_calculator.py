"""
并行可见性计算器

Phase 3优化：多线程并行计算卫星-目标对的可见窗口
- 线程池管理
- JVM attach处理
- 结果聚合

预期性能提升：4-8倍（墙钟时间）
"""

import os
import functools
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

try:
    import jpype
    JPYPE_AVAILABLE = True
except ImportError:
    JPYPE_AVAILABLE = False

logger = logging.getLogger(__name__)


def ensure_jvm_attached(func):
    """装饰器：确保当前线程已挂载到JVM

    非主线程调用Java前必须先attach到JVM，
    否则会抛出JVM Not Attached异常。

    注意：线程池中的线程会复用，不能detach。
    JVM退出时自动清理所有attach的线程。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if JPYPE_AVAILABLE and jpype.isJVMStarted():
            if not jpype.isThreadAttachedToJVM():
                jpype.attachThreadToJVM()
                func_name = getattr(func, '__name__', 'unknown')
                logger.debug(f"Thread attached to JVM: {func_name}")
        return func(*args, **kwargs)
    return wrapper


class ParallelVisibilityCalculator:
    """
    并行可见性计算器

    使用线程池并行计算多个卫星-目标对的可见窗口。
    每个计算任务完全独立，无共享状态。

    Attributes:
        max_workers: 线程池大小，默认CPU核心数×2
        _executor: ThreadPoolExecutor实例
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        初始化并行计算器

        Args:
            max_workers: 线程池大小，默认CPU核心数×2
        """
        self.max_workers = max_workers or (os.cpu_count() * 2)
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="vis_calc"
        )
        logger.info(f"ParallelVisibilityCalculator initialized with {self.max_workers} workers")

    def compute_all_windows(
        self,
        satellites: List[Any],
        targets: List[Any],
        time_range: Tuple[datetime, datetime],
        java_bridge: Any
    ) -> Dict[Tuple[str, str], List[Any]]:
        """
        并行计算所有卫星-目标对的可见窗口

        Args:
            satellites: 卫星列表
            targets: 目标列表
            time_range: (start_time, end_time)
            java_bridge: Java桥接器实例

        Returns:
            Dict[(sat_id, target_id), List[VisibilityWindow]]
        """
        if not satellites or not targets:
            return {}

        # 生成所有任务
        tasks = [
            (sat, target, time_range, java_bridge)
            for sat in satellites
            for target in targets
        ]

        logger.info(f"Submitting {len(tasks)} tasks to thread pool")

        # 提交到线程池
        futures = {
            self._executor.submit(
                self._compute_single_pair,
                sat, target, time_range, java_bridge
            ): (sat.id, target.id)
            for sat, target, time_range, java_bridge in tasks
        }

        # 收集结果
        results = {}
        completed = 0
        total = len(futures)
        progress_interval = max(1, total // 10)  # 每10%报告一次进度

        for future in as_completed(futures):
            sat_id, target_id = futures[future]
            try:
                windows = future.result()
                results[(sat_id, target_id)] = windows
                completed += 1

                # 报告进度
                if completed % progress_interval == 0:
                    progress_pct = 100 * completed // total
                    logger.info(f"Progress: {completed}/{total} ({progress_pct}%)")

            except Exception as e:
                logger.error(f"Failed to compute windows for {sat_id}-{target_id}: {e}")
                results[(sat_id, target_id)] = []

        logger.info(f"Completed {len(results)} tasks")
        return results

    @ensure_jvm_attached
    def _compute_single_pair(
        self,
        satellite: Any,
        target: Any,
        time_range: Tuple[datetime, datetime],
        java_bridge: Any
    ) -> List[Any]:
        """
        计算单个卫星-目标对的可见窗口

        注意：此方法在线程池中执行，需要处理JVM attach。

        Args:
            satellite: 卫星对象
            target: 目标对象
            time_range: (start_time, end_time)
            java_bridge: Java桥接器

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        try:
            # 调用Java批量计算接口（单对版本）
            return java_bridge.compute_visibility_for_pair(
                satellite, target, time_range
            )
        except Exception as e:
            logger.error(f"Computation error for {satellite.id}-{target.id}: {e}")
            return []

    def shutdown(self):
        """关闭线程池"""
        if self._executor:
            self._executor.shutdown(wait=True)
            logger.info("Thread pool shut down")

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取线程池统计信息

        Returns:
            Dict: 统计信息
        """
        return {
            'max_workers': self.max_workers,
            'active_threads': len([t for t in self._executor._threads if t.is_alive()]),
        }
