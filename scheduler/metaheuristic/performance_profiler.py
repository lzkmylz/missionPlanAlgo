"""
GA等元启发式算法性能分析工具

用于精确测量各阶段耗时，识别性能瓶颈
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class TimingRecord:
    """单次计时记录"""
    name: str
    start_time: float
    end_time: float
    count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class PhaseStatistics:
    """阶段统计信息"""
    name: str
    total_time: float = 0.0
    count: int = 0
    min_time: float = float('inf')
    max_time: float = 0.0
    times: List[float] = field(default_factory=list)

    def add(self, duration: float):
        self.total_time += duration
        self.count += 1
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.times.append(duration)

    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0


class PerformanceProfiler:
    """性能分析器 - 用于元启发式算法各阶段计时"""

    def __init__(self, name: str = "profiler"):
        self.name = name
        self.records: List[TimingRecord] = []
        self.phases: Dict[str, PhaseStatistics] = defaultdict(
            lambda: PhaseStatistics(name="")
        )
        self._active_timers: Dict[str, float] = {}
        self._enabled = True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def start(self, name: str, metadata: Dict[str, Any] = None):
        """开始计时"""
        if not self._enabled:
            return
        self._active_timers[name] = time.perf_counter()

    def end(self, name: str, metadata: Dict[str, Any] = None):
        """结束计时"""
        if not self._enabled or name not in self._active_timers:
            return 0.0

        end_time = time.perf_counter()
        start_time = self._active_timers.pop(name)
        duration = end_time - start_time

        record = TimingRecord(
            name=name,
            start_time=start_time,
            end_time=end_time,
            metadata=metadata or {}
        )
        self.records.append(record)

        if name not in self.phases:
            self.phases[name] = PhaseStatistics(name=name)
        self.phases[name].add(duration)

        return duration

    @contextmanager
    def profile(self, name: str, metadata: Dict[str, Any] = None):
        """上下文管理器形式的计时"""
        self.start(name, metadata)
        try:
            yield self
        finally:
            self.end(name, metadata)

    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        total_time = sum(p.total_time for p in self.phases.values())

        summary = {
            "profiler_name": self.name,
            "total_profiled_time": total_time,
            "phases": {}
        }

        for name, phase in sorted(self.phases.items(), key=lambda x: -x[1].total_time):
            summary["phases"][name] = {
                "total_time": phase.total_time,
                "count": phase.count,
                "avg_time": phase.avg_time,
                "min_time": phase.min_time if phase.min_time != float('inf') else 0,
                "max_time": phase.max_time,
                "percentage": (phase.total_time / total_time * 100) if total_time > 0 else 0
            }

        return summary

    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        total_time = summary["total_profiled_time"]

        print(f"\n{'='*80}")
        print(f"性能分析报告: {self.name}")
        print(f"{'='*80}")
        print(f"{'阶段名称':<40} {'总耗时(s)':<12} {'调用次数':<10} {'平均(ms)':<12} {'占比':<8}")
        print(f"{'-'*80}")

        for name, stats in sorted(summary["phases"].items(), key=lambda x: -x[1]["total_time"]):
            print(f"{name:<40} {stats['total_time']:<12.3f} {stats['count']:<10} "
                  f"{stats['avg_time']*1000:<12.3f} {stats['percentage']:<7.1f}%")

        print(f"{'-'*80}")
        print(f"{'总计':<40} {total_time:<12.3f}")
        print(f"{'='*80}\n")

    def save_to_file(self, filepath: str):
        """保存性能数据到文件"""
        data = {
            "summary": self.get_summary(),
            "records": [
                {
                    "name": r.name,
                    "duration": r.duration,
                    "metadata": r.metadata
                }
                for r in self.records
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"性能数据已保存: {filepath}")

    def reset(self):
        """重置所有数据"""
        self.records.clear()
        self.phases.clear()
        self._active_timers.clear()


class MetaheuristicProfiler:
    """专门用于元启发式算法的性能分析器"""

    def __init__(self):
        self.profiler = PerformanceProfiler("metaheuristic")
        self.evaluation_stats = {
            "total_evaluations": 0,
            "total_tasks_evaluated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "constraint_check_calls": 0,
            "constraint_check_time": 0.0
        }

    def profile_evaluation(self, solution_idx: int, task_count: int):
        """记录一次解评估"""
        self.evaluation_stats["total_evaluations"] += 1
        self.evaluation_stats["total_tasks_evaluated"] += task_count

    def profile_constraint_check(self, duration: float):
        """记录一次约束检查"""
        self.evaluation_stats["constraint_check_calls"] += 1
        self.evaluation_stats["constraint_check_time"] += duration

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """获取评估统计摘要"""
        stats = self.evaluation_stats.copy()
        if stats["total_evaluations"] > 0:
            stats["avg_tasks_per_evaluation"] = stats["total_tasks_evaluated"] / stats["total_evaluations"]
            stats["avg_constraint_check_time_ms"] = (
                stats["constraint_check_time"] / stats["constraint_check_calls"] * 1000
                if stats["constraint_check_calls"] > 0 else 0
            )
        return stats

    def print_full_report(self):
        """打印完整报告"""
        self.profiler.print_summary()

        print(f"\n{'='*80}")
        print("评估统计")
        print(f"{'='*80}")
        stats = self.get_evaluation_summary()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        print(f"{'='*80}\n")
