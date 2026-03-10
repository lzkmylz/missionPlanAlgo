"""
批量时间冲突计算器 - Numba向量化优化

功能: 批量处理多个候选的时间冲突检查，使用Numba JIT加速
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Numba JIT优化支持
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def prange(*args):
        return range(*args)

logger = logging.getLogger(__name__)


@dataclass
class BatchTimeConflictCandidate:
    """时间冲突批量检查候选"""
    sat_id: str
    window_start: datetime
    window_end: datetime
    imaging_duration: float = 10.0  # 秒


@dataclass
class BatchTimeConflictResult:
    """时间冲突批量检查结果"""
    feasible: bool
    conflict_count: int
    conflict_tasks: List[Dict[str, Any]] = None
    reason: Optional[str] = None


class BatchTimeConflictData:
    """批量时间冲突数据容器"""

    def __init__(self, n_candidates: int, n_existing_tasks: int = 100):
        self.n = n_candidates
        self.n_existing = n_existing_tasks

        # 候选窗口时间（秒，相对于base_time）
        self.candidate_starts = np.zeros(n_candidates, dtype=np.float64)
        self.candidate_ends = np.zeros(n_candidates, dtype=np.float64)
        self.candidate_sat_idxs = np.zeros(n_candidates, dtype=np.int32)

        # 已有任务时间（秒，相对于base_time）
        self.existing_starts = np.zeros(n_existing_tasks, dtype=np.float64)
        self.existing_ends = np.zeros(n_existing_tasks, dtype=np.float64)
        self.existing_sat_idxs = np.zeros(n_existing_tasks, dtype=np.int32)

        # 卫星ID映射
        self.sat_id_to_idx: Dict[str, int] = {}
        self.idx_to_sat_id: Dict[int, str] = {}

        # 输出数组
        self.out_feasible = np.ones(n_candidates, dtype=np.bool_)
        self.out_conflict_count = np.zeros(n_candidates, dtype=np.int32)


if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def batch_check_time_conflict_numba(
        candidate_starts: np.ndarray,  # [n_candidates]
        candidate_ends: np.ndarray,    # [n_candidates]
        candidate_sat_idxs: np.ndarray, # [n_candidates]
        existing_starts: np.ndarray,   # [n_existing]
        existing_ends: np.ndarray,     # [n_existing]
        existing_sat_idxs: np.ndarray, # [n_existing]
        out_feasible: np.ndarray,      # [n_candidates]
        out_conflict_count: np.ndarray # [n_candidates]
    ):
        """批量时间冲突检查 - Numba并行版本

        Args:
            candidate_starts: 候选开始时间数组 [n_candidates]
            candidate_ends: 候选结束时间数组 [n_candidates]
            candidate_sat_idxs: 候选卫星索引 [n_candidates]
            existing_starts: 已有任务开始时间 [n_existing]
            existing_ends: 已有任务结束时间 [n_existing]
            existing_sat_idxs: 已有任务卫星索引 [n_existing]
            out_feasible: 输出可行性数组 [n_candidates]
            out_conflict_count: 输出冲突次数 [n_candidates]
        """
        n_candidates = len(candidate_starts)
        n_existing = len(existing_starts)

        for i in prange(n_candidates):
            conflicts = 0
            cand_start = candidate_starts[i]
            cand_end = candidate_ends[i]
            cand_sat_idx = candidate_sat_idxs[i]

            for j in range(n_existing):
                # 只检查同一卫星的任务
                if existing_sat_idxs[j] != cand_sat_idx:
                    continue

                # 检查时间重叠: not (cand_end <= existing_start or cand_start >= existing_end)
                if cand_end > existing_starts[j] and cand_start < existing_ends[j]:
                    conflicts += 1

            out_conflict_count[i] = conflicts
            out_feasible[i] = (conflicts == 0)


class BatchTimeConflictCalculator:
    """批量时间冲突计算器"""

    def __init__(self):
        self.use_numba = HAS_NUMBA

    def prepare_batch_data(
        self,
        candidates: List[BatchTimeConflictCandidate],
        existing_tasks: List[Dict[str, Any]],
        sat_id_to_idx: Optional[Dict[str, int]] = None
    ) -> BatchTimeConflictData:
        """准备批量时间冲突检查数据

        Args:
            candidates: 时间冲突检查候选列表
            existing_tasks: 已调度任务列表
            sat_id_to_idx: 卫星ID到索引的映射（可选）

        Returns:
            BatchTimeConflictData: 批量数据容器
        """
        # 创建卫星ID映射
        if sat_id_to_idx is None:
            sat_id_to_idx = {}
            idx = 0
            for cand in candidates:
                if cand.sat_id not in sat_id_to_idx:
                    sat_id_to_idx[cand.sat_id] = idx
                    idx += 1
            for task in existing_tasks:
                sat_id = task.get('satellite_id') or task.get('sat_id')
                if sat_id and sat_id not in sat_id_to_idx:
                    sat_id_to_idx[sat_id] = idx
                    idx += 1

        data = BatchTimeConflictData(len(candidates), len(existing_tasks))
        data.sat_id_to_idx = sat_id_to_idx
        data.idx_to_sat_id = {v: k for k, v in sat_id_to_idx.items()}

        # 使用第一个候选的window_start作为参考时间（处理时区）
        base_time = None
        if candidates:
            base_time = candidates[0].window_start

        # 填充候选数据
        candidate_sat_idxs = []
        for i, cand in enumerate(candidates):
            if base_time is not None:
                data.candidate_starts[i] = (cand.window_start - base_time).total_seconds()
                data.candidate_ends[i] = (cand.window_end - base_time).total_seconds()
            else:
                # 回退：直接使用timestamp
                data.candidate_starts[i] = cand.window_start.timestamp()
                data.candidate_ends[i] = cand.window_end.timestamp()
            candidate_sat_idxs.append(sat_id_to_idx.get(cand.sat_id, -1))
        data.candidate_sat_idxs = np.array(candidate_sat_idxs, dtype=np.int32)

        # 填充已有任务数据
        for j, task in enumerate(existing_tasks):
            sat_id = task.get('satellite_id') or task.get('sat_id')
            task_start = task.get('start')
            task_end = task.get('end')

            if sat_id and task_start and task_end:
                data.existing_sat_idxs[j] = sat_id_to_idx.get(sat_id, -1)
                if isinstance(task_start, datetime):
                    if base_time is not None:
                        data.existing_starts[j] = (task_start - base_time).total_seconds()
                    else:
                        data.existing_starts[j] = task_start.timestamp()
                else:
                    data.existing_starts[j] = float(task_start)

                if isinstance(task_end, datetime):
                    if base_time is not None:
                        data.existing_ends[j] = (task_end - base_time).total_seconds()
                    else:
                        data.existing_ends[j] = task_end.timestamp()
                else:
                    data.existing_ends[j] = float(task_end)

        return data

    def compute_batch(self, data: BatchTimeConflictData) -> List[BatchTimeConflictResult]:
        """执行批量时间冲突计算

        Args:
            data: 批量数据容器

        Returns:
            BatchTimeConflictResult列表
        """
        if self.use_numba and data.n > 0:
            try:
                batch_check_time_conflict_numba(
                    data.candidate_starts,
                    data.candidate_ends,
                    data.candidate_sat_idxs,
                    data.existing_starts,
                    data.existing_ends,
                    data.existing_sat_idxs,
                    data.out_feasible,
                    data.out_conflict_count
                )
            except Exception as e:
                logger.warning(f"Numba batch time conflict computation failed: {e}, falling back to Python")
                self._compute_batch_python(data, data.candidate_sat_idxs)
        else:
            self._compute_batch_python(data, data.candidate_sat_idxs)

        # 转换结果为Python对象
        results = []
        for i in range(data.n):
            reason = None
            if not data.out_feasible[i]:
                reason = f"Time conflict with {data.out_conflict_count[i]} existing task(s)"

            results.append(BatchTimeConflictResult(
                feasible=data.out_feasible[i],
                conflict_count=data.out_conflict_count[i],
                reason=reason
            ))

        return results

    def _compute_batch_python(
        self,
        data: BatchTimeConflictData,
        candidate_sat_idxs: np.ndarray
    ):
        """Python回退版本（当Numba不可用时）"""
        for i in range(data.n):
            conflicts = 0
            cand_start = data.candidate_starts[i]
            cand_end = data.candidate_ends[i]
            cand_sat_idx = candidate_sat_idxs[i]

            for j in range(data.n_existing):
                # 只检查同一卫星的任务
                if data.existing_sat_idxs[j] != cand_sat_idx:
                    continue

                # 检查时间重叠
                if cand_end > data.existing_starts[j] and cand_start < data.existing_ends[j]:
                    conflicts += 1

            data.out_conflict_count[i] = conflicts
            data.out_feasible[i] = (conflicts == 0)
