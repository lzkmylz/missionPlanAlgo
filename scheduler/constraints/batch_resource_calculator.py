"""
批量资源约束计算器 - Numba向量化优化

功能: 批量处理多个候选的资源约束检查，使用Numba JIT加速
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
class BatchResourceCandidate:
    """资源约束批量检查候选"""
    sat_id: str
    power_needed: float = 0.0  # 所需电量
    storage_needed: float = 0.0  # 所需存储
    storage_produced: float = 0.0  # 产生数据量（成像任务）


@dataclass
class BatchResourceResult:
    """资源约束批量检查结果"""
    feasible: bool
    power_feasible: bool
    storage_feasible: bool
    power_after: float
    storage_after: float
    reason: Optional[str] = None


class BatchResourceData:
    """批量资源数据容器"""

    def __init__(self, n_candidates: int, n_satellites: int = 10):
        self.n = n_candidates
        self.n_sats = n_satellites

        # 候选资源需求
        self.power_needed = np.zeros(n_candidates, dtype=np.float64)
        self.storage_needed = np.zeros(n_candidates, dtype=np.float64)
        self.storage_produced = np.zeros(n_candidates, dtype=np.float64)
        self.candidate_sat_idxs = np.zeros(n_candidates, dtype=np.int32)

        # 卫星当前资源状态
        self.sat_power = np.zeros(n_satellites, dtype=np.float64)
        self.sat_storage = np.zeros(n_satellites, dtype=np.float64)
        self.sat_power_capacity = np.zeros(n_satellites, dtype=np.float64)
        self.sat_storage_capacity = np.zeros(n_satellites, dtype=np.float64)

        # 输出数组
        self.out_feasible = np.ones(n_candidates, dtype=np.bool_)
        self.out_power_feasible = np.ones(n_candidates, dtype=np.bool_)
        self.out_storage_feasible = np.ones(n_candidates, dtype=np.bool_)
        self.out_power_after = np.zeros(n_candidates, dtype=np.float64)
        self.out_storage_after = np.zeros(n_candidates, dtype=np.float64)


if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def batch_check_resource_numba(
        power_needed: np.ndarray,        # [n_candidates]
        storage_needed: np.ndarray,      # [n_candidates]
        storage_produced: np.ndarray,    # [n_candidates]
        candidate_sat_idxs: np.ndarray,  # [n_candidates]
        sat_power: np.ndarray,           # [n_satellites]
        sat_storage: np.ndarray,         # [n_satellites]
        sat_power_capacity: np.ndarray,  # [n_satellites]
        sat_storage_capacity: np.ndarray, # [n_satellites]
        out_feasible: np.ndarray,        # [n_candidates]
        out_power_feasible: np.ndarray,  # [n_candidates]
        out_storage_feasible: np.ndarray, # [n_candidates]
        out_power_after: np.ndarray,     # [n_candidates]
        out_storage_after: np.ndarray    # [n_candidates]
    ):
        """批量资源约束检查 - Numba并行版本

        Args:
            power_needed: 所需电量数组 [n_candidates]
            storage_needed: 所需存储数组 [n_candidates]
            storage_produced: 产生数据量数组 [n_candidates]
            candidate_sat_idxs: 候选卫星索引 [n_candidates]
            sat_power: 卫星当前电量 [n_satellites]
            sat_storage: 卫星当前存储 [n_satellites]
            sat_power_capacity: 卫星电量容量 [n_satellites]
            sat_storage_capacity: 卫星存储容量 [n_satellites]
            out_feasible: 输出可行性数组 [n_candidates]
            out_power_feasible: 输出电量可行性 [n_candidates]
            out_storage_feasible: 输出存储可行性 [n_candidates]
            out_power_after: 输出电量使用后 [n_candidates]
            out_storage_after: 输出存储使用后 [n_candidates]
        """
        n_candidates = len(power_needed)

        for i in prange(n_candidates):
            sat_idx = candidate_sat_idxs[i]

            # 检查电量约束
            power_ok = True
            if power_needed[i] > 0:
                power_ok = sat_power[sat_idx] >= power_needed[i]
                out_power_after[i] = sat_power[sat_idx] - power_needed[i]
            else:
                out_power_after[i] = sat_power[sat_idx]

            # 检查存储约束
            storage_ok = True
            new_storage = sat_storage[sat_idx] + storage_produced[i]
            if new_storage > sat_storage_capacity[sat_idx]:
                storage_ok = False
            out_storage_after[i] = new_storage

            out_power_feasible[i] = power_ok
            out_storage_feasible[i] = storage_ok
            out_feasible[i] = power_ok and storage_ok


class BatchResourceCalculator:
    """批量资源计算器"""

    def __init__(self):
        self.use_numba = HAS_NUMBA

    def prepare_batch_data(
        self,
        candidates: List[BatchResourceCandidate],
        satellite_states: Dict[str, Dict[str, float]],
        sat_id_to_idx: Optional[Dict[str, int]] = None
    ) -> BatchResourceData:
        """准备批量资源检查数据

        Args:
            candidates: 资源检查候选列表
            satellite_states: 卫星当前资源状态 {sat_id: {'power': x, 'storage': y, 'power_capacity': pc, 'storage_capacity': sc}}
            sat_id_to_idx: 卫星ID到索引的映射（可选）

        Returns:
            BatchResourceData: 批量数据容器
        """
        # 创建卫星ID映射
        if sat_id_to_idx is None:
            sat_id_to_idx = {}
            idx = 0
            for sat_id in satellite_states.keys():
                sat_id_to_idx[sat_id] = idx
                idx += 1
            for cand in candidates:
                if cand.sat_id not in sat_id_to_idx:
                    sat_id_to_idx[cand.sat_id] = idx
                    idx += 1

        n_sats = len(sat_id_to_idx)
        data = BatchResourceData(len(candidates), n_sats)
        data.sat_id_to_idx = sat_id_to_idx
        data.idx_to_sat_id = {v: k for k, v in sat_id_to_idx.items()}

        # 填充卫星资源状态
        for sat_id, state in satellite_states.items():
            idx = sat_id_to_idx.get(sat_id)
            if idx is not None:
                data.sat_power[idx] = state.get('power', 0.0)
                data.sat_storage[idx] = state.get('storage', 0.0)
                data.sat_power_capacity[idx] = state.get('power_capacity', 1000.0)
                data.sat_storage_capacity[idx] = state.get('storage_capacity', 100.0)

        # 填充候选数据
        for i, cand in enumerate(candidates):
            data.power_needed[i] = cand.power_needed
            data.storage_needed[i] = cand.storage_needed
            data.storage_produced[i] = cand.storage_produced
            data.candidate_sat_idxs[i] = sat_id_to_idx.get(cand.sat_id, -1)

        return data

    def compute_batch(self, data: BatchResourceData) -> List[BatchResourceResult]:
        """执行批量资源计算

        Args:
            data: 批量数据容器

        Returns:
            BatchResourceResult列表
        """
        if self.use_numba and data.n > 0:
            try:
                batch_check_resource_numba(
                    data.power_needed,
                    data.storage_needed,
                    data.storage_produced,
                    data.candidate_sat_idxs,
                    data.sat_power,
                    data.sat_storage,
                    data.sat_power_capacity,
                    data.sat_storage_capacity,
                    data.out_feasible,
                    data.out_power_feasible,
                    data.out_storage_feasible,
                    data.out_power_after,
                    data.out_storage_after
                )
            except Exception as e:
                logger.warning(f"Numba batch resource computation failed: {e}, falling back to Python")
                self._compute_batch_python(data)
        else:
            self._compute_batch_python(data)

        # 转换结果为Python对象
        results = []
        for i in range(data.n):
            reason = None
            if not data.out_feasible[i]:
                reasons = []
                if not data.out_power_feasible[i]:
                    reasons.append(f"insufficient power (need {data.power_needed[i]:.1f}, have {data.sat_power[data.candidate_sat_idxs[i]]:.1f})")
                if not data.out_storage_feasible[i]:
                    storage_cap = data.sat_storage_capacity[data.candidate_sat_idxs[i]]
                    new_storage = data.out_storage_after[i]
                    reasons.append(f"storage overflow ({new_storage:.1f} > {storage_cap:.1f})")
                reason = "; ".join(reasons)

            results.append(BatchResourceResult(
                feasible=data.out_feasible[i],
                power_feasible=data.out_power_feasible[i],
                storage_feasible=data.out_storage_feasible[i],
                power_after=data.out_power_after[i],
                storage_after=data.out_storage_after[i],
                reason=reason
            ))

        return results

    def _compute_batch_python(self, data: BatchResourceData):
        """Python回退版本（当Numba不可用时）"""
        for i in range(data.n):
            sat_idx = data.candidate_sat_idxs[i]

            # 检查电量约束
            power_ok = True
            if data.power_needed[i] > 0:
                power_ok = data.sat_power[sat_idx] >= data.power_needed[i]
                data.out_power_after[i] = data.sat_power[sat_idx] - data.power_needed[i]
            else:
                data.out_power_after[i] = data.sat_power[sat_idx]

            # 检查存储约束
            storage_ok = True
            new_storage = data.sat_storage[sat_idx] + data.storage_produced[i]
            if new_storage > data.sat_storage_capacity[sat_idx]:
                storage_ok = False
            data.out_storage_after[i] = new_storage

            data.out_power_feasible[i] = power_ok
            data.out_storage_feasible[i] = storage_ok
            data.out_feasible[i] = power_ok and storage_ok
