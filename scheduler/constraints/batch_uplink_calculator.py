"""
批量指令上注可行性计算器 - Numba向量化优化

检查每个候选成像任务在执行前是否存在满足条件的上注弧段。
判断标准：
  - 弧段结束时间 <= task_start - command_lead_time_s
  - 弧段可用时长 >= required_uplink_s（扣除切换开销后）
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone as _timezone
from typing import List, Optional

import numpy as np

# Numba JIT 支持
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
class BatchUplinkCandidate:
    """指令上注批量检查候选"""
    sat_id: str
    task_start: datetime             # 任务开始时刻
    required_uplink_s: float = 30.0  # 所需上注时长（秒）
    command_lead_time_s: float = 300.0  # 指令必须提前到达的最短时间（秒）


@dataclass
class BatchUplinkResult:
    """指令上注批量检查结果"""
    feasible: bool
    channel_type: Optional[str] = None   # 'ground_station' | 'relay_satellite' | 'isl'
    channel_id: Optional[str] = None     # 渠道ID
    pass_end: Optional[datetime] = None  # 所用弧段的结束时刻
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Numba 核心：对单颗卫星、单个渠道的弧段池执行批量可行性扫描
# ---------------------------------------------------------------------------

if HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _batch_check_uplink_numba(
        n_candidates: int,
        # 每个候选的 deadline（POSIX秒）= task_start_ts - command_lead_time_s
        deadlines: np.ndarray,          # float64[n_candidates]
        required_s: np.ndarray,         # float64[n_candidates]，所需可用时长
        # 弧段池（同一卫星同一渠道的所有弧段）
        pass_ends: np.ndarray,          # float64[n_passes]
        pass_usable: np.ndarray,        # float64[n_passes]，可用时长（已扣除切换开销）
        # 输出：每候选对应弧段池中最佳弧段的索引，-1 表示无可用弧段
        out_best_idx: np.ndarray,       # int32[n_candidates]
    ) -> None:
        for i in prange(n_candidates):
            deadline_i = deadlines[i]
            req_i = required_s[i]
            best_j = -1
            best_end = -1.0
            for j in range(len(pass_ends)):
                if pass_ends[j] <= deadline_i and pass_usable[j] >= req_i:
                    # 选 end_time 最接近 deadline 的（最靠近任务执行时间）
                    if pass_ends[j] > best_end:
                        best_end = pass_ends[j]
                        best_j = j
            out_best_idx[i] = best_j
else:
    def _batch_check_uplink_numba(
        n_candidates, deadlines, required_s, pass_ends, pass_usable, out_best_idx
    ):
        for i in range(n_candidates):
            deadline_i = deadlines[i]
            req_i = required_s[i]
            best_j = -1
            best_end = -1.0
            for j in range(len(pass_ends)):
                if pass_ends[j] <= deadline_i and pass_usable[j] >= req_i:
                    if pass_ends[j] > best_end:
                        best_end = pass_ends[j]
                        best_j = j
            out_best_idx[i] = best_j


_UTC = _timezone.utc
_EPOCH = datetime(1970, 1, 1, tzinfo=_UTC)


def _to_posix(dt: datetime) -> float:
    """datetime → POSIX 秒（float）。

    项目约定：无时区信息（naive）的 datetime 均视为 UTC。
    使用固定纪元相减而非 datetime.timestamp()，避免受本机时区设置影响。
    """
    if dt.tzinfo is None:
        return (dt - _EPOCH.replace(tzinfo=None)).total_seconds()
    return (dt.astimezone(_UTC) - _EPOCH).total_seconds()


class BatchUplinkCalculator:
    """批量上注可行性计算器

    对 n 个候选批量判断是否存在可用上注弧段。
    对于批次 >= NUMBA_THRESHOLD 的情况，调用 Numba 加速路径；
    较小批次直接使用 Python 循环（避免 JIT 编译开销）。
    """

    NUMBA_THRESHOLD = 10  # 小于此批次大小时跳过 Numba

    def check_batch(
        self,
        candidates: List[BatchUplinkCandidate],
        registry,  # UplinkWindowRegistry
        channel_priority: Optional[List[str]] = None,
    ) -> List[BatchUplinkResult]:
        """批量检查上注可行性。

        Args:
            candidates: 候选列表
            registry: UplinkWindowRegistry 实例
            channel_priority: 渠道优先级（字符串列表），None 使用默认

        Returns:
            与 candidates 等长的 BatchUplinkResult 列表
        """
        if not candidates:
            return []

        from .uplink_window_registry import DEFAULT_CHANNEL_PRIORITY
        from .uplink_channel_type import UplinkChannelType

        priority = channel_priority or DEFAULT_CHANNEL_PRIORITY
        results = [
            BatchUplinkResult(feasible=False, reason="no_uplink_pass_found")
            for _ in candidates
        ]

        # 小批次直接调用 Python 路径
        if len(candidates) < self.NUMBA_THRESHOLD or not HAS_NUMBA:
            for i, cand in enumerate(candidates):
                # 统一使用 naive UTC 计算 deadline，避免时区混用
                deadline_ts = _to_posix(cand.task_start) - cand.command_lead_time_s
                deadline = datetime.utcfromtimestamp(deadline_ts)
                pass_ = registry.find_feasible_pass(
                    cand.sat_id, deadline, cand.required_uplink_s, priority
                )
                if pass_ is not None:
                    results[i] = BatchUplinkResult(
                        feasible=True,
                        channel_type=pass_.channel_type.value,
                        channel_id=pass_.channel_id,
                        pass_end=pass_.end_time,
                    )
            return results

        # 大批次：按卫星 × 渠道分组，调用 Numba 加速
        # 索引：候选序号 → 候选在当前卫星组内的位置
        sat_to_indices: dict = {}
        for i, cand in enumerate(candidates):
            sat_to_indices.setdefault(cand.sat_id, []).append(i)

        for sat_id, indices in sat_to_indices.items():
            # 预计算全部候选的 deadline 和 required，供后续各渠道复用
            sat_candidates = [candidates[i] for i in indices]
            all_deadlines = np.array(
                [_to_posix(c.task_start) - c.command_lead_time_s for c in sat_candidates],
                dtype=np.float64,
            )
            all_required = np.array(
                [c.required_uplink_s for c in sat_candidates],
                dtype=np.float64,
            )

            for channel_str in priority:
                # 仅对尚未满足的候选调用 Numba，节省无效计算
                unsatisfied = [li for li in range(len(indices)) if not results[indices[li]].feasible]
                if not unsatisfied:
                    break  # 该卫星所有候选已满足，无需继续

                try:
                    ch_type = UplinkChannelType(channel_str)
                except ValueError:
                    continue

                passes = [
                    p for p in registry.get_passes_for_satellite(sat_id)
                    if p.channel_type == ch_type
                ]
                if not passes:
                    continue

                pass_ends_arr = np.array([_to_posix(p.end_time) for p in passes], dtype=np.float64)
                pass_usable_arr = np.array([p.usable_duration_s for p in passes], dtype=np.float64)

                # 仅取未满足候选的子数组
                unsatisfied_arr = np.array(unsatisfied, dtype=np.int64)
                deadlines_sub = all_deadlines[unsatisfied_arr]
                required_sub = all_required[unsatisfied_arr]
                out_best = np.full(len(unsatisfied), -1, dtype=np.int32)

                _batch_check_uplink_numba(
                    len(unsatisfied),
                    deadlines_sub,
                    required_sub,
                    pass_ends_arr,
                    pass_usable_arr,
                    out_best,
                )

                for sub_i, local_i in enumerate(unsatisfied):
                    best_j = int(out_best[sub_i])
                    if best_j >= 0:
                        global_i = indices[local_i]
                        p = passes[best_j]
                        results[global_i] = BatchUplinkResult(
                            feasible=True,
                            channel_type=ch_type.value,
                            channel_id=p.channel_id,
                            pass_end=p.end_time,
                        )

        return results
