"""
批量指令上注约束检查器

整合 UplinkWindowRegistry + BatchUplinkCalculator，提供：
  - resolve_uplink_duration(): 两级优先级解析（成像模式级覆盖 > 卫星默认值）
  - check_uplink_feasibility_batch(): 批量上注可行性检查
"""

import logging
from typing import Dict, List, Optional

from .batch_uplink_calculator import BatchUplinkCalculator, BatchUplinkCandidate, BatchUplinkResult
from .uplink_window_registry import UplinkWindowRegistry

logger = logging.getLogger(__name__)

# 无卫星配置时的降级默认值
_FALLBACK_UPLINK_DURATION_S = 30.0
_FALLBACK_COMMAND_LEAD_TIME_S = 300.0


class BatchUplinkConstraintChecker:
    """批量指令上注约束检查器

    使用方法::

        checker = BatchUplinkConstraintChecker(registry)

        candidates = [
            BatchUplinkCandidate(
                sat_id='SAT-01',
                task_start=task.window_start,
                required_uplink_s=checker.resolve_uplink_duration(satellite, imaging_mode),
                command_lead_time_s=300.0,
            )
            for task, satellite, imaging_mode in pending_tasks
        ]
        results = checker.check_uplink_feasibility_batch(candidates)
    """

    def __init__(
        self,
        registry: UplinkWindowRegistry,
        default_command_lead_time_s: float = _FALLBACK_COMMAND_LEAD_TIME_S,
    ) -> None:
        self._registry = registry
        self._calculator = BatchUplinkCalculator()
        self._default_command_lead_time_s = default_command_lead_time_s

        if registry.is_empty():
            logger.warning(
                "BatchUplinkConstraintChecker: 注册表为空，"
                "所有上注约束检查将返回不可行"
            )

    # ------------------------------------------------------------------
    # 上注时长解析（两级优先级）
    # ------------------------------------------------------------------

    def resolve_uplink_duration(
        self,
        satellite,  # Satellite
        imaging_mode: Optional[str] = None,
    ) -> float:
        """解析有效上注时长。

        优先级：成像模式 characteristics['uplink_duration_s'] > 卫星默认值。

        Args:
            satellite: Satellite 对象
            imaging_mode: 成像模式名称（可为 None）

        Returns:
            有效上注时长（秒）
        """
        # 1. 成像模式级覆盖
        if imaging_mode is not None:
            try:
                payload_cfg = getattr(satellite, 'payload_config', None) or getattr(
                    satellite.capabilities, 'payload_config', None
                )
                if payload_cfg is not None:
                    mode_cfg = payload_cfg.get_mode_config(imaging_mode)
                    override = mode_cfg.get_uplink_duration_s()
                    if override is not None:
                        return float(override)
            except (ValueError, AttributeError, KeyError) as exc:
                logger.debug(
                    "resolve_uplink_duration: 成像模式 '%s' 未找到或无覆盖值，"
                    "回退到卫星默认值。详情: %s",
                    imaging_mode, exc,
                )

        # 2. 卫星默认值
        try:
            return float(satellite.capabilities.min_uplink_duration_per_task)
        except AttributeError:
            logger.debug(
                "resolve_uplink_duration: 卫星 '%s' 缺少 min_uplink_duration_per_task 属性，"
                "使用硬编码默认值 %.1fs",
                getattr(satellite, 'id', repr(satellite)),
                _FALLBACK_UPLINK_DURATION_S,
            )

        return _FALLBACK_UPLINK_DURATION_S

    # ------------------------------------------------------------------
    # 批量检查
    # ------------------------------------------------------------------

    def check_uplink_feasibility_batch(
        self,
        candidates: List[BatchUplinkCandidate],
        channel_priority: Optional[List[str]] = None,
    ) -> List[BatchUplinkResult]:
        """批量检查上注可行性。

        Args:
            candidates: 候选列表
            channel_priority: 渠道优先级（字符串列表），None 使用默认顺序

        Returns:
            与 candidates 等长的结果列表
        """
        if not candidates:
            return []

        if self._registry.is_empty():
            return [
                BatchUplinkResult(feasible=False, reason="uplink_registry_empty")
                for _ in candidates
            ]

        return self._calculator.check_batch(candidates, self._registry, channel_priority)

    def check_single(
        self,
        sat_id: str,
        task_start,
        required_uplink_s: Optional[float] = None,
        command_lead_time_s: Optional[float] = None,
        channel_priority: Optional[List[str]] = None,
    ) -> BatchUplinkResult:
        """单候选上注可行性检查（调试/开发用）"""
        candidate = BatchUplinkCandidate(
            sat_id=sat_id,
            task_start=task_start,
            required_uplink_s=required_uplink_s if required_uplink_s is not None
                else _FALLBACK_UPLINK_DURATION_S,
            command_lead_time_s=command_lead_time_s if command_lead_time_s is not None
                else self._default_command_lead_time_s,
        )
        results = self.check_uplink_feasibility_batch([candidate], channel_priority)
        return results[0]
