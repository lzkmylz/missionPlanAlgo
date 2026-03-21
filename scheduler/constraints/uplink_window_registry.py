"""
指令上注窗口注册表

从预计算的可见性窗口（GS: / RELAY: / ISL: 前缀）构建统一的上注弧段索引，
支持按卫星和渠道优先级查询可用弧段。
"""

import bisect
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set

from .uplink_channel_type import UplinkChannelType, UplinkPass

logger = logging.getLogger(__name__)

# 默认渠道优先级（优先地面站，其次中继，最后ISL）
DEFAULT_CHANNEL_PRIORITY = [
    UplinkChannelType.GROUND_STATION.value,
    UplinkChannelType.RELAY_SATELLITE.value,
    UplinkChannelType.ISL.value,
]


class UplinkWindowRegistry:
    """指令上注窗口注册表

    复用 Java Orekit 预计算的可见性窗口，将 GS: / RELAY: / ISL: 前缀窗口
    统一转换为 UplinkPass 对象，提供按卫星 + 渠道优先级的快速查询。

    不做任何重新计算，完全依赖已有的高精度 HPOP 预计算结果。

    用法:
        registry = UplinkWindowRegistry()
        registry.load_from_window_cache(scheduler.window_cache)
        pass_ = registry.find_feasible_pass(
            sat_id='SAT-01',
            latest_deadline=task_start - timedelta(seconds=300),
            min_duration_s=30.0,
            channel_priority=['ground_station', 'relay_satellite', 'isl'],
        )
    """

    def __init__(self) -> None:
        # Dict[satellite_id, List[UplinkPass]]，按 end_time 升序排列
        self._passes: Dict[str, List[UplinkPass]] = defaultdict(list)
        # 与 _passes 同步的 end_time 列表，用于 bisect 二分查找
        self._pass_end_times: Dict[str, List[datetime]] = {}

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load_from_window_cache(
        self,
        window_cache,
        satellite_ids: Optional[Set[str]] = None,
    ) -> None:
        """从调度器 window_cache 加载三种渠道的上注弧段。

        Args:
            window_cache: 调度器的 window_cache 对象（含 _windows 属性）
            satellite_ids: 只加载这些卫星的窗口；None 表示加载全部
        """
        if window_cache is None:
            logger.warning("UplinkWindowRegistry: window_cache 不可用，上注约束将不生效")
            return

        # 优先使用公共接口，降级回退到私有属性
        if hasattr(window_cache, 'get_all_windows'):
            all_windows = window_cache.get_all_windows()
        elif hasattr(window_cache, '_windows'):
            all_windows = window_cache._windows
        else:
            logger.warning("window_cache 缺少有效的窗口访问接口")
            return

        gs_count = relay_count = isl_count = 0

        for (sat_id, target_id), windows in all_windows.items():
            if satellite_ids is not None and sat_id not in satellite_ids:
                continue
            if not isinstance(target_id, str):
                continue

            if target_id.startswith('GS:'):
                channel_type = UplinkChannelType.GROUND_STATION
                channel_id = target_id[3:]
                overhead_s = 30.0
                rate_attr = 'max_data_rate_mbps'
                rate_default = 10.0
                gs_count += len(windows)

            elif target_id.startswith('RELAY:'):
                channel_type = UplinkChannelType.RELAY_SATELLITE
                channel_id = target_id[6:]
                overhead_s = 30.0
                rate_attr = 'max_data_rate_mbps'
                rate_default = 10.0
                relay_count += len(windows)

            elif target_id.startswith('ISL:'):
                channel_type = UplinkChannelType.ISL
                channel_id = target_id[4:]
                overhead_s = 5.0
                # ISL 窗口使用 max_data_rate（无 _mbps 后缀），默认 10000 Mbps
                rate_attr = 'max_data_rate'
                rate_default = 10000.0
                isl_count += len(windows)

            else:
                continue  # 成像目标窗口，跳过

            for win in windows:
                # MEDIUM-1: 直接使用渠道对应的属性名，避免 ISL 被 max_data_rate_mbps 字段遮蔽
                data_rate = getattr(win, rate_attr, rate_default)
                # HIGH-1: 统一归一化为 naive UTC，防止与 bisect 比较时发生
                # TypeError（Orekit 窗口携带 tzinfo=UTC，而 deadline 为 naive）
                start = win.start_time
                end = win.end_time
                if getattr(start, 'tzinfo', None) is not None:
                    start = start.replace(tzinfo=None)
                if getattr(end, 'tzinfo', None) is not None:
                    end = end.replace(tzinfo=None)
                uplink_pass = UplinkPass(
                    channel_type=channel_type,
                    channel_id=channel_id,
                    satellite_id=sat_id,
                    start_time=start,
                    end_time=end,
                    max_data_rate_mbps=float(data_rate),
                    switching_overhead_s=overhead_s,
                )
                self._passes[sat_id].append(uplink_pass)

        # 按 end_time 排序，同步更新二分索引
        for sat_id in self._passes:
            self._passes[sat_id].sort(key=lambda p: p.end_time)
            self._pass_end_times[sat_id] = [p.end_time for p in self._passes[sat_id]]

        logger.info(
            "UplinkWindowRegistry 本次加载: GS=%d, RELAY=%d, ISL=%d 条弧段；"
            "累计 %d 颗卫星，%d 条弧段",
            gs_count, relay_count, isl_count, len(self._passes), self.pass_count(),
        )

        # 对无弧段卫星发出警告
        if satellite_ids:
            missing = satellite_ids - set(self._passes.keys())
            if missing:
                logger.warning(
                    "以下卫星没有任何上注弧段，上注约束将拒绝其所有任务: %s",
                    sorted(missing),
                )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def find_feasible_pass(
        self,
        sat_id: str,
        latest_deadline: datetime,
        min_duration_s: float = 30.0,
        channel_priority: Optional[List[str]] = None,
    ) -> Optional[UplinkPass]:
        """查找满足条件的最优上注弧段。

        按 channel_priority 顺序逐渠道搜索，返回同渠道内最接近 deadline 的弧段
        （即 end_time 最大且 <= latest_deadline 的弧段）。

        Args:
            sat_id: 目标卫星ID
            latest_deadline: 弧段结束时间不得晚于此时刻（通常 = task_start - command_lead_time）
            min_duration_s: 弧段可用时长下限（秒，不含切换开销）
            channel_priority: 渠道优先级列表（字符串），None 使用默认顺序

        Returns:
            满足条件的 UplinkPass，若无则返回 None
        """
        passes = self._passes.get(sat_id)
        if not passes:
            return None

        # MEDIUM-1: 归一化 deadline 为 naive UTC，与存储的 end_time 保持一致。
        # load_from_window_cache 已将窗口时间归一化，但调用方可能直接传入 aware datetime。
        if getattr(latest_deadline, 'tzinfo', None) is not None:
            latest_deadline = latest_deadline.replace(tzinfo=None)

        priority = channel_priority or DEFAULT_CHANNEL_PRIORITY

        # HIGH-2: 防御性守卫——若索引与 _passes 不同步则记录错误并降级为空结果
        end_times = self._pass_end_times.get(sat_id)
        if end_times is None:
            logger.error(
                "find_feasible_pass: %s 存在弧段数据但 _pass_end_times 索引缺失，"
                "请检查 load_from_window_cache 是否正常完成",
                sat_id,
            )
            return None

        # 使用二分查找找到所有满足 end_time <= latest_deadline 的弧段上界
        # _pass_end_times 与 _passes 同步且升序，bisect_right 给出插入点
        cut = bisect.bisect_right(end_times, latest_deadline)
        # passes[:cut] 中的所有弧段 end_time <= latest_deadline

        if cut == 0:
            return None  # 没有任何弧段在 deadline 之前结束

        for channel_str in priority:
            try:
                channel_type = UplinkChannelType(channel_str)
            except ValueError:
                logger.debug("未知渠道类型: %s，已跳过", channel_str)
                continue

            # 从 cut-1 向前扫描：第一个满足条件的即为 end_time 最大的候选
            for j in range(cut - 1, -1, -1):
                p = passes[j]
                if p.channel_type != channel_type:
                    continue
                if p.usable_duration_s < min_duration_s:
                    continue
                return p  # end_time 已满足（j < cut），usable 满足，直接返回

        return None

    def get_passes_for_satellite(self, sat_id: str) -> List[UplinkPass]:
        """获取某卫星的全部上注弧段副本（按 end_time 升序）"""
        return list(self._passes.get(sat_id, []))

    def get_all_passes(self) -> Dict[str, List[UplinkPass]]:
        """获取全部弧段的副本"""
        return {k: list(v) for k, v in self._passes.items()}

    def is_empty(self) -> bool:
        """注册表是否为空"""
        return len(self._passes) == 0

    def satellite_count(self) -> int:
        """已登记弧段的卫星数量"""
        return len(self._passes)

    def pass_count(self, sat_id: Optional[str] = None) -> int:
        """弧段数量（指定卫星或全部）"""
        if sat_id is not None:
            return len(self._passes.get(sat_id, []))
        return sum(len(v) for v in self._passes.values())
