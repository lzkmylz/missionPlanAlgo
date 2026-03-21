"""
ISL (Inter-Satellite Link) 可见性窗口加载与管理

关键设计原则（MANDATORY）：
- ISL可见性窗口数据**必须**来自Java后端预计算（HPOP高精度轨道传播）
- 若未找到预计算文件或文件中无ISL窗口，**立即抛出RuntimeError**
- **禁止**使用任何Python几何回退计算

窗口文件格式（visibility_windows.json）：
    窗口记录中 tgt 字段以 "ISL:" 开头的条目即为ISL窗口：
    {
        "sat": "OPT-01",
        "tgt": "ISL:OPT-02",
        "start": "2026-03-11T00:00:00Z",
        "end": "2026-03-11T00:10:00Z",
        "dur": 600,
        "el": 0.0,
        "isl_link_type": "laser",
        "isl_data_rate_mbps": 8500.0,
        "isl_link_margin_db": 5.2,
        "isl_distance_km": 1850.0,
        "isl_atp_setup_time_s": 32.5,
        "attitude_feasible": true,
        "attitude_samples": []
    }

    窗口键格式（visibility_windows.json中的外层键）：
        {satA_id}_ISL:{satB_id}
"""

import gzip
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from core.models.isl_config import ISLCapabilityConfig
from core.dynamics.isl_physics import (
    ISLPhysicsEngine,
    calculate_laser_link_margin,
    calculate_laser_data_rate,
    calculate_microwave_link_margin,
    calculate_microwave_data_rate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ISL链路数据类
# =============================================================================

@dataclass
class ISLLink:
    """
    星间链路可见窗口

    表示两颗卫星在某个时段内可以建立ISL通信的一个窗口。

    Attributes:
        satellite_a_id: 卫星A的ID（来自窗口文件的 sat 字段）
        satellite_b_id: 卫星B的ID（来自窗口文件的 tgt 字段去掉"ISL:"前缀）
        start_time: 窗口开始时间（UTC）
        end_time: 窗口结束时间（UTC）
        link_type: 链路类型（'laser' 或 'microwave'）
        max_data_rate: 最大可达数据率（Mbps）
        link_margin_db: 链路余量（dB）
        distance_km: 两星间距离（公里），0表示未知
        relative_velocity_km_s: 两星相对速度（km/s），0表示未知
        atp_setup_time_s: ATP建链时间（秒），微波链路为0
        link_quality: 归一化链路质量评分（0-1）
    """
    satellite_a_id: str
    satellite_b_id: str
    start_time: datetime
    end_time: datetime
    link_type: str              # 'laser' 或 'microwave'
    max_data_rate: float        # Mbps
    link_margin_db: float
    distance_km: float
    relative_velocity_km_s: float
    atp_setup_time_s: float     # 激光链路的ATP建链时间；微波为0
    link_quality: float         # 0-1 归一化质量评分

    @property
    def duration_seconds(self) -> float:
        """窗口持续时间（秒）"""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def is_viable(self) -> bool:
        """链路是否可用（数据率>0且链路余量≥0）"""
        return self.max_data_rate > 0.0 and self.link_margin_db >= 0.0

    @property
    def effective_duration_seconds(self) -> float:
        """
        有效通信时长（秒）= 窗口时长 - ATP建链时间

        对于微波链路，ATP时间为0，有效时长等于窗口时长。
        """
        effective = self.duration_seconds - self.atp_setup_time_s
        return max(0.0, effective)

    @property
    def transferable_data_gb(self) -> float:
        """
        窗口内可传输数据量（GB）

        使用有效通信时长和最大数据率计算。
        """
        return self.max_data_rate * self.effective_duration_seconds / 8000.0  # Mbps*s -> GB

    def __repr__(self) -> str:
        return (
            f"ISLLink({self.satellite_a_id}<->{self.satellite_b_id}, "
            f"{self.link_type}, "
            f"{self.start_time.strftime('%H:%M:%S')}-{self.end_time.strftime('%H:%M:%S')}, "
            f"{self.max_data_rate:.0f} Mbps, margin={self.link_margin_db:.1f} dB)"
        )


# =============================================================================
# ISL窗口缓存
# =============================================================================

class ISLWindowCache:
    """
    ISL可见性窗口缓存

    从Java后端预计算的visibility_windows.json中加载ISL窗口（tgt以"ISL:"开头的条目），
    并按卫星对（tuple）索引存储，支持双向查询。

    MANDATORY规则：
    - 若预计算文件不存在，抛出 RuntimeError
    - 若文件中无ISL窗口，抛出 RuntimeError
    - 禁止任何Python几何回退计算

    Args:
        isl_windows_path: 窗口文件路径（记录来源，用于日志）
        physics_engine: ISL物理引擎（用于补全物理参数）
    """

    def __init__(
        self,
        isl_windows_path: str,
        physics_engine: ISLPhysicsEngine,
    ) -> None:
        self._source_path = isl_windows_path
        self._physics = physics_engine
        # 键：(sat_a_id, sat_b_id)，始终按字典序排列（sat_a_id <= sat_b_id）
        self._windows: Dict[Tuple[str, str], List[ISLLink]] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def load_from_visibility_windows(
        self,
        windows_data: list,
        satellite_isl_configs: Dict[str, ISLCapabilityConfig],
        scenario_start: datetime,
    ) -> None:
        """
        从visibility_windows.json的窗口数据列表中解析ISL窗口。

        tgt 字段以 "ISL:" 开头的记录被识别为ISL窗口。
        窗口按卫星对（字典序规范化）存储，支持双向查询。

        Args:
            windows_data: 窗口记录列表（每条记录为字典）
            satellite_isl_configs: 各卫星的ISL能力配置字典，键为卫星ID
            scenario_start: 场景开始时间（UTC，用于处理相对时间戳）
        """
        loaded_count = 0
        skipped_count = 0

        for record in windows_data:
            tgt = record.get('tgt', '')
            if not tgt.startswith('ISL:'):
                continue

            sat_a_id = record.get('sat', '')
            sat_b_id = tgt[4:]  # 去掉 "ISL:" 前缀

            if not sat_a_id or not sat_b_id:
                logger.warning(
                    f"ISL window record missing sat or peer ID: {record}, skipping"
                )
                skipped_count += 1
                continue

            # 解析时间
            start_time = _parse_window_time(record.get('start'), scenario_start)
            end_time = _parse_window_time(record.get('end'), scenario_start)

            if start_time is None or end_time is None:
                logger.warning(
                    f"ISL window for {sat_a_id}<->{sat_b_id} has invalid timestamps, skipping"
                )
                skipped_count += 1
                continue

            if end_time <= start_time:
                logger.debug(
                    f"ISL window for {sat_a_id}<->{sat_b_id}: "
                    f"end_time <= start_time, skipping zero-duration window"
                )
                skipped_count += 1
                continue

            # 读取预计算的物理参数（Java后端已计算）
            link_type = record.get('isl_link_type', 'laser')
            precomputed_rate = record.get('isl_data_rate_mbps', 0.0)
            precomputed_margin = record.get('isl_link_margin_db', -999.0)
            distance_km = record.get('isl_distance_km', 0.0)
            atp_setup_time_s = record.get('isl_atp_setup_time_s', 0.0)

            # 若Java端已提供物理参数，直接使用；否则通过物理引擎补全
            data_rate, link_margin = self._resolve_physics(
                sat_a_id=sat_a_id,
                sat_b_id=sat_b_id,
                link_type=link_type,
                precomputed_rate=float(precomputed_rate),
                precomputed_margin=float(precomputed_margin),
                distance_km=float(distance_km),
                satellite_isl_configs=satellite_isl_configs,
            )

            # 如果Java端未提供ATP时间，通过物理引擎估算
            if atp_setup_time_s <= 0.0 and link_type == 'laser':
                atp_setup_time_s = self._estimate_atp_time(
                    sat_a_id, satellite_isl_configs
                )

            # 归一化链路质量评分（基于余量和数据率）
            link_quality = _compute_link_quality(link_margin, data_rate, link_type)

            link = ISLLink(
                satellite_a_id=sat_a_id,
                satellite_b_id=sat_b_id,
                start_time=start_time,
                end_time=end_time,
                link_type=link_type,
                max_data_rate=data_rate,
                link_margin_db=link_margin,
                distance_km=float(distance_km),
                relative_velocity_km_s=0.0,  # Java端当前不导出相对速度
                atp_setup_time_s=float(atp_setup_time_s),
                link_quality=link_quality,
            )

            # 规范化键（字典序排列，确保双向查询一致性）
            key = _canonical_key(sat_a_id, sat_b_id)
            if key not in self._windows:
                self._windows[key] = []
            self._windows[key].append(link)
            loaded_count += 1

        # 对每对的窗口按开始时间排序
        for key in self._windows:
            self._windows[key].sort(key=lambda w: w.start_time)

        self._loaded = True
        logger.info(
            f"ISL window cache loaded: {loaded_count} windows across "
            f"{len(self._windows)} satellite pairs "
            f"(skipped {skipped_count} invalid records)"
        )

    def _resolve_physics(
        self,
        sat_a_id: str,
        sat_b_id: str,
        link_type: str,
        precomputed_rate: float,
        precomputed_margin: float,
        distance_km: float,
        satellite_isl_configs: Dict[str, ISLCapabilityConfig],
    ) -> Tuple[float, float]:
        """
        解析物理参数：优先使用Java端预计算值，必要时通过物理引擎补全。

        Returns:
            (data_rate_mbps, link_margin_db)
        """
        # Java端已提供有效数值，直接使用
        if precomputed_rate > 0.0 and precomputed_margin > -900.0:
            return precomputed_rate, precomputed_margin

        # Java端未提供，通过物理引擎计算（要求距离已知）
        if distance_km <= 0.0:
            logger.debug(
                f"ISL window {sat_a_id}<->{sat_b_id}: no precomputed physics and "
                f"distance_km={distance_km:.1f}, using fallback zero values"
            )
            return 0.0, -999.0

        sat_config = satellite_isl_configs.get(sat_a_id)
        if sat_config is None:
            logger.debug(
                f"ISL window {sat_a_id}: no ISL config found, using fallback zero values"
            )
            return 0.0, -999.0

        try:
            params = self._physics.compute_link_parameters(
                link_type=link_type,
                laser_config=sat_config.laser,
                microwave_config=sat_config.microwave,
                distance_km=distance_km,
                relative_velocity_km_s=0.0,
            )
            return params['data_rate_mbps'], params['link_margin_db']
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning(
                f"Physics engine failed for ISL {sat_a_id}<->{sat_b_id}: {e}"
            )
            return 0.0, -999.0

    def _estimate_atp_time(
        self,
        sat_id: str,
        satellite_isl_configs: Dict[str, ISLCapabilityConfig],
    ) -> float:
        """估算激光链路ATP建链时间（当Java端未提供时）"""
        config = satellite_isl_configs.get(sat_id)
        if config and config.laser:
            return config.laser.total_atp_time_s
        return 37.0  # 默认值：30s+5s+2s

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_windows(self, sat_a: str, sat_b: str) -> List[ISLLink]:
        """
        获取两颗卫星之间的ISL窗口列表（支持双向查询）。

        Args:
            sat_a: 卫星A的ID
            sat_b: 卫星B的ID

        Returns:
            ISLLink列表（按开始时间升序排列），无窗口时返回空列表
        """
        key = _canonical_key(sat_a, sat_b)
        return list(self._windows.get(key, []))

    def get_active_links(self, satellite_id: str, at_time: datetime) -> List[ISLLink]:
        """
        获取指定时刻某颗卫星的所有活跃ISL链路。

        Args:
            satellite_id: 卫星ID
            at_time: 查询时刻（UTC）

        Returns:
            在 at_time 时刻活跃的 ISLLink 列表
        """
        # 确保时间是 UTC aware
        if at_time.tzinfo is None:
            at_time = at_time.replace(tzinfo=timezone.utc)

        active = []
        for (a_id, b_id), windows in self._windows.items():
            if satellite_id not in (a_id, b_id):
                continue
            for link in windows:
                if link.start_time <= at_time <= link.end_time:
                    active.append(link)

        return active

    def get_all_windows(self) -> Dict[Tuple[str, str], List[ISLLink]]:
        """
        获取所有ISL窗口的字典副本。

        Returns:
            字典，键为 (sat_a_id, sat_b_id)（字典序规范化），值为 ISLLink 列表
        """
        return {key: list(windows) for key, windows in self._windows.items()}

    def get_satellite_ids(self) -> List[str]:
        """获取所有出现在ISL窗口中的卫星ID列表（去重）"""
        ids = set()
        for a_id, b_id in self._windows.keys():
            ids.add(a_id)
            ids.add(b_id)
        return sorted(ids)

    def get_pair_ids(self) -> List[Tuple[str, str]]:
        """获取所有有ISL窗口的卫星对"""
        return list(self._windows.keys())

    def get_window_count(self) -> int:
        """获取总窗口数"""
        return sum(len(v) for v in self._windows.values())

    def get_pair_count(self) -> int:
        """获取有ISL窗口的卫星对数量"""
        return len(self._windows)

    def is_loaded(self) -> bool:
        """是否已成功加载"""
        return self._loaded

    def __len__(self) -> int:
        return self.get_window_count()

    def __repr__(self) -> str:
        return (
            f"ISLWindowCache(source='{self._source_path}', "
            f"pairs={self.get_pair_count()}, "
            f"windows={self.get_window_count()}, "
            f"loaded={self._loaded})"
        )


# =============================================================================
# 加载入口函数（MANDATORY：必须预计算数据）
# =============================================================================

def load_isl_windows_from_cache(
    cache_json_path: str,
    satellite_configs: Dict[str, ISLCapabilityConfig],
    scenario_start: datetime,
    physics_engine: Optional[ISLPhysicsEngine] = None,
) -> ISLWindowCache:
    """
    从Java后端预计算的可见性文件加载ISL窗口。

    MANDATORY规则：
    1. 若文件不存在 → 抛出 RuntimeError（提示运行Java后端）
    2. 若文件中无ISL窗口 → 抛出 RuntimeError（提示配置ISL场景）
    3. 禁止任何Python几何回退计算

    支持格式：
    - .json：标准JSON文件
    - .json.gz：GZIP压缩的JSON文件

    ISL窗口识别规则：tgt字段以"ISL:"开头的记录

    Args:
        cache_json_path: 预计算文件路径（visibility_windows.json 或 .json.gz）
        satellite_configs: 各卫星的ISL能力配置字典（用于物理参数补全）
        scenario_start: 场景开始时间（UTC）
        physics_engine: ISL物理引擎（None时创建默认实例）

    Returns:
        ISLWindowCache 实例

    Raises:
        RuntimeError: 文件不存在或文件中无ISL窗口
        json.JSONDecodeError: 文件格式错误
        OSError: 文件读取错误

    Example::

        from datetime import datetime, timezone
        from core.models.isl_config import ISLCapabilityConfig
        from core.network.isl_visibility import load_isl_windows_from_cache

        cache = load_isl_windows_from_cache(
            cache_json_path='java/output/frequency_scenario/visibility_windows.json',
            satellite_configs={'OPT-01': opt01_isl_config},
            scenario_start=datetime(2026, 3, 11, tzinfo=timezone.utc),
        )
        windows = cache.get_windows('OPT-01', 'OPT-02')
    """
    # 强制文件存在性检查
    if not os.path.exists(cache_json_path):
        raise RuntimeError(
            f"ISL visibility window file not found: {cache_json_path}\n"
            f"\n"
            f"MANDATORY: ISL windows must be precomputed by the Java backend.\n"
            f"Run the Java backend with an ISL-enabled scenario:\n"
            f"\n"
            f"  cd java && java -cp 'classes:lib/*' orekit.visibility.LargeScaleFrequencyTest \\\n"
            f"      --scenario <scenario_with_isl.json>\n"
            f"\n"
            f"Ensure the scenario includes ISL-capable satellites and "
            f"ISLVisibilityCalculator is invoked in the test class.\n"
            f"Python geometry fallback is DISABLED for ISL windows."
        )

    logger.info(f"Loading ISL windows from: {cache_json_path}")

    # 读取文件（支持gzip压缩）
    if cache_json_path.endswith('.gz'):
        with gzip.open(cache_json_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(cache_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    # 提取窗口列表（兼容多种JSON结构）
    windows_data: list = []
    if isinstance(data, list):
        # 文件直接是窗口数组
        windows_data = data
    elif isinstance(data, dict):
        # 可能是 {"windows": [...]} 或 {"visibility_windows": [...]}
        for key in ('windows', 'visibility_windows', 'data'):
            if key in data and isinstance(data[key], list):
                windows_data = data[key]
                break
        if not windows_data:
            # 尝试从字典值中收集列表（键格式：{sat_id}_ISL:{peer_id}）
            # 只处理明确以 '_ISL:' 为键的条目，避免混入非ISL窗口
            for key, val in data.items():
                if isinstance(val, list) and '_ISL:' in key:
                    windows_data.extend(val)

    # 过滤出ISL窗口
    isl_windows = [w for w in windows_data if isinstance(w, dict)
                   and w.get('tgt', '').startswith('ISL:')]

    # 强制要求存在ISL窗口
    if not isl_windows:
        raise RuntimeError(
            f"No ISL windows found in {cache_json_path}.\n"
            f"\n"
            f"MANDATORY: ISL windows must be precomputed by the Java backend.\n"
            f"Possible causes:\n"
            f"  1. The scenario does not include ISL-capable satellites\n"
            f"  2. ISLVisibilityCalculator was not invoked during Java backend execution\n"
            f"  3. The visibility file was generated without ISL support\n"
            f"\n"
            f"Ensure the Java backend scenario has satellites with 'isl_capability' "
            f"configured and ISLVisibilityCalculator is called in LargeScaleFrequencyTest.\n"
            f"Python geometry fallback is DISABLED for ISL windows."
        )

    logger.info(f"Found {len(isl_windows)} ISL window records in {cache_json_path}")

    # 创建物理引擎（若未提供）
    engine = physics_engine if physics_engine is not None else ISLPhysicsEngine()

    # 创建并加载缓存
    cache = ISLWindowCache(cache_json_path, engine)
    cache.load_from_visibility_windows(isl_windows, satellite_configs, scenario_start)

    if not cache.is_loaded() or cache.get_window_count() == 0:
        raise RuntimeError(
            f"ISL window cache loaded successfully from {cache_json_path} "
            f"but resulted in 0 valid windows after parsing. "
            f"Check that ISL window timestamps and satellite IDs are correctly formatted."
        )

    logger.info(
        f"ISL window cache ready: {cache.get_pair_count()} satellite pairs, "
        f"{cache.get_window_count()} windows"
    )
    return cache


# =============================================================================
# 旧版类（保留用于向后兼容，已废弃）
# =============================================================================

# 向后兼容别名（已废弃，请使用 load_isl_windows_from_cache）
# 在文件末尾的别名定义之前先声明，以满足 network_router.py 中的
# `from .isl_visibility import ISLVisibilityCalculator, ISLLink` 导入


class ISLVisibilityCalculatorLegacy:
    """
    旧版ISL可见性计算器（LEGACY，已废弃，生产环境中不使用）

    .. deprecated::
        该类已废弃，不用于生产。
        请改用 :func:`load_isl_windows_from_cache` 加载Java后端预计算的窗口数据。

        旧版实现使用简化几何模型，精度不足，且与HPOP轨道数据不兼容。
    """

    def __init__(
        self,
        link_type: str = 'laser',
        max_link_distance: float = 5000.0,
        min_elevation_angle: float = 0.0,
    ) -> None:
        warnings.warn(
            "ISLVisibilityCalculatorLegacy is deprecated and NOT used in production. "
            "Use load_isl_windows_from_cache() to load Java backend precomputed windows. "
            "Python geometry fallback is disabled for ISL windows.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.link_type = link_type
        self.max_link_distance = max_link_distance

    def compute_isl_windows(self, *args, **kwargs):
        """已废弃，抛出 NotImplementedError。"""
        raise NotImplementedError(
            "ISLVisibilityCalculatorLegacy.compute_isl_windows() is disabled. "
            "ISL windows must come from the Java backend. "
            "Use load_isl_windows_from_cache() instead."
        )


# =============================================================================
# 内部工具函数
# =============================================================================

def _canonical_key(sat_a: str, sat_b: str) -> Tuple[str, str]:
    """
    规范化卫星对键（字典序排列），确保 (A,B) 和 (B,A) 映射到同一个键。

    Args:
        sat_a: 卫星A的ID
        sat_b: 卫星B的ID

    Returns:
        规范化的 (小ID, 大ID) 元组
    """
    if sat_a <= sat_b:
        return (sat_a, sat_b)
    return (sat_b, sat_a)


def _parse_window_time(
    time_value,
    scenario_start: datetime,
) -> Optional[datetime]:
    """
    解析窗口时间字段，支持ISO字符串和相对秒数。

    Args:
        time_value: 时间值（ISO字符串或相对秒数浮点数）
        scenario_start: 场景开始时间（用于相对秒数转换）

    Returns:
        UTC timezone-aware datetime，解析失败返回 None
    """
    if time_value is None:
        return None

    if isinstance(time_value, (int, float)):
        # 相对秒数
        return scenario_start + timedelta(seconds=float(time_value))

    if isinstance(time_value, str):
        # ISO 8601格式
        formats = [
            '%Y-%m-%dT%H:%M:%S.%f%z',
            '%Y-%m-%dT%H:%M:%S%z',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(time_value, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt
            except ValueError:
                continue
        logger.warning(f"Cannot parse ISL window time: '{time_value}'")
        return None

    if isinstance(time_value, datetime):
        if time_value.tzinfo is None:
            return time_value.replace(tzinfo=timezone.utc)
        return time_value.astimezone(timezone.utc)

    logger.warning(f"Unknown time value type: {type(time_value)}, value={time_value}")
    return None


def _compute_link_quality(
    link_margin_db: float,
    data_rate_mbps: float,
    link_type: str,
) -> float:
    """
    计算归一化链路质量评分（0-1）。

    评分逻辑：
    - 链路余量归一化：min_margin=0, good_margin=10 dB → [0, 0.5]
    - 数据率归一化：按链路类型参考值归一化 → [0, 0.5]
    - 总分 = 余量分 + 数据率分，限幅到 [0, 1]

    Args:
        link_margin_db: 链路余量（dB）
        data_rate_mbps: 数据率（Mbps）
        link_type: 链路类型（'laser'/'microwave'）

    Returns:
        质量评分（0-1），0表示不可用，1表示最优
    """
    if link_margin_db < 0.0 or data_rate_mbps <= 0.0:
        return 0.0

    # 余量分 [0, 0.5]：0 dB→0分，≥10 dB→0.5分
    margin_score = min(link_margin_db / 10.0, 1.0) * 0.5

    # 数据率分 [0, 0.5]：按参考值归一化
    if link_type == 'laser':
        ref_rate = 10000.0   # 10 Gbps基准
    else:
        ref_rate = 1000.0    # 1 Gbps基准
    rate_score = min(data_rate_mbps / ref_rate, 1.0) * 0.5

    quality = margin_score + rate_score
    return min(quality, 1.0)


# ---------------------------------------------------------------------------
# 向后兼容别名
# network_router.py 中有 `from .isl_visibility import ISLVisibilityCalculator, ISLLink`
# 此别名保持导入兼容性，同时在实例化时发出废弃警告。
# 注意：公开别名同样在 core/network/__init__.py 中定义，两处保持一致。
# ---------------------------------------------------------------------------

#: 向后兼容别名（已废弃）：指向 ISLVisibilityCalculatorLegacy
ISLVisibilityCalculator = ISLVisibilityCalculatorLegacy
