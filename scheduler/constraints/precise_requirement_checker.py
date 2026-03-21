"""
精准观测需求约束检查工具

提供独立的纯函数供 BaseScheduler、MetaheuristicConstraintChecker 等所有调度器
统一调用，消除多处维护相同逻辑的维护风险。

设计原则
---------
- 纯函数（无副作用），可安全并发调用
- 唯一维护点：精准需求逻辑只在此处定义，调用方薄封装即可
- 空字符串安全：列表字段中的空字符串自动忽略，等价于不限制
"""

from typing import Any, Optional, Set
import logging

from core.models import ImagingMode

logger = logging.getLogger(__name__)


def get_sat_type_category(sat: Any) -> str:
    """
    获取卫星的大类标签（'optical' 或 'sar'）。

    查找顺序：
        1. ``sat.capabilities.payload_config.payload_type``（首选）
        2. ``sat.sat_type.value`` 前缀匹配（回退）

    Returns:
        ``'optical'``、``'sar'`` 或 ``'unknown'``（类型无法识别时）
    """
    caps = getattr(sat, 'capabilities', None)
    if caps is not None:
        pc = getattr(caps, 'payload_config', None)
        if pc is not None:
            pt = getattr(pc, 'payload_type', None)
            if pt:
                return pt.lower()

    sat_type = getattr(sat, 'sat_type', None)
    if sat_type is not None:
        val = sat_type.value.lower()
        if val.startswith('optical'):
            return 'optical'
        if val.startswith('sar'):
            return 'sar'

    logger.warning(
        "卫星 %s 无法识别类型（既无 payload_config.payload_type 也无 sat_type），"
        "请检查卫星配置",
        getattr(sat, 'id', '?')
    )
    return 'unknown'


def check_precise_requirements(sat: Any, task: Any) -> bool:
    """
    检查卫星是否满足任务的精准观测需求约束。

    检查顺序（依次，任一不满足即返回 False）：

    0. **旧字段** ``required_satellite_type``（单字符串）
    1. **新字段** ``allowed_satellite_ids``（列表，过滤空字符串）
    2. **新字段** ``allowed_satellite_types``（列表，过滤空字符串）
    3. **成像模式集合** = ``required_imaging_modes`` 列表 ∪ ``required_imaging_mode`` 旧字段，
       过滤空字符串，大小写不敏感

    列表字段过滤后为空时表示不限制（向后兼容）。

    Returns:
        True 表示满足所有约束，False 表示至少一项不满足
    """
    task_label = getattr(task, 'target_id', getattr(task, 'id', '?'))

    # 懒加载卫星类型标签（调用链路上最多执行一次）
    _sat_category: Optional[str] = None

    def _get_category() -> str:
        nonlocal _sat_category
        if _sat_category is None:
            _sat_category = get_sat_type_category(sat)
        return _sat_category

    # 0. 旧字段 required_satellite_type
    req_sat_type = getattr(task, 'required_satellite_type', None)
    if req_sat_type:
        if req_sat_type.lower() != _get_category():
            logger.debug(
                "精准需求过滤(旧字段): 卫星 %s 类型 '%s' 不满足目标 %s 的 required_satellite_type='%s'",
                sat.id, _get_category(), task_label, req_sat_type,
            )
            return False

    # 1. allowed_satellite_ids（过滤空字符串）
    allowed_ids = [i for i in getattr(task, 'allowed_satellite_ids', []) if i]
    if allowed_ids and sat.id not in allowed_ids:
        logger.debug(
            "精准需求过滤: 卫星 %s 不在目标 %s 的允许ID列表 %s 中",
            sat.id, task_label, allowed_ids,
        )
        return False

    # 2. allowed_satellite_types（过滤空字符串）
    allowed_types = [t for t in getattr(task, 'allowed_satellite_types', []) if t]
    if allowed_types:
        if _get_category() not in {t.lower() for t in allowed_types}:
            logger.debug(
                "精准需求过滤: 卫星 %s 类型 '%s' 不在目标 %s 的允许类型列表 %s 中",
                sat.id, _get_category(), task_label, allowed_types,
            )
            return False

    # 3. 成像模式集合（新字段列表 + 旧字段单值，过滤空字符串）
    required_modes = getattr(task, 'required_imaging_modes', [])
    legacy_mode = getattr(task, 'required_imaging_mode', None)
    effective_modes: Set[str] = {m.lower() for m in required_modes if m}
    if legacy_mode:
        effective_modes.add(legacy_mode.lower())

    if effective_modes:
        caps = getattr(sat, 'capabilities', None)
        sat_modes = getattr(caps, 'imaging_modes', []) if caps else []
        sat_mode_values: Set[str] = set()
        for m in sat_modes:
            if isinstance(m, ImagingMode):
                sat_mode_values.add(m.value.lower())
            elif isinstance(m, str):
                sat_mode_values.add(m.lower())
        if not sat_mode_values.intersection(effective_modes):
            logger.debug(
                "精准需求过滤: 卫星 %s 支持的模式 %s 与目标 %s 要求的模式 %s 无交集",
                sat.id, sat_mode_values, task_label, effective_modes,
            )
            return False

    return True


def select_imaging_mode_for_task(sat: Any, task: Any) -> ImagingMode:
    """
    为卫星-任务对选择成像模式。

    优先级：

    1. 从卫星支持的模式中选第一个满足任务约束的模式
       （``required_imaging_modes`` 列表 ∪ ``required_imaging_mode`` 旧字段，大小写不敏感）
    2. 卫星默认第一个可用模式

    Args:
        sat:  卫星对象（需有 ``capabilities.imaging_modes`` 属性）
        task: 任务/目标对象，``None`` 时直接取默认模式

    Returns:
        :class:`~core.models.ImagingMode` 实例
    """
    caps = getattr(sat, 'capabilities', None)
    modes = getattr(caps, 'imaging_modes', []) if caps else []
    if not modes:
        return ImagingMode.PUSH_BROOM

    # 构建期望模式集合（大小写不敏感，过滤空字符串）
    required_modes_list: list = (
        getattr(task, 'required_imaging_modes', []) if task is not None else []
    )
    legacy_mode_str: Optional[str] = (
        getattr(task, 'required_imaging_mode', None) if task is not None else None
    )
    required_mode_values: Set[str] = {m.lower() for m in required_modes_list if m}
    if legacy_mode_str:
        required_mode_values.add(legacy_mode_str.lower())

    # 优先：从卫星支持模式中选第一个满足约束的
    if required_mode_values:
        for mode in modes:
            if not isinstance(mode, (ImagingMode, str)):
                continue
            mode_val = (mode.value if isinstance(mode, ImagingMode) else mode).lower()
            if mode_val in required_mode_values:
                if isinstance(mode, ImagingMode):
                    return mode
                try:
                    return ImagingMode(mode)
                except ValueError:
                    continue  # 跳过无效枚举字符串，继续找下一个匹配

    # 回退：卫星默认第一个模式
    mode = modes[0]
    if not isinstance(mode, (ImagingMode, str)):
        return ImagingMode.PUSH_BROOM
    if isinstance(mode, ImagingMode):
        return mode
    try:
        return ImagingMode(mode)
    except ValueError:
        return ImagingMode.PUSH_BROOM
