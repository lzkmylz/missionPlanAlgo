"""
姿态管理器

整合所有姿态管理模块，提供高层次的姿态管理API。

功能：
1. 姿态切换规划（plan_transition）
2. 任务后姿态决策（decide_post_task_attitude）
3. 动量卸载判断（should_momentum_dump）
4. 发电功率查询（get_power_generation）

决策逻辑（基于设计文档3.1节）：
1. 首先检查是否需要动量卸载（最高优先级）
2. 如果距离下次任务时间 >= 阈值且优化启用 -> 对日定向
3. 如果SOC < 阈值 -> 强制对日定向
4. 否则 -> 对地定向
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

from core.dynamics.attitude_mode import AttitudeMode, AttitudeTransition, TransitionResult
from core.dynamics.attitude_transition_calculator import (
    AttitudeTransitionCalculator,
    TransitionConfig,
)
from core.dynamics.power_generation_calculator import (
    PowerGenerationCalculator,
    PowerConfig,
)
from core.dynamics.sun_position_calculator import SunPositionCalculator

logger = logging.getLogger(__name__)


@dataclass
class AttitudeManagementConfig:
    """
    姿态管理系统配置参数

    Attributes:
        idle_time_threshold: 空闲时间阈值（秒），超过此值切换到对日定向
        soc_threshold: 电量阈值，低于此值强制对日定向
        momentum_dump_interval: 动量卸载间隔（秒）
        momentum_dump_duration: 动量卸载持续时间（秒）
        settling_time: 机动稳定时间（秒）
        max_slew_rate: 最大机动角速度（度/秒）
        enable_sun_pointing_optimization: 是否启用对日定向优化
        enable_momentum_dumping: 是否启用动量卸载
    """

    idle_time_threshold: float = 300.0  # 5 minutes
    soc_threshold: float = 0.30  # 30%
    momentum_dump_interval: float = 14400.0  # 4 hours
    momentum_dump_duration: float = 600.0  # 10 minutes
    settling_time: float = 5.0
    max_slew_rate: float = 3.0
    enable_sun_pointing_optimization: bool = True
    enable_momentum_dumping: bool = True


class AttitudeManager:
    """
    姿态管理器

    整合姿态切换计算、发电功率计算等功能，提供高层次的姿态管理API。
    用于调度器在任务规划时进行姿态决策。

    Attributes:
        config: 姿态管理配置
        sun_calculator: 太阳位置计算器
        transition_calculator: 姿态切换计算器
        power_calculator: 发电功率计算器
    """

    def __init__(self, config: Optional[AttitudeManagementConfig] = None):
        """
        初始化姿态管理器

        Args:
            config: 姿态管理配置，如果为None则使用默认配置
        """
        self.config = config if config is not None else AttitudeManagementConfig()

        # 创建太阳位置计算器
        self.sun_calculator = SunPositionCalculator(use_orekit=False)

        # 创建姿态切换配置
        transition_config = TransitionConfig(
            max_slew_rate=self.config.max_slew_rate,
            settling_time=self.config.settling_time,
        )

        # 创建姿态切换计算器
        self.transition_calculator = AttitudeTransitionCalculator(
            sun_calculator=self.sun_calculator,
            config=transition_config,
        )

        # 创建发电功率配置
        power_config = PowerConfig()

        # 创建发电功率计算器
        self.power_calculator = PowerGenerationCalculator(
            sun_calculator=self.sun_calculator,
            config=power_config,
        )

        logger.info("AttitudeManager initialized")

    def plan_transition(
        self,
        from_mode: AttitudeMode,
        to_mode: AttitudeMode,
        satellite_position: Tuple[float, float, float],
        timestamp: datetime,
        target_position: Optional[Tuple[float, float]] = None,
        ground_station_position: Optional[Tuple[float, float]] = None,
    ) -> TransitionResult:
        """
        规划姿态切换

        计算从一种姿态模式切换到另一种姿态所需的机动参数。

        Args:
            from_mode: 起始姿态模式
            to_mode: 目标姿态模式
            satellite_position: 卫星ECEF位置（米）
            timestamp: UTC时间
            target_position: 目标位置（纬度、经度），成像模式需要
            ground_station_position: 地面站位置（纬度、经度），数传模式需要

        Returns:
            TransitionResult: 切换计算结果，包含机动时间、角度等
        """
        # 创建切换请求
        transition = AttitudeTransition(
            from_mode=from_mode,
            to_mode=to_mode,
            timestamp=timestamp,
            satellite_position=satellite_position,
            target_position=target_position,
            ground_station_position=ground_station_position,
        )

        # 使用切换计算器计算结果
        return self.transition_calculator.calculate_transition(transition)

    def decide_post_task_attitude(
        self,
        current_mode: AttitudeMode,
        next_task_time: Optional[datetime],
        current_time: datetime,
        soc: float,
        time_since_last_dump: float,
    ) -> AttitudeMode:
        """
        决定任务完成后的姿态模式

        根据当前状态和系统配置，决定成像任务完成后应切换到的姿态模式。
        决策优先级：
        1. 动量卸载（最高优先级，定期执行）
        2. 对日定向（空闲时间长或电量低）
        3. 对地定向（默认）

        Args:
            current_mode: 当前姿态模式
            next_task_time: 下次任务时间，None表示没有后续任务
            current_time: 当前时间
            soc: 电池电量状态（0.0-1.0）
            time_since_last_dump: 距离上次动量卸载的时间（秒）

        Returns:
            AttitudeMode: 建议的下一姿态模式
        """
        # 1. 检查是否需要动量卸载（最高优先级）
        if self.config.enable_momentum_dumping:
            if self.should_momentum_dump(time_since_last_dump):
                logger.debug("Momentum dump required, returning MOMENTUM_DUMP mode")
                return AttitudeMode.MOMENTUM_DUMP

        # 2. 检查电量是否低于阈值
        if soc < self.config.soc_threshold:
            logger.debug(f"SOC {soc} below threshold {self.config.soc_threshold}, returning SUN_POINTING")
            return AttitudeMode.SUN_POINTING

        # 3. 检查空闲时间是否超过阈值
        if next_task_time is not None:
            idle_time = (next_task_time - current_time).total_seconds()

            if idle_time >= self.config.idle_time_threshold:
                if self.config.enable_sun_pointing_optimization:
                    logger.debug(f"Idle time {idle_time}s exceeds threshold, returning SUN_POINTING")
                    return AttitudeMode.SUN_POINTING
                else:
                    logger.debug("Sun pointing optimization disabled, returning NADIR_POINTING")
                    return AttitudeMode.NADIR_POINTING
            else:
                logger.debug(f"Idle time {idle_time}s below threshold, returning NADIR_POINTING")
                return AttitudeMode.NADIR_POINTING
        else:
            # 没有后续任务，切换到对日定向充电
            if self.config.enable_sun_pointing_optimization:
                logger.debug("No next task, returning SUN_POINTING")
                return AttitudeMode.SUN_POINTING
            else:
                logger.debug("No next task but optimization disabled, returning NADIR_POINTING")
                return AttitudeMode.NADIR_POINTING

    def should_momentum_dump(self, time_since_last_dump: float) -> bool:
        """
        判断是否需要动量卸载

        基于距离上次动量卸载的时间判断。

        Args:
            time_since_last_dump: 距离上次动量卸载的时间（秒）

        Returns:
            bool: 如果需要进行动量卸载返回True
        """
        if not self.config.enable_momentum_dumping:
            return False

        return time_since_last_dump >= self.config.momentum_dump_interval

    def get_power_generation(
        self,
        mode: AttitudeMode,
        satellite_position: Tuple[float, float, float],
        timestamp: datetime,
        roll: float = 0.0,
        pitch: float = 0.0,
    ) -> float:
        """
        获取指定姿态模式下的发电功率

        Args:
            mode: 姿态模式
            satellite_position: 卫星ECEF位置（米）
            timestamp: UTC时间
            roll: 滚转角（度），用于IMAGING/DOWNLINK/REALTIME模式
            pitch: 俯仰角（度），用于IMAGING/DOWNLINK/REALTIME模式

        Returns:
            float: 发电功率（W）
        """
        return self.power_calculator.calculate_power(
            attitude_mode=mode,
            satellite_position=satellite_position,
            timestamp=timestamp,
            roll_angle=roll,
            pitch_angle=pitch,
        )
