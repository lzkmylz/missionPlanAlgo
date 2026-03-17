"""
PMC (Pitch Motion Compensation) 动力学计算模块

计算主动前向推扫模式下的姿态机动、成像参数和约束检查。

核心功能:
1. 俯仰角速度计算
2. 成像期间姿态序列生成
3. PMC约束可行性验证
4. SNR增益估算
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any
import math
import logging

from core.models.satellite import Satellite
from core.models.target import Target
from core.models.pmc_config import PitchMotionCompensationConfig
from core.dynamics.attitude_calculator import AttitudeCalculator, AttitudeAngles

logger = logging.getLogger(__name__)


@dataclass
class PMCImagingState:
    """
    PMC成像状态

    记录特定时刻的成像状态和姿态
    """
    timestamp: datetime
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    is_in_pmc_phase: bool  # 是否处于PMC主成像阶段
    ground_velocity_m_s: float  # 等效地面速度


@dataclass
class PMCImagingSequence:
    """
    PMC成像序列

    完整的PMC任务姿态序列，包括进入、成像、退出阶段
    """
    states: List[PMCImagingState]
    pmc_config: PitchMotionCompensationConfig
    satellite_id: str
    target_id: str
    enter_start_time: datetime
    imaging_start_time: datetime
    imaging_end_time: datetime
    exit_end_time: datetime

    def get_duration_s(self) -> float:
        """获取总时长"""
        return (self.exit_end_time - self.enter_start_time).total_seconds()

    def get_imaging_duration_s(self) -> float:
        """获取实际成像时长"""
        return (self.imaging_end_time - self.imaging_start_time).total_seconds()

    def get_pitch_range(self) -> Tuple[float, float]:
        """获取俯仰角范围"""
        pitches = [s.pitch_deg for s in self.states]
        return min(pitches), max(pitches)


class PMCCalculator:
    """
    PMC动力学计算器

    计算PMC模式下的姿态机动和成像参数
    """

    EARTH_RADIUS_M = 6371000.0
    EARTH_GM = 3.986004418e14

    def __init__(self, attitude_calculator: Optional[AttitudeCalculator] = None):
        """
        初始化PMC计算器

        Args:
            attitude_calculator: 姿态角计算器，None则创建默认实例
        """
        self.attitude_calculator = attitude_calculator or AttitudeCalculator()

    def calculate_pitch_rate(
        self,
        orbit_altitude_m: float,
        speed_reduction_ratio: float
    ) -> float:
        """
        计算PMC所需俯仰角速度

        公式: θ_dot = v_orbital * R / h
        其中:
        - v_orbital: 卫星轨道速度
        - R: 降速比
        - h: 轨道高度

        Args:
            orbit_altitude_m: 轨道高度（米）
            speed_reduction_ratio: 降速比（0.1-0.75）

        Returns:
            俯仰角速度（度/秒）
        """
        r = self.EARTH_RADIUS_M + orbit_altitude_m
        v_orbital = math.sqrt(self.EARTH_GM / r)  # m/s

        # 俯仰角速度（弧度/秒）
        pitch_rate_rad_s = v_orbital * speed_reduction_ratio / orbit_altitude_m

        return math.degrees(pitch_rate_rad_s)

    def calculate_ground_velocity(
        self,
        orbit_altitude_m: float,
        speed_reduction_ratio: float
    ) -> float:
        """
        计算PMC模式下的等效地面速度

        Args:
            orbit_altitude_m: 轨道高度（米）
            speed_reduction_ratio: 降速比

        Returns:
            等效地面速度（米/秒）
        """
        r = self.EARTH_RADIUS_M + orbit_altitude_m
        v_orbital = math.sqrt(self.EARTH_GM / r)
        return v_orbital * (1.0 - speed_reduction_ratio)

    def calculate_imaging_sequence(
        self,
        satellite: Satellite,
        target: Target,
        imaging_start: datetime,
        imaging_duration_s: float,
        pmc_config: PitchMotionCompensationConfig,
        time_step_s: float = 1.0
    ) -> PMCImagingSequence:
        """
        计算PMC成像序列

        Args:
            satellite: 卫星对象
            target: 目标对象
            imaging_start: 成像开始时间
            imaging_duration_s: 成像时长（秒）
            pmc_config: PMC配置
            time_step_s: 时间步长（秒）

        Returns:
            PMCImagingSequence
        """
        orbit_altitude = satellite.orbit.get_semi_major_axis() - self.EARTH_RADIUS_M

        # 获取俯仰角速度
        pitch_rate = pmc_config.get_pitch_rate(orbit_altitude)

        # 计算各阶段时间
        enter_duration = pmc_config.enter_exit_time_s
        exit_duration = pmc_config.enter_exit_time_s

        enter_start = imaging_start - timedelta(seconds=enter_duration)
        imaging_end = imaging_start + timedelta(seconds=imaging_duration_s)
        exit_end = imaging_end + timedelta(seconds=exit_duration)

        # 生成时间序列
        states = []
        current_time = enter_start

        while current_time <= exit_end:
            # 计算当前阶段
            if current_time < imaging_start:
                phase = 'enter'
                is_pmc_phase = False
            elif current_time <= imaging_end:
                phase = 'imaging'
                is_pmc_phase = True
            else:
                phase = 'exit'
                is_pmc_phase = False

            # 计算基础姿态（指向目标）
            base_attitude = self.attitude_calculator.calculate_attitude(
                satellite, target, current_time
            )

            # 根据阶段计算PMC俯仰角
            if phase == 'enter':
                # 进入阶段：从0逐渐增加到目标俯仰角
                progress = (current_time - enter_start).total_seconds() / enter_duration
                pmc_pitch_offset = self._calculate_enter_pitch(
                    progress, pitch_rate, enter_duration
                )
            elif phase == 'imaging':
                # 成像阶段：匀速俯仰机动
                imaging_elapsed = (current_time - imaging_start).total_seconds()
                pmc_pitch_offset = pitch_rate * imaging_elapsed
            else:  # exit
                # 退出阶段：从当前俯仰角逐渐回正
                progress = (current_time - imaging_end).total_seconds() / exit_duration
                pmc_pitch_offset = self._calculate_exit_pitch(
                    progress, pitch_rate, imaging_duration_s, exit_duration
                )

            # 等效地面速度
            if is_pmc_phase:
                ground_v = self.calculate_ground_velocity(
                    orbit_altitude, pmc_config.speed_reduction_ratio
                )
            else:
                r = self.EARTH_RADIUS_M + orbit_altitude
                ground_v = math.sqrt(self.EARTH_GM / r)

            state = PMCImagingState(
                timestamp=current_time,
                roll_deg=base_attitude.roll,
                pitch_deg=base_attitude.pitch + pmc_pitch_offset,
                yaw_deg=base_attitude.yaw,
                is_in_pmc_phase=is_pmc_phase,
                ground_velocity_m_s=ground_v
            )
            states.append(state)

            current_time += timedelta(seconds=time_step_s)

        return PMCImagingSequence(
            states=states,
            pmc_config=pmc_config,
            satellite_id=satellite.id,
            target_id=target.id,
            enter_start_time=enter_start,
            imaging_start_time=imaging_start,
            imaging_end_time=imaging_end,
            exit_end_time=exit_end
        )

    def _calculate_enter_pitch(
        self,
        progress: float,
        target_pitch_rate: float,
        enter_duration: float
    ) -> float:
        """
        计算进入阶段的俯仰角偏移

        使用平滑加速曲线避免冲击

        Args:
            progress: 进度（0-1）
            target_pitch_rate: 目标俯仰角速度（度/秒）
            enter_duration: 进入阶段时长（秒）

        Returns:
            俯仰角偏移（度）
        """
        # 使用三次多项式平滑过渡
        # θ(t) = θ_dot * t * (3 - 2t) / T, 其中 t为归一化时间
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        return target_pitch_rate * enter_duration * smooth_progress

    def _calculate_exit_pitch(
        self,
        progress: float,
        pitch_rate: float,
        imaging_duration_s: float,
        exit_duration: float
    ) -> float:
        """
        计算退出阶段的俯仰角偏移

        Args:
            progress: 进度（0-1）
            pitch_rate: 俯仰角速度（度/秒）
            imaging_duration_s: 成像时长（秒）
            exit_duration: 退出阶段时长（秒）

        Returns:
            俯仰角偏移（度）
        """
        # 当前应有的俯仰角（如果没有退出）
        current_pitch = pitch_rate * imaging_duration_s

        # 平滑回正
        smooth_progress = progress * progress * (3.0 - 2.0 * progress)
        return current_pitch * (1.0 - smooth_progress)

    def check_pitch_constraints(
        self,
        sequence: PMCImagingSequence,
        max_pitch_angle_deg: float
    ) -> Tuple[bool, List[str]]:
        """
        检查俯仰角约束

        Args:
            sequence: PMC成像序列
            max_pitch_angle_deg: 最大允许俯仰角（度）

        Returns:
            (是否满足约束, 违反原因列表)
        """
        violations = []

        for state in sequence.states:
            if abs(state.pitch_deg) > max_pitch_angle_deg:
                violations.append(
                    f"Time {state.timestamp.isoformat()}: "
                    f"pitch {state.pitch_deg:.2f}° exceeds limit {max_pitch_angle_deg}°"
                )

        return len(violations) == 0, violations

    def check_roll_constraints(
        self,
        sequence: PMCImagingSequence,
        max_roll_angle_deg: float
    ) -> Tuple[bool, List[str]]:
        """
        检查滚转角约束

        Args:
            sequence: PMC成像序列
            max_roll_angle_deg: 最大允许滚转角（度）

        Returns:
            (是否满足约束, 违反原因列表)
        """
        violations = []

        for state in sequence.states:
            if abs(state.roll_deg) > max_roll_angle_deg:
                violations.append(
                    f"Time {state.timestamp.isoformat()}: "
                    f"roll {state.roll_deg:.2f}° exceeds limit {max_roll_angle_deg}°"
                )

        return len(violations) == 0, violations

    def calculate_snr_gain_db(
        self,
        speed_reduction_ratio: float
    ) -> float:
        """
        计算SNR增益

        SNR与积分时间的平方根成正比
        增益 = 10 * log10(1 / (1 - R))

        Args:
            speed_reduction_ratio: 降速比

        Returns:
            SNR增益（dB）
        """
        integration_gain = 1.0 / (1.0 - speed_reduction_ratio)
        return 10.0 * math.log10(integration_gain)

    def calculate_effective_integration_time(
        self,
        physical_duration_s: float,
        speed_reduction_ratio: float
    ) -> float:
        """
        计算等效积分时间

        Args:
            physical_duration_s: 物理成像时长（秒）
            speed_reduction_ratio: 降速比

        Returns:
            等效积分时间（秒）
        """
        return physical_duration_s / (1.0 - speed_reduction_ratio)

    def estimate_power_consumption(
        self,
        base_power_w: float,
        imaging_duration_s: float,
        pmc_config: PitchMotionCompensationConfig
    ) -> float:
        """
        估算PMC模式总能耗

        Args:
            base_power_w: 基础功耗（瓦特）
            imaging_duration_s: 成像时长（秒）
            pmc_config: PMC配置

        Returns:
            总能耗（Wh）
        """
        # 总时间包含进入和退出
        total_time_s = imaging_duration_s + 2 * pmc_config.enter_exit_time_s

        # PMC模式功耗
        pmc_power = base_power_w * pmc_config.power_overhead_factor

        return (pmc_power * total_time_s) / 3600.0

    def check_feasibility(
        self,
        satellite: Satellite,
        target: Target,
        imaging_start: datetime,
        imaging_duration_s: float,
        pmc_config: PitchMotionCompensationConfig
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        综合检查PMC任务可行性

        Args:
            satellite: 卫星对象
            target: 目标对象
            imaging_start: 成像开始时间
            imaging_duration_s: 成像时长（秒）
            pmc_config: PMC配置

        Returns:
            (是否可行, 详细信息字典)
        """
        orbit_altitude = satellite.orbit.get_semi_major_axis() - self.EARTH_RADIUS_M

        result = {
            'feasible': True,
            'checks': {},
            'errors': [],
            'warnings': [],
        }

        # 1. 检查轨道高度
        altitude_ok, altitude_msg = pmc_config.check_altitude_feasibility(orbit_altitude)
        result['checks']['altitude'] = {'passed': altitude_ok, 'message': altitude_msg}
        if not altitude_ok:
            result['feasible'] = False
            result['errors'].append(altitude_msg)

        # 2. 检查俯仰角速度
        pitch_rate = pmc_config.get_pitch_rate(orbit_altitude)
        if pitch_rate > 2.0:  # 2度/秒为典型上限
            result['checks']['pitch_rate'] = {
                'passed': False,
                'message': f'Pitch rate {pitch_rate:.2f}°/s too high'
            }
            result['feasible'] = False
            result['errors'].append(f'Required pitch rate {pitch_rate:.2f}°/s exceeds limit')
        else:
            result['checks']['pitch_rate'] = {
                'passed': True,
                'message': f'Pitch rate {pitch_rate:.3f}°/s acceptable'
            }

        # 3. 计算成像序列并检查约束
        try:
            sequence = self.calculate_imaging_sequence(
                satellite, target, imaging_start, imaging_duration_s, pmc_config
            )

            # 检查俯仰角约束
            pitch_ok, pitch_violations = self.check_pitch_constraints(
                sequence, pmc_config.max_pitch_angle_deg
            )
            result['checks']['pitch_angle'] = {
                'passed': pitch_ok,
                'violations': pitch_violations
            }
            if not pitch_ok:
                result['feasible'] = False
                result['errors'].extend(pitch_violations)

            # 检查滚转角约束
            roll_ok, roll_violations = self.check_roll_constraints(
                sequence, pmc_config.max_roll_angle_deg
            )
            result['checks']['roll_angle'] = {
                'passed': roll_ok,
                'violations': roll_violations
            }
            if not roll_ok:
                result['feasible'] = False
                result['errors'].extend(roll_violations)

            # 计算SNR增益
            snr_gain = self.calculate_snr_gain_db(pmc_config.speed_reduction_ratio)
            result['snr_gain_db'] = snr_gain

            # 计算等效积分时间
            effective_time = self.calculate_effective_integration_time(
                imaging_duration_s, pmc_config.speed_reduction_ratio
            )
            result['effective_integration_time_s'] = effective_time

            # 估算能耗
            base_power = 150.0 if satellite.capabilities else 150.0
            energy_wh = self.estimate_power_consumption(
                base_power, imaging_duration_s, pmc_config
            )
            result['estimated_energy_wh'] = energy_wh

        except Exception as e:
            result['feasible'] = False
            result['errors'].append(f'Failed to calculate imaging sequence: {str(e)}')

        return result['feasible'], result


def calculate_pmc_parameters(
    orbit_altitude_m: float,
    speed_reduction_ratio: float,
    base_power_w: float = 150.0
) -> Dict[str, float]:
    """
    便捷函数：计算PMC参数

    Args:
        orbit_altitude_m: 轨道高度（米）
        speed_reduction_ratio: 降速比
        base_power_w: 基础功耗（瓦特）

    Returns:
        PMC参数字典
    """
    calculator = PMCCalculator()

    pitch_rate = calculator.calculate_pitch_rate(orbit_altitude_m, speed_reduction_ratio)
    ground_v = calculator.calculate_ground_velocity(orbit_altitude_m, speed_reduction_ratio)
    snr_gain = calculator.calculate_snr_gain_db(speed_reduction_ratio)

    # 计算典型成像时长下的参数
    typical_duration_s = 10.0
    effective_time = calculator.calculate_effective_integration_time(
        typical_duration_s, speed_reduction_ratio
    )

    pmc_config = PitchMotionCompensationConfig(
        speed_reduction_ratio=speed_reduction_ratio
    )
    energy_wh = calculator.estimate_power_consumption(
        base_power_w, typical_duration_s, pmc_config
    )

    return {
        'pitch_rate_dps': pitch_rate,
        'ground_velocity_m_s': ground_v,
        'snr_gain_db': snr_gain,
        'effective_integration_time_s': effective_time,
        'energy_wh': energy_wh,
        'integration_gain': 1.0 / (1.0 - speed_reduction_ratio),
    }
