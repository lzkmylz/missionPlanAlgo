"""
连续状态演化跟踪器

实现第12章设计：
- 卫星状态跟踪器（SatelliteStateTracker）
- 状态枚举（SatelliteState）
- 电量模型（PowerModel）
- 存储积分器（StorageIntegrator）
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta


class SatelliteState(Enum):
    """卫星运行状态枚举"""
    IDLE = "IDLE"               # 空闲/待机
    IMAGING = "IMAGING"         # 成像中
    SLEWING = "SLEWING"         # 姿态机动中
    DOWNLINKING = "DOWNLINKING" # 数传中
    ECLIPSE = "ECLIPSE"         # 地影期（用于未来扩展）
    FAILURE = "FAILURE"         # 故障状态（用于未来扩展）


@dataclass
class StateSnapshot:
    """状态快照 - 记录某时刻的完整状态"""
    timestamp: datetime
    state: SatelliteState
    battery_soc: float
    storage_used_gb: float
    current_task: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SatelliteStateData:
    """卫星状态数据类"""
    satellite_id: str
    timestamp: datetime
    power_wh: float
    storage_gb: float
    is_eclipse: bool = False
    current_activity: Optional[str] = None
    position: Optional[Tuple[float, float, float]] = None
    velocity: Optional[Tuple[float, float, float]] = None


class ImagingState(Enum):
    """成像状态细分"""
    PAYLOAD_OFF = auto()        # 载荷关机
    PAYLOAD_WARMUP = auto()     # 预热中
    SLEWING = auto()            # 姿态机动中
    SHUTTER_OPEN = auto()       # 快门打开
    EXPOSING = auto()           # 曝光中
    SHUTTER_CLOSE = auto()      # 快门关闭
    SLEW_RECOVER = auto()       # 恢复姿态
    DATA_BUFFERING = auto()     # 数据缓存中


@dataclass
class PowerModelConfig:
    """电源模型配置"""
    max_capacity_wh: float = 1000.0           # 最大容量(Wh)
    initial_charge_wh: float = 1000.0         # 初始电量(Wh)，默认满电保持兼容
    nominal_generation_wh_per_sec: float = 10.0  # 正常发电速率(Wh/s)
    eclipse_generation_wh_per_sec: float = 0.0   # 地影中发电速率(Wh/s)
    min_safe_soc: float = 0.1                 # 最小安全电量


class PowerModel:
    """
    电量模型 - H5修复：支持日食充电逻辑

    基于第12章设计：
    - 不同任务状态有不同功耗
    - 考虑卫星类型差异
    - 支持地影中充电逻辑（考虑核电池等）
    """

    # 状态功耗系数（相对于总功率容量的比例）
    POWER_CONSUMPTION_RATES = {
        'IDLE': 0.10,
        'IMAGING_OPTICAL': 0.60,
        'IMAGING_SAR': 0.80,
        'SLEWING': 0.40,
        'DOWNLINKING': 0.80,
    }

    def __init__(self, satellite: Any = None, config: PowerModelConfig = None):
        """
        初始化电源模型

        Args:
            satellite: 卫星对象（可选）
            config: 电源配置，如果为None则使用默认值
        """
        self.satellite = satellite

        if satellite and hasattr(satellite, 'capabilities'):
            self.total_capacity = satellite.capabilities.power_capacity  # Wh
        else:
            self.total_capacity = (config.max_capacity_wh if config else 1000.0)

        # 如果提供了自定义配置，使用配置的初始电量；否则默认满电
        if config:
            self.config = config
            self.current_charge_wh = config.initial_charge_wh
        else:
            # 默认配置，但使用total_capacity作为初始电量（满电）
            self.config = PowerModelConfig(
                max_capacity_wh=self.total_capacity,
                initial_charge_wh=self.total_capacity
            )
            self.current_charge_wh = self.total_capacity

    @property
    def max_capacity_wh(self) -> float:
        """获取最大容量"""
        return self.total_capacity

    @property
    def current_battery_wh(self) -> float:
        """获取当前电量"""
        return self.current_charge_wh

    def get_soc(self) -> float:
        """
        获取当前电量百分比（0-1）

        Returns:
            float: 电量状态(State of Charge)
        """
        return self.current_charge_wh / self.total_capacity if self.total_capacity > 0 else 0.0

    def get_battery_level(self) -> float:
        """获取当前电量百分比（兼容旧接口）"""
        return self.get_soc()

    def calculate_imaging_power(self, duration_seconds: float, imaging_mode: str) -> float:
        """计算成像任务功耗"""
        if self.satellite and hasattr(self.satellite, 'sat_type'):
            sat_type = self.satellite.sat_type.value
            if 'sar' in sat_type.lower():
                rate = self.POWER_CONSUMPTION_RATES['IMAGING_SAR']
            else:
                rate = self.POWER_CONSUMPTION_RATES['IMAGING_OPTICAL']
        else:
            rate = self.POWER_CONSUMPTION_RATES['IMAGING_OPTICAL']

        # 功耗 = 功率 * 时间
        power_wh = self.total_capacity * rate * (duration_seconds / 3600)
        return power_wh

    def calculate_downlink_power(self, duration_seconds: float, data_rate_mbps: float) -> float:
        """计算数传任务功耗"""
        rate = self.POWER_CONSUMPTION_RATES['DOWNLINKING']
        # 数传功耗与数据速率相关
        rate_adjusted = rate * (data_rate_mbps / 300.0)  # 基准300Mbps
        power_wh = self.total_capacity * rate_adjusted * (duration_seconds / 3600)
        return power_wh

    def calculate_idle_power(self, duration_seconds: float) -> float:
        """计算空闲功耗"""
        rate = self.POWER_CONSUMPTION_RATES['IDLE']
        power_wh = self.total_capacity * rate * (duration_seconds / 3600)
        return power_wh

    def calculate_slewing_power(self, duration_seconds: float, slew_angle: float) -> float:
        """计算姿态机动功耗"""
        rate = self.POWER_CONSUMPTION_RATES['SLEWING']
        # 大角度机动功耗更高
        angle_factor = min(slew_angle / 30.0, 2.0)  # 最大2倍
        power_wh = self.total_capacity * rate * angle_factor * (duration_seconds / 3600)
        return power_wh

    def consume_power(self, power_wh: float, duration_seconds: float = 1.0) -> bool:
        """
        消耗电量

        Args:
            power_wh: 功耗(Wh)
            duration_seconds: 持续时间(秒)

        Returns:
            bool: 是否有足够电量支持此操作
        """
        total_consumption = power_wh * duration_seconds
        if total_consumption > self.current_charge_wh:
            # 电量不足，消耗到0但返回False
            self.current_charge_wh = 0.0
            return False
        self.current_charge_wh = max(0.0, self.current_charge_wh - total_consumption)
        return True

    def charge(self, duration_seconds: float, in_eclipse: bool = False) -> float:
        """
        充电 - H5关键修复：支持地影中充电逻辑

        Args:
            duration_seconds: 充电时间(秒)
            in_eclipse: 是否在地影中

        Returns:
            float: 实际充电量(Wh)
        """
        # 根据地影状态选择发电速率
        if in_eclipse:
            generation_rate = self.config.eclipse_generation_wh_per_sec
        else:
            generation_rate = self.config.nominal_generation_wh_per_sec

        # 计算充电量
        charge_amount = generation_rate * duration_seconds

        # 限制不超过最大容量
        old_charge = self.current_charge_wh
        self.current_charge_wh = min(self.total_capacity, self.current_charge_wh + charge_amount)
        actual_charged = self.current_charge_wh - old_charge

        return actual_charged

    def simulate_activity(
        self,
        activity_type: str,
        duration_seconds: float,
        in_eclipse: bool = False,
        **kwargs
    ) -> Dict[str, float]:
        """
        模拟活动期间的电量变化 - H5关键功能

        Args:
            activity_type: 活动类型 ('imaging', 'downlink', 'idle', 'slewing')
            duration_seconds: 持续时间(秒)
            in_eclipse: 是否在地影中
            **kwargs: 其他参数

        Returns:
            Dict: 包含电量变化信息的字典
        """
        initial_charge = self.current_charge_wh

        # 计算活动功耗
        if activity_type == 'imaging':
            imaging_mode = kwargs.get('imaging_mode', 'optical')
            power_consumption = self.calculate_imaging_power(duration_seconds, imaging_mode)
        elif activity_type == 'downlink':
            data_rate = kwargs.get('data_rate_mbps', 300.0)
            power_consumption = self.calculate_downlink_power(duration_seconds, data_rate)
        elif activity_type == 'slewing':
            slew_angle = kwargs.get('slew_angle', 0.0)
            power_consumption = self.calculate_slewing_power(duration_seconds, slew_angle)
        else:  # idle
            power_consumption = self.calculate_idle_power(duration_seconds)

        # 充电（根据地影状态）
        charge_amount = self.charge(duration_seconds, in_eclipse)

        # 耗电
        self.current_charge_wh = max(0.0, self.current_charge_wh - power_consumption)

        # 净变化
        net_change = self.current_charge_wh - initial_charge

        return {
            'initial_charge_wh': initial_charge,
            'final_charge_wh': self.current_charge_wh,
            'power_consumed_wh': power_consumption,
            'power_generated_wh': charge_amount,
            'net_change_wh': net_change,
            'soc': self.get_soc()
        }

    def can_support_activity(
        self,
        power_consumption_rate_wh_per_sec: float,
        duration_seconds: float,
        in_eclipse: bool = False
    ) -> bool:
        """
        检查是否支持指定活动

        Args:
            power_consumption_rate_wh_per_sec: 功耗速率(Wh/s)
            duration_seconds: 持续时间(秒)
            in_eclipse: 是否在地影中

        Returns:
            bool: 是否支持
        """
        total_consumption = power_consumption_rate_wh_per_sec * duration_seconds

        # 计算活动期间的发电量
        if in_eclipse:
            generation = self.config.eclipse_generation_wh_per_sec * duration_seconds
        else:
            generation = self.config.nominal_generation_wh_per_sec * duration_seconds

        # 净消耗
        net_consumption = total_consumption - generation

        # 检查电量是否足够
        if net_consumption <= 0:
            return True  # 发电大于耗电，总是支持

        return self.current_charge_wh >= net_consumption + (self.total_capacity * self.config.min_safe_soc)

    def get_power_status(self) -> Dict[str, Any]:
        """
        获取电源状态

        Returns:
            Dict: 电源状态信息
        """
        return {
            'current_charge_wh': self.current_charge_wh,
            'max_capacity_wh': self.total_capacity,
            'soc': self.get_soc(),
            'nominal_generation_wh_per_sec': self.config.nominal_generation_wh_per_sec,
            'eclipse_generation_wh_per_sec': self.config.eclipse_generation_wh_per_sec,
            'min_safe_soc': self.config.min_safe_soc
        }


class StorageIntegrator:
    """
    存储积分器

    跟踪卫星存储空间的使用情况
    """

    def __init__(self, satellite: Any):
        self.satellite = satellite
        self.total_capacity = satellite.capabilities.storage_capacity  # GB
        self._data_log: List[Dict[str, Any]] = []  # [(timestamp, target_id, size_gb, operation)]

    def get_used_storage(self) -> float:
        """获取当前已用存储空间"""
        total = sum(
            entry['size_gb'] if entry['operation'] == 'ADD' else -entry['size_gb']
            for entry in self._data_log
        )
        return max(0.0, total)

    def get_available_storage(self) -> float:
        """获取可用存储空间"""
        return self.total_capacity - self.get_used_storage()

    def add_imaging_data(self, target_id: str, data_size_gb: float, timestamp: datetime) -> None:
        """添加成像数据"""
        if data_size_gb > self.get_available_storage():
            raise ValueError(
                f"Storage overflow: trying to add {data_size_gb}GB, "
                f"but only {self.get_available_storage()}GB available"
            )

        self._data_log.append({
            'timestamp': timestamp,
            'target_id': target_id,
            'size_gb': data_size_gb,
            'operation': 'ADD'
        })

    def remove_downlinked_data(self, data_size_gb: float, timestamp: datetime) -> float:
        """移除已下传的数据"""
        actual_removed = min(data_size_gb, self.get_used_storage())
        if actual_removed > 0:
            self._data_log.append({
                'timestamp': timestamp,
                'target_id': 'downlink',
                'size_gb': actual_removed,
                'operation': 'REMOVE'
            })
        return actual_removed

    def get_storage_at_time(self, timestamp: datetime) -> float:
        """获取指定时间点的存储状态"""
        total = 0.0
        for entry in self._data_log:
            if entry['timestamp'] <= timestamp:
                if entry['operation'] == 'ADD':
                    total += entry['size_gb']
                else:
                    total -= entry['size_gb']
        return max(0.0, total)

    def get_data_by_target(self, target_id: str) -> float:
        """获取特定目标的数据量"""
        total = 0.0
        for entry in self._data_log:
            if entry['target_id'] == target_id:
                if entry['operation'] == 'ADD':
                    total += entry['size_gb']
                else:
                    total -= entry['size_gb']
        return max(0.0, total)


@dataclass
class SatelliteStateInfo:
    """
    卫星状态信息结构

    对应第12章的SatelliteState枚举
    """
    satellite_id: str
    state: SatelliteState
    battery_soc: float  # 电量百分比 0-1
    storage_used_gb: float
    current_task: Optional[str] = None
    imaging_progress: float = 0.0  # 成像进度 0-1
    temperature: float = 20.0  # 温度（摄氏度）
    temperature_kelvin: float = 293.15  # 温度（开尔文）


class SatelliteStateTracker:
    """
    卫星状态跟踪器

    实现第12章设计：
    - 记录卫星随时间的状态变化
    - 支持电量、存储、任务状态的演化计算
    - 提供任意时间点的状态查询
    - 集成热控模型（第16章）
    """

    def __init__(self, satellite: Any, thermal_integrator: Any = None):
        self.satellite = satellite
        self.power_model = PowerModel(satellite)
        self.storage = StorageIntegrator(satellite)
        self.thermal_integrator = thermal_integrator  # 可选的热控积分器

        # 状态变化记录
        self._state_log: List[Dict[str, Any]] = []

        # 初始化状态
        self._record_state(
            timestamp=datetime(2024, 1, 1, 0, 0),
            state=SatelliteState.IDLE,
            context={'init': True}
        )

    def _record_state(self, timestamp: datetime, state: SatelliteState, context: Dict = None) -> None:
        """记录状态变化"""
        log_entry = {
            'timestamp': timestamp,
            'state': state,
            'battery_soc': self.power_model.get_battery_level(),
            'storage_gb': self.storage.get_used_storage(),
            'context': context or {}
        }

        # 如果有热控积分器，记录温度
        if self.thermal_integrator is not None:
            log_entry['temperature_k'] = self.thermal_integrator.temperature

        self._state_log.append(log_entry)

    def record_imaging_task(self, target_id: str, start_time: datetime, end_time: datetime, data_size_gb: float) -> None:
        """记录成像任务"""
        duration = (end_time - start_time).total_seconds()

        # 计算功耗
        power_consumed = self.power_model.calculate_imaging_power(
            duration, "push_broom"
        )
        has_power = self.power_model.consume_power(power_consumed)
        if not has_power:
            # 电量不足，记录失败状态
            self._record_state(
                timestamp=start_time,
                state=SatelliteState.IDLE,
                context={'error': 'insufficient_power', 'task': 'imaging_failed'}
            )
            return

        # 添加数据到存储
        self.storage.add_imaging_data(target_id, data_size_gb, start_time)

        # 更新热控状态（如果有热控积分器）
        if self.thermal_integrator is not None:
            # 确定成像模式
            activity_mode = 'imaging_stripmap'
            if hasattr(self.satellite, 'capabilities') and self.satellite.capabilities.imaging_modes:
                mode = self.satellite.capabilities.imaging_modes[0]
                if mode.value == 'spotlight':
                    activity_mode = 'imaging_spotlight'
                elif mode.value == 'sliding_spotlight':
                    activity_mode = 'imaging_sliding_spotlight'

            self.thermal_integrator.update(start_time, activity_mode)
            self.thermal_integrator.update(end_time, 'idle')

        # 记录状态
        self._record_state(
            timestamp=start_time,
            state=SatelliteState.IMAGING,
            context={'target_id': target_id, 'task': 'imaging_start'}
        )
        self._record_state(
            timestamp=end_time,
            state=SatelliteState.IDLE,
            context={'target_id': target_id, 'task': 'imaging_end'}
        )

    def record_downlink_task(self, ground_station_id: str, start_time: datetime, end_time: datetime, data_size_gb: float) -> None:
        """记录数传任务"""
        duration = (end_time - start_time).total_seconds()

        # 计算功耗
        power_consumed = self.power_model.calculate_downlink_power(
            duration, 300  # 假设300Mbps
        )
        has_power = self.power_model.consume_power(power_consumed)
        if not has_power:
            self._record_state(
                timestamp=start_time,
                state=SatelliteState.IDLE,
                context={'error': 'insufficient_power', 'task': 'downlink_failed'}
            )
            return

        # 移除已下传的数据
        self.storage.remove_downlinked_data(data_size_gb, start_time)

        # 记录状态
        self._record_state(
            timestamp=start_time,
            state=SatelliteState.DOWNLINKING,
            context={'gs_id': ground_station_id, 'task': 'downlink_start'}
        )
        self._record_state(
            timestamp=end_time,
            state=SatelliteState.IDLE,
            context={'gs_id': ground_station_id, 'task': 'downlink_end'}
        )

    def record_slewing(self, from_target: str, to_target: str, start_time: datetime, end_time: datetime, slew_angle: float) -> None:
        """记录姿态机动"""
        duration = (end_time - start_time).total_seconds()

        # 计算机动功耗
        power_consumed = self.power_model.calculate_slewing_power(
            duration, slew_angle
        )
        has_power = self.power_model.consume_power(power_consumed)
        if not has_power:
            self._record_state(
                timestamp=start_time,
                state=SatelliteState.IDLE,
                context={'error': 'insufficient_power', 'task': 'slew_failed'}
            )
            return

        # 记录状态
        self._record_state(
            timestamp=start_time,
            state=SatelliteState.SLEWING,
            context={'from': from_target, 'to': to_target, 'angle': slew_angle}
        )
        self._record_state(
            timestamp=end_time,
            state=SatelliteState.IDLE,
            context={'from': from_target, 'to': to_target, 'task': 'slew_complete'}
        )

    def get_state_at(self, timestamp: datetime) -> SatelliteStateInfo:
        """获取指定时间点的状态"""
        # 查找该时间点之前的最新状态记录
        relevant_states = [
            s for s in self._state_log
            if s['timestamp'] <= timestamp
        ]

        if not relevant_states:
            # 如果没有找到，返回初始状态
            return SatelliteStateInfo(
                satellite_id=self.satellite.id,
                state=SatelliteState.IDLE,
                battery_soc=1.0,
                storage_used_gb=0.0
            )

        # 获取最新状态
        latest = max(relevant_states, key=lambda s: s['timestamp'])

        # 计算空闲期间的电量消耗
        idle_duration = (timestamp - latest['timestamp']).total_seconds()
        if idle_duration > 0:
            idle_power = self.power_model.calculate_idle_power(idle_duration)
            temp_battery = max(0.0, latest['battery_soc'] * self.power_model.total_capacity - idle_power)
            battery_soc = temp_battery / self.power_model.total_capacity
        else:
            battery_soc = latest['battery_soc']

        # 获取温度信息
        temperature_c = 20.0
        temperature_k = 293.15
        if self.thermal_integrator is not None:
            temperature_k = self.thermal_integrator.temperature
            temperature_c = temperature_k - 273.15

        return SatelliteStateInfo(
            satellite_id=self.satellite.id,
            state=latest['state'],
            battery_soc=battery_soc,
            storage_used_gb=self.storage.get_storage_at_time(timestamp),
            current_task=latest['context'].get('target_id'),
            temperature=temperature_c,
            temperature_kelvin=temperature_k
        )

    def get_state_log(self) -> List[Dict[str, Any]]:
        """获取完整的状态日志"""
        return self._state_log.copy()

    def check_thermal_constraint(self, activity: str, duration: float) -> Tuple[bool, str]:
        """
        检查热控约束

        Args:
            activity: 活动类型
            duration: 活动持续时间（秒）

        Returns:
            (是否可行, 原因)
        """
        if self.thermal_integrator is None:
            return True, ""

        is_valid, predicted_temp = self.thermal_integrator.is_temperature_valid(
            activity, duration
        )

        if not is_valid:
            return False, f"Thermal constraint violation: predicted temperature {predicted_temp:.2f}K exceeds limit"

        return True, ""

    def validate_schedule(self, tasks: List[Any]) -> Tuple[bool, List[str]]:
        """
        验证调度方案的可行性

        Returns:
            (是否可行, 违规列表)
        """
        violations = []

        # 检查电量约束
        min_battery = min(s['battery_soc'] for s in self._state_log)
        if min_battery < 0.1:  # 电量低于10%
            violations.append(f"Battery level dropped to {min_battery:.1%}, below safety threshold")

        # 检查存储约束
        max_storage = max(s['storage_gb'] for s in self._state_log)
        if max_storage > self.satellite.capabilities.storage_capacity:
            violations.append(f"Storage overflow: {max_storage:.1f}GB > {self.satellite.capabilities.storage_capacity}GB")

        # 检查热控约束
        if self.thermal_integrator is not None:
            temp_status = self.thermal_integrator.get_thermal_status()
            if not temp_status['is_safe']:
                violations.append(f"Thermal constraint violation: temperature margin {temp_status['temperature_margin_k']:.2f}K below safety threshold")

            # 检查是否超过最大工作温度
            if temp_status['current_temperature_k'] > self.thermal_integrator.params.max_operating_temp:
                violations.append(f"Temperature {temp_status['current_temperature_k']:.2f}K exceeds max operating temperature")

            # 检查是否低于最小工作温度
            if temp_status['current_temperature_k'] < self.thermal_integrator.params.min_operating_temp:
                violations.append(f"Temperature {temp_status['current_temperature_k']:.2f}K below min operating temperature")

        return len(violations) == 0, violations
