"""
卫星模型 - 定义异构卫星的属性和能力

支持四种卫星类型：
- 光学1型：基础光学成像
- 光学2型：增强光学成像
- SAR-1型：支持聚束/滑动聚束/条带模式
- SAR-2型：增强SAR成像
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import math


class SatelliteType(Enum):
    """卫星类型枚举"""
    OPTICAL_1 = "optical_1"
    OPTICAL_2 = "optical_2"
    SAR_1 = "sar_1"
    SAR_2 = "sar_2"


class ImagingMode(Enum):
    """成像模式枚举"""
    # SAR模式
    SPOTLIGHT = "spotlight"
    SLIDING_SPOTLIGHT = "sliding_spotlight"
    STRIPMAP = "stripmap"
    SCAN = "scan"
    # 光学模式
    PUSH_BROOM = "push_broom"
    FRAME = "frame"


class OrbitType(Enum):
    """轨道类型"""
    SSO = "SSO"  # 太阳同步轨道
    LEO = "LEO"  # 近地轨道
    MEO = "MEO"  # 中地球轨道
    GEO = "GEO"  # 地球静止轨道


class OrbitSource(Enum):
    """轨道数据来源"""
    ELEMENTS = "elements"  # 轨道六根数
    TLE = "tle"  # TLE两行根数
    SIMPLIFIED = "simplified"  # 简化参数（仅高度和倾角）


@dataclass
class Orbit:
    """
    轨道参数

    支持三种配置方式：
    1. 轨道六根数：semi_major_axis, eccentricity, inclination, raan, arg_of_perigee, mean_anomaly
    2. TLE两行根数：tle_line1, tle_line2
    3. 简化参数：altitude, inclination（用于快速配置）

    优先级：TLE > 六根数 > 简化参数
    """
    orbit_type: OrbitType = OrbitType.SSO

    # 轨道六根数（经典开普勒轨道元素）
    semi_major_axis: Optional[float] = None  # 半长轴（米），None时从altitude计算
    eccentricity: float = 0.0  # 偏心率（0-1）
    inclination: float = 97.4  # 轨道倾角（度）
    raan: float = 0.0  # 升交点赤经/RAAN（度）
    arg_of_perigee: float = 0.0  # 近地点幅角（度）
    mean_anomaly: float = 0.0  # 平近点角（度）

    # 简化参数（向后兼容）
    altitude: float = 500000.0  # 轨道高度（米）

    # TLE两行根数
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None

    # 轨道数据来源标志
    source: OrbitSource = field(default=OrbitSource.SIMPLIFIED)

    def __post_init__(self):
        """初始化后确定轨道数据来源"""
        if self.tle_line1 and self.tle_line2:
            self.source = OrbitSource.TLE
        elif self.semi_major_axis is not None:
            self.source = OrbitSource.ELEMENTS
            # 从半长轴计算高度
            EARTH_RADIUS = 6371000.0
            self.altitude = self.semi_major_axis - EARTH_RADIUS
        elif self.altitude > 0:
            self.source = OrbitSource.SIMPLIFIED
            # 从高度计算半长轴
            EARTH_RADIUS = 6371000.0
            self.semi_major_axis = EARTH_RADIUS + self.altitude
        else:
            # 默认值
            self.source = OrbitSource.SIMPLIFIED
            self.altitude = 500000.0
            EARTH_RADIUS = 6371000.0
            self.semi_major_axis = EARTH_RADIUS + self.altitude

    def get_semi_major_axis(self) -> float:
        """获取半长轴（米）"""
        if self.semi_major_axis is not None:
            return self.semi_major_axis
        EARTH_RADIUS = 6371000.0
        return EARTH_RADIUS + self.altitude

    def get_period(self) -> float:
        """获取轨道周期（秒）"""
        GM = 3.986004418e14  # 地球引力常数
        a = self.get_semi_major_axis()
        return 2 * math.pi * math.sqrt(a**3 / GM)


@dataclass
class SatelliteCapabilities:
    """卫星能力配置"""
    # 成像能力
    imaging_modes: List[ImagingMode] = field(default_factory=list)
    max_off_nadir: float = 30.0  # 最大侧摆角（度）
    agility: Dict[str, float] = field(default_factory=lambda: {
        'max_slew_rate': 3.0,  # 度/秒
        'slew_acceleration': 1.5,  # 度/秒²
        'settling_time': 5.0  # 稳定时间（秒）
    })

    # 存储和能源
    storage_capacity: float = 500.0  # GB
    power_capacity: float = 2000.0  # Wh
    data_rate: float = 300.0  # Mbps

    # 成像参数
    resolution: float = 10.0  # 分辨率（米）
    swath_width: float = 10000.0  # 幅宽（米）

    # 详细成像器配置（JSON模板中的详细参数）
    imager: Dict[str, Any] = field(default_factory=dict)

    # 详细成像模式参数（JSON模板中的模式详细配置）
    imaging_mode_details: List[Dict[str, Any]] = field(default_factory=list)

    def supports_mode(self, mode: ImagingMode) -> bool:
        """检查是否支持指定成像模式"""
        return mode in self.imaging_modes


@dataclass
class Satellite:
    """
    卫星模型

    Attributes:
        id: 卫星唯一标识
        name: 卫星名称
        sat_type: 卫星类型
        orbit: 轨道参数
        capabilities: 能力配置
        tle_line1: TLE第一行（可选）
        tle_line2: TLE第二行（可选）
    """
    id: str
    name: str
    sat_type: SatelliteType
    orbit: Orbit = field(default_factory=Orbit)
    capabilities: SatelliteCapabilities = field(default_factory=SatelliteCapabilities)

    # TLE（用于SGP4传播）
    tle_line1: Optional[str] = None
    tle_line2: Optional[str] = None

    # 当前状态（运行时被更新）
    current_power: float = field(default=0.0)  # 当前电量
    current_storage: float = field(default=0.0)  # 当前存储使用

    def __post_init__(self):
        """初始化后设置默认值"""
        if not self.capabilities.imaging_modes:
            self._set_default_capabilities()
        if self.current_power == 0.0:
            self.current_power = self.capabilities.power_capacity

    def _set_default_capabilities(self):
        """根据卫星类型设置默认能力"""
        if self.sat_type == SatelliteType.OPTICAL_1:
            self.capabilities.imaging_modes = [ImagingMode.PUSH_BROOM]
            self.capabilities.max_off_nadir = 30.0
            self.capabilities.storage_capacity = 500.0
            self.capabilities.power_capacity = 2000.0
            self.capabilities.resolution = 10.0
        elif self.sat_type == SatelliteType.OPTICAL_2:
            self.capabilities.imaging_modes = [ImagingMode.PUSH_BROOM, ImagingMode.FRAME]
            self.capabilities.max_off_nadir = 45.0
            self.capabilities.storage_capacity = 800.0
            self.capabilities.power_capacity = 2500.0
            self.capabilities.resolution = 5.0
        elif self.sat_type == SatelliteType.SAR_1:
            self.capabilities.imaging_modes = [
                ImagingMode.SPOTLIGHT,
                ImagingMode.SLIDING_SPOTLIGHT,
                ImagingMode.STRIPMAP
            ]
            self.capabilities.max_off_nadir = 35.0
            self.capabilities.storage_capacity = 1000.0
            self.capabilities.power_capacity = 3000.0
            self.capabilities.resolution = 3.0
        elif self.sat_type == SatelliteType.SAR_2:
            self.capabilities.imaging_modes = [
                ImagingMode.SPOTLIGHT,
                ImagingMode.SLIDING_SPOTLIGHT,
                ImagingMode.STRIPMAP
            ]
            self.capabilities.max_off_nadir = 50.0
            self.capabilities.storage_capacity = 1500.0
            self.capabilities.power_capacity = 4000.0
            self.capabilities.resolution = 1.0

    def get_position_sgp4(self, dt: datetime) -> tuple:
        """
        使用SGP4计算卫星位置

        Returns:
            (x, y, z) in ECI coordinates (meters)
        """
        from sgp4.api import Satrec, jday

        if not self.tle_line1 or not self.tle_line2:
            # 如果没有TLE，使用简化轨道模型
            return self._get_position_simplified(dt)

        sat = Satrec.twoline2rv(self.tle_line1, self.tle_line2)
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)

        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            raise RuntimeError(f"SGP4 propagation error: {e}")

        return r  # (x, y, z) in km

    def _get_position_simplified(self, dt: datetime) -> tuple:
        """简化的轨道位置计算（用于没有TLE的情况）"""
        # 这是一个简化的圆轨道模型
        period = self.orbit.get_period()
        mean_motion = 2 * math.pi / period

        # 计算从参考时间开始的秒数
        ref_time = datetime(2024, 1, 1)
        delta_t = (dt - ref_time).total_seconds()

        # 平近点角
        M = self.orbit.mean_anomaly + math.radians(mean_motion * delta_t)

        # 简化假设：圆轨道
        a = self.orbit.get_semi_major_axis()
        i = math.radians(self.orbit.inclination)
        raan = math.radians(self.orbit.raan)

        # 在轨道平面内的位置
        x_orb = a * math.cos(M)
        y_orb = a * math.sin(M)

        # 转换到ECI坐标系（简化，假设轨道倾角为i，升交点赤经为raan）
        x = x_orb * math.cos(raan) - y_orb * math.cos(i) * math.sin(raan)
        y = x_orb * math.sin(raan) + y_orb * math.cos(i) * math.cos(raan)
        z = y_orb * math.sin(i)

        return (x / 1000, y / 1000, z / 1000)  # 转换为km以与SGP4一致

    def get_subpoint(self, dt: datetime) -> tuple:
        """
        获取星下点坐标

        Returns:
            (latitude, longitude, altitude) in degrees and meters
        """
        from sgp4.api import jday

        r = self.get_position_sgp4(dt)

        # ECI to LLA conversion (simplified)
        x, y, z = r
        r_norm = math.sqrt(x**2 + y**2 + z**2)

        # 纬度
        lat = math.degrees(math.asin(z / r_norm))

        # 经度（需要计算GMST，这里简化）
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
        # 简化格林尼治恒星时计算
        gmst = (280.46061837 + 360.98564736629 * (jd - 2451545.0)) % 360
        lon = (math.degrees(math.atan2(y, x)) - gmst) % 360
        if lon > 180:
            lon -= 360

        alt = r_norm * 1000 - 6371000  # 转换为米

        return (lat, lon, alt)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # 根据轨道来源选择序列化格式
        orbit_dict: Dict[str, Any] = {
            'orbit_type': self.orbit.orbit_type.value,
            'source': self.orbit.source.value,
        }

        if self.orbit.source == OrbitSource.TLE and self.orbit.tle_line1 and self.orbit.tle_line2:
            # TLE格式
            orbit_dict['tle_line1'] = self.orbit.tle_line1
            orbit_dict['tle_line2'] = self.orbit.tle_line2
        elif self.orbit.source == OrbitSource.ELEMENTS:
            # 六根数格式
            orbit_dict['semi_major_axis'] = self.orbit.semi_major_axis
            orbit_dict['eccentricity'] = self.orbit.eccentricity
            orbit_dict['inclination'] = self.orbit.inclination
            orbit_dict['raan'] = self.orbit.raan
            orbit_dict['arg_of_perigee'] = self.orbit.arg_of_perigee
            orbit_dict['mean_anomaly'] = self.orbit.mean_anomaly
        else:
            # 简化格式
            orbit_dict['altitude'] = self.orbit.altitude
            orbit_dict['inclination'] = self.orbit.inclination

        return {
            'id': self.id,
            'name': self.name,
            'sat_type': self.sat_type.value,
            'orbit': orbit_dict,
            'capabilities': {
                'imaging_modes': [m.value for m in self.capabilities.imaging_modes],
                'max_off_nadir': self.capabilities.max_off_nadir,
                'storage_capacity': self.capabilities.storage_capacity,
                'power_capacity': self.capabilities.power_capacity,
                'data_rate': self.capabilities.data_rate,
                'imager': self.capabilities.imager,
                'imaging_mode_details': self.capabilities.imaging_mode_details,
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Satellite':
        """从字典创建，支持多种轨道配置格式"""
        sat_type = SatelliteType(data['sat_type'])

        orbit_data = data.get('orbit', {})

        # 优先级1: 检查是否有TLE（支持orbit.tle_line1/2或orbit.tle数组）
        tle_line1 = orbit_data.get('tle_line1')
        tle_line2 = orbit_data.get('tle_line2')
        if not tle_line1 and not tle_line2:
            tle_array = orbit_data.get('tle', [])
            if len(tle_array) >= 2:
                tle_line1 = tle_array[0]
                tle_line2 = tle_array[1]

        # 优先级2: 检查是否有完整的六根数
        semi_major_axis = orbit_data.get('semi_major_axis')

        # 创建Orbit对象
        if tle_line1 and tle_line2:
            # TLE格式
            orbit = Orbit(
                orbit_type=OrbitType(orbit_data.get('orbit_type', 'SSO')),
                tle_line1=tle_line1,
                tle_line2=tle_line2,
                inclination=orbit_data.get('inclination', 97.4),
                raan=orbit_data.get('raan', 0.0),
            )
        elif semi_major_axis is not None:
            # 六根数格式
            orbit = Orbit(
                orbit_type=OrbitType(orbit_data.get('orbit_type', 'SSO')),
                semi_major_axis=semi_major_axis,
                eccentricity=orbit_data.get('eccentricity', 0.0),
                inclination=orbit_data.get('inclination', 97.4),
                raan=orbit_data.get('raan', 0.0),
                arg_of_perigee=orbit_data.get('arg_of_perigee', 0.0),
                mean_anomaly=orbit_data.get('mean_anomaly', 0.0),
            )
        else:
            # 简化格式（向后兼容）
            orbit = Orbit(
                orbit_type=OrbitType(orbit_data.get('orbit_type', 'SSO')),
                altitude=orbit_data.get('altitude', 500000.0),
                inclination=orbit_data.get('inclination', 97.4),
                eccentricity=orbit_data.get('eccentricity', 0.0),
                raan=orbit_data.get('raan', 0.0),
                arg_of_perigee=orbit_data.get('arg_of_perigee', 0.0),
                mean_anomaly=orbit_data.get('mean_anomaly', 0.0),
            )

        cap_data = data.get('capabilities', {})

        # 读取详细成像器配置（支持多种可能的字段名）
        imager_data = cap_data.get('imager', {})
        if not imager_data:
            # 尝试从旧格式的字段构建imager配置
            imager_data = {}

        # 读取详细成像模式配置（支持多种可能的字段名）
        imaging_mode_details = cap_data.get('imaging_mode_details', [])
        if not imaging_mode_details:
            imaging_mode_details = cap_data.get('imaging_modes_details', [])

        # 解析成像模式列表（支持字符串列表或对象列表）
        raw_imaging_modes = cap_data.get('imaging_modes', [])
        imaging_modes = []
        for m in raw_imaging_modes:
            if isinstance(m, str):
                # 旧格式：字符串列表
                imaging_modes.append(ImagingMode(m))
            elif isinstance(m, dict) and 'mode_id' in m:
                # 新格式：对象列表，提取mode_id
                imaging_modes.append(ImagingMode(m['mode_id']))
                # 同时将对象添加到imaging_mode_details（如果还没有的话）
                if m not in imaging_mode_details:
                    imaging_mode_details.append(m)

        capabilities = SatelliteCapabilities(
            imaging_modes=imaging_modes,
            max_off_nadir=cap_data.get('max_off_nadir', 30.0),
            storage_capacity=cap_data.get('storage_capacity', 500.0),
            power_capacity=cap_data.get('power_capacity', 2000.0),
            data_rate=cap_data.get('data_rate', 300.0),
            imager=imager_data,
            imaging_mode_details=imaging_mode_details,
        )

        # 读取TLE（支持多种格式）
        tle_line1 = data.get('tle_line1')
        tle_line2 = data.get('tle_line2')
        # 如果根级别没有，尝试介orbit.tle数组
        if not tle_line1 and not tle_line2:
            tle_array = orbit_data.get('tle', [])
            if len(tle_array) >= 2:
                tle_line1 = tle_array[0]
                tle_line2 = tle_array[1]

        # 读取TLE用于Satellite对象（优先级：orbit.tle_line1/2 > orbit.tle数组 > 根级别tle）
        sat_tle_line1 = orbit.tle_line1 if orbit.tle_line1 else data.get('tle_line1')
        sat_tle_line2 = orbit.tle_line2 if orbit.tle_line2 else data.get('tle_line2')

        return cls(
            id=data['id'],
            name=data['name'],
            sat_type=sat_type,
            orbit=orbit,
            capabilities=capabilities,
            tle_line1=sat_tle_line1,
            tle_line2=sat_tle_line2,
        )
