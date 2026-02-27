"""
轨道工具函数

提供轨道计算相关的共享工具函数和常量
"""

import math
from typing import Tuple

# =============================================================================
# 轨道常数
# =============================================================================

# 地球J2项系数（扁率）
EARTH_J2 = 1.08263e-3

# 地球半径（米）
EARTH_RADIUS_M = 6371000.0

# 地球引力常数（m^3/s^2）
EARTH_GM = 3.986004418e14

# 地球自转角速度（rad/s）
EARTH_ROTATION_RATE = 7.2921159e-5


# =============================================================================
# 通用工具函数
# =============================================================================

def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    将值限制在指定范围内

    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值

    Returns:
        限制在[min_val, max_val]范围内的值
    """
    return max(min_val, min(max_val, value))


# =============================================================================
# J2摄动计算
# =============================================================================

def calculate_j2_perturbations(
    semi_major_axis: float,
    inclination: float,
    eccentricity: float,
    mean_motion: float
) -> Tuple[float, float]:
    """
    计算J2摄动引起的轨道参数长期变化率

    公式来源：
    - RAAN进动率: dRAAN/dt = -3/2 * n * J2 * (R/a)^2 * cos(i) / (1-e^2)^2
    - 近地点幅角变化率: dω/dt = 3/4 * n * J2 * (R/a)^2 * (5*cos^2(i) - 1) / (1-e^2)^2

    Args:
        semi_major_axis: 轨道半长轴（米）
        inclination: 轨道倾角（度）
        eccentricity: 轨道偏心率
        mean_motion: 平均运动（rad/s）

    Returns:
        (raan_dot, arg_perigee_dot): RAAN进动率和近地点幅角变化率（rad/s）
    """
    # 计算中间量
    cos_i = math.cos(math.radians(inclination))
    cos_i_sq = cos_i ** 2
    factor = (1 - eccentricity**2)**2

    # 避免除零
    if factor <= 0:
        return 0.0, 0.0

    # 计算J2项系数
    j2_factor = EARTH_J2 * (EARTH_RADIUS_M / semi_major_axis)**2

    # RAAN进动率（rad/s）
    # 太阳同步轨道维持的关键
    raan_dot = -1.5 * mean_motion * j2_factor * cos_i / factor

    # 近地点幅角变化率（rad/s）
    arg_perigee_dot = 0.75 * mean_motion * j2_factor * (5*cos_i_sq - 1) / factor

    return raan_dot, arg_perigee_dot


def apply_j2_perturbations(
    raan: float,
    arg_of_perigee: float,
    delta_t: float,
    semi_major_axis: float,
    inclination: float,
    eccentricity: float,
    mean_motion: float
) -> Tuple[float, float]:
    """
    应用J2摄动修正，计算指定时间后的轨道参数

    Args:
        raan: 初始升交点赤经（度）
        arg_of_perigee: 初始近地点幅角（度）
        delta_t: 时间偏移（秒）
        semi_major_axis: 轨道半长轴（米）
        inclination: 轨道倾角（度）
        eccentricity: 轨道偏心率
        mean_motion: 平均运动（rad/s）

    Returns:
        (raan_corrected, arg_perigee_corrected): 修正后的RAAN和近地点幅角（度）
    """
    raan_dot, arg_perigee_dot = calculate_j2_perturbations(
        semi_major_axis, inclination, eccentricity, mean_motion
    )

    # 应用变化率
    raan_corrected = raan + math.degrees(raan_dot * delta_t)
    arg_perigee_corrected = arg_of_perigee + math.degrees(arg_perigee_dot * delta_t)

    return raan_corrected, arg_perigee_corrected


# =============================================================================
# 轨道几何计算
# =============================================================================

def calculate_orbital_period(semi_major_axis: float) -> float:
    """
    计算轨道周期

    Args:
        semi_major_axis: 轨道半长轴（米）

    Returns:
        轨道周期（秒）
    """
    return 2 * math.pi * math.sqrt(semi_major_axis**3 / EARTH_GM)


def calculate_mean_motion(semi_major_axis: float) -> float:
    """
    计算平均运动

    Args:
        semi_major_axis: 轨道半长轴（米）

    Returns:
        平均运动（rad/s）
    """
    return math.sqrt(EARTH_GM / semi_major_axis**3)


def calculate_orbital_position(
    semi_major_axis: float,
    mean_anomaly: float,
    inclination: float,
    raan: float
) -> Tuple[float, float, float]:
    """
    计算轨道平面内的位置（ECI坐标系）

    Args:
        semi_major_axis: 轨道半长轴（米）
        mean_anomaly: 平近点角（弧度）
        inclination: 轨道倾角（度）
        raan: 升交点赤经（度）

    Returns:
        (x, y, z): ECI坐标系位置（米）
    """
    # 在轨道平面内的位置（圆轨道近似）
    x_orb = semi_major_axis * math.cos(mean_anomaly)
    y_orb = semi_major_axis * math.sin(mean_anomaly)

    # 转换到ECI坐标系
    i = math.radians(inclination)
    raan_rad = math.radians(raan)

    x_eci = x_orb * math.cos(raan_rad) - y_orb * math.cos(i) * math.sin(raan_rad)
    y_eci = x_orb * math.sin(raan_rad) + y_orb * math.cos(i) * math.cos(raan_rad)
    z_eci = y_orb * math.sin(i)

    return x_eci, y_eci, z_eci


def eci_to_ecef(
    x_eci: float,
    y_eci: float,
    z_eci: float,
    theta: float
) -> Tuple[float, float, float]:
    """
    将ECI坐标转换为ECEF坐标

    Args:
        x_eci, y_eci, z_eci: ECI坐标
        theta: 地球自转角度（rad）

    Returns:
        (x, y, z): ECEF坐标
    """
    x = x_eci * math.cos(theta) + y_eci * math.sin(theta)
    y = -x_eci * math.sin(theta) + y_eci * math.cos(theta)
    z = z_eci

    return x, y, z


def calculate_ecef_velocity(
    vx_eci: float,
    vy_eci: float,
    vz_eci: float,
    x_ecef: float,
    y_ecef: float,
    theta: float
) -> Tuple[float, float, float]:
    """
    将ECI速度转换为ECEF速度（考虑地球自转）

    Args:
        vx_eci, vy_eci, vz_eci: ECI速度（m/s）
        x_ecef, y_ecef: ECEF位置（用于计算科里奥利效应）
        theta: 地球自转角度（rad）

    Returns:
        (vx, vy, vz): ECEF速度（m/s）
    """
    vx = vx_eci * math.cos(theta) + vy_eci * math.sin(theta) - EARTH_ROTATION_RATE * y_ecef
    vy = -vx_eci * math.sin(theta) + vy_eci * math.cos(theta) + EARTH_ROTATION_RATE * x_ecef
    vz = vz_eci

    return vx, vy, vz
