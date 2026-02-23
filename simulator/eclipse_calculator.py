"""
地影计算器

实现设计文档第12章的地影计算：
- 圆柱形地影模型
- 计算卫星在规划周期内的地影区间
"""

import math
from typing import List, Tuple, Optional
from datetime import datetime, timedelta

from core.models import Orbit


class EclipseCalculator:
    """
    地影计算器

    计算卫星在规划周期内的地影区间（本影+半影）
    简化模型：圆柱形地影

    参考：SpaceSim地影计算设计
    """

    # 物理常数
    EARTH_RADIUS = 6371000.0       # 地球半径（米）
    SUN_RADIUS = 696340000.0       # 太阳半径（米）
    AU = 149597870700.0            # 天文单位（米）

    # 地球轨道参数
    EARTH_ORBIT_ECCENTRICITY = 0.0167
    EARTH_ORBIT_PERIOD_DAYS = 365.256363004

    def __init__(self):
        """初始化地影计算器"""
        pass

    def calculate_eclipse_intervals(
        self,
        satellite_orbit: Orbit,
        start_time: datetime,
        end_time: datetime,
        time_step: int = 60
    ) -> List[Tuple[datetime, datetime]]:
        """
        计算卫星在给定时间段内的地影区间

        Args:
            satellite_orbit: 卫星轨道参数
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长（秒）

        Returns:
            List[Tuple[datetime, datetime]]: 地影区间列表 [(start, end), ...]
        """
        if end_time <= start_time:
            return []

        eclipse_intervals = []
        current_time = start_time

        in_eclipse = False
        eclipse_start = None

        while current_time <= end_time:
            # 获取卫星在当前时刻的位置
            sat_pos = self._get_satellite_position(satellite_orbit, current_time)

            # 获取太阳在当前时刻的位置
            sun_pos = self._get_sun_position(current_time)

            # 判断是否在地影中
            is_eclipse = self._is_in_eclipse(sat_pos, sun_pos)

            if is_eclipse and not in_eclipse:
                # 进入地影
                in_eclipse = True
                eclipse_start = current_time
            elif not is_eclipse and in_eclipse:
                # 离开地影
                in_eclipse = False
                if eclipse_start:
                    eclipse_intervals.append((eclipse_start, current_time))
                    eclipse_start = None

            current_time += timedelta(seconds=time_step)

        # 如果结束时还在地影中
        if in_eclipse and eclipse_start:
            eclipse_intervals.append((eclipse_start, end_time))

        return self._merge_intervals(eclipse_intervals)

    def _is_in_eclipse(
        self,
        sat_pos: Tuple[float, float, float],
        sun_pos: Tuple[float, float, float]
    ) -> bool:
        """
        判断卫星是否在地影中（圆柱近似模型）

        圆柱模型：
        - 以日地连线为轴
        - 地球阴影投影为圆柱
        - 卫星位于圆柱内即为地影

        Args:
            sat_pos: 卫星位置 (x, y, z) in meters
            sun_pos: 太阳位置 (x, y, z) in meters

        Returns:
            bool: 是否在地影中
        """
        # 向量：太阳 -> 地球原点
        # 简化：假设地球在原点附近
        earth_pos = (0.0, 0.0, 0.0)

        # 向量：太阳 -> 地球
        sun_to_earth = (
            earth_pos[0] - sun_pos[0],
            earth_pos[1] - sun_pos[1],
            earth_pos[2] - sun_pos[2]
        )

        # 向量：太阳 -> 卫星
        sun_to_sat = (
            sat_pos[0] - sun_pos[0],
            sat_pos[1] - sun_pos[1],
            sat_pos[2] - sun_pos[2]
        )

        # 计算太阳到卫星连线与地影轴的夹角
        # 地影轴方向 = 日地连线方向
        sun_to_earth_dist = math.sqrt(sum(x**2 for x in sun_to_earth))
        sun_to_sat_dist = math.sqrt(sum(x**2 for x in sun_to_sat))

        if sun_to_sat_dist < 1e-10:
            return False

        # 计算投影（卫星在日地连线上的投影位置）
        # 投影点 = 太阳位置 + 投影长度 * 单位向量
        dot_product = sum(sun_to_sat[i] * sun_to_earth[i] for i in range(3))
        projection_length = dot_product / sun_to_earth_dist

        # 投影点坐标
        sun_to_earth_unit = (
            sun_to_earth[0] / sun_to_earth_dist,
            sun_to_earth[1] / sun_to_earth_dist,
            sun_to_earth[2] / sun_to_earth_dist
        )

        projection_point = (
            sun_pos[0] + projection_length * sun_to_earth_unit[0],
            sun_pos[1] + projection_length * sun_to_earth_unit[1],
            sun_pos[2] + projection_length * sun_to_earth_unit[2]
        )

        # 卫星到投影点的距离（垂直于地影轴的距离）
        sat_to_projection = (
            sat_pos[0] - projection_point[0],
            sat_pos[1] - projection_point[1],
            sat_pos[2] - projection_point[2]
        )
        perpendicular_dist = math.sqrt(sum(x**2 for x in sat_to_projection))

        # 地影半径 = 地球半径（圆柱近似）
        # 实际上应该考虑半影，但圆柱模型简化为地球半径
        eclipse_radius = self.EARTH_RADIUS

        # 判断条件：
        # 1. 卫星到地影轴的垂直距离 < 地影半径
        # 2. 投影点在地球后面（即投影长度 > 日地距离）

        if perpendicular_dist > eclipse_radius:
            return False

        # 检查投影位置是否在地球阴影区
        if projection_length < sun_to_earth_dist:
            return False

        # 卫星到地球的距离
        sat_to_earth_dist = math.sqrt(sum(sat_pos[i]**2 for i in range(3)))

        # 如果卫星距离地球太远（比如GEO），可能不在本影中
        # 简化处理：如果距离 > 地球半径 * 10，不考虑地影
        if sat_to_earth_dist > 10 * self.EARTH_RADIUS:
            return False

        return True

    def _get_sun_position(self, dt: datetime) -> Tuple[float, float, float]:
        """
        计算太阳在ECI坐标系中的位置（简化模型）

        使用近似公式计算太阳位置，精度足够用于地影判断。

        Args:
            dt: 时间

        Returns:
            Tuple[float, float, float]: 太阳位置 (x, y, z) in meters
        """
        # J2000.0纪元
        J2000 = datetime(2000, 1, 1, 12, 0, 0)

        # 从J2000起算的天数
        days_since_j2000 = (dt - J2000).total_seconds() / 86400.0

        # 平黄经（mean longitude）
        # L = 280.46061837 + 0.98564736629 * days
        mean_longitude = math.radians(
            (280.46061837 + 0.98564736629 * days_since_j2000) % 360
        )

        # 平近点角（mean anomaly）
        # M = 357.52911 + 0.9856002831 * days
        mean_anomaly = math.radians(
            (357.52911 + 0.9856002831 * days_since_j2000) % 360
        )

        # 黄道倾角（obliquity of ecliptic）
        obliquity = math.radians(23.4397 - 0.0000003568 * days_since_j2000)

        # 距离（AU）- 简化，使用1AU
        distance_au = 1.0

        # 黄道坐标
        x_ecliptic = distance_au * math.cos(mean_longitude)
        y_ecliptic = distance_au * math.sin(mean_longitude)
        z_ecliptic = 0.0  # 太阳在黄道面上

        # 转换到赤道坐标（ECI）
        x = x_ecliptic
        y = y_ecliptic * math.cos(obliquity) - z_ecliptic * math.sin(obliquity)
        z = y_ecliptic * math.sin(obliquity) + z_ecliptic * math.cos(obliquity)

        # 转换为米
        return (x * self.AU, y * self.AU, z * self.AU)

    def _get_satellite_position(
        self,
        orbit: Orbit,
        dt: datetime
    ) -> Tuple[float, float, float]:
        """
        计算卫星在ECI坐标系中的位置（简化圆轨道模型）

        Args:
            orbit: 轨道参数
            dt: 时间

        Returns:
            Tuple[float, float, float]: 卫星位置 (x, y, z) in meters
        """
        # 获取轨道周期
        period = orbit.get_period()

        # 计算轨道半径（地心距离）
        semi_major_axis = orbit.get_semi_major_axis()

        # 参考时间
        ref_time = datetime(2024, 1, 1, 0, 0, 0)
        delta_t = (dt - ref_time).total_seconds()

        # 平近点角（假设圆轨道）
        mean_motion = 2 * math.pi / period
        mean_anomaly = math.radians(orbit.mean_anomaly) + mean_motion * delta_t

        # 轨道面内的位置
        x_orb = semi_major_axis * math.cos(mean_anomaly)
        y_orb = semi_major_axis * math.sin(mean_anomaly)
        z_orb = 0.0

        # 转换到ECI坐标系
        # 简化：只考虑轨道倾角和升交点赤经
        i = math.radians(orbit.inclination)
        raan = math.radians(orbit.raan)

        # 旋转矩阵（简化）
        x = x_orb * math.cos(raan) - y_orb * math.cos(i) * math.sin(raan)
        y = x_orb * math.sin(raan) + y_orb * math.cos(i) * math.cos(raan)
        z = y_orb * math.sin(i)

        return (x, y, z)

    def _merge_intervals(
        self,
        intervals: List[Tuple[datetime, datetime]]
    ) -> List[Tuple[datetime, datetime]]:
        """
        合并重叠或相邻的地影区间

        Args:
            intervals: 原始区间列表

        Returns:
            List[Tuple[datetime, datetime]]: 合并后的区间列表
        """
        if not intervals:
            return []

        # 按开始时间排序
        sorted_intervals = sorted(intervals, key=lambda x: x[0])

        merged = [sorted_intervals[0]]

        for current in sorted_intervals[1:]:
            last = merged[-1]

            # 如果当前区间与上一个重叠或相邻（间隔小于1分钟）
            gap = (current[0] - last[1]).total_seconds()
            if gap <= 60:  # 1分钟内认为是连续的
                # 合并区间
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)

        return merged
