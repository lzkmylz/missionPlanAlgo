"""
Footprint Calculator - Coverage Geometry Calculator

Calculates satellite imaging footprint geometry and off-nadir angles.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import math

from core.models import Target, TargetType, ImagingMode


@dataclass
class Footprint:
    """地面成像条带足迹"""
    center: Tuple[float, float]  # (lon, lat)
    polygon: List[Tuple[float, float]]  # 多边形顶点
    width_km: float
    length_km: float


class FootprintCalculator:
    """
    卫星成像足迹计算器

    计算卫星成像条带的地面足迹、侧摆角以及覆盖能力分析。
    """

    # 地球半径（米）
    EARTH_RADIUS_M = 6371000.0

    def __init__(self, satellite_altitude_km: float = 500.0):
        """
        初始化足迹计算器

        Args:
            satellite_altitude_km: 卫星轨道高度（公里），默认500km（典型LEO）
        """
        self.satellite_altitude_km = satellite_altitude_km

    def calculate_footprint(
        self,
        satellite_position: Tuple[float, float, float],  # (x, y, z) in ECEF
        nadir_position: Tuple[float, float],  # (lon, lat)
        look_angle: float,  # degrees from nadir
        swath_width_km: float,
        imaging_mode: ImagingMode
    ) -> Footprint:
        """
        计算成像条带地面足迹

        Args:
            satellite_position: 卫星ECEF坐标（米）
            nadir_position: 星下点经纬度（度）
            look_angle: 观测角度（度，从星下点算起）
            swath_width_km: 幅宽（公里）
            imaging_mode: 成像模式

        Returns:
            Footprint: 地面足迹对象
        """
        # 计算足迹中心（考虑侧摆位移）
        center_lon, center_lat = self._calculate_footprint_center(
            nadir_position, look_angle
        )

        # 计算足迹多边形
        polygon = self._calculate_footprint_polygon(
            center_lon, center_lat, swath_width_km, look_angle
        )

        # 计算足迹长度（沿轨道方向，约为幅宽的5-10倍）
        length_km = self._calculate_footprint_length(swath_width_km, imaging_mode)

        return Footprint(
            center=(center_lon, center_lat),
            polygon=polygon,
            width_km=swath_width_km,
            length_km=length_km
        )

    def calculate_off_nadir_angle(
        self,
        satellite_position: Tuple[float, float, float],
        target_position: Tuple[float, float]  # (lon, lat)
    ) -> float:
        """
        计算观测目标所需的侧摆角

        Args:
            satellite_position: 卫星ECEF坐标（米）
            target_position: 目标经纬度（度）

        Returns:
            float: 所需侧摆角（度）
        """
        if satellite_position is None:
            raise ValueError("Satellite position cannot be None")

        # 将目标转换为ECEF坐标
        target_ecef = self._lla_to_ecef(
            target_position[1],  # lat
            target_position[0],  # lon
            0.0  # altitude
        )

        # 计算卫星到地心的距离
        sat_radius = math.sqrt(
            satellite_position[0]**2 +
            satellite_position[1]**2 +
            satellite_position[2]**2
        )

        # 计算卫星高度
        altitude = sat_radius - self.EARTH_RADIUS_M

        # 计算卫星到目标的地面距离
        ground_distance = self._calculate_ground_distance(
            satellite_position, target_ecef
        )

        # 计算侧摆角: atan2(地面距离, 高度)
        off_nadir = math.degrees(math.atan2(ground_distance, altitude))

        return off_nadir

    def is_target_in_footprint(
        self,
        target: Target,
        footprint: Footprint
    ) -> bool:
        """
        判断目标是否在足迹内

        使用射线投射算法（ray-casting）判断点是否在多边形内。

        Args:
            target: 目标对象
            footprint: 足迹对象

        Returns:
            bool: 目标是否在足迹内
        """
        # 获取目标坐标
        if target.target_type == TargetType.POINT:
            point = (target.longitude, target.latitude)
        else:
            # 对于区域目标，检查其中心点
            point = target.get_center()

        # 使用射线投射算法
        return self._point_in_polygon(point, footprint.polygon)

    def can_cover_targets(
        self,
        targets: List[Target],
        satellite_position: Tuple[float, float, float],
        nadir_position: Tuple[float, float],
        max_off_nadir: float,
        swath_width_km: float
    ) -> Tuple[bool, float]:
        """
        判断卫星位置是否可以覆盖所有目标

        Args:
            targets: 目标列表
            satellite_position: 卫星ECEF坐标（米）
            nadir_position: 星下点经纬度（度）
            max_off_nadir: 最大允许侧摆角（度）
            swath_width_km: 幅宽（公里）

        Returns:
            Tuple[bool, float]: (是否可覆盖, 所需侧摆角)
        """
        if not targets:
            return True, 0.0

        if len(targets) == 1:
            # 单目标：检查是否在最大侧摆角内
            angle = self.calculate_off_nadir_angle(
                satellite_position,
                targets[0].get_center()
            )
            can_cover = abs(angle) <= max_off_nadir
            return can_cover, angle

        # 多目标：计算包含所有目标的最优观测角度
        target_centers = [t.get_center() for t in targets]

        # 计算目标分布范围
        lons = [c[0] for c in target_centers]
        lats = [c[1] for c in target_centers]

        # 计算目标跨度
        lon_span = max(lons) - min(lons)
        lat_span = max(lats) - min(lats)

        # 将跨度转换为地面距离（公里）
        avg_lat = sum(lats) / len(lats)
        lon_distance = self._lon_to_km(lon_span, avg_lat)
        lat_distance = self._lat_to_km(lat_span)

        # 计算目标分布的最大距离
        max_target_distance = math.sqrt(lon_distance**2 + lat_distance**2)

        # 如果目标分布超过幅宽，无法同时覆盖
        if max_target_distance > swath_width_km:
            return False, 0.0

        # 计算覆盖所有目标所需的中心角度
        # 使用目标中心作为观测中心
        center_lon = sum(lons) / len(lons)
        center_lat = sum(lats) / len(lats)

        required_angle = self.calculate_off_nadir_angle(
            satellite_position,
            (center_lon, center_lat)
        )

        # 检查是否超过最大侧摆角
        if abs(required_angle) > max_off_nadir:
            return False, required_angle

        # 验证所有目标是否都在幅宽内
        half_swath = swath_width_km / 2
        for center in target_centers:
            distance = self._calculate_ground_distance_between(
                (center_lon, center_lat), center
            )
            if distance > half_swath:
                return False, required_angle

        return True, required_angle

    def _calculate_footprint_center(
        self,
        nadir_position: Tuple[float, float],
        look_angle: float
    ) -> Tuple[float, float]:
        """
        计算足迹中心位置

        Args:
            nadir_position: 星下点经纬度
            look_angle: 观测角度（度）

        Returns:
            Tuple[float, float]: 足迹中心经纬度
        """
        if abs(look_angle) < 0.001:
            # 星下点观测，无位移
            return nadir_position

        # 计算地面位移距离
        # 简化模型：displacement = altitude * tan(look_angle)
        displacement_km = self.satellite_altitude_km * math.tan(
            math.radians(look_angle)
        )

        # 将距离转换为经度变化（假设在赤道附近，1度~111km）
        # 实际应用中应考虑纬度影响
        lon_offset = self._km_to_lon(displacement_km, nadir_position[1])

        return (nadir_position[0] + lon_offset, nadir_position[1])

    def _calculate_footprint_polygon(
        self,
        center_lon: float,
        center_lat: float,
        swath_width_km: float,
        look_angle: float
    ) -> List[Tuple[float, float]]:
        """
        计算足迹多边形顶点

        Args:
            center_lon: 中心经度
            center_lat: 中心纬度
            swath_width_km: 幅宽（公里）
            look_angle: 观测角度（度）

        Returns:
            List[Tuple[float, float]]: 多边形顶点列表
        """
        # 计算半幅宽对应的经纬度偏移
        half_swath_km = swath_width_km / 2

        # 假设足迹为矩形，沿东西方向展开
        # 实际足迹形状取决于观测几何
        lon_offset = self._km_to_lon(half_swath_km, center_lat)
        lat_offset = self._km_to_lat(half_swath_km)

        # 创建矩形足迹（简化模型）
        # 实际应用中可能需要更复杂的多边形
        polygon = [
            (center_lon - lon_offset, center_lat - lat_offset),  # 左下
            (center_lon + lon_offset, center_lat - lat_offset),  # 右下
            (center_lon + lon_offset, center_lat + lat_offset),  # 右上
            (center_lon - lon_offset, center_lat + lat_offset),  # 左上
        ]

        return polygon

    def _calculate_footprint_length(
        self,
        swath_width_km: float,
        imaging_mode: ImagingMode
    ) -> float:
        """
        计算足迹长度（沿轨道方向）

        Args:
            swath_width_km: 幅宽（公里）
            imaging_mode: 成像模式

        Returns:
            float: 足迹长度（公里）
        """
        # 根据成像模式确定长度倍数
        length_multipliers = {
            ImagingMode.PUSH_BROOM: 10.0,      # 推扫模式：长条带
            ImagingMode.FRAME: 1.0,             # 帧模式：近似正方形
            ImagingMode.SPOTLIGHT: 1.0,         # 聚束模式：近似正方形
            ImagingMode.STRIPMAP: 10.0,         # 条带模式：长条带
            ImagingMode.SLIDING_SPOTLIGHT: 5.0, # 滑动聚束：中等长度
            ImagingMode.SCAN: 20.0,             # 扫描模式：很长条带
        }

        multiplier = length_multipliers.get(imaging_mode, 5.0)
        return swath_width_km * multiplier

    def _lla_to_ecef(
        self,
        lat: float,
        lon: float,
        alt: float
    ) -> Tuple[float, float, float]:
        """
        将经纬度高程转换为ECEF坐标

        Args:
            lat: 纬度（度）
            lon: 经度（度）
            alt: 高程（米）

        Returns:
            Tuple[float, float, float]: ECEF坐标（米）
        """
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)

        r = self.EARTH_RADIUS_M + alt

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        return (x, y, z)

    def _calculate_ground_distance(
        self,
        satellite_position: Tuple[float, float, float],
        target_ecef: Tuple[float, float, float]
    ) -> float:
        """
        计算卫星到目标的地面距离（投影到地球表面）

        Args:
            satellite_position: 卫星ECEF坐标
            target_ecef: 目标ECEF坐标

        Returns:
            float: 地面距离（米）
        """
        # 将卫星位置投影到地球表面
        sat_radius = math.sqrt(
            satellite_position[0]**2 +
            satellite_position[1]**2 +
            satellite_position[2]**2
        )

        scale = self.EARTH_RADIUS_M / sat_radius
        sat_surface = (
            satellite_position[0] * scale,
            satellite_position[1] * scale,
            satellite_position[2] * scale
        )

        # 计算表面距离
        dx = sat_surface[0] - target_ecef[0]
        dy = sat_surface[1] - target_ecef[1]
        dz = sat_surface[2] - target_ecef[2]

        # 弦长
        chord_length = math.sqrt(dx**2 + dy**2 + dz**2)

        # 转换为弧长
        # 对于小角度，弧长 ≈ 弦长
        # 对于大角度，使用：arc = R * 2 * asin(chord / (2R))
        if chord_length < self.EARTH_RADIUS_M:
            arc_length = self.EARTH_RADIUS_M * 2 * math.asin(
                chord_length / (2 * self.EARTH_RADIUS_M)
            )
        else:
            arc_length = chord_length

        return arc_length

    def _calculate_ground_distance_between(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float]
    ) -> float:
        """
        计算两个地面位置之间的距离（公里）

        Args:
            pos1: (lon, lat) 位置1
            pos2: (lon, lat) 位置2

        Returns:
            float: 距离（公里）
        """
        # 使用Haversine公式
        R = self.EARTH_RADIUS_M / 1000  # 转换为公里

        lat1, lon1 = math.radians(pos1[1]), math.radians(pos1[0])
        lat2, lon2 = math.radians(pos2[1]), math.radians(pos2[0])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (math.sin(dlat / 2)**2 +
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _point_in_polygon(
        self,
        point: Tuple[float, float],
        polygon: List[Tuple[float, float]]
    ) -> bool:
        """
        使用射线投射算法判断点是否在多边形内

        包含边界检测：点在顶点或边上时返回True。

        Args:
            point: (lon, lat) 点坐标
            polygon: 多边形顶点列表

        Returns:
            bool: 点是否在多边形内（包含边界）
        """
        x, y = point
        n = len(polygon)
        inside = False

        # 首先检查点是否在顶点或边上
        for i in range(n):
            xi, yi = polygon[i]
            # 检查是否精确匹配顶点
            if abs(xi - x) < 1e-9 and abs(yi - y) < 1e-9:
                return True

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            # 检查点是否在当前边上（不包括端点，避免重复计数）
            if self._point_on_segment(x, y, xi, yi, xj, yj):
                return True

            # 射线投射算法
            # 检查点是否在边的y范围内（一个端点包含，另一个不包含）
            if ((yi > y) != (yj > y)):
                # 计算边与水平线的交点
                xinters = (xj - xi) * (y - yi) / (yj - yi) + xi
                if x <= xinters:
                    inside = not inside

            j = i

        return inside

    def _point_on_segment(
        self,
        x: float, y: float,
        x1: float, y1: float,
        x2: float, y2: float
    ) -> bool:
        """
        检查点是否在线段上（不包括端点）

        Args:
            x, y: 待检查点
            x1, y1: 线段起点
            x2, y2: 线段终点

        Returns:
            bool: 点是否在线段上
        """
        # 检查点是否在包围盒内
        if (min(x1, x2) <= x <= max(x1, x2) and
            min(y1, y2) <= y <= max(y1, y2)):
            # 检查是否共线（叉积为0）
            cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            if abs(cross) < 1e-9:
                # 排除端点（已在前面检查）
                if not ((abs(x - x1) < 1e-9 and abs(y - y1) < 1e-9) or
                       (abs(x - x2) < 1e-9 and abs(y - y2) < 1e-9)):
                    return True
        return False

    def _km_to_lon(self, km: float, lat: float) -> float:
        """
        将公里转换为经度差

        Args:
            km: 距离（公里）
            lat: 纬度（度）

        Returns:
            float: 经度差（度）
        """
        # 1度经度 = 111km * cos(lat)
        lat_rad = math.radians(lat)
        km_per_degree = 111.32 * math.cos(lat_rad)
        if km_per_degree < 0.001:
            km_per_degree = 0.001  # 避免除零
        return km / km_per_degree

    def _km_to_lat(self, km: float) -> float:
        """
        将公里转换为纬度差

        Args:
            km: 距离（公里）

        Returns:
            float: 纬度差（度）
        """
        # 1度纬度 ≈ 111km
        return km / 111.32

    def _lon_to_km(self, lon_diff: float, lat: float) -> float:
        """
        将经度差转换为公里

        Args:
            lon_diff: 经度差（度）
            lat: 纬度（度）

        Returns:
            float: 距离（公里）
        """
        lat_rad = math.radians(lat)
        return abs(lon_diff) * 111.32 * math.cos(lat_rad)

    def _lat_to_km(self, lat_diff: float) -> float:
        """
        将纬度差转换为公里

        Args:
            lat_diff: 纬度差（度）

        Returns:
            float: 距离（公里）
        """
        return abs(lat_diff) * 111.32
