"""
Footprint Calculator - Coverage Geometry Calculator

Calculates satellite imaging footprint geometry and off-nadir angles.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math

from core.models import Target, TargetType, ImagingMode
from core.constants import EARTH_RADIUS_M as CONST_EARTH_RADIUS_M, METERS_TO_KM


@dataclass
class Footprint:
    """地面成像条带足迹"""
    center: Tuple[float, float]  # (lon, lat)
    polygon: List[Tuple[float, float]]  # 多边形顶点（四脚点）
    width_km: float
    length_km: float
    fov_config: Optional[Dict[str, Any]] = None  # 使用的FOV配置


class FootprintCalculator:
    """
    卫星成像足迹计算器

    计算卫星成像条带的地面足迹、侧摆角以及覆盖能力分析。
    """

    # 地球半径（米）
    EARTH_RADIUS_M = CONST_EARTH_RADIUS_M

    def __init__(self, satellite_altitude_km: float = 500.0):
        """
        初始化足迹计算器

        Args:
            satellite_altitude_km: 卫星轨道高度（公里），默认500km（典型LEO）
        """
        self.satellite_altitude_km = satellite_altitude_km

    def calculate_footprint_from_fov(
        self,
        satellite_position: Tuple[float, float, float],  # (x, y, z) in ECEF
        nadir_position: Tuple[float, float],  # (lon, lat)
        look_angle: float,  # degrees from nadir
        look_direction: float,  # degrees, 0=north, 90=east
        fov_config: Dict[str, Any],
        imaging_mode: ImagingMode
    ) -> Footprint:
        """
        基于FOV配置计算精确足迹

        Args:
            satellite_position: 卫星ECEF坐标（米）
            nadir_position: 星下点经纬度（度）
            look_angle: 观测角度（度，从星下点算起）
            look_direction: 观测方向（度，0=北，90=东）
            fov_config: FOV配置字典
                - fov_type: 'cone' 或 'rectangular'
                - half_angle: 圆锥视场半角（度）
                - half_angle_x: 矩形视场X方向半角（度）
                - half_angle_y: 矩形视场Y方向半角（度）
            imaging_mode: 成像模式

        Returns:
            Footprint: 包含四脚点坐标的足迹对象
        """
        # 1. 计算卫星高度
        sat_radius = math.sqrt(sum(x**2 for x in satellite_position))
        altitude_m = sat_radius - self.EARTH_RADIUS_M
        altitude_km = altitude_m / 1000.0

        # 2. 根据FOV类型计算幅宽和长度
        fov_type = fov_config.get('fov_type', 'cone')
        if fov_type == 'cone':
            half_angle = math.radians(fov_config.get('half_angle', 0.5))
            # 圆锥视场：圆形 footprint，近似为矩形
            swath_width_km = 2 * altitude_km * math.tan(half_angle)
            length_km = swath_width_km  # 近似圆形
        elif fov_type == 'rectangular':
            half_angle_x = math.radians(fov_config.get('half_angle_x', 0.5))
            half_angle_y = math.radians(fov_config.get('half_angle_y', 0.35))
            # 矩形视场：沿轨迹方向（X）和垂直轨迹方向（Y）
            swath_width_km = 2 * altitude_km * math.tan(half_angle_y)
            length_km = 2 * altitude_km * math.tan(half_angle_x)
        else:
            # 回退到配置的swath_width
            swath_width_km = fov_config.get('swath_width_km', 10.0)
            length_km = self._calculate_footprint_length(swath_width_km, imaging_mode)

        # 3. 计算足迹中心（考虑侧摆位移）
        center_lon, center_lat = self._calculate_footprint_center(
            nadir_position, look_angle
        )

        # 4. 计算精确四脚点坐标
        corners = self._calculate_footprint_corners_precise(
            center_lon=center_lon,
            center_lat=center_lat,
            swath_width_km=swath_width_km,
            length_km=length_km,
            look_direction=look_direction,
            altitude_km=altitude_km
        )

        return Footprint(
            center=(center_lon, center_lat),
            polygon=corners,
            width_km=swath_width_km,
            length_km=length_km,
            fov_config=fov_config
        )

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

    def calculate_required_roll_angle(
        self,
        satellite_position: Tuple[float, float, float],
        target_position: Tuple[float, float]  # (lon, lat)
    ) -> float:
        """
        计算观测目标所需的滚转角（原侧摆角）

        滚转角是绕X轴旋转的角度，控制卫星的侧摆（左右）。
        与俯仰角（绕Y轴）不同，滚转角是主要的观测姿态控制。

        Args:
            satellite_position: 卫星ECEF坐标（米）
            target_position: 目标经纬度（度）

        Returns:
            float: 所需滚转角（度）
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

        # 计算滚转角: atan2(地面距离, 高度)
        roll_angle = math.degrees(math.atan2(ground_distance, altitude))

        return roll_angle

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
        max_roll_angle: float,
        swath_width_km: float
    ) -> Tuple[bool, float]:
        """
        判断卫星位置是否可以覆盖所有目标

        Args:
            targets: 目标列表
            satellite_position: 卫星ECEF坐标（米）
            nadir_position: 星下点经纬度（度）
            max_roll_angle: 最大允许滚转角（度）
            swath_width_km: 幅宽（公里）

        Returns:
            Tuple[bool, float]: (是否可覆盖, 所需滚转角)
        """
        if not targets:
            return True, 0.0

        if len(targets) == 1:
            return self._check_single_target(targets[0], satellite_position, max_roll_angle)

        return self._check_multiple_targets(
            targets, satellite_position, max_roll_angle, swath_width_km
        )

    def _check_single_target(
        self,
        target: Target,
        satellite_position: Tuple[float, float, float],
        max_roll_angle: float
    ) -> Tuple[bool, float]:
        """Check if a single target is within coverage."""
        angle = self.calculate_required_roll_angle(
            satellite_position,
            target.get_center()
        )
        can_cover = abs(angle) <= max_roll_angle
        return can_cover, angle

    def _check_multiple_targets(
        self,
        targets: List[Target],
        satellite_position: Tuple[float, float, float],
        max_roll_angle: float,
        swath_width_km: float
    ) -> Tuple[bool, float]:
        """Check if multiple targets can be covered together."""
        target_centers = [t.get_center() for t in targets]
        center_lon, center_lat = self._calculate_target_centroid(target_centers)

        if not self._are_targets_within_swath(target_centers, center_lon, center_lat, swath_width_km):
            return False, 0.0

        required_angle = self.calculate_required_roll_angle(
            satellite_position,
            (center_lon, center_lat)
        )

        if abs(required_angle) > max_roll_angle:
            return False, required_angle

        if not self._all_targets_in_swath(target_centers, center_lon, center_lat, swath_width_km):
            return False, required_angle

        return True, required_angle

    def _calculate_target_centroid(
        self,
        target_centers: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """Calculate the centroid of target centers."""
        lons = [c[0] for c in target_centers]
        lats = [c[1] for c in target_centers]
        return sum(lons) / len(lons), sum(lats) / len(lats)

    def _are_targets_within_swath(
        self,
        target_centers: List[Tuple[float, float]],
        center_lon: float,
        center_lat: float,
        swath_width_km: float
    ) -> bool:
        """Check if target span is within swath width."""
        lons = [c[0] for c in target_centers]
        lats = [c[1] for c in target_centers]

        lon_span = max(lons) - min(lons)
        lat_span = max(lats) - min(lats)

        lon_distance = self._lon_to_km(lon_span, center_lat)
        lat_distance = self._lat_to_km(lat_span)

        max_target_distance = math.sqrt(lon_distance**2 + lat_distance**2)
        return max_target_distance <= swath_width_km

    def _all_targets_in_swath(
        self,
        target_centers: List[Tuple[float, float]],
        center_lon: float,
        center_lat: float,
        swath_width_km: float
    ) -> bool:
        """Verify all targets are within half swath from center."""
        half_swath = swath_width_km / 2
        for center in target_centers:
            distance = self._calculate_ground_distance_between(
                (center_lon, center_lat), center
            )
            if distance > half_swath:
                return False
        return True

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

    def _calculate_footprint_corners_precise(
        self,
        center_lon: float,
        center_lat: float,
        swath_width_km: float,
        length_km: float,
        look_direction: float,
        altitude_km: float
    ) -> List[Tuple[float, float]]:
        """
        计算足迹四脚点坐标（精确计算）

        考虑地球曲率和观测几何，计算精确的经纬度坐标。
        四脚点顺序：左前、右前、右后、左后（沿飞行方向）

        Args:
            center_lon: 中心经度
            center_lat: 中心纬度
            swath_width_km: 幅宽（公里）
            length_km: 长度（公里，沿飞行方向）
            look_direction: 观测方向（度，0=北，90=东）
            altitude_km: 卫星高度（公里）

        Returns:
            List[Tuple[float, float]]: 四脚点坐标列表 [(lon1, lat1), (lon2, lat2), ...]
        """
        # 计算半幅宽和半长度
        half_width_km = swath_width_km / 2
        half_length_km = length_km / 2

        # 将距离转换为角度（考虑纬度影响）
        km_per_deg_lat = 111.32
        km_per_deg_lon = 111.32 * math.cos(math.radians(center_lat))

        if km_per_deg_lon < 0.001:
            km_per_deg_lon = 0.001  # 避免除零

        half_width_deg = half_width_km / km_per_deg_lon
        half_length_deg = half_length_km / km_per_deg_lat

        # 考虑观测方向的旋转
        direction_rad = math.radians(look_direction)
        cos_d = math.cos(direction_rad)
        sin_d = math.sin(direction_rad)

        # 计算四脚点（相对于中心，沿飞行方向）
        # 顺序：左前、右前、右后、左后
        corners_local = [
            (-half_width_deg, half_length_deg),   # 左前
            (half_width_deg, half_length_deg),    # 右前
            (half_width_deg, -half_length_deg),   # 右后
            (-half_width_deg, -half_length_deg),  # 左后
        ]

        # 应用旋转和平移
        rotated_corners = []
        for dx, dy in corners_local:
            # 旋转
            rot_x = dx * cos_d - dy * sin_d
            rot_y = dx * sin_d + dy * cos_d
            # 平移到中心
            rotated_corners.append((center_lon + rot_x, center_lat + rot_y))

        return rotated_corners

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
