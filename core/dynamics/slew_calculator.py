"""
卫星姿态机动计算器

计算卫星在不同目标间机动所需的姿态角度和时间。

关键公式:
1. 机动角度 = 两个目标视线向量间的夹角
2. 机动时间 = 机动角度 / 最大机动角速度 + 稳定时间
3. 可行性 = 总时间 <= 可用时间 AND 机动角度 <= 最大机动角度
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import math

from core.models.target import Target, TargetType
from core.clustering.target_clusterer import TargetCluster


# 地球半径 (米)
EARTH_RADIUS = 6371000.0


@dataclass
class SlewManeuver:
    """姿态机动信息

    Attributes:
        start_target_id: 起始目标ID
        end_target_id: 结束目标ID
        roll_angle: 滚转角变化（度）
        pitch_angle: 俯仰角变化（度）
        total_slew_angle: 总机动角度
        slew_time: 机动时间（秒）
        feasible: 是否可行
    """
    start_target_id: str
    end_target_id: str
    roll_angle: float      # 滚转角变化（度）
    pitch_angle: float     # 俯仰角变化（度）
    total_slew_angle: float # 总机动角度
    slew_time: float       # 机动时间（秒）
    feasible: bool         # 是否可行


class SlewCalculator:
    """卫星姿态机动计算器

    计算卫星在不同目标间机动所需的姿态角度和时间。

    Attributes:
        max_slew_rate: 最大机动角速度（度/秒）
        max_slew_angle: 最大侧摆角（度）
        settling_time: 稳定时间（秒）
    """

    def __init__(
        self,
        max_slew_rate: float = 2.0,      # 最大机动角速度（度/秒）
        max_slew_angle: float = 45.0,     # 最大侧摆角（度）
        settling_time: float = 5.0        # 稳定时间（秒）
    ):
        """初始化机动计算器

        Args:
            max_slew_rate: 最大机动角速度（度/秒）
            max_slew_angle: 最大侧摆角（度）
            settling_time: 稳定时间（秒）

        Raises:
            ValueError: 如果参数无效
        """
        if max_slew_rate <= 0:
            raise ValueError(f"max_slew_rate must be positive, got {max_slew_rate}")
        if max_slew_angle <= 0:
            raise ValueError(f"max_slew_angle must be positive, got {max_slew_angle}")
        if settling_time < 0:
            raise ValueError(f"settling_time must be non-negative, got {settling_time}")

        self.max_slew_rate = max_slew_rate
        self.max_slew_angle = max_slew_angle
        self.settling_time = settling_time

    def calculate_slew_angles(
        self,
        satellite_position: Tuple[float, float, float],  # ECEF
        target1_position: Tuple[float, float],  # (lon, lat)
        target2_position: Tuple[float, float]   # (lon, lat)
    ) -> Tuple[float, float, float]:
        """计算两个目标间的姿态机动角度

        计算从target1到target2所需的姿态变化角度。
        使用ECEF坐标系中的向量夹角计算。

        Args:
            satellite_position: 卫星位置 (x, y, z) in ECEF meters
            target1_position: 目标1位置 (longitude, latitude) in degrees
            target2_position: 目标2位置 (longitude, latitude) in degrees

        Returns:
            (roll_delta, pitch_delta, total_angle) in degrees

        Raises:
            ValueError: 如果输入参数无效
        """
        self._validate_slew_inputs(satellite_position, target1_position, target2_position)

        total_angle = self._calculate_total_slew_angle(
            satellite_position, target1_position, target2_position
        )
        roll_delta, pitch_delta = self._calculate_angle_components(
            target1_position, target2_position, total_angle
        )

        return (roll_delta, pitch_delta, total_angle)

    def _validate_slew_inputs(
        self,
        satellite_position: Tuple[float, float, float],
        target1_position: Tuple[float, float],
        target2_position: Tuple[float, float]
    ) -> None:
        """Validate inputs for slew angle calculation.

        Raises:
            ValueError: If inputs are invalid.
        """
        if len(satellite_position) != 3:
            raise ValueError(f"satellite_position must have 3 components, got {len(satellite_position)}")
        if len(target1_position) != 2:
            raise ValueError(f"target1_position must have 2 components, got {len(target1_position)}")
        if len(target2_position) != 2:
            raise ValueError(f"target2_position must have 2 components, got {len(target2_position)}")

    def _calculate_total_slew_angle(
        self,
        satellite_position: Tuple[float, float, float],
        target1_position: Tuple[float, float],
        target2_position: Tuple[float, float]
    ) -> float:
        """Calculate total slew angle between two targets.

        Args:
            satellite_position: Satellite ECEF position.
            target1_position: First target (lon, lat).
            target2_position: Second target (lon, lat).

        Returns:
            Total angle in degrees.
        """
        sat_x, sat_y, sat_z = satellite_position

        target1_ecef = self._geodetic_to_ecef(target1_position[0], target1_position[1])
        target2_ecef = self._geodetic_to_ecef(target2_position[0], target2_position[1])

        los1 = (
            target1_ecef[0] - sat_x,
            target1_ecef[1] - sat_y,
            target1_ecef[2] - sat_z
        )
        los2 = (
            target2_ecef[0] - sat_x,
            target2_ecef[1] - sat_y,
            target2_ecef[2] - sat_z
        )

        los1_norm = self._normalize_vector(los1)
        los2_norm = self._normalize_vector(los2)

        return self._vector_angle(los1_norm, los2_norm)

    def _calculate_angle_components(
        self,
        target1_position: Tuple[float, float],
        target2_position: Tuple[float, float],
        total_angle: float
    ) -> Tuple[float, float]:
        """Calculate roll and pitch components.

        Args:
            target1_position: First target (lon, lat).
            target2_position: Second target (lon, lat).
            total_angle: Total slew angle.

        Returns:
            (roll_delta, pitch_delta) in degrees.
        """
        roll_delta = target2_position[0] - target1_position[0]
        pitch_delta = target2_position[1] - target1_position[1]

        if abs(total_angle) > 0.001:
            component_total = math.sqrt(roll_delta**2 + pitch_delta**2)
            if component_total > 0:
                scale = total_angle / component_total
                roll_delta *= scale
                pitch_delta *= scale

        return (roll_delta, pitch_delta)

    def calculate_slew_time(
        self,
        slew_angle: float,
        from_target_id: Optional[str] = None,
        to_target_id: Optional[str] = None
    ) -> float:
        """计算机动所需时间

        公式: slew_time = slew_angle / max_slew_rate + settling_time

        Args:
            slew_angle: 机动角度（度）
            from_target_id: 起始目标ID（可选）
            to_target_id: 结束目标ID（可选）

        Returns:
            机动时间（秒）

        Raises:
            ValueError: 如果slew_angle为负
        """
        if slew_angle < 0:
            raise ValueError(f"slew_angle must be non-negative, got {slew_angle}")

        # 机动时间 = 转动时间 + 稳定时间
        # 即使角度为0，也需要稳定时间
        slew_time = slew_angle / self.max_slew_rate + self.settling_time

        return slew_time

    def calculate_maneuver(
        self,
        satellite_position: Tuple[float, float, float],
        target1: Target,
        target2: Target
    ) -> SlewManeuver:
        """计算完整的机动信息

        Args:
            satellite_position: 卫星位置 (x, y, z) in ECEF meters
            target1: 起始目标
            target2: 结束目标

        Returns:
            SlewManeuver对象，包含完整机动信息

        Raises:
            ValueError: 如果目标是区域目标或None
        """
        self._validate_maneuver_targets(target1, target2)

        target1_pos = (target1.longitude, target1.latitude)
        target2_pos = (target2.longitude, target2.latitude)

        roll_delta, pitch_delta, total_angle = self.calculate_slew_angles(
            satellite_position, target1_pos, target2_pos
        )

        slew_time = self.calculate_slew_time(
            total_angle,
            from_target_id=target1.id,
            to_target_id=target2.id
        )

        feasible = total_angle <= self.max_slew_angle

        return SlewManeuver(
            start_target_id=target1.id,
            end_target_id=target2.id,
            roll_angle=roll_delta,
            pitch_angle=pitch_delta,
            total_slew_angle=total_angle,
            slew_time=slew_time,
            feasible=feasible
        )

    def _validate_maneuver_targets(self, target1: Target, target2: Target) -> None:
        """Validate targets for maneuver calculation.

        Args:
            target1: First target.
            target2: Second target.

        Raises:
            ValueError: If targets are invalid.
        """
        if target1 is None or target2 is None:
            raise ValueError("Targets cannot be None")

        if target1.target_type == TargetType.AREA:
            raise ValueError(f"Target {target1.id} is an area target, point targets required")
        if target2.target_type == TargetType.AREA:
            raise ValueError(f"Target {target2.id} is an area target, point targets required")

    def is_maneuver_feasible(
        self,
        slew_angle: float,
        available_time: float
    ) -> bool:
        """判断在给定时间内是否可以完成机动

        Args:
            slew_angle: 机动角度（度）
            available_time: 可用时间（秒）

        Returns:
            如果可行返回True，否则返回False

        Raises:
            ValueError: 如果available_time为负
        """
        if available_time < 0:
            raise ValueError(f"available_time must be non-negative, got {available_time}")

        # 检查角度限制
        if slew_angle > self.max_slew_angle:
            return False

        # 检查时间限制
        required_time = self.calculate_slew_time(slew_angle)
        if required_time > available_time:
            return False

        return True

    def _geodetic_to_ecef(self, lon: float, lat: float, alt: float = 0.0) -> Tuple[float, float, float]:
        """将地理坐标转换为ECEF坐标

        Args:
            lon: 经度（度）
            lat: 纬度（度）
            alt: 海拔高度（米），默认0

        Returns:
            (x, y, z) in ECEF meters
        """
        lon_rad = math.radians(lon)
        lat_rad = math.radians(lat)
        r = EARTH_RADIUS + alt

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        return (x, y, z)

    def _normalize_vector(self, v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """归一化向量

        Args:
            v: 输入向量 (x, y, z)

        Returns:
            归一化后的向量
        """
        length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
        if length < 1e-10:
            return (0.0, 0.0, 0.0)
        return (v[0]/length, v[1]/length, v[2]/length)

    def _vector_angle(
        self,
        v1: Tuple[float, float, float],
        v2: Tuple[float, float, float]
    ) -> float:
        """计算两个向量间的夹角（度）

        Args:
            v1: 向量1 (x, y, z)
            v2: 向量2 (x, y, z)

        Returns:
            夹角（度）
        """
        # 点积
        dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

        # 限制点积范围（处理数值误差）
        dot = max(-1.0, min(1.0, dot))

        # 计算角度
        angle_rad = math.acos(dot)
        angle_deg = math.degrees(angle_rad)

        return angle_deg


class ClusterSlewCalculator(SlewCalculator):
    """聚类内部机动计算器

    计算覆盖聚类内多个目标所需的机动角度和时间。
    """

    def calculate_cluster_slew_coverage(
        self,
        satellite_position: Tuple[float, float, float],
        cluster: TargetCluster,
        look_angle: float = 0.0
    ) -> Tuple[float, float]:
        """计算覆盖整个聚类所需的总机动角度和时间

        基于聚类边界框计算从一侧到另一侧的扫描角度。

        Args:
            satellite_position: 卫星位置 (x, y, z) in ECEF meters
            cluster: 目标聚类
            look_angle: 观测角度（度），用于调整计算

        Returns:
            (total_sweep_angle, total_slew_time)
        """
        if not cluster.targets or len(cluster.targets) == 1:
            return (0.0, 0.0)

        total_sweep_angle = self._calculate_sweep_angle(satellite_position, cluster)
        total_slew_time = self.calculate_slew_time(total_sweep_angle)

        return (total_sweep_angle, total_slew_time)

    def _calculate_sweep_angle(
        self,
        satellite_position: Tuple[float, float, float],
        cluster: TargetCluster
    ) -> float:
        """Calculate sweep angle from cluster bounding box corners.

        Args:
            satellite_position: Satellite ECEF position.
            cluster: Target cluster.

        Returns:
            Sweep angle in degrees.
        """
        min_lon, max_lon, min_lat, max_lat = cluster.bounding_box
        centroid_lon, centroid_lat = cluster.centroid

        corners = [
            (min_lon, min_lat),
            (min_lon, max_lat),
            (max_lon, min_lat),
            (max_lon, max_lat),
        ]

        angles = self._calculate_corner_angles(
            satellite_position, centroid_lon, centroid_lat, corners
        )

        sweep_angle = max(angles) - min(angles)
        return min(sweep_angle, self.max_slew_angle)

    def _calculate_corner_angles(
        self,
        satellite_position: Tuple[float, float, float],
        centroid_lon: float,
        centroid_lat: float,
        corners: List[Tuple[float, float]]
    ) -> List[float]:
        """Calculate angles to each corner from centroid.

        Args:
            satellite_position: Satellite ECEF position.
            centroid_lon: Centroid longitude.
            centroid_lat: Centroid latitude.
            corners: List of corner coordinates.

        Returns:
            List of angles in degrees.
        """
        angles = []
        for corner in corners:
            _, _, angle = self.calculate_slew_angles(
                satellite_position,
                (centroid_lon, centroid_lat),
                corner
            )
            angles.append(angle)
        return angles

    def can_cover_cluster_in_time(
        self,
        cluster: TargetCluster,
        visibility_duration: float,
        imaging_time: float
    ) -> bool:
        """判断在可见性窗口内是否有足够时间完成成像+机动

        Args:
            cluster: 目标聚类
            visibility_duration: 可见性窗口持续时间（秒）
            imaging_time: 成像所需时间（秒）

        Returns:
            如果时间足够返回True，否则返回False
        """
        if not cluster.targets:
            return True  # 空聚类总是可以"覆盖"

        # 估算机动时间
        # 对于聚类，我们假设需要扫描跨越聚类的角度
        # 使用一个简化的估算：基于目标分布

        if len(cluster.targets) == 1:
            # 单目标只需要稳定时间
            total_time = imaging_time + self.settling_time
        else:
            # 多目标需要计算扫描角度
            # 使用边界框估算角度跨度
            min_lon, max_lon, min_lat, max_lat = cluster.bounding_box

            # 简化的角度估算（假设在赤道附近，1度约111km）
            # 使用边界框的对角线作为估算
            lon_span = max_lon - min_lon
            lat_span = max_lat - min_lat

            # 简化的角度估算（实际应该使用卫星位置计算）
            # 这里假设500km轨道高度，1度地面距离对应约0.5度视角
            estimated_angle = math.sqrt(lon_span**2 + lat_span**2) * 0.5

            # 限制在最大角度内
            estimated_angle = min(estimated_angle, self.max_slew_angle)

            # 计算机动时间
            slew_time = self.calculate_slew_time(estimated_angle)

            total_time = imaging_time + slew_time

        return total_time <= visibility_duration
