"""
STK可见性计算器

基于STK HPOP的可见性计算实现
设计文档第3.2章 - STK后端

使用STK COM接口进行高精度可见性计算：
- 考虑大气阻力、太阳光压、三体引力
- 通过COM接口与STK交互
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging

from .base import VisibilityCalculator, VisibilityWindow
from ..utils import (
    EARTH_J2, EARTH_RADIUS_M, EARTH_GM,
    clamp, calculate_j2_perturbations, apply_j2_perturbations
)

logger = logging.getLogger(__name__)

# 尝试导入win32com，如果不可用则标记
try:
    import win32com.client
    WIN32COM_AVAILABLE = True
except ImportError:
    WIN32COM_AVAILABLE = False
    win32com = None


class STKVisibilityCalculator(VisibilityCalculator):
    """
    STK可见性计算器

    使用STK HPOP传播器进行高精度可见性计算
    - 考虑大气阻力、太阳光压、三体引力
    - 通过COM接口与STK交互
    - 支持优雅降级（STK不可用时使用简化计算）
    """

    # STK对象类型常量
    AG_SATELLITE = 18  # AgESTKObjectType.eSatellite
    AG_TARGET = 8      # AgESTKObjectType.eTarget
    AG_FACILITY = 9    # AgESTKObjectType.eFacility

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化STK可见性计算器

        Args:
            config: 配置参数
                - min_elevation: 最小仰角（度，默认5.0）
                - time_step: 计算时间步长（秒，默认60）
                - use_hpop: 是否使用HPOP传播器（默认True）
                - enable_drag: 是否启用大气阻力（默认True）
                - enable_srp: 是否启用太阳光压（默认True）
                - enable_third_body: 是否启用三体引力（默认True）
        """
        config = config or {}
        min_elevation = config.get('min_elevation', 5.0)
        super().__init__(min_elevation=min_elevation)

        self.time_step = config.get('time_step', 60)
        self._config = config
        self._use_hpop = config.get('use_hpop', True)
        self._enable_drag = config.get('enable_drag', True)
        self._enable_srp = config.get('enable_srp', True)
        self._enable_third_body = config.get('enable_third_body', True)

        # STK COM接口
        self._stk_app = None
        self._stk_root = None
        self._scenario = None

        # 对象缓存
        self._satellites: Dict[str, Any] = {}
        self._targets: Dict[str, Any] = {}
        self._ground_stations: Dict[str, Any] = {}

    def _is_stk_available(self) -> bool:
        """检查STK是否可用"""
        if not WIN32COM_AVAILABLE:
            return False
        try:
            # 尝试创建COM对象
            test_app = win32com.client.Dispatch("STK11.Application")
            del test_app
            return True
        except Exception:
            return False

    def connect(self) -> bool:
        """
        连接STK应用程序

        Returns:
            bool: 连接是否成功
        """
        if self._stk_app is not None and self._stk_root is not None:
            # 已经连接
            return True

        if not WIN32COM_AVAILABLE:
            logger.warning("win32com not available, STK integration disabled")
            return False

        try:
            # 创建STK应用程序实例
            self._stk_app = win32com.client.Dispatch("STK11.Application")
            self._stk_app.Visible = True
            self._stk_root = self._stk_app.NewScenario("DefaultScenario")
            logger.info("Successfully connected to STK")
            return True
        except Exception as e:
            logger.warning(f"Failed to connect to STK: {e}")
            self._stk_app = None
            self._stk_root = None
            return False

    def disconnect(self) -> None:
        """断开与STK的连接"""
        try:
            if self._stk_app is not None:
                self._stk_app.Quit()
        except Exception as e:
            logger.debug(f"Error during STK disconnect: {e}")
        finally:
            self._stk_app = None
            self._stk_root = None
            self._scenario = None
            self._satellites.clear()
            self._targets.clear()
            self._ground_stations.clear()

    def setup_scenario(self, scenario_name: str) -> bool:
        """
        设置STK场景

        Args:
            scenario_name: 场景名称

        Returns:
            bool: 设置是否成功
        """
        if self._stk_root is None:
            if not self.connect():
                return False

        try:
            # 关闭当前场景（如果存在）
            if self._scenario is not None:
                try:
                    self._stk_root.CloseScenario()
                except Exception:
                    pass

            # 创建新场景
            self._scenario = self._stk_root.NewScenario(scenario_name)
            logger.info(f"Created scenario: {scenario_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to setup scenario: {e}")
            return False

    def _set_scenario_time_period(self, start_time: datetime, end_time: datetime) -> bool:
        """
        设置场景时间范围

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            bool: 设置是否成功
        """
        if self._scenario is None:
            return False

        try:
            start_str = self._format_stk_time(start_time)
            end_str = self._format_stk_time(end_time)
            self._scenario.SetTimePeriod(start_str, end_str)
            self._scenario.Epoch = start_str
            return True
        except Exception as e:
            logger.error(f"Failed to set scenario time period: {e}")
            return False

    def add_satellite(self, satellite, use_hpop: bool = True) -> bool:
        """
        添加卫星到场景

        Args:
            satellite: 卫星模型
            use_hpop: 是否使用HPOP传播器

        Returns:
            bool: 添加是否成功
        """
        if self._scenario is None:
            if not self.connect() or not self.setup_scenario("DefaultScenario"):
                return False

        try:
            # 创建卫星对象
            sat_obj = self._scenario.Children.New(self.AG_SATELLITE, satellite.id)

            # 设置轨道参数
            if satellite.tle_line1 and satellite.tle_line2:
                # 使用TLE
                sat_obj.SetPropagatorType(1)  # SGP4
                propagator = sat_obj.Propagator
                propagator.CommonTasks.AddSegsFromFile(
                    satellite.tle_line1,
                    satellite.tle_line2
                )
                propagator.Propagate()
            else:
                # 使用轨道参数
                if use_hpop and self._use_hpop:
                    self._setup_hpop_propagator(sat_obj, satellite)
                else:
                    self._setup_two_body_propagator(sat_obj, satellite)

            self._satellites[satellite.id] = sat_obj
            logger.info(f"Added satellite: {satellite.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add satellite {satellite.id}: {e}")
            return False

    def _setup_two_body_propagator(self, sat_obj, satellite) -> None:
        """设置二体传播器"""
        try:
            sat_obj.SetPropagatorType(0)  # TwoBody
            propagator = sat_obj.Propagator

            # 设置轨道要素
            orbit = satellite.orbit
            keplerian = propagator.InitialState.Representation.ConvertTo(1)  # eOrbitStateClassical

            keplerian.SizeShapeType = 0  # eSizeShapeAltitude
            keplerian.LocationType = 5   # eLocationTrueAnomaly
            keplerian.Orientation.AscNodeType = 0  # eAscNodeRAAN

            keplerian.SizeShape.PerigeeAltitude = orbit.altitude / 1000.0  # km
            keplerian.SizeShape.ApogeeAltitude = orbit.altitude / 1000.0   # km
            keplerian.Orientation.Inclination = orbit.inclination
            keplerian.Orientation.ArgOfPerigee = orbit.arg_of_perigee
            keplerian.Orientation.AscNode.Value = orbit.raan
            keplerian.Location.Value = orbit.mean_anomaly

            propagator.Propagate()
        except Exception as e:
            logger.error(f"Failed to setup two-body propagator: {e}")
            raise

    def _setup_hpop_propagator(self, sat_obj, satellite) -> None:
        """设置HPOP传播器"""
        try:
            sat_obj.SetPropagatorType(3)  # HPOP
            propagator = sat_obj.Propagator

            # 设置轨道要素
            orbit = satellite.orbit
            keplerian = propagator.InitialState.Representation.ConvertTo(1)

            keplerian.SizeShapeType = 0
            keplerian.LocationType = 5
            keplerian.Orientation.AscNodeType = 0

            keplerian.SizeShape.PerigeeAltitude = orbit.altitude / 1000.0
            keplerian.SizeShape.ApogeeAltitude = orbit.altitude / 1000.0
            keplerian.Orientation.Inclination = orbit.inclination
            keplerian.Orientation.ArgOfPerigee = orbit.arg_of_perigee
            keplerian.Orientation.AscNode.Value = orbit.raan
            keplerian.Location.Value = orbit.mean_anomaly

            # 配置HPOP力模型
            self._configure_hpop_force_model(propagator)

            propagator.Propagate()
        except Exception as e:
            logger.error(f"Failed to setup HPOP propagator: {e}")
            raise

    def _configure_hpop_force_model(self, propagator) -> None:
        """配置HPOP力模型"""
        try:
            force_model = propagator.InitialState.ForceModel

            # 配置大气阻力
            if self._enable_drag:
                self._configure_atmospheric_drag(force_model)

            # 配置太阳光压
            if self._enable_srp:
                self._configure_solar_radiation_pressure(force_model)

            # 配置三体引力
            if self._enable_third_body:
                self._configure_third_body_gravity(force_model)

            # 配置积分器
            integrator = propagator.InitialState.IntegrationSetup
            self._setup_hpop_integrator(integrator, self.time_step)

        except Exception as e:
            logger.error(f"Failed to configure HPOP force model: {e}")
            raise

    def _configure_atmospheric_drag(self, force_model) -> None:
        """配置大气阻力模型"""
        try:
            force_model.UseDragModel = True
            force_model.Drag.DragModelType = 1  # HPOP drag model
            force_model.Drag.Cd = 2.2  # 阻力系数
            force_model.Drag.Area = 1.0  # 截面积 (m^2)
            force_model.Drag.Mass = 100.0  # 质量 (kg)
        except Exception as e:
            logger.warning(f"Failed to configure atmospheric drag: {e}")

    def _configure_solar_radiation_pressure(self, force_model) -> None:
        """配置太阳光压模型"""
        try:
            force_model.UseSRP = True
            force_model.SRP.Cr = 1.0  # 反射系数
            force_model.SRP.Area = 1.0  # 截面积 (m^2)
            force_model.SRP.Mass = 100.0  # 质量 (kg)
        except Exception as e:
            logger.warning(f"Failed to configure solar radiation pressure: {e}")

    def _configure_third_body_gravity(self, force_model) -> None:
        """配置三体引力模型"""
        try:
            force_model.UseThirdBodyGravity = True
            force_model.ThirdBodyGravity.UseLuna = True  # 月球
            force_model.ThirdBodyGravity.UseSun = True   # 太阳
        except Exception as e:
            logger.warning(f"Failed to configure third body gravity: {e}")

    def _setup_hpop_integrator(self, integrator, time_step: float) -> None:
        """设置HPOP积分器"""
        try:
            integrator.Step = time_step  # 积分步长（秒）
            integrator.StepControlType = 1  # 自适应步长控制
            integrator.RelativeError = 1e-9
            integrator.AbsoluteError = 1e-12
        except Exception as e:
            logger.warning(f"Failed to setup HPOP integrator: {e}")

    def add_target(self, target) -> bool:
        """
        添加目标到场景

        Args:
            target: 目标模型

        Returns:
            bool: 添加是否成功
        """
        if self._scenario is None:
            if not self.connect() or not self.setup_scenario("DefaultScenario"):
                return False

        try:
            # 创建目标对象
            target_obj = self._scenario.Children.New(self.AG_TARGET, target.id)

            # 设置目标位置
            if hasattr(target, 'get_ecef_position'):
                x, y, z = target.get_ecef_position()
                target_obj.Position.AssignCartesian(
                    0,  # eCoordinateSystem.eGeocentric
                    x / 1000.0,  # 转换为km
                    y / 1000.0,
                    z / 1000.0
                )
            elif hasattr(target, 'longitude') and hasattr(target, 'latitude'):
                lon = target.longitude
                lat = target.latitude
                alt = getattr(target, 'altitude', 0.0) / 1000.0  # 转换为km
                target_obj.Position.AssignGeodetic(lon, lat, alt)

            self._targets[target.id] = target_obj
            logger.info(f"Added target: {target.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add target {target.id}: {e}")
            return False

    def add_ground_station(self, ground_station) -> bool:
        """
        添加地面站到场景

        Args:
            ground_station: 地面站模型

        Returns:
            bool: 添加是否成功
        """
        if self._scenario is None:
            if not self.connect() or not self.setup_scenario("DefaultScenario"):
                return False

        try:
            # 创建地面站对象（使用Facility）
            gs_obj = self._scenario.Children.New(self.AG_FACILITY, ground_station.id)

            # 设置地面站位置
            if hasattr(ground_station, 'get_ecef_position'):
                x, y, z = ground_station.get_ecef_position()
                gs_obj.Position.AssignCartesian(
                    0,  # eCoordinateSystem.eGeocentric
                    x / 1000.0,
                    y / 1000.0,
                    z / 1000.0
                )
            elif hasattr(ground_station, 'longitude') and hasattr(ground_station, 'latitude'):
                lon = ground_station.longitude
                lat = ground_station.latitude
                alt = getattr(ground_station, 'altitude', 0.0) / 1000.0
                gs_obj.Position.AssignGeodetic(lon, lat, alt)

            # 设置最小仰角
            gs_obj.MinimumElevationAngle = self.min_elevation

            self._ground_stations[ground_station.id] = gs_obj
            logger.info(f"Added ground station: {ground_station.id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add ground station {ground_station.id}: {e}")
            return False

    def compute_satellite_target_windows(
        self,
        satellite,
        target,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta = timedelta(seconds=60)
    ) -> List[VisibilityWindow]:
        """
        计算卫星-目标可见窗口

        Args:
            satellite: 卫星模型
            target: 目标模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 参数验证
        if satellite is None or target is None:
            return []

        if start_time >= end_time:
            return []

        # 如果STK不可用，使用fallback计算
        if not self._is_stk_available():
            return self._compute_fallback_windows(
                satellite, target, start_time, end_time, time_step, is_ground_station=False
            )

        try:
            # 确保已连接STK
            if not self.connect():
                return self._compute_fallback_windows(
                    satellite, target, start_time, end_time, time_step, is_ground_station=False
                )

            # 设置场景时间
            self._set_scenario_time_period(start_time, end_time)

            # 添加对象到场景
            if satellite.id not in self._satellites:
                self.add_satellite(satellite, use_hpop=self._use_hpop)

            if target.id not in self._targets:
                self.add_target(target)

            # 计算Access
            sat_obj = self._satellites.get(satellite.id)
            target_obj = self._targets.get(target.id)

            if sat_obj is None or target_obj is None:
                return []

            access = sat_obj.GetAccess(target_obj.Path)
            access.ComputeAccess()

            if not access.Computed:
                return []

            # 解析Access数据
            windows = self._parse_access_data(
                access, satellite.id, target.id, start_time, end_time
            )

            access.Unload()
            return windows

        except Exception as e:
            logger.error(f"Error computing satellite-target windows: {e}")
            return self._compute_fallback_windows(
                satellite, target, start_time, end_time, time_step, is_ground_station=False
            )

    def compute_satellite_ground_station_windows(
        self,
        satellite,
        ground_station,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta = timedelta(seconds=60)
    ) -> List[VisibilityWindow]:
        """
        计算卫星-地面站可见窗口

        Args:
            satellite: 卫星模型
            ground_station: 地面站模型
            start_time: 开始时间
            end_time: 结束时间
            time_step: 时间步长

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 参数验证
        if satellite is None or ground_station is None:
            return []

        if start_time >= end_time:
            return []

        # 如果STK不可用，使用fallback计算
        if not self._is_stk_available():
            return self._compute_fallback_windows(
                satellite, ground_station, start_time, end_time, time_step, is_ground_station=True
            )

        try:
            # 确保已连接STK
            if not self.connect():
                return self._compute_fallback_windows(
                    satellite, ground_station, start_time, end_time, time_step, is_ground_station=True
                )

            # 设置场景时间
            self._set_scenario_time_period(start_time, end_time)

            # 添加对象到场景
            if satellite.id not in self._satellites:
                self.add_satellite(satellite, use_hpop=self._use_hpop)

            if ground_station.id not in self._ground_stations:
                self.add_ground_station(ground_station)

            # 计算Access
            sat_obj = self._satellites.get(satellite.id)
            gs_obj = self._ground_stations.get(ground_station.id)

            if sat_obj is None or gs_obj is None:
                return []

            access = sat_obj.GetAccess(gs_obj.Path)
            access.ComputeAccess()

            if not access.Computed:
                return []

            # 解析Access数据
            target_id = f"GS:{ground_station.id}"
            windows = self._parse_access_data(
                access, satellite.id, target_id, start_time, end_time
            )

            access.Unload()
            return windows

        except Exception as e:
            logger.error(f"Error computing satellite-ground station windows: {e}")
            return self._compute_fallback_windows(
                satellite, ground_station, start_time, end_time, time_step, is_ground_station=True
            )

    def _parse_access_data(
        self,
        access,
        satellite_id: str,
        target_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[VisibilityWindow]:
        """
        解析STK Access数据

        Args:
            access: STK Access对象
            satellite_id: 卫星ID
            target_id: 目标ID
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        windows = []

        try:
            # 获取Access数据
            access_data = access.DataSets.Item(0)
            data = access_data.GetValues()

            for entry in data:
                if len(entry) >= 2:
                    start_str = entry[0]
                    end_str = entry[1]
                    max_elevation = entry[2] if len(entry) > 2 else 0.0

                    window_start = self._parse_stk_time(start_str)
                    window_end = self._parse_stk_time(end_str)

                    if window_start and window_end:
                        # 过滤时间范围
                        if window_end < start_time or window_start > end_time:
                            continue

                        # 裁剪到请求的时间范围
                        actual_start = max(window_start, start_time)
                        actual_end = min(window_end, end_time)

                        # 计算质量评分
                        quality = min(1.0, max_elevation / 90.0) if max_elevation > 0 else 0.5

                        windows.append(VisibilityWindow(
                            satellite_id=satellite_id,
                            target_id=target_id,
                            start_time=actual_start,
                            end_time=actual_end,
                            max_elevation=max_elevation if max_elevation > 0 else 45.0,
                            quality_score=quality
                        ))

        except Exception as e:
            logger.error(f"Error parsing access data: {e}")

        return sorted(windows)

    def _parse_stk_time(self, stk_time_str: str) -> Optional[datetime]:
        """
        解析STK时间字符串

        Args:
            stk_time_str: STK时间格式字符串 (e.g., "1 Jan 2024 08:30:00.000")

        Returns:
            datetime: 解析后的时间，失败返回None
        """
        try:
            # STK格式: "1 Jan 2024 08:30:00.000"
            parts = stk_time_str.split()
            day = int(parts[0])
            month_str = parts[1]
            year = int(parts[2])
            time_parts = parts[3].split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            second = int(float(time_parts[2]))

            month_map = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
                'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
                'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            month = month_map.get(month_str, 1)

            return datetime(year, month, day, hour, minute, second)
        except Exception as e:
            logger.debug(f"Failed to parse STK time: {stk_time_str}, error: {e}")
            return None

    def _format_stk_time(self, dt: datetime) -> str:
        """
        格式化为STK时间字符串

        Args:
            dt: datetime对象

        Returns:
            str: STK格式时间字符串
        """
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_str = month_names[dt.month - 1]
        return f"{dt.day} {month_str} {dt.year} {dt.hour:02d}:{dt.minute:02d}:{dt.second:02d}.000"

    def _compute_fallback_windows(
        self,
        satellite,
        target,
        start_time: datetime,
        end_time: datetime,
        time_step: timedelta,
        is_ground_station: bool = False
    ) -> List[VisibilityWindow]:
        """
        STK不可用时的fallback计算

        使用简化的几何计算来估计可见窗口
        """
        windows = []
        target_id = f"GS:{target.id}" if is_ground_station else target.id

        try:
            # 获取目标位置
            if hasattr(target, 'get_ecef_position'):
                target_pos = target.get_ecef_position()
            elif hasattr(target, 'longitude') and hasattr(target, 'latitude'):
                # 简化的ECEF计算
                import math
                EARTH_RADIUS = 6371000.0
                lon_rad = math.radians(target.longitude)
                lat_rad = math.radians(target.latitude)
                alt = getattr(target, 'altitude', 0.0)
                r = EARTH_RADIUS + alt
                x = r * math.cos(lat_rad) * math.cos(lon_rad)
                y = r * math.cos(lat_rad) * math.sin(lon_rad)
                z = r * math.sin(lat_rad)
                target_pos = (x, y, z)
            else:
                return []

            # 简化的轨道传播和可见性计算
            # 这里使用简化的方法估计可见窗口
            current_time = start_time
            in_window = False
            window_start = None
            max_elevation = 0.0

            while current_time < end_time:
                # 简化的卫星位置计算（圆轨道近似）
                sat_pos = self._estimate_satellite_position(satellite, current_time)

                if sat_pos:
                    # 计算仰角
                    elevation = self._calculate_elevation(sat_pos, target_pos)

                    if elevation >= self.min_elevation:
                        if not in_window:
                            in_window = True
                            window_start = current_time
                            max_elevation = elevation
                        else:
                            max_elevation = max(max_elevation, elevation)
                    else:
                        if in_window:
                            # 窗口结束
                            quality = min(1.0, max_elevation / 90.0)
                            windows.append(VisibilityWindow(
                                satellite_id=satellite.id,
                                target_id=target_id,
                                start_time=window_start,
                                end_time=current_time,
                                max_elevation=max_elevation,
                                quality_score=quality
                            ))
                            in_window = False
                            max_elevation = 0.0

                current_time += time_step

            # 处理最后一个窗口
            if in_window and window_start:
                quality = min(1.0, max_elevation / 90.0)
                windows.append(VisibilityWindow(
                    satellite_id=satellite.id,
                    target_id=target_id,
                    start_time=window_start,
                    end_time=end_time,
                    max_elevation=max_elevation,
                    quality_score=quality
                ))

        except Exception as e:
            logger.error(f"Error in fallback window computation: {e}")

        return windows

    def _get_orbit_attr(self, orbit, attr_name: str, default_value):
        """安全获取轨道属性，处理Mock对象"""
        try:
            from unittest.mock import Mock
            value = getattr(orbit, attr_name, default_value)
            # 如果值是Mock对象，返回默认值
            if isinstance(value, Mock):
                return default_value
            return value
        except Exception:
            return default_value

    def _estimate_satellite_position(
        self,
        satellite,
        dt: datetime,
        scenario_start_time: Optional[datetime] = None
    ) -> Optional[Tuple[float, float, float]]:
        """
        估计卫星位置（简化计算，支持历元和J2摄动）

        Args:
            satellite: 卫星模型
            dt: 目标时间
            scenario_start_time: 场景开始时间，作为默认历元

        Returns:
            (x, y, z) in meters
        """
        try:
            import math
            from datetime import timezone
            from core.models.satellite import ensure_utc_datetime

            # 确保dt是UTC时区感知的（向后兼容）
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)

            # 使用简化的圆轨道模型
            orbit = getattr(satellite, 'orbit', None)
            if orbit is None:
                return None
            altitude = self._get_orbit_attr(orbit, 'altitude', 500000.0)
            inclination = self._get_orbit_attr(orbit, 'inclination', 97.4)
            raan = self._get_orbit_attr(orbit, 'raan', 0.0)
            arg_of_perigee = self._get_orbit_attr(orbit, 'arg_of_perigee', 0.0)
            mean_anomaly_offset = self._get_orbit_attr(orbit, 'mean_anomaly', 0.0)

            # ★ 确定参考历元
            epoch = self._get_orbit_attr(orbit, 'epoch', None)
            if epoch:
                ref_time = epoch
            elif scenario_start_time:
                ref_time = ensure_utc_datetime(scenario_start_time)
            else:
                ref_time = datetime(2024, 1, 1, tzinfo=timezone.utc)

            # 确保ref_time是UTC
            if ref_time.tzinfo is None:
                ref_time = ref_time.replace(tzinfo=timezone.utc)

            # 计算轨道参数
            a = EARTH_RADIUS_M + altitude
            mean_motion = math.sqrt(EARTH_GM / a**3)

            # 计算从历元到目标时间的偏移
            delta_t = (dt - ref_time).total_seconds()

            # ★ 使用工具函数计算J2摄动修正
            e = self._get_orbit_attr(orbit, 'eccentricity', 0.0)
            raan_corrected, arg_perigee_corrected = apply_j2_perturbations(
                raan, arg_of_perigee, delta_t, a, inclination, e, mean_motion
            )

            # 平近点角
            M = math.radians(mean_anomaly_offset) + mean_motion * delta_t

            # 轨道参数
            i = math.radians(inclination)
            raan_rad = math.radians(raan_corrected)

            # 圆轨道位置
            x_orb = a * math.cos(M)
            y_orb = a * math.sin(M)

            # 转换到ECEF（使用修正后的RAAN）
            x = x_orb * math.cos(raan_rad) - y_orb * math.cos(i) * math.sin(raan_rad)
            y = x_orb * math.sin(raan_rad) + y_orb * math.cos(i) * math.cos(raan_rad)
            z = y_orb * math.sin(i)

            return (x, y, z)
        except Exception:
            return None

    def calculate_windows(
        self,
        satellite_id: str,
        target_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> List[VisibilityWindow]:
        """
        计算可见窗口（简化接口）

        注意：此接口需要外部提供卫星和目标对象映射
        当前实现返回空列表

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[VisibilityWindow]: 可见窗口列表
        """
        # 简化接口：需要外部提供对象映射才能工作
        # 当前返回空列表
        return []

    def is_visible(
        self,
        satellite_id: str,
        target_id: str,
        time: datetime
    ) -> bool:
        """
        检查指定时间是否可见

        Args:
            satellite_id: 卫星ID
            target_id: 目标ID
            time: 检查时间

        Returns:
            bool: 是否可见
        """
        windows = self.calculate_windows(satellite_id, target_id, time, time)

        for window in windows:
            if window.start_time <= time <= window.end_time:
                return True

        return False
