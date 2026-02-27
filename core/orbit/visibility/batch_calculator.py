"""
批量可见性计算器

提供高性能的批量可见性计算功能，通过单次JNI调用Java端完成全部计算。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import time

from .base import VisibilityWindow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchComputationConfig:
    """批量计算配置"""

    # 步长配置
    coarse_step_seconds: float = 300.0  # 粗扫描步长（秒）
    fine_step_seconds: float = 60.0  # 精化步长（秒）

    # 精度控制
    min_elevation_degrees: float = 0.0  # 最小仰角
    elevation_tolerance_degrees: float = 0.01  # 仰角容差

    # 性能控制
    use_parallel_propagation: bool = True  # Java端是否并行
    max_batch_size: int = 100  # 最大批次大小（防内存溢出）

    # 容错控制
    max_retries: int = 3  # 最大重试次数
    retry_delay_seconds: float = 1.0  # 重试间隔
    fallback_on_error: bool = True  # 错误时回退到逐对计算


@dataclass
class BatchComputationStats:
    """计算性能统计"""

    total_computation_time_ms: int
    jni_call_time_ms: int
    java_computation_time_ms: int
    python_overhead_ms: int
    n_satellites: int
    n_targets: int
    n_ground_stations: int
    n_windows_found: int
    memory_usage_mb: float
    cache_hit_rate: float = 0.0


@dataclass
class BatchVisibilityResult:
    """批量计算结果"""

    target_windows: Dict[Tuple[str, str], List[VisibilityWindow]]
    ground_station_windows: Dict[Tuple[str, str], List[VisibilityWindow]]
    computation_stats: Optional[BatchComputationStats]
    is_fallback_result: bool = False

    @property
    def total_window_count(self) -> int:
        """总窗口数"""
        total = sum(len(w) for w in self.target_windows.values())
        total += sum(len(w) for w in self.ground_station_windows.values())
        return total

    def get_windows_for_satellite_target(
        self, satellite_id: str, target_id: str
    ) -> List[VisibilityWindow]:
        """获取指定卫星-目标对的窗口"""
        return self.target_windows.get((satellite_id, target_id), [])

    def to_cache_format(self) -> Dict:
        """转换为缓存格式"""
        return {
            "target_windows": [
                {
                    "satellite_id": w.satellite_id,
                    "target_id": w.target_id,
                    "start_time": w.start_time.isoformat(),
                    "end_time": w.end_time.isoformat(),
                    "duration": (w.end_time - w.start_time).total_seconds(),
                    "max_elevation": w.max_elevation,
                }
                for windows in self.target_windows.values()
                for w in windows
            ],
            "ground_station_windows": [
                {
                    "satellite_id": w.satellite_id,
                    "target_id": w.target_id,
                    "start_time": w.start_time.isoformat(),
                    "end_time": w.end_time.isoformat(),
                    "duration": (w.end_time - w.start_time).total_seconds(),
                    "max_elevation": w.max_elevation,
                }
                for windows in self.ground_station_windows.values()
                for w in windows
            ],
        }


class BatchComputationError(Exception):
    """批量计算错误"""
    pass


class BatchVisibilityCalculator:
    """
    批量可见性计算器

    设计原则：
    1. 单次JNI调用完成全部计算
    2. 错误时自动回退到逐对计算
    3. 详细的性能监控
    """

    def __init__(self, java_bridge: Optional[Any] = None):
        self._bridge = java_bridge
        self._cache = {}
        self._stats = None

    def compute_all_windows(
        self,
        satellites: List,
        targets: List,
        ground_stations: List,
        start_time: datetime,
        end_time: datetime,
        config: Optional[BatchComputationConfig] = None,
    ) -> BatchVisibilityResult:
        """
        批量计算所有可见窗口（主入口）

        Args:
            satellites: 卫星列表
            targets: 目标列表
            ground_stations: 地面站列表
            start_time: 开始时间
            end_time: 结束时间
            config: 计算配置

        Returns:
            BatchVisibilityResult: 包含所有窗口的结果
        """
        config = config or BatchComputationConfig()

        # 尝试批量计算
        try:
            result = self._compute_batch(
                satellites, targets, ground_stations, start_time, end_time, config
            )
            return result

        except Exception as e:
            logger.warning(f"Batch computation failed: {e}. Falling back to pairwise.")
            if config.fallback_on_error:
                return self._fallback_pairwise(
                    satellites, targets, ground_stations, start_time, end_time, config
                )
            raise BatchComputationError(f"Batch computation failed: {e}") from e

    def _compute_batch(
        self,
        satellites: List,
        targets: List,
        ground_stations: List,
        start_time: datetime,
        end_time: datetime,
        config: BatchComputationConfig,
    ) -> BatchVisibilityResult:
        """执行批量计算"""

        # 延迟初始化Java桥接器
        if self._bridge is None:
            from .orekit_java_bridge import OrekitJavaBridge

            self._bridge = OrekitJavaBridge()

        start_ns = time.perf_counter_ns()

        # 准备数据（序列化）
        prep_start = time.perf_counter_ns()
        sat_params = [self._serialize_satellite(s) for s in satellites]
        target_params = [self._serialize_target(t) for t in targets]
        gs_params = [self._serialize_ground_station(gs) for gs in ground_stations]
        prep_time = (time.perf_counter_ns() - prep_start) // 1_000_000  # ms

        # JNI调用Java端
        jni_start = time.perf_counter_ns()
        result = self._bridge.compute_visibility_batch(
            sat_params,
            target_params,
            gs_params,
            start_time,
            end_time,
            {
                "coarseStep": config.coarse_step_seconds,
                "fineStep": config.fine_step_seconds,
                "minElevation": config.min_elevation_degrees,
                "useParallel": config.use_parallel_propagation,
            },
        )
        jni_time = (time.perf_counter_ns() - jni_start) // 1_000_000  # ms

        # 解析结果
        parse_start = time.perf_counter_ns()
        parsed_result = self._parse_java_result(
            result, satellites, targets, ground_stations
        )
        parse_time = (time.perf_counter_ns() - parse_start) // 1_000_000  # ms

        total_time = (time.perf_counter_ns() - start_ns) // 1_000_000  # ms

        # 记录统计
        self._stats = BatchComputationStats(
            total_computation_time_ms=total_time,
            jni_call_time_ms=jni_time,
            java_computation_time_ms=result.get("stats", {}).get(
                "computationTimeMs", 0
            ),
            python_overhead_ms=prep_time + parse_time,
            n_satellites=len(satellites),
            n_targets=len(targets),
            n_ground_stations=len(ground_stations),
            n_windows_found=parsed_result.total_window_count,
            memory_usage_mb=result.get("stats", {}).get("memoryUsageMb", 0.0),
        )

        logger.info(f"Batch computation completed: {self._stats}")
        return parsed_result

    def _fallback_pairwise(
        self,
        satellites: List,
        targets: List,
        ground_stations: List,
        start_time: datetime,
        end_time: datetime,
        config: BatchComputationConfig,
    ) -> BatchVisibilityResult:
        """回退到逐对计算（确保可用性）"""
        logger.info("Using fallback pairwise computation")

        from .orekit_visibility import OrekitVisibilityCalculator

        calc = OrekitVisibilityCalculator(
            config={
                "use_java_orekit": True,
                "use_adaptive_step": True,
                "min_elevation": config.min_elevation_degrees,
            }
        )

        target_windows: Dict[Tuple[str, str], List[VisibilityWindow]] = {}
        gs_windows: Dict[Tuple[str, str], List[VisibilityWindow]] = {}

        # 逐对计算
        for sat in satellites:
            for target in targets:
                windows = calc.compute_satellite_target_windows(
                    sat, target, start_time, end_time
                )
                if windows:
                    target_windows[(sat.id, target.id)] = windows

            for gs in ground_stations:
                windows = calc.compute_satellite_ground_station_windows(
                    sat, gs, start_time, end_time
                )
                if windows:
                    gs_windows[(sat.id, gs.id)] = windows

        return BatchVisibilityResult(
            target_windows=target_windows,
            ground_station_windows=gs_windows,
            computation_stats=None,
            is_fallback_result=True,
        )

    def _serialize_satellite(self, satellite) -> Dict[str, Any]:
        """序列化卫星参数"""
        orbit = getattr(satellite, "orbit", None)
        orbit_type = getattr(orbit, "orbit_type", "SSO")
        # Ensure orbitType is a string, not an enum
        if hasattr(orbit_type, 'value'):
            orbit_type = orbit_type.value
        return {
            "id": satellite.id,
            "name": getattr(satellite, "name", satellite.id),
            "orbitType": str(orbit_type),
            "semiMajorAxis": getattr(orbit, "semi_major_axis", 7016000.0),
            "eccentricity": getattr(orbit, "eccentricity", 0.001),
            "inclination": getattr(orbit, "inclination", 97.9),
            "raan": getattr(orbit, "raan", 0.0),
            "argOfPerigee": getattr(orbit, "arg_of_perigee", 90.0),
            "meanAnomaly": getattr(orbit, "mean_anomaly", 0.0),
            "altitude": getattr(orbit, "altitude", 645000.0),
        }

    def _serialize_target(self, target) -> Dict[str, Any]:
        """序列化目标参数"""
        return {
            "id": target.id,
            "name": getattr(target, "name", target.id),
            "longitude": target.longitude,
            "latitude": target.latitude,
            "altitude": getattr(target, "altitude", 0.0),
        }

    def _serialize_ground_station(self, gs) -> Dict[str, Any]:
        """序列化地面站参数"""
        return {
            "id": gs.id,
            "name": getattr(gs, "name", gs.id),
            "longitude": gs.longitude,
            "latitude": gs.latitude,
            "altitude": getattr(gs, "altitude", 0.0),
            "minElevation": getattr(gs, "min_elevation", 5.0),
        }

    def _parse_java_result(
        self,
        java_result: Dict,
        satellites: List,
        targets: List,
        ground_stations: List,
    ) -> BatchVisibilityResult:
        """解析Java返回的结果"""

        # 解析目标窗口
        target_windows: Dict[Tuple[str, str], List[VisibilityWindow]] = {}
        for window_data in java_result.get("targetWindows", []):
            key = (window_data["satelliteId"], window_data["targetId"])
            if key not in target_windows:
                target_windows[key] = []

            target_windows[key].append(
                VisibilityWindow(
                    satellite_id=window_data["satelliteId"],
                    target_id=window_data["targetId"],
                    start_time=datetime.fromisoformat(window_data["startTime"]),
                    end_time=datetime.fromisoformat(window_data["endTime"]),
                    max_elevation=window_data["maxElevation"],
                    quality_score=min(1.0, window_data["maxElevation"] / 90.0),
                )
            )

        # 解析地面站窗口
        gs_windows: Dict[Tuple[str, str], List[VisibilityWindow]] = {}
        for window_data in java_result.get("groundStationWindows", []):
            key = (window_data["satelliteId"], window_data["targetId"])
            if key not in gs_windows:
                gs_windows[key] = []

            gs_windows[key].append(
                VisibilityWindow(
                    satellite_id=window_data["satelliteId"],
                    target_id=window_data["targetId"],
                    start_time=datetime.fromisoformat(window_data["startTime"]),
                    end_time=datetime.fromisoformat(window_data["endTime"]),
                    max_elevation=window_data["maxElevation"],
                    quality_score=min(1.0, window_data["maxElevation"] / 90.0),
                )
            )

        return BatchVisibilityResult(
            target_windows=target_windows,
            ground_station_windows=gs_windows,
            computation_stats=self._stats,
            is_fallback_result=False,
        )

    @property
    def last_computation_stats(self) -> Optional[BatchComputationStats]:
        """获取上次计算的统计"""
        return self._stats
