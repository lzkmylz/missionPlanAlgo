"""
Orekit Java桥接层

管理JVM生命周期，提供Python到Java Orekit库的桥接功能。
包含JVM单例管理、线程安全、缓存机制和异常转换。
"""

import functools
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable, List

# 配置日志
logger = logging.getLogger(__name__)

# 尝试导入jpype
try:
    import jpype
    from jpype import JClass
    JPYPE_AVAILABLE = True
except ImportError:
    JPYPE_AVAILABLE = False
    logger.warning("JPype not available. Orekit Java bridge will not function.")

# 导入配置
from core.orbit.visibility.orekit_config import (
    DEFAULT_OREKIT_CONFIG,
    merge_config,
    get_jvm_classpath
)


class OrbitPropagationError(Exception):
    """轨道传播错误

    当Orekit计算失败或Java异常发生时抛出。
    """
    pass


def ensure_jvm_attached(func: Callable) -> Callable:
    """装饰器：确保当前线程已挂载到JVM

    非启动JVM的Python线程调用Java前，必须先attachThreadToJVM()，
    否则会抛出JVM Not Attached异常。

    Args:
        func: 需要JVM访问的函数

    Returns:
        Callable: 包装后的函数
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if JPYPE_AVAILABLE and jpype.isJVMStarted() and not jpype.isThreadAttachedToJVM():
            jpype.attachThreadToJVM()
        return func(*args, **kwargs)
    return wrapper


def translate_java_exception(func: Callable) -> Callable:
    """装饰器：将Java异常转换为Python异常

    将jpype.JException转换为更易处理的Python异常类型。

    Args:
        func: 可能抛出Java异常的函数

    Returns:
        Callable: 包装后的函数

    Raises:
        OrbitPropagationError: Orekit计算失败
        ValueError: 参数错误
        RuntimeError: 其他Java异常
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as ex:
            # 检查是否是Java异常
            if JPYPE_AVAILABLE and hasattr(ex, 'javaClass'):
                try:
                    java_class = ex.javaClass().getName()
                    error_msg = str(ex)

                    if "OrekitException" in java_class:
                        raise OrbitPropagationError(f"Orekit计算失败: {error_msg}") from None
                    elif "IllegalArgumentException" in java_class:
                        raise ValueError(f"参数错误: {error_msg}") from None
                    else:
                        raise RuntimeError(f"Java异常 [{java_class}]: {error_msg}") from None
                except AttributeError:
                    # 无法获取Java类信息，重新抛出原始异常
                    raise
            else:
                # 不是Java异常，直接抛出
                raise
    return wrapper


class OrekitJavaBridge:
    """Orekit Java桥接单例

    管理JVM生命周期，提供对Java Orekit库的线程安全访问。
    使用单例模式确保整个Python进程中只有一个JVM实例。

    Attributes:
        _instance: 单例实例
        _jvm_started: JVM是否已启动
        _lock: 线程锁，用于单例创建和JVM启动
        _cached_frames: 坐标系缓存
        _cached_time_scales: 时间尺度缓存
        _cached_gravity_field: 引力场模型缓存
        _cached_atmosphere: 大气模型缓存
    """

    _instance: Optional['OrekitJavaBridge'] = None
    _jvm_started: bool = False
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, config: Optional[Dict[str, Any]] = None) -> 'OrekitJavaBridge':
        """单例模式实现

        Args:
            config: 可选的自定义配置

        Returns:
            OrekitJavaBridge: 单例实例
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize(config)
        return cls._instance

    def _initialize(self, config: Optional[Dict[str, Any]] = None) -> None:
        """初始化桥接器

        Args:
            config: 可选的自定义配置
        """
        # 合并配置
        self._config = merge_config(config)

        # 初始化缓存
        self._cached_frames: Dict[str, Any] = {}
        self._cached_time_scales: Dict[str, Any] = {}
        self._cached_gravity_field: Optional[Any] = None
        self._cached_atmosphere: Optional[Any] = None

        # 数据目录
        self._data_root_dir = self._config.get('data', {}).get('root_dir', '/usr/local/share/orekit')

        logger.debug("OrekitJavaBridge initialized")

    def _ensure_jvm_started(self) -> None:
        """确保JVM已启动（延迟启动）"""
        if not JPYPE_AVAILABLE:
            raise RuntimeError("JPype not available. Cannot start JVM.")

        if not OrekitJavaBridge._jvm_started and not jpype.isJVMStarted():
            with OrekitJavaBridge._lock:
                if not OrekitJavaBridge._jvm_started and not jpype.isJVMStarted():
                    self._start_jvm()

    def _start_jvm(self) -> None:
        """启动JVM并配置DataContext

        启动JVM，加载Orekit类库，配置数据上下文。
        线程安全，由调用者确保在锁内执行。
        """
        if not JPYPE_AVAILABLE:
            raise RuntimeError("JPype not available")

        try:
            # 获取JVM配置
            jvm_config = self._config.get('jvm', {})
            classpath = get_jvm_classpath(self._config)
            max_memory = jvm_config.get('max_memory', '2g')

            # 构建JVM启动参数
            jvm_args = [f"-Xmx{max_memory}"]

            logger.info(f"Starting JVM with classpath: {classpath}")

            # 启动JVM
            jpype.startJVM(
                classpath=[classpath],
                *jvm_args
            )

            # 配置DataContext
            self._configure_data_context()

            OrekitJavaBridge._jvm_started = True
            logger.info("JVM started successfully")

        except Exception as e:
            logger.error(f"Failed to start JVM: {e}")
            raise RuntimeError(f"JVM启动失败: {e}") from e

    def _configure_data_context(self) -> None:
        """配置Orekit数据上下文

        配置DataProvidersManager，加载IERS/EGM96/DE440数据。
        """
        if not JPYPE_AVAILABLE:
            return

        try:
            File = JClass("java.io.File")
            DataContext = JClass("org.orekit.data.DataContext")
            DirectoryCrawler = JClass("org.orekit.data.DirectoryCrawler")

            # 获取默认上下文
            context = DataContext.getDefault()
            manager = context.getDataProvidersManager()

            # 配置数据目录
            data_dir = File(self._data_root_dir)
            manager.addProvider(DirectoryCrawler(data_dir))

            logger.info(f"DataContext configured with data directory: {self._data_root_dir}")

        except Exception as e:
            logger.warning(f"Failed to configure DataContext: {e}")
            # 不抛出异常，允许在没有数据的情况下继续

    def _clear_cache(self) -> None:
        """清除所有缓存"""
        self._cached_frames.clear()
        self._cached_time_scales.clear()
        self._cached_gravity_field = None
        self._cached_atmosphere = None
        logger.debug("Cache cleared")

    @ensure_jvm_attached
    @translate_java_exception
    def get_frame(self, name: str) -> Any:
        """获取坐标系（带缓存）

        Args:
            name: 坐标系名称，如 'EME2000', 'ITRF'

        Returns:
            Any: Java坐标系对象
        """
        self._ensure_jvm_started()

        if name not in self._cached_frames:
            FramesFactory = JClass("org.orekit.frames.FramesFactory")

            # 处理Orekit 12.x ITRF API变化
            if name == "ITRF":
                # Orekit 12.x 需要额外参数: getITRF(IERSConventions, boolean)
                IERSConventions = JClass("org.orekit.utils.IERSConventions")
                self._cached_frames[name] = FramesFactory.getITRF(
                    IERSConventions.IERS_2010, True
                )
            else:
                method_name = f"get{name}"
                if hasattr(FramesFactory, method_name):
                    self._cached_frames[name] = getattr(FramesFactory, method_name)()
                else:
                    raise ValueError(f"Unknown frame: {name}")

        return self._cached_frames[name]

    @ensure_jvm_attached
    @translate_java_exception
    def get_time_scale(self, name: str) -> Any:
        """获取时间尺度（带缓存）

        Args:
            name: 时间尺度名称，如 'UTC', 'GPS', 'TAI'

        Returns:
            Any: Java时间尺度对象
        """
        self._ensure_jvm_started()

        if name not in self._cached_time_scales:
            TimeScalesFactory = JClass("org.orekit.time.TimeScalesFactory")
            method_name = f"get{name}"
            if hasattr(TimeScalesFactory, method_name):
                self._cached_time_scales[name] = getattr(TimeScalesFactory, method_name)()
            else:
                raise ValueError(f"Unknown time scale: {name}")

        return self._cached_time_scales[name]

    @ensure_jvm_attached
    @translate_java_exception
    def get_gravity_field(self, degree: int, order: int) -> Any:
        """获取地球引力场模型（带缓存）

        使用EGM96模型获取地球引力场提供者。
        结果会被缓存以避免重复加载。

        Args:
            degree: 引力场阶数
            order: 引力场次阶数（必须 <= degree）

        Returns:
            Any: Java NormalizedSphericalHarmonicsProvider对象

        Raises:
            ValueError: 如果order > degree
            OrbitPropagationError: 如果无法加载引力场数据
        """
        self._ensure_jvm_started()

        if order > degree:
            raise ValueError(f"Order ({order}) cannot exceed degree ({degree})")

        cache_key = f"gravity_{degree}_{order}"
        if cache_key not in self._cached_frames:
            try:
                GravityFieldFactory = JClass(
                    "org.orekit.forces.gravity.potential.GravityFieldFactory"
                )
                self._cached_frames[cache_key] = GravityFieldFactory.getNormalizedProvider(
                    degree, order
                )
            except Exception as e:
                logger.warning(f"Failed to load gravity field data: {e}")
                raise OrbitPropagationError(
                    f"无法加载引力场数据: {e}. "
                    f"请确保EGM96数据文件已安装在 {self._data_root_dir}/EGM96/"
                ) from e

        return self._cached_frames[cache_key]

    @ensure_jvm_attached
    @translate_java_exception
    def get_atmosphere_model(self, use_simple: bool = False) -> Any:
        """获取大气模型（带缓存）

        使用简单指数大气模型（不需要JPL星历）。
        结果会被缓存以避免重复创建。

        Args:
            use_simple: 是否使用简单指数模型（不需要JPL星历）

        Returns:
            Any: Java Atmosphere对象

        Raises:
            OrbitPropagationError: 如果无法创建大气模型
        """
        self._ensure_jvm_started()

        if self._cached_atmosphere is None:
            try:
                # 使用简单指数大气模型
                SimpleExponentialAtmosphere = JClass(
                    "org.orekit.models.earth.atmosphere.SimpleExponentialAtmosphere"
                )
                OneAxisEllipsoid = JClass(
                    "org.orekit.bodies.OneAxisEllipsoid"
                )
                Constants = JClass("org.orekit.utils.Constants")

                # 创建地球椭球模型
                earth = OneAxisEllipsoid(
                    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                    Constants.WGS84_EARTH_FLATTENING,
                    self.get_frame("ITRF")
                )

                # 标准大气参数
                rho0 = 1.225e-9  # kg/m^3 at sea level
                h0 = 0.0  # reference altitude (m)
                scale_height = 8500.0  # scale height (m)

                self._cached_atmosphere = SimpleExponentialAtmosphere(
                    earth, rho0, h0, scale_height
                )
                logger.debug("Created SimpleExponentialAtmosphere model")
            except Exception as e:
                logger.error(f"Failed to create atmosphere model: {e}")
                raise OrbitPropagationError(
                    f"无法创建大气模型: {e}"
                ) from e

        return self._cached_atmosphere

    @ensure_jvm_attached
    @translate_java_exception
    def create_numerical_propagator(self, orbit: Any, date: Any, frame: Any,
                                     config: Optional[Dict[str, Any]] = None) -> Any:
        """创建数值传播器

        创建配置完整摄动力模型的数值传播器。

        Args:
            orbit: 初始轨道（Java Orbit对象）
            date: 初始日期（Java AbsoluteDate对象）
            frame: 坐标系（Java Frame对象）
            config: 可选的传播器配置

        Returns:
            Any: Java NumericalPropagator对象
        """
        self._ensure_jvm_started()

        # 合并配置
        propagator_config = config if config else {}
        perturbations = propagator_config.get('perturbations', {})

        # 创建积分器（默认使用Dormand-Prince 8(5,3)）
        DormandPrince853Integrator = JClass(
            "org.hipparchus.ode.nonstiff.DormandPrince853Integrator"
        )

        # 积分器参数
        min_step = propagator_config.get('min_step', 0.001)
        max_step = propagator_config.get('max_step', 1000.0)

        integrator = DormandPrince853Integrator(
            min_step, max_step, 1e-13, 1e-13
        )

        # 创建数值传播器
        NumericalPropagator = JClass(
            "org.orekit.propagation.numerical.NumericalPropagator"
        )
        propagator = NumericalPropagator(integrator)

        # 设置初始状态 - orbit应该是Java Orbit对象
        SpacecraftState = JClass(
            "org.orekit.propagation.SpacecraftState"
        )
        initial_state = SpacecraftState(orbit)
        propagator.setInitialState(initial_state)

        # 配置摄动力模型
        if perturbations:
            self._configure_perturbations(propagator, perturbations, frame)

        return propagator

    def _configure_perturbations(self, propagator: Any,
                                  perturbations: Dict[str, Any],
                                  frame: Any) -> None:
        """配置摄动力模型

        Args:
            propagator: Java传播器对象
            perturbations: 摄动力配置字典
            frame: 坐标系
        """
        if not perturbations:
            return

        # 地球引力场
        gravity_config = perturbations.get('earth_gravity', {})
        if gravity_config.get('enabled', True):
            self._add_gravity_force(propagator, gravity_config, frame)

        # 大气阻力
        drag_config = perturbations.get('drag', {})
        if drag_config.get('enabled', True):
            self._add_drag_force(propagator, drag_config, frame)

        # 太阳光压
        srp_config = perturbations.get('solar_radiation', {})
        if srp_config.get('enabled', True):
            self._add_solar_radiation_pressure(propagator, srp_config)

        # 第三体引力
        third_body_config = perturbations.get('third_body', {})
        if third_body_config.get('enabled', True):
            self._add_third_body_forces(propagator, third_body_config)

        # 相对论效应
        relativity_config = perturbations.get('relativity', {})
        if relativity_config.get('enabled', True):
            self._add_relativity_force(propagator)

    def _add_gravity_force(self, propagator: Any,
                           config: Dict[str, Any], frame: Any) -> None:
        """添加地球引力摄动力

        Args:
            propagator: Java传播器对象
            config: 引力场配置
            frame: 坐标系
        """
        model = config.get('model', 'EGM96')
        degree = config.get('degree', 36)
        order = config.get('order', 36)

        # 获取引力场提供者
        gravity_field = self.get_gravity_field(degree, order)

        # 创建牛顿引力摄动力
        HolmesFeatherstoneAttractionModel = JClass(
            "org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel"
        )
        gravity_force = HolmesFeatherstoneAttractionModel(frame, gravity_field)

        propagator.addForceModel(gravity_force)
        logger.debug(f"Added gravity force model: {model} ({degree}x{order})")

    def _add_drag_force(self, propagator: Any,
                        config: Dict[str, Any], frame: Any) -> None:
        """添加大气阻力摄动力

        Args:
            propagator: Java传播器对象
            config: 阻力配置
            frame: 坐标系
        """
        cd = config.get('cd', 2.2)
        area = config.get('area', 1.0)

        # 获取大气模型
        atmosphere = self.get_atmosphere_model()

        # 创建阻力模型
        DragForce = JClass("org.orekit.forces.drag.DragForce")
        IsotropicDrag = JClass(
            "org.orekit.forces.drag.IsotropicDrag"
        )

        drag_sensitive = IsotropicDrag(area, cd)
        drag_force = DragForce(atmosphere, drag_sensitive)

        propagator.addForceModel(drag_force)
        logger.debug(f"Added drag force model: Cd={cd}, area={area}")

    def _add_solar_radiation_pressure(self, propagator: Any,
                                      config: Dict[str, Any]) -> None:
        """添加太阳光压摄动力 (Orekit 12.x API)

        Args:
            propagator: Java传播器对象
            config: 太阳光压配置
        """
        cr = config.get('cr', 1.0)
        area = config.get('area', 1.0)

        # 获取太阳和地球
        CelestialBodyFactory = JClass(
            "org.orekit.bodies.CelestialBodyFactory"
        )
        sun = CelestialBodyFactory.getSun()

        # Orekit 12.x: 使用OneAxisEllipsoid作为地球参数
        OneAxisEllipsoid = JClass("org.orekit.bodies.OneAxisEllipsoid")
        Constants = JClass("org.orekit.utils.Constants")
        earth = OneAxisEllipsoid(
            Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
            Constants.WGS84_EARTH_FLATTENING,
            self.get_frame("ITRF")
        )

        # 创建太阳光压模型 (Orekit 12.x API)
        SolarRadiationPressure = JClass(
            "org.orekit.forces.radiation.SolarRadiationPressure"
        )
        IsotropicRadiationSingleCoefficient = JClass(
            "org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient"
        )

        radiation_sensitive = IsotropicRadiationSingleCoefficient(area, cr)
        # Orekit 12.x: SolarRadiationPressure(sun, earth, radiation_sensitive)
        srp_force = SolarRadiationPressure(sun, earth, radiation_sensitive)

        propagator.addForceModel(srp_force)
        logger.debug(f"Added SRP force model: Cr={cr}, area={area}")

    def _add_third_body_forces(self, propagator: Any,
                                config: Dict[str, Any]) -> None:
        """添加第三体引力摄动力

        Args:
            propagator: Java传播器对象
            config: 第三体配置
        """
        bodies = config.get('bodies', ['SUN', 'MOON'])

        CelestialBodyFactory = JClass(
            "org.orekit.bodies.CelestialBodyFactory"
        )
        ThirdBodyAttraction = JClass(
            "org.orekit.forces.gravity.ThirdBodyAttraction"
        )

        for body_name in bodies:
            try:
                if body_name.upper() == 'SUN':
                    body = CelestialBodyFactory.getSun()
                elif body_name.upper() == 'MOON':
                    body = CelestialBodyFactory.getMoon()
                else:
                    logger.warning(f"Unknown third body: {body_name}")
                    continue

                third_body_force = ThirdBodyAttraction(body)
                propagator.addForceModel(third_body_force)
                logger.debug(f"Added third-body force: {body_name}")
            except Exception as e:
                logger.warning(f"Failed to add third-body force for {body_name}: {e}")

    def _add_relativity_force(self, propagator: Any) -> None:
        """添加相对论效应摄动力

        Args:
            propagator: Java传播器对象
        """
        Relativity = JClass(
            "org.orekit.forces.gravity.Relativity"
        )

        # 获取引力场常数（用于相对论计算）
        gravity_field = self.get_gravity_field(2, 0)
        mu = gravity_field.getMu()

        relativity_force = Relativity(mu)
        propagator.addForceModel(relativity_force)
        logger.debug("Added relativity force model")

    @ensure_jvm_attached
    @translate_java_exception
    def propagate_batch(self, propagator: Any, start_date: Any, end_date: Any,
                       step_size: float) -> Any:
        """高性能批量传播

        使用BatchStepHandler在Java端收集所有状态，
        传播结束后一次性返回double[][]数组，通过JPype零拷贝映射为numpy数组。

        Args:
            propagator: Java传播器对象
            start_date: 开始日期（AbsoluteDate）
            end_date: 结束日期（AbsoluteDate）
            step_size: 步长（秒）

        Returns:
            numpy.ndarray: shape为(n_steps, 7)的数组，列分别为
                          [seconds_since_j2000, px, py, pz, vx, vy, vz]
        """
        self._ensure_jvm_started()

        # 获取BatchStepHandler类
        BatchStepHandler = JClass("orekit.helper.BatchStepHandler")
        handler = BatchStepHandler()

        # Orekit 12.x: 使用setStepHandler代替setMasterMode
        FixedStepHandler = JClass(
            "org.orekit.propagation.sampling.OrekitStepHandler"
        )
        # 设置固定步长处理器
        try:
            # Orekit 12.x API
            propagator.setStepHandler(step_size, handler)
        except AttributeError:
            # 回退到旧版API (Orekit 11.x)
            propagator.setMasterMode(step_size, handler)

        # 执行传播
        propagator.propagate(start_date, end_date)

        # 获取结果
        results = handler.getResults()

        # 转换为numpy数组
        import numpy as np
        return np.array(results)

    @ensure_jvm_attached
    @translate_java_exception
    def reload_data_context(self) -> None:
        """重新加载Orekit数据文件

        用于IERS数据更新后重新加载数据文件。
        """
        self._ensure_jvm_started()

        try:
            DataContext = JClass("org.orekit.data.DataContext")
            DirectoryCrawler = JClass("org.orekit.data.DirectoryCrawler")
            File = JClass("java.io.File")

            # 获取默认上下文
            context = DataContext.getDefault()
            manager = context.getDataProvidersManager()

            # 清除现有提供者
            manager.clearProviders()

            # 清除缓存
            self._clear_cache()

            # 重新添加数据目录
            data_dir = File(self._data_root_dir)
            manager.addProvider(DirectoryCrawler(data_dir))

            logger.info("DataContext reloaded successfully")

        except Exception as e:
            logger.error(f"Failed to reload DataContext: {e}")
            raise OrbitPropagationError(f"数据上下文重载失败: {e}") from e

    def is_jvm_running(self) -> bool:
        """检查JVM是否正在运行

        Returns:
            bool: JVM是否已启动
        """
        if not JPYPE_AVAILABLE:
            return False
        return jpype.isJVMStarted()

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置

        Returns:
            Dict[str, Any]: 配置字典
        """
        return self._config.copy()

    @translate_java_exception
    def _get_java_class(self, class_name: str) -> Any:
        """获取Java类

        辅助方法，用于动态获取Java类。

        Args:
            class_name: 完整的Java类名，如 "org.orekit.time.AbsoluteDate"

        Returns:
            Any: Java类对象

        Raises:
            RuntimeError: 如果JPype不可用或类不存在
        """
        if not JPYPE_AVAILABLE:
            raise RuntimeError("JPype not available")

        # 确保JVM已启动
        self._ensure_jvm_started()

        # 确保当前线程已附加到JVM（多线程环境下必需）
        if jpype.isJVMStarted() and not jpype.isThreadAttachedToJVM():
            jpype.attachThreadToJVM()

        return JClass(class_name)

    # =========================================================================
    # Phase 2: Java端批量可见性计算接口
    # =========================================================================

    @ensure_jvm_attached
    @translate_java_exception
    def compute_visibility_batch(
        self,
        satellite_configs: List[Dict[str, Any]],
        target_configs: List[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime,
        coarse_step_seconds: int = 300,
        fine_step_seconds: int = 60,
        min_elevation_degrees: float = 5.0,
        min_window_duration_seconds: int = 60,
    ) -> Dict[str, Any]:
        """
        Java端批量可见窗口计算

        Phase 2优化：单次JNI调用完成所有计算，大幅减少JNI开销。
        在Java端实现自适应步长算法（粗扫描+精化）。

        Args:
            satellite_configs: 卫星配置列表
                [{ 'id': str, 'tle_line1': str, 'tle_line2': str,
                   'min_elevation': float (可选), 'sensor_fov': float (可选) }]
            target_configs: 目标配置列表
                [{ 'id': str, 'longitude': float, 'latitude': float,
                   'altitude': float (默认0), 'min_observation_duration': int (默认60),
                   'priority': int (默认5) }]
            start_time: 开始时间
            end_time: 结束时间
            coarse_step_seconds: 粗扫描步长（秒，默认300）
            fine_step_seconds: 精化步长（秒，默认60）
            min_elevation_degrees: 最小仰角（度，默认5.0）
            min_window_duration_seconds: 最小窗口持续时间（秒，默认60）

        Returns:
            Dict: {
                'windows': {
                    (sat_id, target_id): [
                        {
                            'satellite_id': str,
                            'target_id': str,
                            'start_time': datetime,
                            'end_time': datetime,
                            'duration_seconds': float,
                            'max_elevation': float,
                            'max_elevation_time': datetime,
                            'entry_azimuth': float,
                            'exit_azimuth': float,
                            'quality_score': float,
                            'confidence': str
                        },
                        ...
                    ]
                },
                'statistics': {
                    'total_pairs': int,
                    'pairs_with_windows': int,
                    'total_windows': int,
                    'computation_time_ms': int,
                    'coarse_scan_points': int,
                    'fine_scan_points': int
                },
                'errors': [
                    {
                        'satellite_id': str,
                        'target_id': str,
                        'error_type': str,
                        'error_message': str
                    },
                    ...
                ]
            }

        Raises:
            OrbitPropagationError: 如果Java计算失败
        """
        self._ensure_jvm_started()

        try:
            # 获取Java类
            BatchCalculator = JClass(
                "orekit.visibility.calculator.VisibilityBatchCalculator"
            )
            SatelliteConfig = JClass(
                "orekit.visibility.model.SatelliteConfig"
            )
            TargetConfig = JClass(
                "orekit.visibility.model.TargetConfig"
            )
            AbsoluteDate = JClass("org.orekit.time.AbsoluteDate")

            # 创建计算器实例
            calculator = BatchCalculator(
                float(coarse_step_seconds),
                float(fine_step_seconds),
                float(min_elevation_degrees),
                float(min_window_duration_seconds)
            )

            # 转换卫星配置为Java对象
            java_satellites = []
            for sat in satellite_configs:
                java_sat = SatelliteConfig(
                    sat['id'],
                    sat.get('tle_line1', ''),
                    sat.get('tle_line2', ''),
                    float(sat.get('min_elevation', min_elevation_degrees)),
                    float(sat.get('sensor_fov', 0.0))
                )
                java_satellites.append(java_sat)

            # 转换目标配置为Java对象
            java_targets = []
            for tgt in target_configs:
                java_tgt = TargetConfig(
                    tgt['id'],
                    float(tgt['longitude']),
                    float(tgt['latitude']),
                    float(tgt.get('altitude', 0.0)),
                    int(tgt.get('min_observation_duration', min_window_duration_seconds)),
                    int(tgt.get('priority', 5))
                )
                java_targets.append(java_tgt)

            # 转换时间
            java_start = self._datetime_to_java_date(start_time, AbsoluteDate)
            java_end = self._datetime_to_java_date(end_time, AbsoluteDate)

            # 执行批量计算
            java_result = calculator.computeBatch(
                java_satellites,
                java_targets,
                java_start,
                java_end
            )

            # 转换结果为Python字典
            return self._convert_batch_result(java_result)

        except Exception as e:
            logger.error(f"Java batch computation failed: {e}")
            raise OrbitPropagationError(
                f"Java批量计算失败: {e}"
            ) from e

    def _datetime_to_java_date(
        self,
        dt: datetime,
        AbsoluteDate_class: Any
    ) -> Any:
        """将Python datetime转换为Java AbsoluteDate"""
        utc = self.get_time_scale("UTC")

        # 使用JClass获取DateTimeComponents
        DateTimeComponents = JClass(
            "org.orekit.time.DateTimeComponents"
        )

        # 创建DateTimeComponents
        components = DateTimeComponents(
            dt.year,
            dt.month,
            dt.day,
            dt.hour,
            dt.minute,
            float(dt.second + dt.microsecond / 1e6)
        )

        return AbsoluteDate_class(components, utc)

    def _convert_batch_result(self, java_result: Any) -> Dict[str, Any]:
        """将Java BatchResult转换为Python字典"""
        result = {
            'windows': {},
            'statistics': {},
            'errors': []
        }

        try:
            # 转换统计信息
            java_stats = java_result.getStatistics()
            result['statistics'] = {
                'total_pairs': java_stats.getTotalPairs(),
                'pairs_with_windows': java_stats.getPairsWithWindows(),
                'total_windows': java_stats.getTotalWindows(),
                'computation_time_ms': java_stats.getComputationTimeMs(),
                'coarse_scan_points': java_stats.getCoarseScanPoints(),
                'fine_scan_points': java_stats.getFineScanPoints(),
            }

            # 转换错误信息
            java_errors = java_result.getErrors()
            for error in java_errors:
                result['errors'].append({
                    'satellite_id': error.getSatelliteId(),
                    'target_id': error.getTargetId(),
                    'error_type': error.getErrorType(),
                    'error_message': error.getErrorMessage()
                })

            # 转换窗口数据
            java_windows_map = java_result.getAllWindows()
            for entry in java_windows_map.entrySet():
                key = entry.getKey()  # "sat_id-target_id"格式
                window_list = entry.getValue()

                # 解析键
                parts = key.split('-', 1)
                if len(parts) == 2:
                    sat_id, target_id = parts
                else:
                    sat_id = key
                    target_id = key

                result['windows'][(sat_id, target_id)] = []

                for java_window in window_list:
                    # 转换Java日期为Python datetime
                    start_dt = self._java_date_to_datetime(
                        java_window.getStartTime()
                    )
                    end_dt = self._java_date_to_datetime(
                        java_window.getEndTime()
                    )
                    max_el_time = self._java_date_to_datetime(
                        java_window.getMaxElevationTime()
                    )

                    result['windows'][(sat_id, target_id)].append({
                        'satellite_id': java_window.getSatelliteId(),
                        'target_id': java_window.getTargetId(),
                        'start_time': start_dt,
                        'end_time': end_dt,
                        'duration_seconds': java_window.getDurationSeconds(),
                        'max_elevation': java_window.getMaxElevation(),
                        'max_elevation_time': max_el_time,
                        'entry_azimuth': java_window.getEntryAzimuth(),
                        'exit_azimuth': java_window.getExitAzimuth(),
                        'quality_score': java_window.getQualityScore(),
                        'confidence': java_window.getConfidence()
                    })

        except Exception as e:
            logger.error(f"Failed to convert batch result: {e}")
            result['errors'].append({
                'satellite_id': 'CONVERSION',
                'target_id': 'CONVERSION',
                'error_type': 'CONVERSION_ERROR',
                'error_message': str(e)
            })

        return result

    def _java_date_to_datetime(self, java_date: Any) -> datetime:
        """将Java AbsoluteDate转换为Python datetime"""
        utc = self.get_time_scale("UTC")

        # 获取日期组件
        components = java_date.getComponents(utc)
        date = components.getDate()
        time = components.getTime()

        return datetime(
            int(date.getYear()),
            int(date.getMonth()),
            int(date.getDay()),
            int(time.getHour()),
            int(time.getMinute()),
            int(time.getSecond()),
            int((time.getSecond() - int(time.getSecond())) * 1e6)
        )

    @ensure_jvm_attached
    @translate_java_exception
    def compute_visibility_batch(
        self,
        satellites: List[Dict],
        targets: List[Dict],
        ground_stations: List[Dict],
        start_time: datetime,
        end_time: datetime,
        config: Dict
    ) -> Dict:
        """
        批量计算所有可见窗口（高性能版本）

        单次JNI调用完成全部计算，避免多次往返开销。

        Args:
            satellites: 卫星参数列表
            targets: 目标参数列表
            ground_stations: 地面站参数列表
            start_time: 开始时间
            end_time: 结束时间
            config: 计算配置
                - coarseStep: 粗扫描步长（秒）
                - fineStep: 精化步长（秒）
                - minElevation: 最小仰角（度）
                - useParallel: 是否并行传播

        Returns:
            Dict: 包含targetWindows、groundStationWindows和stats的结果

        Raises:
            JavaError: Java计算失败
        """
        self._ensure_jvm_started()

        try:
            # 获取Java类
            PythonBridge = self._get_java_class(
                "orekit.visibility.PythonBridge"
            )
            SatelliteParameters = self._get_java_class(
                "orekit.visibility.SatelliteParameters"
            )
            GroundPoint = self._get_java_class(
                "orekit.visibility.GroundPoint"
            )
            ComputationConfig = self._get_java_class(
                "orekit.visibility.ComputationConfig"
            )
            ArrayList = self._get_java_class("java.util.ArrayList")

            # 转换卫星参数
            sat_list = ArrayList()
            for sat in satellites:
                sat_param = SatelliteParameters()
                sat_param.setId(sat['id'])
                sat_param.setName(sat.get('name', sat['id']))
                sat_param.setOrbitType(sat.get('orbitType', 'SSO'))
                sat_param.setSemiMajorAxis(sat.get('semiMajorAxis', 7016000.0))
                sat_param.setEccentricity(sat.get('eccentricity', 0.001))
                sat_param.setInclination(sat.get('inclination', 97.9))
                sat_param.setRaan(sat.get('raan', 0.0))
                sat_param.setArgOfPerigee(sat.get('argOfPerigee', 90.0))
                sat_param.setMeanAnomaly(sat.get('meanAnomaly', 0.0))
                sat_param.setAltitude(sat.get('altitude', 645000.0))
                sat_param.setEpoch(start_time.isoformat())
                sat_list.add(sat_param)

            # 转换目标参数
            target_list = ArrayList()
            for target in targets:
                point = GroundPoint()
                point.setId(target['id'])
                point.setName(target.get('name', target['id']))
                point.setLongitude(target['longitude'])
                point.setLatitude(target['latitude'])
                point.setAltitude(target.get('altitude', 0.0))
                point.setMinElevation(0.0)
                target_list.add(point)

            # 转换地面站参数
            gs_list = ArrayList()
            for gs in ground_stations:
                point = GroundPoint()
                point.setId(gs['id'])
                point.setName(gs.get('name', gs['id']))
                point.setLongitude(gs['longitude'])
                point.setLatitude(gs['latitude'])
                point.setAltitude(gs.get('altitude', 0.0))
                point.setMinElevation(gs.get('minElevation', 5.0))
                gs_list.add(point)

            # 转换配置
            java_config = ComputationConfig()
            java_config.setCoarseStep(config.get('coarseStep', 300.0))
            java_config.setFineStep(config.get('fineStep', 60.0))
            java_config.setMinElevation(config.get('minElevation', 0.0))
            java_config.setUseParallel(config.get('useParallel', True))
            java_config.setMaxBatchSize(config.get('maxBatchSize', 100))

            # 调用Java批量计算方法
            result = PythonBridge.computeVisibilityBatch(
                sat_list,
                target_list,
                gs_list,
                start_time.isoformat(),
                end_time.isoformat(),
                java_config
            )

            # 检查结果是否有错误 (Java Map doesn't support get(key, default))
            if result.containsKey('error') and result.get('error'):
                error_msg = result.get('errorMessage') if result.containsKey('errorMessage') else 'Unknown error'
                raise RuntimeError(
                    f"Java computation failed: {error_msg}"
                )

            # 转换结果为Python原生类型
            return self._convert_batch_result(result)

        except Exception as e:
            logger.error(f"Batch computation via Java failed: {e}")
            raise

    def _convert_batch_result(self, java_result: Any) -> Dict:
        """将Java返回的批量结果转换为Python原生类型"""
        result = {
            'targetWindows': [],
            'groundStationWindows': [],
            'stats': {}
        }

        # Helper to safely get from Java Map (no default value support)
        def safe_get(map_obj, key, default=None):
            if map_obj and map_obj.containsKey(key):
                return map_obj.get(key)
            return default

        # 转换目标窗口
        target_windows = safe_get(java_result, 'targetWindows', [])
        if target_windows:
            for window in target_windows:
                result['targetWindows'].append({
                    'satelliteId': str(safe_get(window, 'satelliteId')),
                    'targetId': str(safe_get(window, 'targetId')),
                    'startTime': str(safe_get(window, 'startTime')),
                    'endTime': str(safe_get(window, 'endTime')),
                    'maxElevation': float(safe_get(window, 'maxElevation', 0.0)),
                    'durationSeconds': float(safe_get(window, 'durationSeconds', 0.0)),
                })

        # 转换地面站窗口
        gs_windows = safe_get(java_result, 'groundStationWindows', [])
        if gs_windows:
            for window in gs_windows:
                result['groundStationWindows'].append({
                    'satelliteId': str(safe_get(window, 'satelliteId')),
                    'targetId': str(safe_get(window, 'targetId')),
                    'startTime': str(safe_get(window, 'startTime')),
                    'endTime': str(safe_get(window, 'endTime')),
                    'maxElevation': float(safe_get(window, 'maxElevation', 0.0)),
                    'durationSeconds': float(safe_get(window, 'durationSeconds', 0.0)),
                })

        # 转换统计信息
        stats = safe_get(java_result, 'stats', {})
        if stats:
            result['stats'] = {
                'computationTimeMs': int(safe_get(stats, 'computationTimeMs', 0)),
                'nWindows': int(safe_get(stats, 'nWindows', 0)),
                'memoryUsageMb': float(safe_get(stats, 'memoryUsageMb', 0.0)),
            }

        return result
