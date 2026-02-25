"""
Orekit Java桥接层

管理JVM生命周期，提供Python到Java Orekit库的桥接功能。
包含JVM单例管理、线程安全、缓存机制和异常转换。
"""

import functools
import threading
import logging
from typing import Dict, Any, Optional, Callable

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
        """添加太阳光压摄动力

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
        earth = CelestialBodyFactory.getEarth()

        # 创建太阳光压模型
        SolarRadiationPressure = JClass(
            "org.orekit.forces.radiation.SolarRadiationPressure"
        )
        IsotropicRadiationSingleCoefficient = JClass(
            "org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient"
        )

        radiation_sensitive = IsotropicRadiationSingleCoefficient(area, cr)
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

        # 设置固定步长处理器
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

    @ensure_jvm_attached
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

        self._ensure_jvm_started()
        return JClass(class_name)
