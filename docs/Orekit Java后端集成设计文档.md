# Orekit Java后端集成设计文档

## 概述

将当前的纯Python实现的`OrekitVisibilityCalculator`升级为通过JPype桥接调用真正的Java Orekit库（v12.x），实现高精度轨道传播计算。

## 系统要求

- **Java**: 17 LTS（长期支持版本）
- **Python**: 3.8+
- **JPype**: 1.5.0+
- **Orekit**: 12.x
- **Hipparchus**: Orekit数学库依赖

## 架构设计

### 整体架构

```
Python Application
       │
       ▼
┌──────────────────────┐
│ OrekitVisibilityCalculator │  (修改现有类)
│  - _propagate_satellite()  │  (自动选择传播器)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ OrekitJavaBridge     │  (新增桥接层)
│  - 单例模式管理JVM   │
│  - 自动启动/关闭     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│    JPype Bridge      │  (第三方库)
│  - Python/Java桥接   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Java Orekit 12.x   │  (高精度轨道力学库)
│  - NumericalPropagator    │
│  - EGM96地球引力场        │
│  - NRLMSISE-00大气模型    │
│  - 太阳光压/三体引力      │
└──────────────────────┘
```

### JVM生命周期管理

采用**方案A：Python进程级生命周期**

```
Python进程启动
      │
      ▼
┌─────────────┐
│ 延迟初始化  │  首次调用Orekit时启动JVM
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  JVM运行中  │  多次调用Orekit（预计算窗口）
│  Orekit计算 │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Python结束  │  atexit钩子自动关闭JVM
└─────────────┘
```

**特点**:
- JVM只启动一次，性能好
- **不主动关闭JVM**：让操作系统在Python进程结束时自然回收，避免shutdownJVM()导致的僵死/段错误风险
- 对现有代码无侵入性

**⚠️ 重要**：根据JPype官方建议，不推荐显式调用`shutdownJVM()`。JVM一旦关闭无法在同一进程重启，且可能导致资源释放问题。

## 关键设计优化

### 1. 性能优化：避免频繁JNI调用

**问题**：Python for循环中每次调用Java propagate()，24小时窗口(1s步长)会产生86,400次JNI调用，开销极其恐怖。

**解决方案演进**：

#### 方案A：Python StepHandler（基础优化）

```python
@jpype.JImplements("org.orekit.propagation.sampling.OrekitFixedStepHandler")
class PythonStepHandler:
    def __init__(self):
        self.states = []

    @jpype.JOverride
    def handleStep(self, currentState, isLast):
        pv = currentState.getPVCoordinates()
        self.states.append({
            'date': currentState.getDate(),
            'position': pv.getPosition(),
            'velocity': pv.getVelocity()
        })
```

**隐患**：虽然减少了Python调用Java的次数，但Java每走一步仍需通过JNI回调Python 86,400次，Python GIL和代理转换仍是瓶颈。

#### 方案B：Java辅助类 + 零拷贝（推荐极致优化）

利用JPype对Java基本类型数组与NumPy的零拷贝特性，将86,400次双向通信降为**1次调用+1次大块内存读取**。

**Java辅助类**（编译为 `orekit-helper.jar`）：
```java
package orekit.helper;

import org.orekit.propagation.sampling.OrekitFixedStepHandler;
import org.orekit.propagation.SpacecraftState;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;
import java.util.ArrayList;

public class BatchStepHandler implements OrekitFixedStepHandler {
    private final ArrayList<double[]> data = new ArrayList<>();

    @Override
    public void handleStep(SpacecraftState currentState, boolean isLast) {
        AbsoluteDate date = currentState.getDate();
        PVCoordinates pv = currentState.getPVCoordinates();

        // [seconds_since_epoch, px, py, pz, vx, vy, vz]
        double[] row = new double[7];
        row[0] = date.durationFrom(AbsoluteDate.J2000_EPOCH);
        row[1] = pv.getPosition().getX();
        row[2] = pv.getPosition().getY();
        row[3] = pv.getPosition().getZ();
        row[4] = pv.getVelocity().getX();
        row[5] = pv.getVelocity().getY();
        row[6] = pv.getVelocity().getZ();

        data.add(row);
    }

    public double[][] getResults() {
        return data.toArray(new double[0][]);
    }
}
```

**Python端使用**：
```python
class OrekitJavaBridge:
    def propagate_batch(self, propagator, start_date, end_date, step_size):
        """批量传播：Java内部收集，一次返回所有数据"""
        BatchStepHandler = jpype.JClass("orekit.helper.BatchStepHandler")
        handler = BatchStepHandler()

        # 1次JNI调用，Java内部完成86,400步
        propagator.propagate(start_date, end_date, handler)

        # 零拷贝：Java double[][] → Python numpy数组
        results = handler.getResults()
        import numpy as np
        return np.array(results)  # shape: (86400, 7)
```

**性能对比**：
| 方案 | 24小时窗口计算时间 | JNI调用次数 | 跨语言通信方向 |
|------|-------------------|-------------|----------------|
| Python循环调用 | ~60-120s | 86,400 | Python→Java |
| Python StepHandler | ~3-5s | 86,400 | Java→Python (回调) |
| **Java辅助类+零拷贝（推荐）** | **~0.5-1s** | **1** | **单向返回** |
| 星历Ephemeris | ~3-5s | 1 | 单向返回 |

### 2. 线程安全：JVM线程挂载

**问题**：非启动JVM的Python线程调用Java前，必须先`attachThreadToJVM()`，否则会抛出`JVM Not Attached`异常。

**解决方案**：

```python
def ensure_jvm_attached(func):
    """装饰器：确保当前线程已挂载到JVM"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if jpype.isJVMStarted() and not jpype.isThreadAttachedToJVM():
            jpype.attachThreadToJVM()
        return func(*args, **kwargs)
    return wrapper

class OrekitJavaBridge:
    @ensure_jvm_attached
    def propagate_with_handler(self, propagator, start_date, end_date, handler):
        """线程安全的方法"""
        return propagator.propagate(start_date, end_date, handler)
```

**线程池使用建议**：

对于使用 `ThreadPoolExecutor` 或高并发Web框架（FastAPI/Celery）的场景，如果线程频繁创建销毁，被attach的线程若不detach可能阻止JVM释放线程本地变量。

**推荐做法**：在线程池的initializer中统一attach

```python
from concurrent.futures import ThreadPoolExecutor
import jpype

def init_worker():
    """线程池工作线程初始化器"""
    if jpype.isJVMStarted() and not jpype.isThreadAttachedToJVM():
        jpype.attachThreadToJVM()

# 创建线程池时指定initializer
executor = ThreadPoolExecutor(
    max_workers=4,
    initializer=init_worker
)

# 后续提交的任务所在的线程都已挂载到JVM
executor.submit(orekit_task)
```

### 3. DataContext配置与数据时效性

**问题**：Orekit 12.x严格依赖DataContext，不配置数据提供者会导致时间转换、坐标系转换抛出异常。

**解决方案**：

```python
def _configure_data_context(self, data_root_dir):
    """配置Orekit数据上下文"""
    File = jpype.JClass("java.io.File")
    DataContext = jpype.JClass("org.orekit.data.DataContext")
    DataProvidersManager = jpype.JClass("org.orekit.data.DataProvidersManager")
    DirectoryCrawler = jpype.JClass("org.orekit.data.DirectoryCrawler")

    # 获取默认上下文
    context = DataContext.getDefault()
    manager = context.getDataProvidersManager()

    # 配置数据目录
    data_dir = File(data_root_dir)
    manager.addProvider(DirectoryCrawler(data_dir))

    # 验证关键数据加载
    # - IERS finals.all (闰秒、EOP参数)
    # - EGM96.gfc (地球引力场)
    # - DE440.bsp (JPL星历)
```

**⚠️ 生产环境数据时效性风险**：

对于高精度轨道传播（尤其是使用大气阻力模型、太阳光压、涉及ECI到ECEF坐标转换），地球自转参数(EOP)和闰秒数据随时间变化会过期。

**应对措施**：

1. **定时同步**：配置Cron任务定期从IERS官网同步最新 `finals.all`
2. **运行时重载接口**：Python端暴露DataContext刷新方法
3. **定期重启**：对于长期运行的Python Worker，定期重启以重新加载数据

```python
class OrekitJavaBridge:
    def reload_data_context(self):
        """重新加载Orekit数据文件（用于数据更新后）"""
        DataContext = jpype.JClass("org.orekit.data.DataContext")
        # 清除缓存，强制重新加载
        context = DataContext.getDefault()
        manager = context.getDataProvidersManager()
        manager.clearProviders()
        # 重新添加数据目录
        self._configure_data_context(self.data_root_dir)

# 运维脚本示例
# crontab -e
# 0 2 * * * /usr/local/bin/update_orekit_data.sh && curl -X POST http://localhost:8000/api/orekit/reload
```

### 4. 异常处理映射

**问题**：Orekit异常以`jpype.JException`形式抛出，直接暴露给业务层难以维护。

**解决方案**：

```python
class OrbitPropagationError(Exception):
    """轨道传播错误"""
    pass

def translate_java_exception(func):
    """装饰器：将Java异常转换为Python异常"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except jpype.JException as ex:
            java_class = ex.javaClass().getName()
            error_msg = ex.getMessage()

            if "OrekitException" in java_class:
                raise OrbitPropagationError(f"Orekit计算失败: {error_msg}") from None
            elif "IllegalArgumentException" in java_class:
                raise ValueError(f"参数错误: {error_msg}") from None
            else:
                raise RuntimeError(f"Java异常 [{java_class}]: {error_msg}") from None
    return wrapper
```

### 5. 静态对象缓存

**问题**：频繁创建引力场模型、大气模型等对象开销较大。

**解决方案**：

```python
class OrekitJavaBridge:
    _cache = {
        'frames': {},  # {'EME2000': frame_obj, 'ITRF': frame_obj}
        'time_scales': {},  # {'UTC': ts_obj, 'GPS': ts_obj}
        'gravity_field': None,  # EGM96加载后缓存（重要！）
        'atmosphere': None,  # NRLMSISE00加载后缓存（重要！）
    }

    def get_frame(self, name):
        """获取坐标系（带缓存）"""
        if name not in self._cache['frames']:
            FramesFactory = jpype.JClass("org.orekit.frames.FramesFactory")
            self._cache['frames'][name] = getattr(FramesFactory, f"get{name}")()
        return self._cache['frames'][name]

    def get_gravity_field(self, degree, order):
        """获取地球引力场模型（带缓存）- 真正需要缓存的对象"""
        cache_key = f"gravity_{degree}_{order}"
        if cache_key not in self._cache:
            GravityFieldFactory = jpype.JClass("org.orekit.forces.gravity.GravityFieldFactory")
            self._cache[cache_key] = GravityFieldFactory.getNormalizedProvider(degree, order)
        return self._cache[cache_key]

    def get_atmosphere_model(self, solar_activity=None):
        """获取大气模型（带缓存）- 真正需要缓存的对象"""
        if self._cache['atmosphere'] is None:
            # NRLMSISE00模型初始化需要解析数据文件，开销较大
            from org.orekit.models.earth.atmosphere import NRLMSISE00
            from org.orekit.models.earth.atmosphere.data import MarshallSolarActivityFutureEstimation
            # ... 初始化代码
            self._cache['atmosphere'] = atmosphere
        return self._cache['atmosphere']
```

**缓存策略说明**：

| 对象类型 | Java端是否已缓存 | Python端缓存价值 | 建议 |
|----------|------------------|------------------|------|
| Frames (EME2000等) | ✅ FramesFactory内部已用LazyLoadedDataContext实现单例 | 低（省去JNI微小开销） | 可做可不做 |
| TimeScales (UTC等) | ✅ TimeScalesFactory内部已缓存 | 低 | 可做可不做 |
| GravityField | ❌ 每次getNormalizedProvider都创建新对象 | **高** | **必须缓存** |
| Atmosphere模型 | ❌ 初始化需解析数据文件 | **高** | **必须缓存** |
| CelestialBody | ✅ 内部已缓存 | 低 | 无需缓存 |

## 核心组件设计

### 1. OrekitJavaBridge (桥接层)

**文件**: `core/orbit/visibility/orekit_java_bridge.py`

**职责**:
- JVM生命周期管理（启动/关闭）
- Orekit数据文件加载
- Java对象创建封装
- 摄动力模型配置

**关键方法**:
```python
class OrekitJavaBridge:
    """Orekit Java桥接单例"""

    _instance = None
    _jvm_started = False
    _lock = threading.Lock()

    # 缓存的静态Java对象（避免重复创建）
    _cached_frames = {}
    _cached_time_scales = {}
    _cached_gravity_field = None
    _cached_atmosphere = None

    def __new__(cls):
        # 单例模式

    def _start_jvm(self):
        """启动JVM（线程安全）并配置DataContext"""
        # 1. 启动JVM
        # 2. 配置DataProvidersManager（加载IERS/EGM96/DE440数据）
        # 3. 预加载并缓存静态对象（FramesFactory, TimeScalesFactory等）

    def ensure_jvm_attached(self, func):
        """装饰器：确保当前线程已挂载到JVM"""
        # 非启动线程调用Java前必须先attachThreadToJVM()

    def create_numerical_propagator(self, orbit, date, frame, config):
        """创建数值传播器，配置摄动力模型（使用缓存的静态对象）"""

    def propagate_with_handler(self, propagator, start_date, end_date, step_handler):
        """使用OrekitFixedStepHandler批量传播（避免频繁JNI调用）"""
        # 方案A：Java引擎循环，回调Python收集状态
        # 性能：比Python循环调用快10-100倍

    def propagate_ephemeris(self, propagator, start_date, end_date):
        """生成星历对象，支持后续插值（替代方案B）"""

    def propagate_batch(self, propagator, start_date, end_date, step_size):
        """高性能批量传播（推荐方案）

        使用Java辅助类BatchStepHandler在Java端收集所有状态，
        传播结束后一次性返回double[][]数组，通过JPype零拷贝映射为numpy数组。
        将86,400次JNI调用降为1次，性能提升一个数量级。

        Returns:
            numpy.ndarray: shape为(n_steps, 7)的数组，列分别为
                          [seconds_since_j2000, px, py, pz, vx, vy, vz]
        """

    def reload_data_context(self):
        """重新加载Orekit数据文件（用于IERS数据更新后）"""
```

### 2. OrekitVisibilityCalculator (修改)

**文件**: `core/orbit/visibility/orekit_visibility.py`

**修改点**:
```python
class OrekitVisibilityCalculator(VisibilityCalculator):
    def __init__(self, config=None):
        super().__init__(config)
        self.use_java_orekit = config.get('use_java_orekit', True)
        self.orekit_config = config.get('orekit', {})

    def _propagate_satellite(self, satellite, dt):
        if self.use_java_orekit:
            return self._propagate_with_java_orekit(satellite, dt)
        else:
            return self._propagate_simplified(satellite, dt)  # fallback

    def _propagate_range_with_java_orekit(self, satellite, start_time, end_time, time_step):
        """调用真正的Java Orekit进行批量传播（高性能方案）

        性能优化：使用OrekitFixedStepHandler，让Java引擎负责循环，
        避免Python for循环中86,400次JNI调用（24小时/1秒步长）。
        """
        bridge = OrekitJavaBridge()

        # 1. 创建AbsoluteDate
        # 2. 创建Orbit对象（Keplerian/Cartesian）
        # 3. 创建NumericalPropagator（使用桥接层缓存的静态对象）
        # 4. 配置摄动力模型

        # 5. 高性能批量传播（关键优化）
        # 方案A（推荐）：使用StepHandler
        #   - Python实现Java的OrekitFixedStepHandler接口（@JImplements）
        #   - Java引擎循环，每一步回调Python收集PV坐标
        #   - 仅需1次JNI调用发起传播，86,400步在Java内部完成

        # 方案B（备选）：生成星历Ephemeris
        #   - propagator.setEphemerisMode()
        #   - 一次性传播生成完整星历
        #   - 后续从星历对象插值获取点位

        # 6. 返回PV坐标列表

    @OrekitJavaBridge.ensure_jvm_attached  # 线程安全装饰器
    def _propagate_single(self, satellite, dt):
        """单点传播（内部使用，已被批量传播替代）"""
```

### 3. OrekitConfig (配置管理)

**文件**: `core/orbit/visibility/orekit_config.py`

**配置项**:
```python
DEFAULT_OREKIT_CONFIG = {
    'jvm': {
        'java_home': '/usr/lib/jvm/java-17',  # 自动检测
        'classpath': ['/usr/local/share/orekit/orekit-12.0.jar', ...],
        'max_memory': '2g',
    },
    'data': {
        'root_dir': '/usr/local/share/orekit',
        'iers_dir': 'IERS',
        'gravity_dir': 'EGM96',
        'ephemeris_dir': 'DE440',
    },
    'propagator': {
        'integrator': 'DormandPrince853',  # RK78
        'min_step': 0.001,  # s
        'max_step': 300.0,  # s
        'position_tolerance': 10.0,  # m
    },
    'perturbations': {
        'earth_gravity': {
            'enabled': True,
            'model': 'EGM96',
            'degree': 36,
            'order': 36,
        },
        'drag': {
            'enabled': True,
            'model': 'NRLMSISE00',
            'cd': 2.2,
            'area': 10.0,  # m²
        },
        'solar_radiation': {
            'enabled': True,
            'cr': 1.5,
            'area': 10.0,  # m²
        },
        'third_body': {
            'enabled': True,
            'bodies': ['SUN', 'MOON'],
        },
        'relativity': {
            'enabled': True,
        },
    },
}
```

## 数据文件目录结构

```
/usr/local/share/orekit/          # Linux/macOS系统目录
├── orekit-12.0.jar               # Orekit主库
├── orekit-helper.jar             # 自定义Java辅助类（性能优化）
├── hipparchus-core-3.0.jar       # 数学库依赖（Orekit 12.x需3.0+）
├── hipparchus-geometry-3.0.jar
├── hipparchus-ode-3.0.jar
├── hipparchus-filtering-3.0.jar
├── IERS/
│   └── finals.all                # 地球自转数据(IERS)
├── DE440/
│   └── de440.bsp                 # JPL星历
├── EGM96/
│   └── egm96.gfc                 # 地球引力场
└── config/
    └── orekit.properties         # Orekit配置文件
```

**安装脚本**: `scripts/install_orekit_data.sh`
- 自动下载Orekit jar包（**使用Hipparchus 3.0+**）
- 编译自定义Java辅助类（`orekit-helper.jar`）
- 下载数据文件
- 配置系统环境变量

**编译辅助类**: `scripts/compile_helper_jar.sh`
- 编译 `BatchStepHandler.java` 为 `orekit-helper.jar`
- 该辅助类实现零拷贝高性能批量传播

## 摄动力模型配置

### 默认配置（全部启用）

| 摄动模型 | Java类 | 参数 | 影响 |
|---------|--------|------|------|
| 地球引力场 | HolmesFeatherstoneAttractionModel | EGM96 36x36 | 轨道进动、共振 |
| 大气阻力 | DragForce + NRLMSISE00 | Cd=2.2, Area=10m² | 轨道衰减(低轨) |
| 太阳光压 | SolarRadiationPressure | Cr=1.5, Area=10m² | 轨道偏心率变化 |
| 第三体引力 | ThirdBodyAttraction | 太阳、月球 | 长周期摄动 |
| 相对论 | Relativity | - | GPS卫星必须 |

### 配置示例

```python
from core.orbit.visibility.orekit_visibility import OrekitVisibilityCalculator

calculator = OrekitVisibilityCalculator(config={
    'use_java_orekit': True,
    'min_elevation': 5.0,
    'time_step': 1,
    'orekit': {
        'perturbations': {
            'earth_gravity': {'enabled': True, 'degree': 36, 'order': 36},
            'drag': {'enabled': True, 'cd': 2.2, 'area': 10.0},
            'solar_radiation': {'enabled': True, 'cr': 1.5, 'area': 10.0},
            'third_body': {'enabled': True, 'bodies': ['SUN', 'MOON']},
            'relativity': {'enabled': True},
        }
    }
})
```

## 关键类映射

### Java → Python 类映射

| 功能 | Java类 | Python封装 |
|------|--------|-----------|
| 绝对时间 | `org.orekit.time.AbsoluteDate` | `create_absolute_date()` |
| 时间尺度 | `org.orekit.time.TimeScalesFactory` | `get_utc()`, `get_gps()` |
| 坐标系 | `org.orekit.frames.FramesFactory` | `get_frame(name)` |
| 开普勒轨道 | `org.orekit.orbits.KeplerianOrbit` | `create_orbit()` |
| 数值传播器 | `org.orekit.propagation.numerical.NumericalPropagator` | `create_propagator()` |
| 地球引力场 | `org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel` | `add_earth_gravity()` |
| 大气模型 | `org.orekit.models.earth.atmosphere.NRLMSISE00` | `create_atmosphere()` |
| 大气阻力 | `org.orekit.forces.drag.DragForce` | `add_drag_force()` |
| 太阳光压 | `org.orekit.forces.radiation.SolarRadiationPressure` | `add_srp_force()` |
| 第三体引力 | `org.orekit.forces.gravity.ThirdBodyAttraction` | `add_third_body()` |
| 相对论 | `org.orekit.forces.gravity.Relativity` | `add_relativity()` |
| 星历 | `org.orekit.bodies.JPLEphemeridesLoader` | `load_ephemeris()` |
| PV坐标 | `org.orekit.utils.PVCoordinates` | `to_ecef()` |

## 性能预期

| 指标 | 纯Python (简化) | Java Orekit (基础) | Java Orekit (零拷贝优化) | 说明 |
|------|----------------|-------------------|-------------------------|------|
| 单点传播 | ~0.1ms | ~0.05ms | ~0.05ms | JVM调用开销 |
| 24小时窗口计算(1s步长) | ~10s | ~3-5s | **~0.5-1s** | Python循环→StepHandler→零拷贝 |
| JNI调用次数 | N/A | 86,400 | **1** | 零拷贝大幅降低跨语言开销 |
| 内存占用 | ~50MB | ~500MB+ | ~500MB+ | JVM堆内存 |
| 启动时间 | 0s | 1-2s | 1-2s | JVM启动 |
| 精度 | 米级(简化模型) | 厘米级 | 厘米级 | 完整摄动模型 |

**性能优化要点**：
1. **零拷贝是关键**：使用 `BatchStepHandler` 将86,400次JNI调用降为1次
2. **预计算缓存**：调度运行时不调用Orekit，只读预计算缓存
3. **JVM生命周期**：进程级单例，启动开销只发生在初始化阶段

**优化策略**:
- 预计算阶段一次性计算所有窗口，缓存结果
- 调度器运行时不调用Orekit，只读缓存
- JVM启动开销只发生在实验初始化阶段

## 测试计划

### 1. 单元测试
- `test_orekit_java_bridge.py`: JVM启动/关闭，Java对象创建
- `test_orekit_propagator.py`: 轨道传播精度
- `test_orekit_perturbations.py`: 摄动力模型效果

### 2. 精度验证
- 与STK HPOP对比（相同轨道参数）
- 与已知星历数据对比（ISS TLE传播）
- 与SGP4对比（短期/长期误差分析）

### 3. 性能测试
- JVM启动时间
- 单点传播时间
- 大批量窗口计算时间

## 实施步骤

### Phase 1: 环境准备
1. 升级Java 9 → 17
2. 安装JPype: `pip install JPype1>=1.5.0`
3. 下载Orekit 12.x jar包和数据文件（**Hipparchus必须使用3.0+**）
4. 编译自定义Java辅助类: `javac -cp orekit-12.0.jar BatchStepHandler.java && jar cf orekit-helper.jar orekit/helper/`
5. 运行安装脚本: `scripts/install_orekit_data.sh`

### Phase 2: 桥接层开发
1. 创建 `orekit_java_bridge.py`
2. 实现JVM生命周期管理
3. 实现Java对象封装
4. 实现摄动力模型配置

### Phase 3: 集成开发
1. 修改 `orekit_visibility.py`
2. 实现 `_propagate_with_java_orekit()`
3. 保持 `_propagate_simplified()` 作为fallback

### Phase 4: 测试验证
1. 编写单元测试
2. 精度对比测试
3. 性能基准测试

### Phase 5: 文档更新
1. 更新API文档
2. 编写配置指南
3. 编写故障排查手册

## 依赖清单

### Python依赖
```
JPype1>=1.5.0
```

### Java依赖

> **⚠️ 重要版本兼容性提示**: Orekit 12.x 强制依赖 Hipparchus 3.0+，请勿使用 2.x 版本，否则初始化数值积分器时会抛出 `ClassNotFoundException` 或 `NoSuchMethodError`。

```
orekit-12.0.jar
hipparchus-core-3.0.jar
hipparchus-geometry-3.0.jar
hipparchus-ode-3.0.jar
hipparchus-filtering-3.0.jar
```

### 数据文件
```
finals.all (IERS地球自转数据)
egm96.gfc (地球引力场模型)
de440.bsp (JPL星历)
sun-moon.epm (太阳月球星历，可选)
```

## 风险与应对

| 风险 | 可能性 | 影响 | 应对措施 |
|------|--------|------|----------|
| **Hipparchus版本不匹配** | **高** | **高** | **使用Hipparchus 3.0+，与Orekit 12.x兼容** |
| IERS数据过期 | 中 | 高 | 配置定时同步Cron任务；暴露reload接口 |
| JVM启动失败 | 中 | 高 | 自动fallback到纯Python实现 |
| 内存不足 | 低 | 高 | 配置JVM堆内存上限；分批计算 |
| 数据文件缺失 | 中 | 高 | 启动时检查；提供自动下载脚本 |
| Java版本不兼容 | 低 | 高 | 文档明确Java 17+要求；版本检查 |
| 线程安全问题 | 中 | 中 | `ensure_jvm_attached`装饰器自动处理 |
| 线程池内存泄漏 | 低 | 中 | 使用initializer统一attach；避免频繁创建销毁线程 |

## 附录

### 参考文档
- [Orekit官方文档](https://www.orekit.org/)
- [JPype文档](https://jpype.readthedocs.io/)
- [Orekit Python Wrapper (废弃)](https://gitlab.com/orekit/orekit-python-wrapper) (参考用)

### 相关文件
- `core/orbit/visibility/orekit_java_bridge.py` (新增)
- `core/orbit/visibility/orekit_config.py` (新增)
- `core/orbit/visibility/orekit_visibility.py` (修改)
- `scripts/install_orekit_data.sh` (新增)
- `scripts/compile_helper_jar.sh` (新增)
- `java/src/orekit/helper/BatchStepHandler.java` (新增)
- `tests/unit/test_orekit_java_bridge.py` (新增)

### 相关目录
- `/usr/local/share/orekit/` - Orekit数据和jar文件目录
