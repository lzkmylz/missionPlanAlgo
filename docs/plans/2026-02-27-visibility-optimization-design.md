# 可见性计算性能优化设计文档

**日期:** 2026-02-27
**目标:** 将可见性计算时间从400秒降至5秒（80倍提升）
**优先级:** Phase 1 > Phase 2 > Phase 3

---

## 1. 概述

### 1.1 当前性能瓶颈

当前可见性计算流程：
- 时间步长: 60秒
- 15颗卫星 × 6个目标 = 90对
- 1440点/天 × 90对 = 129,600次计算
- 每次计算: Python调用Java获取位置 → Python计算仰角
- **总耗时: 400+秒**

### 1.2 优化策略总览

| 优化方向 | 核心思想 | 预期提升 | 复杂度 |
|----------|----------|----------|--------|
| 自适应时间步长 | 先粗扫描定位窗口，再精化边界 | 4-5倍 | 低 |
| Java端批量计算 | Java完成全部计算，Python仅调用一次 | 10-20倍 | 中 |
| 多线程并行 | 90个卫星-目标对并行计算 | 4-8倍 | 低 |

**综合预期: 400秒 → 5秒（80倍提升）**

---

## 2. 自适应时间步长优化（Phase 1）

### 2.1 算法设计

**两阶段搜索策略：**

```
阶段1 - 粗扫描 (Coarse Scan)
├── 步长: 300秒 (5分钟)
├── 计算所有卫星-目标对的仰角
├── 检测"潜在窗口": 连续可见点序列
└── 输出: 粗略窗口边界 [(start1, end1), (start2, end2), ...]

阶段2 - 窗口精化 (Window Refinement)
├── 在每个潜在窗口边界附近
├── 步长: 60秒 (1分钟)
├── 向前/向后扩展搜索直到不可见
└── 输出: 精确窗口边界
```

### 2.2 精化算法伪代码

```python
def refine_window(
    self,
    satellite,
    target,
    coarse_start: datetime,
    coarse_end: datetime,
    time_step: timedelta
) -> Tuple[datetime, datetime]:
    """精化窗口边界"""

    # 向前扩展找精确开始时间
    start = coarse_start
    while start > mission_start_time:
        prev_time = start - time_step
        if not is_visible(satellite, target, prev_time):
            break
        start = prev_time

    # 向后扩展找精确结束时间
    end = coarse_end
    while end < mission_end_time:
        next_time = end + time_step
        if not is_visible(satellite, target, next_time):
            break
        end = next_time

    return (start, end)
```

### 2.3 计算量对比

| 方案 | 计算点数/对 | 总计算点数 | 相比当前 |
|------|-------------|------------|----------|
| 当前(60秒步长) | 1,440 | 129,600 | 100% |
| 粗扫描(300秒) | 288 | 25,920 | 20% |
| 精化(平均) | ~100 | ~9,000 | 7% |
| **总计** | **~388** | **~35,000** | **27%** |

### 2.4 实现文件

- `core/orbit/visibility/adaptive_step_calculator.py` - 新增
- 修改 `orekit_visibility.py` 中的 `_propagate_range` 方法

### 2.5 验证标准

- 窗口数量误差: < 1个
- 窗口时间误差: < 60秒
- 性能提升: > 4倍

---

## 3. Java端批量计算优化（Phase 2）

### 3.1 核心思想

**当前问题:** Python调用Java 129,600次获取位置，每次数据传输6个float

**优化方案:** Python调用Java 1次，Java完成全部计算，返回窗口列表

### 3.2 Python接口设计

```python
class OrekitJavaBridge:
    def compute_visibility_batch(
        self,
        satellite_configs: List[SatelliteConfig],
        target_configs: List[TargetConfig],
        time_config: TimeConfig,
        calculation_config: CalculationConfig,
        quality_config: QualityConfig
    ) -> BatchVisibilityResult:
        """
        批量计算所有卫星-目标对的可见窗口

        单次JNI调用完成全部计算，Java端实现自适应步长算法
        """
```

**配置类：**

```python
@dataclass
class SatelliteConfig:
    id: str
    tle_line1: str
    tle_line2: str
    min_elevation: float = 5.0
    sensor_fov: float = 0.0

@dataclass
class TargetConfig:
    id: str
    longitude: float
    latitude: float
    altitude: float = 0.0
    min_observation_duration: int = 60
    priority: int = 5

@dataclass
class TimeConfig:
    start_time: datetime
    end_time: datetime
    coarse_step: int = 300
    fine_step: int = 60
    max_windows_per_pair: int = 10
```

### 3.3 Java端实现

```java
public class VisibilityBatchCalculator {
    public BatchResult computeBatch(
        List<SatelliteConfig> satellites,
        List<TargetConfig> targets,
        TimeConfig timeConfig,
        CalculationConfig calcConfig,
        QualityConfig qualityConfig
    ) {
        BatchResult result = new BatchResult();

        for (SatelliteConfig sat : satellites) {
            Propagator propagator = propagatorFactory.create(sat);

            for (TargetConfig target : targets) {
                try {
                    List<VisibilityWindow> windows = computeWindows(
                        propagator, sat, target, timeConfig, calcConfig, qualityConfig
                    );
                    result.addWindows(sat.id, target.id, windows);
                } catch (Exception e) {
                    result.addError(sat.id, target.id, e);
                }
            }
        }

        return result;
    }
}
```

### 3.4 JNI调用对比

| 指标 | 当前方案 | 优化后 |
|------|----------|--------|
| JNI调用次数 | 129,600 | 1 |
| 单次传输数据量 | 6 floats (24 bytes) | 配置参数 (~1KB) |
| 总数据传输 | ~3.1 MB | ~1 KB |

---

## 4. 多线程并行优化（Phase 3）

### 4.1 并行策略

**粒度选择: 卫星-目标对级别**

- **优点:** 完全独立，无共享状态，无锁竞争
- **任务数:** 90个 (15卫星 × 6目标)
- **理想并行度:** min(90, CPU核心数 × 2)

### 4.2 线程池设计

```python
class ParallelVisibilityCalculator:
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or (os.cpu_count() * 2)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def compute_all_windows(self, satellites, targets, time_range, java_bridge):
        # 生成所有任务
        tasks = [(sat, target, time_range, java_bridge)
                 for sat in satellites for target in targets]

        # 提交到线程池
        futures = {
            self._executor.submit(self._compute_single_pair, task): 
            (sat.id, target.id) for task in tasks
        }

        # 收集结果
        results = {}
        for future in as_completed(futures):
            sat_id, target_id = futures[future]
            results[(sat_id, target_id)] = future.result()

        return results
```

### 4.3 JVM线程安全

```python
def ensure_jvm_attached(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if jpype.isJVMStarted() and not jpype.isThreadAttachedToJVM():
            jpype.attachThreadToJVM()
        return func(*args, **kwargs)
    return wrapper
```

---

## 5. 三阶段集成方案

### 5.1 Phase 1: 自适应时间步长（Week 1）

**目标:** 独立实现，验证算法正确性

**任务清单:**
- [ ] 实现 `AdaptiveStepCalculator` 类
- [ ] 实现粗扫描算法
- [ ] 实现窗口精化算法
- [ ] 与原始实现对比验证
- [ ] 性能测试

**验证标准:**
- 窗口数量误差 < 1个
- 窗口时间误差 < 60秒
- 性能提升 > 4倍

### 5.2 Phase 2: Java端批量计算（Week 2-3）

**目标:** 减少JNI调用开销

**任务清单:**
- [ ] 设计Java端数据模型
- [ ] 实现 `VisibilityBatchCalculator` Java类
- [ ] 实现自适应步长算法（Java版）
- [ ] 修改Python端接口
- [ ] JNI接口测试

**验证标准:**
- 结果与Phase 1一致
- JNI调用次数 = 1
- 性能提升 > 10倍

### 5.3 Phase 3: 多线程并行（Week 4）

**目标:** 充分利用多核CPU

**任务清单:**
- [ ] 实现 `ParallelVisibilityCalculator`
- [ ] 实现JVM attach装饰器
- [ ] 线程池配置优化
- [ ] 8线程 vs 单线程对比测试

**验证标准:**
- 8线程加速比 > 4倍
- 内存使用 < 2GB
- 无线程泄漏

### 5.4 总体时间线

```
Week 1: Phase 1 - 自适应步长
├── Day 1-2: 实现粗扫描
├── Day 3-4: 实现精化算法
└── Day 5: 验证与测试

Week 2-3: Phase 2 - Java批量计算
├── Day 1-3: Java数据模型 + 粗扫描
├── Day 4-6: Java精化算法
├── Day 7-8: Python接口 + JNI
└── Day 9-10: 集成测试

Week 4: Phase 3 - 多线程
├── Day 1-2: 线程池实现
├── Day 3-4: JVM attach处理
└── Day 5: 性能优化

Week 5: 集成与优化
├── 三阶段集成
├── 端到端测试
└── 文档更新
```

---

## 6. 风险与回退方案

### 6.1 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| Java JNI异常 | 中 | 高 | 完善异常转换，提供Python回退 |
| 多线程内存泄漏 | 低 | 中 | 使用上下文管理器，严格测试 |
| 精度损失 | 低 | 高 | 与原始实现对比验证 |

### 6.2 回退策略

```python
# Level 1 - 配置回退
visibility_calculator = VisibilityCalculator(
    use_adaptive_step=False,  # 回退到固定步长
    use_java_batch=False,     # 回退到逐点调用
    use_parallel=False,       # 回退到单线程
)
```

---

## 7. 预期收益总结

| 指标 | 当前 | 优化后 | 提升倍数 |
|------|------|--------|----------|
| 计算时间 | 400秒 | 5秒 | **80倍** |
| JNI调用 | 129,600 | 1 | **129,600倍** |
| 计算点数 | 129,600 | 35,000 | **3.7倍** |
| 并行度 | 1 | 8-16 | **8-16倍** |

---

**文档版本:** 1.0
**下一步:** 评审后开始Phase 1实现
