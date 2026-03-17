# missionPlanAlgo - Claude Code 记忆文件

## 项目概述

卫星任务规划算法库，包含可见性计算、任务调度、轨道传播等功能。

---

## 可见性计算系统架构

### 系统组件流程

```
场景配置文件 (JSON)
    │
    ├─ 卫星配置 (60颗: 30光学 + 30SAR)
    ├─ 目标配置 (1000个地面目标)
    ├─ 地面站配置 (12个)
    └─ 观测需求 (频次约束)
    │
    ▼
JsonScenarioLoader
    ├─ 解析轨道参数 (六根数/TLE)
    ├─ 读取物理参数 (mass, dragArea, reflectivity, Cd)
    └─ 加载目标地理位置
    │
    ▼
SmartOrbitInitializer ← 核心组件
    │
    ├─ [TLE数据] → SGP4外推到场景开始
    ├─ [六根数+历元<3天] → 直接使用历元轨道
    └─ [六根数+历元>3天] → J4解析外推
    │
    ▼
场景开始时刻的初始状态 (SpacecraftState)
    │
    ▼
HPOP数值传播器 (仅传播场景持续时间24h)
    │
    ▼
OrbitStateCache (并行计算60颗卫星)
    │
    ▼
OptimizedVisibilityCalculator
    ├─ 粗扫描 (5秒步长)
    ├─ 精化扫描 (1秒步长)
    └─ 卫星-目标窗口 + 卫星-地面站窗口 (默认一起计算)
    │
    ▼
输出文件
    ├─ visibility_windows.json (279,386窗口: 277,946目标 + 1,440地面站)
    ├─ satellites.json (轨道数据)
    └─ ground_stations.json

### 地面站可见窗口计算

**重要变更**: Java后端现在**默认同时计算**卫星-地面站可见性窗口，不再需要在Python端用简化模型生成。

**计算方法** (`OptimizedVisibilityCalculator.java`):
```java
// 新的重载方法，接受地面站列表
public BatchResult computeAllVisibilityWindows(
    List<SatelliteConfig> satellites,
    List<TargetConfig> targets,
    List<GroundStationConfig> groundStations,  // 新增
    AbsoluteDate startTime,
    AbsoluteDate endTime,
    double coarseStep,
    double fineStep
)
```

**计算流程**:
1. 预计算所有卫星轨道（HPOP高精度模型）
2. 并行计算卫星-目标窗口（60卫星 × 1000目标）
3. **并行计算卫星-地面站窗口（60卫星 × 12地面站）**
4. 统一输出到 `visibility_windows.json`

**地面站窗口标识**: 目标ID使用 `GS:` 前缀，如 `GS:GS-BEIJING`

**性能影响**:
- 额外计算720对卫星-地面站组合
- 地面站计算更快（固定位置 vs 移动目标）
- 总体增加约1-2秒计算时间
```

---

## 关键配置参数

### 时间阈值
| 参数 | 值 | 说明 |
|------|-----|------|
| DIRECT_HPOP_THRESHOLD_DAYS | 3.0 | 历元距场景<3天直接HPOP，>3天用J4外推 |
| 粗扫描步长 | 5.0s | 快速定位窗口 |
| 精化扫描步长 | 1.0s | 精确窗口边界 |
| **轨道数据导出步长** | **1.0s** | **与精扫描步长一致，确保与姿态计算配置匹配** |
| 场景持续时间 | 24h | HPOP实际传播时长 |

### 卫星物理参数默认值
| 参数 | 光学卫星 | SAR卫星 | 单位 |
|------|----------|---------|------|
| mass | 100.0 | 150.0 | kg |
| dragArea | 5.0 | 8.0 | m² |
| reflectivity | 1.5 | 1.3 | - |
| dragCoefficient | 2.2 | 2.2 | - |

### HPOP积分器配置
```java
minStep = 0.001s          // 1毫秒
maxStep = 300.0s          // 5分钟
positionTolerance = 10.0m // 位置容差
```

### 引力场配置
- HPOP传播: EGM2008 90x90 (最高精度)
- J4解析外推: 6x6 (Eckstein-Hechler需要)

---

## 关键代码文件

| 文件 | 路径 | 职责 |
|------|------|------|
| SmartOrbitInitializer | java/src/orekit/visibility/ | 智能选择SGP4/J4/Direct HPOP策略 |
| OrbitStateCache | java/src/orekit/visibility/ | 轨道预计算、缓存、插值查询 |
| JsonScenarioLoader | java/src/orekit/visibility/ | 加载场景配置、解析物理参数 |
| OptimizedVisibilityCalculator | java/src/orekit/visibility/ | 可见性窗口计算 |
| LargeScaleFrequencyTest | java/src/orekit/visibility/ | 大规模场景测试入口 |

---

## 性能基准

### 可见性计算
| 场景 | 耗时 | 窗口数 |
|------|------|--------|
| 60卫星×1000目标×24h | 80秒 | 318,312 |
| 每对计算耗时 | 1.34 ms | - |

### 批量约束检查（大规模场景实测）
| 检查环节 | 调用次数 | 总耗时 | 平均耗时 | 性能评估 |
|---------|---------|--------|---------|---------|
| Batch slew check | 2,638 | 35.75s | **13.55ms** | ✅ 优秀 |
| Unified batch constraint check | 2,638 | 15.69s | **5.95ms** | ✅ 优秀 |
| 姿态约束检查（含Numba加速） | 2,638 | ~30ms | ~11ms | ✅ 优秀 |

### 完整调度结果（2026-03-11测试）
**场景**: 60卫星×1000目标×24h，频次需求2-5次/目标
| 指标 | 结果 |
|------|------|
| 调度任务数 | **2,638个** |
| 频次满足率 | **100%** (1000/1000目标) |
| 卫星利用率 | 13.20% |
| 完成时间跨度 | 8.81小时 |
| 总计算时间 | 69分钟 |
| 姿态机动计算 | 精确模型+批量优化 |

---

## 轨道数据持久化与姿态计算

### Java端HPOP轨道数据持久化

可见性计算完成后，Java端**默认自动导出**预计算的HPOP轨道数据：

**默认保存路径**: `java/output/frequency_scenario/orbits.json.gz`

**导出步长**: **1.0秒**（与精扫描步长一致，确保与姿态计算配置匹配）

**关键代码** (`OptimizedVisibilityCalculator.java:133`):
```java
// 使用精扫描步长（fineStep）而非粗扫描步长，确保轨道数据精度与可见性计算一致
orbitCache.precomputeAllOrbits(satellites, startTime, endTime, fineStep);  // fineStep = 1.0s
```

**文件格式**: JSON + GZIP压缩，包含：
- 卫星ID、时间戳（秒）
- 位置向量（ECEF，米）
- 速度向量（米/秒）
- 地理坐标（纬度/经度/高度）

**导出代码** (`LargeScaleFrequencyTest.java:92-103`):
```java
OrbitStateCache orbitCache = calculator.getOrbitCache();
OrbitDataExporter exporter = new OrbitDataExporter();
exporter.exportToJson(orbitCache.getCache(), "output/frequency_scenario/orbits.json.gz");
```

### Python端轨道数据加载

**所有调度器默认自动加载**Java预计算的轨道数据：

**加载优先级**:
1. 检查 `use_precomputed_orbits`（默认True）
2. 尝试加载 `java/output/frequency_scenario/orbits.json.gz`
3. 成功 → 使用O(1)缓存查询，**跳过Python端预计算**
4. 失败 → Python端自己预计算（1秒步长）

**关键代码** (`base_scheduler.py:360-395`):
```python
def _load_precomputed_orbits_from_java(self) -> bool:
    json_path = self.config.get('orbit_json_path',
                                'java/output/frequency_scenario/orbits.json.gz')
    propagator.load_precomputed_orbits(json_path, start_time)
```

### 默认姿态计算配置

**所有规划算法默认精确计算姿态角** (`scripts/config.py`):

```python
DEFAULT_IMAGING_CONFIG = {
    # 高精度要求：始终使用精确模式，简化模式已移除
    'enable_attitude_calculation': True,    # 启用姿态角计算
    'precompute_positions': True,           # 预计算位置（如未加载Java数据）
    'precompute_step_seconds': 1.0,         # 1秒步长（与HPOP精扫描匹配）
    'use_precomputed_orbits': True,         # 默认加载Java预计算轨道
}
```

**姿态计算**: 只要 `enable_attitude_calculation=True` 就始终计算姿态角。

### 默认频次需求配置

**所有规划算法默认启用目标观测频次需求处理** (`scripts/config.py`):

```python
default_algorithm_config = {
    'enable_frequency': True,   # 默认启用频次需求
    'consider_frequency': True,  # 默认考虑频次约束
}
```

**调度器默认行为**:
- 场景中的 `observation_requirements` 自动解析为目标频次需求
- 每个目标的 `required_observations` 决定最少观测次数
- 调度结果包含 `frequency_satisfaction` 字段显示满足情况

**关键代码** (`scripts/run_scheduler.py:132-148`):
```python
parser.add_argument(
    '--frequency',
    action='store_true',
    default=True,  # 默认启用
    help='启用观测频次需求处理 (默认: 启用)'
)
parser.add_argument(
    '--no-frequency',
    dest='frequency',
    action='store_false',
    help='禁用观测频次需求处理'
)
```

### 默认数传规划配置

**所有规划算法默认启用地⾯站数传规划** (`scripts/config.py`):

```python
default_algorithm_config = {
    'enable_downlink': True,    # 默认启用数传规划
}
```

**调度器默认行为**:
- 同时规划成像任务和数传任务
- 数传任务安排在成像任务之后，考虑卫星存储和地面站可见窗口
- 调度结果包含 `downlink_result` 字段显示数传任务

**关键代码** (`scripts/run_scheduler.py:138-154`):
```python
parser.add_argument(
    '--downlink',
    action='store_true',
    default=True,  # 默认启用
    help='启用地面站数传规划 (默认: 启用)'
)
parser.add_argument(
    '--no-downlink',
    dest='downlink',
    action='store_false',
    help='禁用地面站数传规划'
)
```

### 默认姿态机动计算模型

**所有姿态机动计算统一使用精确模型** (`scheduler/unified_scheduler.py`):

```python
# 统一使用精确姿态机动模型
self.slew_checker = PreciseSlewConstraintChecker(
    mission=self.mission,
    use_precise_model=True
)
```

**模型特点**:
- 基于刚体动力学的精确计算 (`core/dynamics/precise/rigid_body_dynamics.py`)
- 时间最优轨迹规划 (`core/dynamics/precise/trajectory_planner.py`)
- 能量消耗精确建模 (`core/dynamics/precise/energy_model.py`)
- 飞轮动量管理 (`core/dynamics/precise/momentum_manager.py`)

**已移除的配置** (不再支持):
- 简化估算模型 (SlewCalculator)
- 混合模式 (hybrid)
- `--slew-model` 命令行参数

**关键代码** (`scheduler/unified_scheduler.py:152-161`):
```python
def _initialize_constraint_checkers(self) -> None:
    """初始化精确姿态机动约束检查器"""
    from .constraints import PreciseSlewConstraintChecker

    self.slew_checker = PreciseSlewConstraintChecker(
        mission=self.mission,
        use_precise_model=True
    )
    logger.info("使用精确姿态机动模型")
```

---

### 分轴姿态机动限制（滚转/俯仰区分）

**姿态机动约束现在支持分轴限制**，区分滚转（roll/X轴）和俯仰（pitch/Y轴）的角速度和角加速度：

| 参数 | 滚转轴 (X) | 俯仰轴 (Y) | 说明 |
|------|-----------|-----------|------|
| 最大角速度 | `max_roll_rate` | `max_pitch_rate` | 度/秒 |
| 最大角加速度 | `max_roll_acceleration` | `max_pitch_acceleration` | 度/秒² |

**卫星配置示例** (`data/entity_lib/satellites/optical_1.json`):
```json
"agility": {
    "max_slew_rate": 3.0,
    "max_pitch_rate": 2.0,
    "max_roll_acceleration": 1.5,
    "max_pitch_acceleration": 1.0,
    "settling_time": 5.0
}
```

**实现文件**:
- `core/models/satellite.py` - `SatelliteCapabilities` 支持分轴限制
- `core/dynamics/precise/trajectory_planner.py` - 轨迹规划器根据旋转轴选择有效限制
- `scheduler/constraints/batch_slew_calculator.py` - Numba批量计算支持分轴限制

**限制选择策略**（保守估计法）:
```python
# 根据旋转轴主要方向选择对应轴的限制
if abs(rotation_axis[0]) > abs(rotation_axis[1]):
    # 主要滚转运动，使用滚转限制
    effective_rate = max_roll_rate
else:
    # 主要俯仰运动，使用俯仰限制
    effective_rate = max_pitch_rate
```

**向后兼容**:
- 未配置分轴限制时，自动从标量限制派生
- 俯仰默认使用滚转限制的 2/3

---

### 默认批量姿态约束计算（向量化优化）

**所有调度器默认使用批量姿态约束检查器** (`scheduler/base_scheduler.py`):

```python
def _initialize_slew_checker(self) -> None:
    """初始化机动约束检查器 - 默认使用向量化批量优化"""
    # ...
    # 默认使用批量姿态约束检查器（向量化优化）
    self._slew_checker = BatchSlewConstraintChecker(
        self.mission,
        use_precise_model=True
    )
```

**实现文件**:
- `scheduler/constraints/batch_slew_calculator.py` - Numba批量计算核心
- `scheduler/constraints/batch_slew_constraint_checker.py` - 批量检查器类

**优化特点**:
- **Numba JIT并行计算**: 使用 `@njit(parallel=True)` 和 `prange` 实现C级并行
- **向量化数据布局**: Python对象 → NumPy数组 → Numba加速计算
- **批量处理**: 一次性计算多个候选的姿态约束，减少Python函数调用开销
- **自动回退**: Numba不可用时自动使用Python实现

**性能提升**:
| 批次大小 | 逐个检查 | 批量检查 | 加速比 |
|----------|----------|----------|--------|
| 10 | 16.8 ms | 9.0 ms | **1.9x** |
| 50 | 1.0 ms | 0.1 ms | **9.4x** |
| 100 | 1.9 ms | 0.2 ms | **8.2x** |
| 200 | 3.6 ms | 0.3 ms | **10.7x** |

**使用方式**:
```python
# 调度器自动使用（默认）
from scheduler.greedy.greedy_scheduler import GreedyScheduler
scheduler = GreedyScheduler()  # 自动使用BatchSlewConstraintChecker

# 手动批量检查
from scheduler.constraints import BatchSlewConstraintChecker, BatchSlewCandidate
checker = BatchSlewConstraintChecker(mission)
results = checker.check_slew_feasibility_batch(candidates)
```

**启用条件**:
- 高精度要求：始终使用批量约束检查器
- 多个候选（大于1个）时自动启用批量优化
- 单候选时回退到单个检查（保持接口兼容）

---

### 默认批量SAA约束计算（向量化优化）

**所有调度器默认使用批量SAA约束检查器** (`scheduler/base_scheduler.py`):

```python
def _initialize_saa_checker(self) -> None:
    """初始化SAA约束检查器 - 默认使用向量化批量优化"""
    # 默认使用批量SAA约束检查器（向量化优化）
    self._saa_checker = BatchSAAConstraintChecker(self.mission)
```

**实现文件**:
- `scheduler/constraints/batch_saa_calculator.py` - Numba批量计算核心
- `scheduler/constraints/batch_saa_constraint_checker.py` - 批量检查器类

**优化特点**:
- **Numba JIT并行计算**: 使用 `@njit(parallel=True)` 和 `prange` 实现C级并行
- **向量化数据布局**: Python对象 → NumPy数组 → Numba加速计算
- **批量处理**: 一次性检查多个候选的SAA约束，减少Python函数调用开销
- **椭圆SAA模型**: 使用标准椭圆模型（中心-45°,-25°，半长轴40°，半短轴30°）
- **自动回退**: Numba不可用时自动使用Python实现

**性能提升**:
| 批次大小 | 逐个检查 | 批量检查 | 加速比 |
|----------|----------|----------|--------|
| 10 | ~5 ms | ~2 ms | **2.5x** |
| 50 | ~25 ms | ~5 ms | **5x** |
| 100 | ~50 ms | ~8 ms | **6x** |
| 200 | ~100 ms | ~15 ms | **6.7x** |

**使用方式**:
```python
# 调度器自动使用（默认）
from scheduler.greedy.greedy_scheduler import GreedyScheduler
scheduler = GreedyScheduler()  # 自动使用BatchSAAConstraintChecker

# 手动批量检查
from scheduler.constraints import BatchSAAConstraintChecker, BatchSAACandidate
checker = BatchSAAConstraintChecker(mission)
results = checker.check_window_feasibility_batch(candidates)
```

---

### 默认批量时间冲突检查（向量化优化）

**实现文件**:
- `scheduler/constraints/batch_time_conflict_calculator.py` - Numba批量计算核心
- `scheduler/constraints/batch_time_conflict_checker.py` - 批量检查器类

**优化特点**:
- **Numba JIT并行计算**: 使用 `@njit(parallel=True)` 和 `prange` 实现C级并行
- **向量化数据布局**: Python对象 → NumPy数组 → Numba加速计算
- **批量处理**: 一次性检查多个候选的时间冲突，减少Python函数调用开销
- **卫星分离**: 只检查同一卫星的任务冲突
- **自动回退**: Numba不可用时自动使用Python实现

**性能提升**:
| 批次大小 | 逐个检查 | 批量检查 | 加速比 |
|----------|----------|----------|--------|
| 10 | ~2 ms | ~0.5 ms | **4x** |
| 50 | ~10 ms | ~1 ms | **10x** |
| 100 | ~20 ms | ~2 ms | **10x** |
| 200 | ~40 ms | ~3 ms | **13x** |

**使用方式**:
```python
from scheduler.constraints import BatchTimeConflictChecker, BatchTimeConflictCandidate

checker = BatchTimeConflictChecker()
candidates = [
    BatchTimeConflictCandidate(sat_id='SAT-001', window_start=t1, window_end=t2)
    for t1, t2 in time_windows
]
results = checker.check_time_conflict_batch(candidates, existing_tasks)
```

---

### 默认批量资源约束检查（向量化优化）

**实现文件**:
- `scheduler/constraints/batch_resource_calculator.py` - Numba批量计算核心
- `scheduler/constraints/batch_resource_checker.py` - 批量检查器类

**优化特点**:
- **Numba JIT并行计算**: 使用 `@njit(parallel=True)` 和 `prange` 实现C级并行
- **向量化数据布局**: Python对象 → NumPy数组 → Numba加速计算
- **批量处理**: 一次性检查多个候选的资源约束
- **支持电量和存储**: 同时检查电量充足性和存储容量
- **自动回退**: Numba不可用时自动使用Python实现

**性能提升**:
| 批次大小 | 逐个检查 | 批量检查 | 加速比 |
|----------|----------|----------|--------|
| 10 | ~1 ms | ~0.3 ms | **3x** |
| 50 | ~5 ms | ~0.6 ms | **8x** |
| 100 | ~10 ms | ~1 ms | **10x** |
| 200 | ~20 ms | ~1.5 ms | **13x** |

**使用方式**:
```python
from scheduler.constraints import BatchResourceChecker, BatchResourceCandidate

checker = BatchResourceChecker()
candidates = [
    BatchResourceCandidate(
        sat_id='SAT-001',
        power_needed=100.0,
        storage_produced=10.0
    )
    for _ in tasks
]
results = checker.check_resources_batch(candidates, satellite_states)
```

---

### 统一批量约束检查器

**实现文件**:
- `scheduler/constraints/unified_batch_constraint_checker.py` - 统一批量检查入口

**功能**:
- **阶段式检查**: 姿态 → SAA → 时间 → 资源
- **早期终止**: 某阶段失败则跳过后续检查
- **快速筛选**: `check_fast_phase_batch()` 只检查姿态、SAA、时间
- **完整检查**: `check_all_constraints_batch()` 检查所有约束
- **统一接口**: 单一入口调用所有批量检查器

**性能提升**:
- 整合所有批量优化，整体调度性能提升 **5-10倍**
- 早期终止减少不必要的计算

**使用方式**:
```python
from scheduler.constraints import (
    UnifiedBatchConstraintChecker,
    UnifiedBatchCandidate
)

checker = UnifiedBatchConstraintChecker(mission)
candidates = [UnifiedBatchCandidate(
    sat_id='SAT-001',
    satellite=sat,
    target=target,
    window_start=start,
    window_end=end,
    prev_end_time=prev_end,
    power_needed=100.0,
    storage_produced=10.0
) for ...]

# 完整检查
results = checker.check_all_constraints_batch(
    candidates=candidates,
    existing_tasks=scheduled_tasks,
    satellite_states=satellite_states
)

# 快速筛选
results = checker.check_fast_phase_batch(
    candidates=candidates,
    existing_tasks=scheduled_tasks
)
```

---

### 批量约束检查开发规范

**所有调度器必须使用批量约束检查**:
- GreedyScheduler、GA、ACO、SA、PSO、Tabu等所有调度算法
- 自定义调度器开发时必须继承批量检查模式
- 禁止直接逐个调用约束检查（性能瓶颈）

**未来新增约束的实现要求**:
1. **必须提供批量检查接口**: `check_<constraint>_batch(candidates, ...)`
2. **必须使用Numba JIT**: `@njit(parallel=True, cache=True)`
3. **必须提供候选数据类**: `<Constraint>BatchCandidate` dataclass
4. **必须提供结果数据类**: `<Constraint>BatchResult` dataclass
5. **必须集成到UnifiedBatchConstraintChecker**: 在`check_fast_phase_batch()`或`check_all_constraints_batch()`中调用

**批量检查器标准结构**:
```python
# 1. 数据类
@dataclass
class BatchNewConstraintCandidate:
    sat_id: str
    window_start: datetime
    window_end: datetime
    # ... 其他参数

@dataclass
class BatchNewConstraintResult:
    feasible: bool
    reason: Optional[str] = None

# 2. Numba计算核心
@njit(parallel=True, cache=True)
def batch_check_new_constraint_numba(
    # 输入数组
    # ...
    # 输出数组
    out_feasible: np.ndarray,
):
    for i in prange(n):
        # 约束检查逻辑
        pass

# 3. 批量检查器类
class BatchNewConstraintChecker:
    def check_batch(self, candidates: List[BatchNewConstraintCandidate]) -> List[BatchNewConstraintResult]:
        # 准备数据
        # 调用Numba函数
        # 返回结果
        pass
```

---

### 默认结果保存配置

**所有规划算法默认自动保存详细任务列表** (`scripts/run_scheduler.py`):

```python
# 默认自动保存，除非指定 --no-save
parser.add_argument(
    '--no-save',
    action='store_true',
    help='禁用自动保存结果到文件'
)
```

**默认保存路径**:
- 单一算法: `results/{algorithm}_schedule_{timestamp}.json`
- 对比模式: `results/compare_{algorithms}_{timestamp}.json`

**输出文件内容**:
- `metadata`: 场景、缓存路径、时间戳等元信息
- `results`: 算法结果列表，包含详细的 `scheduled_tasks` 数组
- 每个任务包含: 任务ID、卫星ID、目标ID、时间窗口、姿态角、存储变化等

**关键代码** (`scripts/run_scheduler.py:482-498`):
```python
# 保存结果 (默认自动保存，除非指定 --no-save)
if not parsed_args.no_save:
    if parsed_args.output:
        output_path = parsed_args.output
    else:
        # 生成默认路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"results/{algorithm}_schedule_{timestamp}.json"

    save_results_to_file(
        results=[result],
        output_path=output_path,
        ...
    )
```

---

## 使用命令

```bash
# 编译
cd java && make build

# 运行大规模场景测试（使用默认配置）
cd java && java -cp "classes:lib/*" orekit.visibility.LargeScaleFrequencyTest

# 指定场景文件和输出目录
cd java && java -cp "classes:lib/*" orekit.visibility.LargeScaleFrequencyTest \
    --scenario ../scenarios/my_scenario.json \
    --output output/my_results \
    --orbit-output output/my_results/orbits.json.gz

# 自定义扫描步长
cd java && java -cp "classes:lib/*" orekit.visibility.LargeScaleFrequencyTest \
    --coarse-step 10.0 \
    --fine-step 2.0
```

**命令行参数:**
| 参数 | 短选项 | 说明 | 默认值 |
|------|--------|------|--------|
| `--scenario` | `-s` | 场景配置文件路径 | `../scenarios/large_scale_frequency.json` |
| `--output` | `-o` | 输出目录 | `output/frequency_scenario` |
| `--orbit-output` | | 轨道数据输出路径 | `<output>/orbits.json.gz` |
| `--coarse-step` | | 粗扫描步长(秒) | `5.0` |
| `--fine-step` | | 精化步长(秒) | `1.0` |
| `--help` | `-h` | 显示帮助 | |

---

## 环境配置

### Python 命令
| 命令 | 说明 |
|------|------|
| `python3` | 调用Python解释器 (注意: 不是 `python`) |
| `pip3` | 管理Python包 (注意: 不是 `pip`) |

示例:
```bash
# 运行脚本
python3 scripts/run_scheduler.py -c cache.json -s scenario.json

# 查看已安装库
pip3 list

# 安装依赖
pip3 install -r requirements.txt
```

---

## 重要约束

1. **HPOP强制使用**: 无论TLE还是六根数，最终都用HPOP计算（从场景开始时刻）
2. **时间阈值3天**: 历元距场景>3天必须用解析方法外推，禁止HPOP长期传播
3. **J4外推精度**: Eckstein-Hechler传播器，6x6引力场，考虑J2/J3/J4项
4. **并行计算**: 60颗卫星同时处理，使用parallelStream
5. **地面站可见窗口计算强制使用Java后端**: 卫星-地面站可见窗口计算**必须**使用Java Orekit后端（`OptimizedVisibilityCalculator.computeAllVisibilityWindows()`），严禁使用Python简化模型。统一通过 `BatchVisibilityCalculator` → `OrekitJavaBridge` → Java后端流程计算，确保地面站窗口使用与目标窗口相同的高精度HPOP轨道数据。

---

## Git LFS大文件推送

### 推送成功的关键 - SSH保活
当推送大文件到GitHub时，可能会遇到连接超时问题。解决方案是使用SSH保活参数：

```bash
GIT_SSH_COMMAND="ssh -o ServerAliveInterval=60 -o ServerAliveCountMax=30" git push origin main
```

**参数说明：**
- `ServerAliveInterval=60`: 每60秒发送一次保活包
- `ServerAliveCountMax=30`: 最多允许30次保活失败（约30分钟总超时）

**适用场景：**
- 推送Git LFS大文件（>50MB）
- 网络连接不稳定
- 远程主机关闭连接

### 2026-03-08: Python脚本支持Java轨道数据导出
**问题**: Python脚本 `compute_visibility.py` 只计算可见性窗口，不导出轨道数据，导致调度器需要单独预计算
**解决**: 修改 `PythonBridge.java`、`orekit_java_bridge.py` 和 `compute_visibility.py`
- Java端新增 `computeVisibilityBatchWithOrbitExport` 方法，使用 `OptimizedVisibilityCalculator` 计算并导出轨道数据
- Python端新增 `compute_visibility_batch_with_orbit_export` 方法调用Java接口
- 脚本默认启用轨道数据导出，支持 `--orbit-output` 和 `--no-orbit-export` 参数
**关键配置**: 轨道数据导出使用 **1.0秒步长**（与精扫描步长一致），与 `precompute_step_seconds: 1.0` 配置匹配

### 2026-03-08: Java后端默认计算卫星-地面站可见性
**问题**: 地面站可见窗口原本在Python端用简化模型生成（`compute_gs_visibility.py`），不是精确的Orekit计算
**解决**: 修改 `OptimizedVisibilityCalculator.java` 和 `LargeScaleFrequencyTest.java`
- 新增 `computeAllVisibilityWindows()` 重载方法，接受地面站列表
- 新增 `computeGroundStationWindows()` 方法，使用HPOP轨道缓存计算
- 新增 `computeGsWindowsUsingCache()` 方法，纯几何计算地面站仰角
- 输出文件统一包含目标窗口和地面站窗口
**结果**:
- 地面站窗口现在使用与目标窗口相同的高精度HPOP轨道数据
- 无需额外的Python后处理步骤
- 输出文件直接可用于数传规划

### 2026-03-15: 删除简化模型代码，统一使用Java Orekit计算
**问题**: `examples/compute_gs_visibility.py` 使用伪随机数据生成地面站窗口，与高精度要求不符
**解决**:
- 删除 `examples/compute_gs_visibility.py` 简化模型文件
- 统一使用 `BatchVisibilityCalculator` → `OrekitJavaBridge` → Java后端流程
- 所有地面站窗口现在通过HPOP高精度轨道传播计算
**验证**:
- 调度器通过 `load_window_cache_from_json()` 正确加载地面站窗口
- `unified_scheduler._get_ground_station_windows()` 正确使用 `GS:` 前缀键
- 数传规划使用高精度计算的地面站窗口

### 2026-03-06: 智能轨道初始化器
**问题**: 历元(J2000)距场景时间(2024)24年，HPOP传播耗时226+分钟
**解决**: 实现SmartOrbitInitializer
- TLE → SGP4外推
- 历元>3天 → J4解析外推
- 历元<3天 → Direct HPOP
**结果**: 性能提升170倍 (226分钟 → 80秒)

### 2026-03-11: 空间换时间优化 - 姿态预计算缓存
**问题**: Phase2/3姿态预计算成为瓶颈（1386秒，占82%），每任务需要实时计算50个候选的姿态角
**解决**: 实现AttitudePrecacheManager，调度前预计算所有可见窗口的姿态角
**核心策略**:
1. **预加载轨道数据**: 将5,184,060条轨道记录加载为NumPy数组（~445MB）
2. **批量预计算姿态**: 使用Numba并行计算277,946个窗口的姿态角（~30MB）
3. **O(1)缓存查询**: 调度时使用字典直接查找，替代实时计算

**性能提升**:
| 指标 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| Phase2/3耗时 | 1386秒 | 1.1秒 | **1227x** |
| 总调度时间 | 1708秒 | 332秒 | **5.1x** |
| 单次查询耗时 | ~10-20ms | ~0.001ms | **10000x** |

**内存成本**: ~475MB（姿态缓存30MB + 轨道数据445MB）

**适用场景**:
- 内存充足（>1GB可用）
- 重复调度相同场景（缓存可复用）
- 实时性要求高（查询延迟敏感）

**不适用场景**:
- 内存受限环境
- 一次性调度任务（预计算 overhead 不划算）
- 动态变化场景（窗口频繁变更）

**关键代码**:
```python
# core/dynamics/attitude_precache.py
class AttitudePrecacheManager:
    def precompute_attitudes_for_windows(self, visibility_windows):
        # Numba批量计算所有姿态
        # 存储到字典: {(sat_id, window_start): (roll, pitch)}

    def get_attitude(self, sat_id, window_start):
        # O(1)字典查询
        return self._attitude_cache.get((sat_id, window_start.isoformat()))
```

**配置启用**:
```python
# greedy_scheduler.py 配置
config = {
    'enable_attitude_precache': True,  # 启用姿态预计算缓存
    'orbit_json_path': 'java/output/frequency_scenario/orbits.json.gz'
}
```

**经验教训**:
1. 小批量数据（<1000）使用Python循环比Numba批量更快（避免Numba开销）
2. 大批量数据（>5000）使用Numba批量查询可获10-40倍加速
3. 预计算时间约15秒（一次性），适合长时间运行的调度任务
4. 缓存命中率通常>99%，因为所有窗口都在调度前预计算

**进一步优化方向**:
- 将预计算结果持久化到磁盘（如Parquet格式），避免每次重新计算
- 使用内存映射（mmap）管理大缓存，减少内存占用
- GPU加速姿态预计算（CUDA/Numba.cuda）

---

## API 变更记录

### 2026-03-12: 移除简化模式（向后兼容性破坏）

**变更类型**: 向后兼容性破坏 (Breaking Change)

**原因**: 项目要求保持高精度计算，简化模式（近似计算）与这一目标冲突。

**移除的API**:

| 类/函数 | 移除的参数/方法 | 替代方案 |
|---------|----------------|----------|
| `ConstraintConfig` | `mode="simplified"` | 仅支持 `mode="standard"` 或 `"full"` |
| `MetaheuristicConstraintChecker` | `use_simplified_slew` 参数 | 强制使用 `UnifiedBatchConstraintChecker` |
| `UnifiedManeuverChecker.__init__` | `use_simplified_slew` 参数 | 始终使用精确模型 |
| `UnifiedSpatiotemporalChecker.__init__` | `use_simplified_slew` 参数 | 始终使用精确模型 |
| `SlewConstraintChecker.check_slew` | `use_simplified` 参数 | 始终使用精确计算 |
| `PreciseSlewConstraintChecker.check_slew` | `use_simplified` 参数 | 始终使用精确计算 |
| `ConstraintChecker.check_slew` | `use_simplified` 参数 | 始终使用精确计算 |
| `UnifiedManeuverChecker` | `_check_simplified()` 方法 | 已完全移除 |
| `UnifiedSpatiotemporalChecker` | `_simplified_slew_check()` 方法 | 已完全移除 |
| `SlewConstraintChecker` | `_calculate_simplified_slew_angle()` 方法 | 使用ECEF精确计算 |
| `BaseScheduler` | `_use_simplified_slew` 属性 | 强制使用批量检查器 |
| `scripts/run_scheduler.py` | `--simplified` 参数 | 已隐藏，使用时报错 |
| `scripts/config.py` | `use_simplified_slew` 配置项 | 已移除 |

**迁移指南**:

```python
# 旧代码（不再支持）
checker = MetaheuristicConstraintChecker(
    mission,
    config={"use_simplified_slew": True}  # ValueError\!
)

# 新代码（强制精确模式）
checker = MetaheuristicConstraintChecker(
    mission,
    config={"consider_power": True}  # 始终使用精确计算
)
```

**错误处理变化**:
- 旧行为：无法获取卫星位置时使用简化估算
- 新行为：抛出 `RuntimeError` 要求精确位置数据

```python
# 旧行为（静默回退到估算）
if sat_position is None:
    slew_angle = self._calculate_simplified_slew_angle(...)

# 新行为（抛出错误）
if sat_position is None:
    raise RuntimeError(
        "Cannot get satellite position. "
        "High precision mode requires exact position data."
    )
```

**受影响的外部接口**:
- 所有调度器初始化参数
- 约束检查器构造函数
- 命令行参数 `--simplified`

**验证方法**:
```python
# 验证简化模式被拒绝
try:
    config = ConstraintConfig(mode="simplified")
except ValueError as e:
    print(f"Correctly rejected: {e}")

# 验证精确模式正常工作
config = ConstraintConfig(mode="standard")  # OK
checker = MetaheuristicConstraintChecker(mission)  # OK
```

---

## 2026-03-14: 姿态角术语重构 - 统一使用滚转角/俯仰角

### 重构背景

通过查阅文献，发现**侧摆角**和**滚转角**实际上是同一个物理现象的不同表述，都控制绕X轴的旋转。原代码中同时使用 `max_off_nadir`（最大侧摆角）和 `roll`（滚转角）造成了概念混淆。

### 术语统一

| 旧术语 | 新术语 | 说明 |
|--------|--------|------|
| max_off_nadir (侧摆角) | **max_roll_angle** (滚转角) | 绕X轴旋转，控制左右侧摆 |
| - | **max_pitch_angle** (俯仰角) | 绕Y轴旋转，控制前后斜视（新增） |

### 约束计算方案修正

**原来（错误）**:
```python
# 使用合成角度检查
total_angle = sqrt(roll² + pitch²)
if total_angle > max_off_nadir:
    feasible = False
```

**现在（正确）**:
```python
# 分别检查滚转角和俯仰角
if abs(roll) > max_roll_angle:
    feasible = False
if abs(pitch) > max_pitch_angle:
    feasible = False
```

### 卫星能力配置更新

| 卫星类型 | 最大滚转角 | 最大俯仰角 |
|----------|-----------|-----------|
| 光学卫星 | **±35°** | **±20°** |
| SAR卫星 | **±45°** | **±30°** |

### 修改的文件列表

**场景配置** (5个文件):
- `data/entity_lib/satellites/optical_1.json`
- `data/entity_lib/satellites/optical_2.json`
- `data/entity_lib/satellites/sar_1.json`
- `data/entity_lib/satellites/sar_2.json`
- `scenarios/large_scale_frequency.json` (60颗卫星)
- `scenarios/point_group_scenario.json`
- `scenarios/point_group_scenario.yaml`
- `scenarios/archive/large_scale_experiment.json`

**核心数据模型** (2个文件):
- `core/constants.py` - 添加新的常量定义
- `core/models/satellite.py` - SatelliteCapabilities类更新

**姿态约束检查** (4个文件):
- `scheduler/constraints/batch_attitude_calculator.py` - 分别检查roll/pitch
- `scheduler/constraints/slew_constraint_checker.py`
- `scheduler/constraints/precise_slew_constraint_checker.py`
- `scheduler/constraints/unified_maneuver_checker.py`

**覆盖计算** (1个文件):
- `core/coverage/footprint_calculator.py` - `calculate_off_nadir_angle` → `calculate_required_roll_angle`

**调度器** (3个文件):
- `scheduler/greedy/greedy_scheduler.py`
- `scheduler/greedy/heuristic_scheduler.py`
- `scheduler/common/constraint_checker.py`

**Java后端** (1个文件):
- `java/src/orekit/visibility/JsonScenarioLoader.java` - 兼容新旧字段

**工具脚本** (4个文件):
- `scripts/generate_scenario.py`
- `utils/yaml_loader.py`
- `utils/entity/library.py`
- `utils/entity/cli/commands/feasibility.py`

**数据库Schema** (2个文件):
- `storage/schema.py`
- `core/models/schema/config_tables.py`

**测试文件** (13个文件):
- `tests/unit/core/models/test_satellite.py`
- `tests/unit/core/coverage/test_footprint_calculator.py`
- `tests/unit/scheduler/test_*.py`
- `tests/integration/test_*.py`
- 其他相关测试文件

### 向后兼容性

Java后端 (`JsonScenarioLoader.java`) 保持向后兼容:
```java
// 优先读取max_roll_angle（新字段），回退到max_off_nadir（旧字段）
if (capabilities.has("max_roll_angle")) {
    maxOffNadir = capabilities.getDouble("max_roll_angle");
} else if (capabilities.has("max_off_nadir")) {
    maxOffNadir = capabilities.getDouble("max_off_nadir");
}
```

### 关键代码变更示例

**SatelliteCapabilities类**:
```python
@dataclass
class SatelliteCapabilities:
    # 旧:
    max_off_nadir: float = DEFAULT_MAX_OFF_NADIR_DEG
    
    # 新:
    max_roll_angle: float = DEFAULT_MAX_ROLL_ANGLE_DEG   # 35°光学 / 45°SAR
    max_pitch_angle: float = DEFAULT_MAX_PITCH_ANGLE_DEG  # 20°光学 / 30°SAR
```

**批量姿态约束检查**:
```python
# scheduler/constraints/batch_attitude_calculator.py
@njit(parallel=True, cache=True)
def _batch_filter_by_attitude_numba(
    rolls: np.ndarray,
    pitches: np.ndarray,
    max_roll_angles: np.ndarray,    # 新增
    max_pitch_angles: np.ndarray    # 新增
) -> np.ndarray:
    for i in prange(n):
        # 分别检查（替代原来的合成角度检查）
        if abs(rolls[i]) > max_roll_angles[i] * 1.1:
            feasible_mask[i] = False
        elif abs(pitches[i]) > max_pitch_angles[i] * 1.1:
            feasible_mask[i] = False
```

---

## 2026-03-15: 姿态预计算系统重大修复

### 问题背景
在大规模场景（60卫星×1000目标）姿态预计算测试中，发现计算出的目标可见性窗口为**0个**，而地面站窗口正常（3884个）。

### 问题排查过程

#### 1. 姿态计算Bug（180°误差）
**问题**: 姿态角计算结果异常（如118-180°），超出正常范围（±90°）

**根因**: `AttitudeCalculator.java` 中使用了错误的公式：
```java
// 错误代码
roll = Math.toDegrees(Math.atan2(y, -z));  // -z 导致180°误差
pitch = Math.toDegrees(Math.atan2(x, -z));
```

**修复**: 移除负号
```java
// 正确代码
roll = Math.toDegrees(Math.atan2(y, z));
pitch = Math.toDegrees(Math.atan2(x, z));
```

#### 2. 轨道参数显示错误（单位转换错误）
**问题**: 验证程序显示轨道倾角为4125.3°（应为55°或72°）

**根因**: `OrbitVerify.java` 中对已经是度数的角度再次使用 `Math.toDegrees()` 转换：
```java
// 错误代码
double i = Math.toDegrees(es.inclination);  // 72° -> 4125.3°
```

**修复**: 直接使用存储的度数
```java
// 正确代码
double i = es.inclination;  // 已经是度数
```

#### 3. 姿态约束检查逻辑错误（核心问题）
**问题**: 原实现中，只要窗口内有**任何一点**超出姿态限制，整个窗口被丢弃

**错误代码**:
```java
// 原逻辑：全有或全无
for (sample : samples) {
    if (Math.abs(roll) > maxRoll || Math.abs(pitch) > maxPitch) {
        return false;  // 整个窗口丢弃！
    }
}
```

**物理现实**: 卫星经过目标时，姿态角变化规律：
- 接近目标：俯仰角最大（如60-70°）
- 正上方：俯仰角接近0°
- 远离目标：俯仰角反向最大
- **滚转角在侧面观测时可能始终较大**

**正确逻辑**: 提取窗口内**姿态约束满足的具体子时段**

**修复代码**:
```java
// 新逻辑：提取可行子时段
public static class AttitudeFeasibleInterval {
    public final double startTime;  // 相对于窗口开始
    public final double endTime;
    public final List<AttitudeSample> samples;
}

private List<AttitudeFeasibleInterval> findAttitudeFeasibleIntervals(
        List<AttitudeSample> samples,
        double maxRoll, double maxPitch,
        String satId, String targetId,
        double minDuration) {

    List<AttitudeFeasibleInterval> intervals = new ArrayList<>();
    boolean inFeasibleInterval = false;
    double intervalStart = 0;
    List<AttitudeSample> currentSamples = new ArrayList<>();

    for (AttitudeSample sample : samples) {
        boolean feasible = Math.abs(sample.roll) <= maxRoll
                        && Math.abs(sample.pitch) <= maxPitch;

        if (feasible) {
            if (!inFeasibleInterval) {
                // 开始新的可行区间
                inFeasibleInterval = true;
                intervalStart = sample.timestamp;
                currentSamples = new ArrayList<>();
            }
            currentSamples.add(sample);
        } else {
            if (inFeasibleInterval) {
                // 结束当前可行区间
                inFeasibleInterval = false;
                double intervalEnd = sample.timestamp;

                // 检查最小持续时间
                if (intervalEnd - intervalStart >= minDuration) {
                    intervals.add(new AttitudeFeasibleInterval(
                        intervalStart, intervalEnd,
                        new ArrayList<>(currentSamples)));
                }
                currentSamples.clear();
            }
        }
    }
    // ... 处理最后一个区间
    return intervals;
}
```

### 关键经验总结

#### 1. 可见性窗口定义
**正确的可见性窗口**必须同时满足：
- 几何可见（仰角≥最小仰角）
- **姿态可行（滚转/俯仰角在限制范围内）**
- 持续时间≥最小观测时长

#### 2. 姿态角变化规律
卫星经过目标时：
| 阶段 | 俯仰角 | 滚转角 | 说明 |
|------|--------|--------|------|
| 接近 | 大→小 | 变化 | 需要俯视 |
| 正上方 | 接近0° | 较小 | 最佳观测点 |
| 远离 | 小→大 | 变化 | 需要仰视 |
| 侧面 | 中等 | 可能大 | 需要侧摆 |

#### 3. 设计原则
- **提取子时段**：一个几何窗口可能产生多个姿态可行子窗口
- **全时段采样**：1秒步长采样确保不遗漏可行时段
- **严格约束**：每个采样点都必须满足姿态限制

### 修复结果

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 目标窗口数 | 0 | **184,357** |
| 平均每目标 | 0 | 184.2个 |
| 总窗口数 | 2,843 | 188,241 |
| 平均每卫星 | 47.4 | 3,137.4个 |

### 相关文件变更

**Java后端**:
- `java/src/orekit/visibility/AttitudeCalculator.java` - 修复姿态角计算
- `java/src/orekit/visibility/OptimizedVisibilityCalculator.java` - 实现子时段提取
- `java/src/orekit/visibility/OrbitVerify.java` - 修复角度显示

**场景生成**:
- `scripts/generate_scenario.py` - 修正Walker星座倾角参数

---

## 2026-03-16: 成像中心点距离优化

### 问题背景
调度器选择任务时，成像中心点（footprint_center）与目标坐标存在偏差（通常2-8度）。优化目标是在满足约束条件下，优先选择成像中心更靠近目标的可见窗口。

### 解决方案

#### 1. 新增工具函数 (`scheduler/common/footprint_utils.py`)
- `calculate_haversine_distance()` - Haversine公式计算大地线距离
- `calculate_footprint_center_from_attitude()` - 根据姿态角计算成像中心
- `calculate_center_distance_score()` - 计算距离评分（指数衰减模型）

#### 2. 调度器集成
- **GreedyScheduler**: 在 `_calculate_assignment_score()` 中添加中心点距离评分
- **HeuristicScheduler**: 重写 `_select_best_assignment()`，使用综合评分
- **MetaheuristicScheduler**: 在 `_evaluate_task_assignment()` 中添加评分因子

#### 3. 配置项 (`scheduler/common/config.py`)
```python
enable_center_distance_score: bool = True   # 是否启用
center_distance_weight: float = 15.0        # 评分权重（分/度）
```

### 评分模型
- **指数衰减**: `score = exp(-distance / scale)`
- **参数**: max_distance=10°, scale=3°
- **效果**: 0°偏差=1.0分，3°偏差≈0.37分，10°偏差≈0.04分

### 关键设计决策

#### 1. 仅对非聚类任务启用
```python
if not getattr(task, 'is_cluster', False) and not getattr(task, 'is_cluster_task', False):
    # 应用中心点距离评分
```
- **原因**: 聚类任务覆盖多个目标，无法定义单一的"最佳中心"
- **影响**: 约50%的任务类型不受影响

#### 2. 防御性编程
```python
# 参数边界检查
if max_distance <= 0:
    max_distance = 10.0  # 使用默认值
if not satellite_position or len(satellite_position) != 3:
    return 0.5  # 默认中等评分
```

#### 3. 失败回退策略
- 任何异常都返回默认评分0.5
- 不影响调度主流程
- 记录debug日志供排查

### 测试覆盖
- **单元测试**: `tests/unit/scheduler/test_center_distance_score.py`
  - Haversine距离计算（相同点、赤道、极点）
  - 成像中心计算（星下点、侧摆、俯仰）
  - 评分函数（边界值、极端情况、无效参数）
- **22个测试用例全部通过**

### 使用示例
```python
# 启用优化（默认）
config = {
    'enable_center_distance_score': True,
    'center_distance_weight': 15.0,
}
scheduler = GreedyScheduler(config=config)

# 禁用优化
config = {
    'enable_center_distance_score': False,
}
```

### 性能影响
- **额外计算**: 每任务增加一次卫星位置查询和姿态计算
- **优化措施**: 导入优化（避免方法内重复导入）
- **实测**: 在2412任务场景下，调度时间增加<5%

### 后续调参建议
根据实际运行效果调整权重：
- **偏差改善不明显**: 增加权重到 20-25
- **任务调度数量下降**: 降低权重到 10-12
- **平衡推荐**: 15（默认值）

---

## 2026-03-16: 场景缓存管理功能

### 功能概述

实现了基于内容哈希的智能缓存识别和复用机制，支持：

1. **识别内容相同的场景** - 即使文件名不同，只要内容相同就能复用缓存
2. **检测可复用的部分配置** - 如卫星配置相同但目标不同，可复用轨道数据
3. **自动缓存管理** - 缓存索引自动维护，支持过期清理

### 核心组件

| 组件 | 文件 | 职责 |
|------|------|------|
| **指纹计算器** | `core/cache/fingerprint_calculator.py` | 计算场景各组件的SHA256哈希 |
| **指纹数据结构** | `core/cache/fingerprint.py` | ScenarioFingerprint和ComponentHash定义 |
| **缓存索引** | `core/cache/index_manager.py` | 管理缓存索引的CRUD操作 |
| **索引条目** | `core/cache/index.py` | CacheIndexEntry和CacheStatus定义 |
| **CLI工具** | `scripts/cache_manager.py` | 命令行管理工具 |

### 使用方法

#### 检查场景缓存状态
```bash
python scripts/cache_manager.py check -s scenarios/my_scene.json
```

#### 列出所有缓存
```bash
python scripts/cache_manager.py list
python scripts/cache_manager.py list --sort size  # 按大小排序
```

#### 清理过期缓存
```bash
python scripts/cache_manager.py clean --older-than 30  # 删除30天未访问的缓存
```

#### 分析场景复用可能性
```bash
python scripts/cache_manager.py analyze -s scene1.json -S scene2.json
```

### 与现有工具集成

#### compute_visibility.py
计算完成后自动注册缓存到索引。

#### run_scheduler.py
自动检测并使用匹配的场景缓存：
```
[自动缓存检测] 找到匹配的场景缓存:
  场景指纹: e61c1944140d1161...
  缓存文件: results/visibility_cache.json
  ✓ 将使用索引中的缓存
```

### 缓存目录结构
```
cache/
├── index.json              # 缓存索引文件
├── windows/                # 可见性窗口缓存
└── orbits/                 # 轨道数据缓存
```

### 性能收益
| 场景 | 优化前 | 优化后 |
|------|--------|--------|
| 相同场景重复运行 | 每次都重新计算 | 自动复用缓存 (节省100%) |
| 相同卫星不同目标 | 完全重新计算 | 复用轨道计算 (节省60-80%) |

### API使用示例
```python
from core.cache.fingerprint_calculator import FingerprintCalculator, FingerprintComparator
from core.cache.index_manager import CacheIndexManager

# 计算场景指纹
calculator = FingerprintCalculator()
fingerprint = calculator.calculate("scenes/my_scene.json")

# 注册缓存
manager = CacheIndexManager()
entry = manager.register(fingerprint, "cache.json", "orbits.json.gz")

# 查找缓存
found = manager.find(fingerprint)

# 查找可复用的轨道缓存
orbit_entry = manager.find_reusable_orbit_cache(fingerprint)
```

### 测试覆盖
- `tests/unit/cache/test_fingerprint.py` - 指纹计算测试 (15个用例)
- `tests/unit/cache/test_index_manager.py` - 索引管理测试 (9个用例)
- 总计 **24个测试用例全部通过**


---

## 2026-03-17: 主动前向推扫模式（PMC）实现

### 功能概述

实现了主动前向推扫模式（Pitch Motion Compensation），通过俯仰匀速机动降低相机相对地面的推扫速度，延长积分时间，提高信噪比（SNR）。

### 技术原理

- **俯仰机动补偿**: 成像过程中以固定俯仰角速度机动
- **等效降速**: 地速 = 轨道速度 × (1 - 降速比)
- **积分时间增益**: Gain = 1 / (1 - R)，其中R为降速比

### 支持的降速比

| 降速比 | 俯仰角速度(500km) | 积分时间增益 | SNR增益 |
|--------|------------------|-------------|---------|
| 10%    | 0.087°/s        | 1.11x       | 0.46dB  |
| 25%    | 0.218°/s        | 1.33x       | 1.25dB  |
| 50%    | 0.436°/s        | 2.00x       | 3.01dB  |
| 75%    | 0.655°/s        | 4.00x       | 6.02dB  |

### 新增文件

**数据模型**:
- `core/models/pmc_config.py` - PMC配置类
- 扩展 `core/models/imaging_mode.py` - 添加ImagingMode枚举和PMC模板
- 扩展 `core/models/payload_config.py` - 支持PMC模式管理
- 扩展 `core/models/target.py` - 支持PMC任务需求

**动力学计算**:
- `core/dynamics/pmc_calculator.py` - PMC动力学计算核心

**约束检查**:
- `scheduler/constraints/pmc_constraint_checker.py` - PMC约束检查器
- 扩展 `scheduler/constraints/unified_batch_constraint_checker.py` - 集成PMC检查

**示例**:
- `examples/pmc_mode_example.py` - 使用示例

### 使用示例

```python
# 创建PMC模式
from core.models.imaging_mode import create_pmc_mode_config

pmc_mode = create_pmc_mode_config(
    base_resolution_m=0.5,
    base_swath_width_m=15000,
    speed_reduction_ratio=0.25,
    mode_type='optical'
)

# 创建需要PMC的目标
from core.models.target import Target, TargetType

target = Target(
    id='TGT-001',
    target_type=TargetType.POINT,
    longitude=116.4,
    latitude=39.9,
    required_imaging_mode='forward_pushbroom_pmc',
    pmc_priority=2
)

# PMC约束检查
from scheduler.constraints.pmc_constraint_checker import PMCConstraintChecker

checker = PMCConstraintChecker()
result = checker.check_pmc_feasibility(candidate)
print(f"SNR增益: {result.snr_gain_db:.2f}dB")
```

### 目标任务需求扩展

目标现在可以指定：
- `required_imaging_mode`: 要求的成像模式（如 'forward_pushbroom_pmc'）
- `required_satellite_type`: 要求的卫星类型（'optical' 或 'sar'）
- `pmc_priority`: PMC模式优先级（0-3）
- `pmc_speed_reduction_range`: 期望降速比范围

