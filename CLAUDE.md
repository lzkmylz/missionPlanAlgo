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

| 场景 | 耗时 | 窗口数 |
|------|------|--------|
| 60卫星×1000目标×24h | 80秒 | 318,312 |
| 每对计算耗时 | 1.34 ms | - |

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
    'use_simplified_slew': False,           # 禁用简化模式
    'enable_attitude_calculation': True,    # 启用姿态角计算
    'precompute_positions': True,           # 预计算位置（如未加载Java数据）
    'precompute_step_seconds': 1.0,         # 1秒步长（与HPOP精扫描匹配）
    'use_precomputed_orbits': True,         # 默认加载Java预计算轨道
}
```

**姿态计算解耦**: 姿态计算与简化模式完全解耦，只要 `enable_attitude_calculation=True` 就始终计算。

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

### 2026-03-06: 智能轨道初始化器
**问题**: 历元(J2000)距场景时间(2024)24年，HPOP传播耗时226+分钟
**解决**: 实现SmartOrbitInitializer
- TLE → SGP4外推
- 历元>3天 → J4解析外推
- 历元<3天 → Direct HPOP
**结果**: 性能提升170倍 (226分钟 → 80秒)
