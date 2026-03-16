# 卫星星座任务规划算法研究平台

面向大规模遥感卫星星座的任务规划算法研究与仿真平台，支撑异构卫星、多类观测目标、多成像模式的调度优化研究。

**核心亮点**：60颗卫星×1000目标×24小时的可见性计算仅需 **80秒**，性能提升 **170倍**。

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/zhaolin/missionplanalgo.git
cd missionplanalgo

# 安装依赖
pip install -e .

# 或使用 conda
conda env create -f environment.yml
conda activate mpa
```

### CLI 使用

```bash
# 计算可见性窗口
mpa visibility compute -s scenarios/large_scale_frequency.json -o output/windows.json

# 执行调度
mpa schedule run -s scenarios/large_scale_frequency.json \
                 -c output/windows.json \
                 -a greedy \
                 -o output/schedule.json

# 启动 API 服务
mpa serve start --host 0.0.0.0 --port 8000
```

### Python API 使用

```python
import missionplanalgo as mpa

# 计算可见性
windows = mpa.compute_visibility("scenarios/test.json")
print(f"计算了 {windows['total_windows']} 个窗口")

# 执行调度
result = mpa.schedule("scenarios/test.json",
                      algorithm="greedy",
                      cache_path="output/windows.json")
print(f"调度了 {len(result['scheduled_tasks'])} 个任务")

# 评估结果
metrics = mpa.evaluate_schedule(result)
print(f"频次满足率: {metrics['frequency_satisfaction']:.1%}")
```

### RESTful API 使用

```bash
# 启动服务
mpa serve start

# 查看 API 文档
open http://127.0.0.1:8000/docs

# 调用 API
curl -X POST http://localhost:8000/api/v1/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": {"satellites": [...], "targets": [...]},
    "algorithm": "greedy"
  }'
```

---

## 📦 功能特性

- **CLI 工具** (`mpa`): 统一的命令行接口，支持调度、可见性计算、配置管理
- **Python API**: 函数式接口，一行代码完成任务
- **RESTful API**: FastAPI 实现，支持异步任务处理
- **单机/分布式**: 开箱即用，也可配置 Redis 实现分布式

---

## 🔧 配置

```bash
# 初始化配置
mpa config init

# 设置默认算法
mpa config set scheduler.default_algorithm ga

# 启用 Redis 分布式模式（可选）
mpa config set task_backend.type celery
mpa config set task_backend.broker_url redis://localhost:6379/0
```

配置位置：`~/.config/mpa/config.yaml`

---

## 📦 Python 包安装与使用

### 作为 Python 包安装

```bash
# 从源码安装（开发模式）
git clone https://github.com/zhaolin/missionplanalgo.git
cd missionplanalgo
pip install -e .

# 打包分发
python -m build

# 离线安装
pip install dist/missionplanalgo-*.whl
```

### 三种使用方式

#### 1. CLI 命令行工具

```bash
# 查看帮助
mpa --help

# 计算可见性
mpa visibility compute -s scenarios/large_scale_frequency.json

# 执行调度
mpa schedule run -s scenarios/large_scale_frequency.json -a greedy

# 启动 API 服务
mpa serve start --host 0.0.0.0 --port 8000
```

#### 2. Python API

```python
import missionplanalgo as mpa

# 加载场景
scenario = mpa.load_scenario("scenarios/test.json")

# 计算可见性
windows = mpa.compute_visibility(scenario)

# 执行调度
result = mpa.schedule(scenario, algorithm="greedy")

# 评估结果
metrics = mpa.evaluate_schedule(result)
```

#### 3. RESTful API 服务

```bash
# 启动服务
mpa serve start

# 查看 API 文档
open http://127.0.0.1:8000/docs

# 提交调度任务
curl -X POST http://localhost:8000/api/v1/schedule \
  -H "Content-Type: application/json" \
  -d '{"scenario": {...}, "algorithm": "greedy"}'
```

### 配置管理

```bash
# 初始化配置
mpa config init

# 设置默认算法
mpa config set scheduler.default_algorithm ga

# 启用 Redis 分布式模式（可选）
mpa config set task_backend.type celery
mpa config set task_backend.broker_url redis://localhost:6379/0
```

配置文件位置：`~/.config/mpa/config.yaml`

---

## 核心特性

- **大规模星座支持**：60颗异构卫星（30光学 + 30 SAR）
- **多类型目标**：点群目标、大区域目标（支持网格/条带分解）
- **多成像模式**：聚束/滑动聚束/条带（SAR）、推扫/框幅（光学）
- **地面站资源调度**：多地理位置、多天线约束、频次约束
- **算法即插即用**：统一接口支持贪心、EDD、SPT、GA、SA、ACO、PSO、禁忌搜索等
- **双后端可见性计算**：STK HPOP（高精度）+ Orekit（自主可控，Java实现）
- **智能轨道初始化**：自动处理历元与场景时间差距（SGP4/J4/HPOP自适应选择）
- **姿态预计算缓存**：调度前预计算所有窗口姿态角，O(1)查询，1227倍加速
- **批量约束检查**：Numba向量化优化，约束检查性能提升5-10倍
- **精确姿态机动模型**：基于刚体动力学的时间最优轨迹规划
- **成像中心点距离优化**：非聚类任务成像中心更靠近目标坐标（偏差降低8.3%）
- **星载边缘计算支持**：在轨数据预处理，减少下传压力
- **完整实验流程**：场景配置 → 可见性计算 → 算法执行 → 结果验证 → 性能评估 → 批量对比

---

## 技术栈

- **语言**: Python 3.10+, Java 17+
- **轨道计算**: SGP4 / STK HPOP / Orekit 12.0 (EGM2008 90x90重力场)
- **数值计算**: NumPy, SciPy, Hipparchus
- **可视化**: Matplotlib, Plotly
- **数据**: JSON/YAML 配置, Parquet/HDF5 结果存储
- **测试**: pytest, 覆盖率 80%+
- **构建**: Make, javac

---

## 项目结构

```
missionPlanAlgo/
├── core/                    # Python核心层
│   ├── models/             # 实体模型（卫星、目标、地面站、任务）
│   ├── orbit/              # 轨道模块（传播器、可见性计算）
│   │   └── visibility/     # Orekit可见性计算Python接口
│   ├── dynamics/           # 动力学模块（姿态预计算、刚体动力学）
│   │   ├── attitude_precache.py       # 姿态预计算缓存管理
│   │   └── precise/        # 精确姿态机动模型
│   ├── resources/          # 资源管理（卫星池、地面站池）
│   ├── decomposer/         # 目标分解（条带、网格）
│   ├── processing/         # 在轨处理管理
│   ├── network/            # 星间链路网络与路由
│   ├── dynamic_scheduler/  # 事件驱动和滚动时域调度
│   ├── validators/         # 约束验证
│   └── telecommand/        # 指令序列生成
├── java/                   # Java后端（Orekit高性能计算）
│   ├── src/orekit/visibility/  # 可见性计算Java实现
│   │   ├── SmartOrbitInitializer.java   # 智能轨道初始化
│   │   ├── OrbitStateCache.java         # 轨道状态缓存
│   │   ├── OptimizedVisibilityCalculator.java  # 优化可见性计算
│   │   └── JsonScenarioLoader.java      # 场景配置加载
│   ├── lib/                # Orekit/Hipparchus JAR包
│   └── classes/            # 编译后的class文件
├── payload/                # 载荷模块（光学/SAR成像器、成像模式）
├── scheduler/              # 调度算法
│   ├── greedy/            # 启发式算法（Greedy、EDD、SPT）
│   ├── metaheuristic/     # 元启发式算法（GA、SA、ACO、PSO、Tabu）
│   └── constraints/       # 批量约束检查器（Numba向量化优化）
├── simulator/             # 离散事件仿真引擎
├── evaluation/            # 性能指标与结果分析
├── experiments/           # 实验管理与基准测试
├── visualization/         # 甘特图、星下点轨迹、覆盖图
├── scripts/               # 工具脚本
│   └── download_orekit_data.sh  # Orekit数据下载
├── scenarios/             # JSON/YAML 场景配置文件
├── docs/                  # 架构设计文档与实验流程文档
└── tests/                 # 单元测试与集成测试（80%+ 覆盖率）
```

---

## 可见性计算系统架构

高性能可见性计算采用 **Java Orekit 后端 + Python 前端** 混合架构：

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
场景开始时刻的初始状态
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
    └─ 精化扫描 (1秒步长)
    │
    ▼
输出文件
    ├─ visibility_windows.json
    ├─ satellites.json
    └─ ground_stations.json
```

### 关键技术创新

**1. 智能轨道初始化器 (SmartOrbitInitializer)**

解决历元时间与场景时间差距大的问题（如历元2000年，场景2024年）：

| 历元距场景时间 | 处理策略 | 说明 |
|--------------|---------|------|
| < 3天 | 直接使用历元轨道 | HPOP从历元开始传播 |
| > 3天 | J4解析外推到场景开始 | Eckstein-Hechler传播器，避免长期HPOP误差累积 |
| TLE数据 | SGP4外推到场景开始 | 标准TLE处理方法 |

**性能提升**：从226分钟 → 80秒（提升170倍）

**2. 姿态预计算缓存系统 (AttitudePrecacheManager)**

调度前预计算所有可见窗口的姿态角，实现O(1)查询：

| 指标 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| Phase2/3耗时 | 1386秒 | 1.1秒 | **1227x** |
| 总调度时间 | 1708秒 | 332秒 | **5.1x** |
| 单次查询耗时 | ~10-20ms | ~0.001ms | **10000x** |

**3. 批量约束检查（Numba向量化优化）**

所有约束检查器使用Numba JIT并行计算：

| 约束类型 | 逐个检查 | 批量检查(200批次) | 加速比 |
|---------|---------|------------------|--------|
| 姿态约束 | 3.6ms | 0.3ms | **10.7x** |
| SAA约束 | ~100ms | ~15ms | **6.7x** |
| 时间冲突 | ~40ms | ~3ms | **13x** |
| 资源约束 | ~20ms | ~1.5ms | **13x** |

**4. 精确姿态机动模型**

基于刚体动力学的时间最优轨迹规划：
- 飞轮动量管理
- 能量消耗精确建模
- 时间最优轨迹规划
- 统一使用精确模型（简化模式已移除）

---

## 性能基准

### 可见性计算

| 场景 | 耗时 | 窗口数 | 每对耗时 |
|------|------|--------|----------|
| 60卫星×1000目标×24h | **80秒** | 188,241 | 1.34 ms |
| 卫星-目标窗口 | - | 184,357 | - |
| 卫星-地面站窗口 | - | 3,884 | - |

测试环境：Intel i7, 16GB RAM, Java 17, EGM2008 90x90

### 调度性能（60卫星×1000目标×24h）

| 检查环节 | 调用次数 | 总耗时 | 平均耗时 |
|---------|---------|--------|---------|
| 批量姿态检查 | 2,638 | 35.75s | **13.55ms** |
| 批量约束检查 | 2,638 | 15.69s | **5.95ms** |
| 姿态预计算 | 1次 | 15s | 一次性 |

### 姿态预计算缓存效果

| 模式 | 调度时间 | 适用场景 |
|-----|---------|---------|
| 无预计算 | 1708秒 | 内存受限 |
| 预计算缓存 | **332秒** | 内存充足（推荐）|

测试场景：60卫星×1000目标，调度任务数2,638个，频次满足率100%

---

## 技术亮点详解

### 1. 批量约束检查器（Numba向量化优化）

所有调度器默认使用批量约束检查器，通过Numba JIT并行计算实现C级性能：

```python
from scheduler.constraints import (
    BatchSlewConstraintChecker,    # 批量姿态约束
    BatchSAAConstraintChecker,     # 批量SAA约束
    BatchTimeConflictChecker,      # 批量时间冲突
    BatchResourceChecker,          # 批量资源约束
    UnifiedBatchConstraintChecker  # 统一批量检查入口
)

# 统一检查所有约束
checker = UnifiedBatchConstraintChecker(mission)
results = checker.check_all_constraints_batch(
    candidates=candidates,
    existing_tasks=scheduled_tasks,
    satellite_states=satellite_states
)
```

**优化特点**：
- **Numba JIT并行计算**：`@njit(parallel=True)` + `prange` 实现C级并行
- **向量化数据布局**：Python对象 → NumPy数组 → Numba加速计算
- **早期终止**：阶段式检查，失败即跳过后续检查

### 2. 姿态预计算缓存

调度前预计算所有可见窗口的姿态角，调度时O(1)查询：

```python
from core.dynamics.attitude_precache import AttitudePrecacheManager

# 初始化预计算管理器
precache = AttitudePrecacheManager(
    orbit_json_path='java/output/frequency_scenario/orbits.json.gz'
)

# 预计算所有窗口姿态角（一次性，约15秒）
precache.precompute_attitudes_for_windows(visibility_windows)

# 调度时O(1)查询
roll, pitch = precache.get_attitude(sat_id, window_start)
```

**内存成本**：约475MB（姿态缓存30MB + 轨道数据445MB）

### 3. 精确姿态机动模型

基于刚体动力学的时间最优轨迹规划：

```python
from scheduler.constraints import PreciseSlewConstraintChecker

# 所有调度器默认使用精确模型
checker = PreciseSlewConstraintChecker(
    mission=mission,
    use_precise_model=True  # 强制精确计算
)
```

**模型组成**：
- `core/dynamics/precise/rigid_body_dynamics.py` - 刚体动力学
- `core/dynamics/precise/trajectory_planner.py` - 时间最优轨迹
- `core/dynamics/precise/energy_model.py` - 能量消耗建模
- `core/dynamics/precise/momentum_manager.py` - 飞轮动量管理

### 4. 姿态角术语统一

统一使用滚转角（Roll）和俯仰角（Pitch）：

| 卫星类型 | 最大滚转角 | 最大俯仰角 |
|----------|-----------|-----------|
| 光学卫星 | ±35° | ±20° |
| SAR卫星 | ±45° | ±30° |

**约束检查**：分别检查滚转角和俯仰角（替代原来的合成角度检查）

### 5. 成像中心点距离优化

在任务调度评分中引入成像中心点与目标坐标的距离因子，使非聚类任务的成像中心更靠近目标：

**评分模型**：
```python
score = exp(-distance / scale)  # 指数衰减模型
# 0°偏差 = 1.0分, 3°偏差 ≈ 0.37分, 10°+ = 0分
```

**配置方式**：
```python
config = {
    'enable_center_distance_score': True,   # 启用优化
    'center_distance_weight': 15.0,         # 评分权重（分/度）
}
scheduler = GreedyScheduler(config)
```

**优化效果**（60卫星×1000目标场景）：
| 指标 | 优化前 | 优化后 | 改善 |
|------|--------|--------|------|
| 平均偏差 | 6.51° | 5.97° | **8.3%** ↓ |
| 最小偏差 | 3.80° | 0.16° | **95.8%** ↓ |
| 高精度任务占比(<2°) | - | 9.5% | 新增 |

**设计特点**：
- 仅对非聚类任务启用（聚类任务覆盖多目标，无法定义单一最佳中心）
- 防御性编程：无效参数自动回退到默认值
- 失败安全：任何异常返回中等评分0.5，不中断调度流程

**权重调参建议**：
- 保守配置（15.0）：默认推荐，平衡精度与覆盖率
- 平衡配置（20.0）：科学观测场景，轻微提升精度
- 激进配置（25.0）：高精度需求，可能牺牲部分边缘任务

---

## 支持的算法

### 启发式算法

| 算法 | 说明 | 特点 |
|-----|------|------|
| **Greedy** | 贪心算法 | 按优先级排序，选择最佳分配 |
| **EDD** | 最早截止时间优先 | 最小化最大延误 |
| **SPT** | 最短处理时间优先 | 最小化平均流程时间 |

### 元启发式算法

| 算法 | 说明 | 核心机制 |
|-----|------|---------|
| **GA** | 遗传算法 | 选择、交叉、变异、精英保留 |
| **SA** | 模拟退火 | 温度控制、Metropolis准则 |
| **ACO** | 蚁群优化 | 信息素更新、启发函数 |
| **PSO** | 粒子群优化 | 速度位置更新、全局/局部搜索 |
| **Tabu** | 禁忌搜索 | 禁忌表、邻域搜索、藐视准则 |

---

## 快速开始

### 1. 安装依赖

#### 1.1 安装 Git LFS（必需）

本项目使用 [Git LFS](https://git-lfs.github.com/) 管理大文件（Java依赖库、计算输出数据）。**克隆前必须安装 Git LFS**。

**安装 Git LFS：**

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
# 下载安装程序：https://git-lfs.github.com/
```

**初始化 Git LFS：**

```bash
git lfs install
```

#### 1.2 克隆项目

```bash
git clone <repository-url>
cd missionPlanAlgo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装Python依赖
pip install -r requirements.txt
```

#### 1.2 下载 Orekit 数据文件

本项目使用 [Orekit](https://www.orekit.org/) 进行高精度轨道计算，需要下载以下数据文件：

| 数据文件 | 大小 | 说明 | 必需 |
|---------|------|------|------|
| **EGM2008** | 70MB (压缩) | 地球重力场模型 (90x90阶) | ✅ 必须 |
| Orekit JARs | ~15MB | Java 运行时库 | ✅ 必须 |
| IERS | ~3MB | 地球自转参数 | ⚠️ 推荐 |
| DE440 | 120MB | 行星历表 | ⚠️ 推荐 |

**一键下载（推荐）：**

```bash
# 下载所有必需数据到 ~/orekit-data
./scripts/download_orekit_data.sh

# 或使用自定义目录
./scripts/download_orekit_data.sh -d /opt/orekit-data

# 设置环境变量（添加到 ~/.bashrc 或 ~/.zshrc）
export OREKIT_DATA_DIR=$HOME/orekit-data
```

**验证安装：**

```bash
# 验证数据文件完整性
./scripts/download_orekit_data.sh --verify

# 测试 Python 配置
python -c "from core.orbit.visibility.orekit_config import get_orekit_data_dir; print(get_orekit_data_dir())"
```

**手动安装（备用）：**

如果自动下载失败，可手动下载：

1. **EGM2008 重力场数据**（必须）
   - 访问：https://earth-info.nga.mil/
   - 下载：EGM2008 Spherical Harmonics (104MB ZIP)
   - 解压：`EGM2008_to2190_TideFree` 文件
   - 压缩：`gzip -c EGM2008_to2190_TideFree > ~/orekit-data/potential/egm-format/EGM2008_to2190_TideFree.gz`

2. **Orekit JAR 包**
   - https://repo1.maven.org/maven2/org/orekit/orekit/12.0/orekit-12.0.jar
   - https://repo1.maven.org/maven2/org/hipparchus/hipparchus-core/3.0/hipparchus-core-3.0.jar
   - https://repo1.maven.org/maven2/org/hipparchus/hipparchus-geometry/3.0/hipparchus-geometry-3.0.jar
   - https://repo1.maven.org/maven2/org/hipparchus/hipparchus-ode/3.0/hipparchus-ode-3.0.jar
   - 放入：`~/orekit-data/lib/`

3. **IERS 地球自转数据**（推荐）
   - 下载：https://datacenter.iers.org/products/eop/rapid/standard/finals2000A.all
   - 放入：`~/orekit-data/IERS/finals2000A.all`

#### 1.3 安装 Java 运行时

Orekit Java 后端需要 Java 17+：

```bash
# Ubuntu/Debian
sudo apt-get install openjdk-17-jre

# macOS
brew install openjdk@17

# 验证
java -version  # 应显示 17 或更高版本
```

### 2. 运行示例

```bash
# 使用贪心算法运行场景
python main.py --scenario scenarios/point_group_scenario.json --algorithm greedy

# 使用遗传算法并生成可视化
python main.py --scenario scenarios/point_group_scenario.json --algorithm ga --visualize

# 使用EDD算法并保存结果
python main.py --scenario scenarios/point_group_scenario.json --algorithm edd --save-result

# 查看帮助
python main.py --help
```

### 编程式使用

```python
from core.models import Mission
from scheduler.greedy import GreedyScheduler
from evaluation.metrics import MetricsCalculator
from core.dynamics.attitude_precache import AttitudePrecacheManager

# 加载任务场景
mission = Mission.load("scenarios/point_group_50sats.json")

# 配置调度器（启用姿态预计算缓存和成像中心优化）
config = {
    'name': 'Greedy-EDD',
    'enable_attitude_precache': True,      # 启用姿态预计算
    'orbit_json_path': 'java/output/frequency_scenario/orbits.json.gz',
    'use_batch_constraints': True,          # 使用批量约束检查
    'consider_frequency': True,             # 考虑频次约束
    'enable_downlink': True,                # 启用数传规划
    'enable_center_distance_score': True,   # 启用成像中心点距离优化
    'center_distance_weight': 15.0,         # 距离评分权重（分/度）
}

# 选择调度算法
scheduler = GreedyScheduler(config)
scheduler.initialize(mission)

# 运行调度
result = scheduler.schedule()

# 验证解的可行性
is_valid = scheduler.validate_solution(result)

# 计算性能指标
metrics_calc = MetricsCalculator(mission)
metrics = metrics_calc.calculate_all(result)

# 输出结果
print(f"需求满足率: {metrics.demand_satisfaction_rate:.2%}")
print(f"完工时间: {metrics.makespan/3600:.2f} 小时")
print(f"求解时间: {metrics.computation_time:.2f} 秒")
print(f"频次满足率: {result.frequency_satisfaction:.2%}")
```

---

---

## 常用命令

### Java后端编译与运行

```bash
# 进入Java目录
cd java/

# 编译所有Java类
make build

# 运行大规模场景测试（使用默认配置）
java -cp "classes:lib/*" orekit.visibility.LargeScaleFrequencyTest

# 指定场景文件和输出目录
java -cp "classes:lib/*" orekit.visibility.LargeScaleFrequencyTest \
    --scenario ../scenarios/my_scenario.json \
    --output output/my_results \
    --orbit-output output/my_results/orbits.json.gz

# 自定义扫描步长
cd java && java -cp "classes:lib/*" orekit.visibility.LargeScaleFrequencyTest \
    --coarse-step 10.0 \
    --fine-step 2.0

# 运行快速测试
java -cp "classes:lib/*" orekit.visibility.QuickFrequencyTest
```

**命令行参数:**

| 参数 | 短选项 | 说明 | 默认值 |
|------|--------|------|--------|
| `--scenario` | `-s` | 场景配置文件路径 | `../scenarios/large_scale_frequency.json` |
| `--output` | `-o` | 输出目录 | `output/frequency_scenario` |
| `--orbit-output` | | 轨道数据输出路径 | `<output>/orbits.json.gz` |
| `--coarse-step` | | 粗扫描步长(秒) | `5.0` |
| `--fine-step` | | 精化步长(秒) | `1.0` |

### 可见性计算

```bash
# Python方式 - 使用Orekit计算可见性
python scripts/compute_visibility.py \
    --scenario scenarios/large_scale_frequency.json \
    --output output/frequency_scenario/

# 计算并导出轨道数据（用于姿态预计算）
python scripts/compute_visibility.py \
    --scenario scenarios/large_scale_frequency.json \
    --output output/frequency_scenario/ \
    --orbit-output output/frequency_scenario/orbits.json.gz

# 计算单颗卫星可见性
python scripts/compute_visibility.py \
    --scenario scenarios/single_satellite.json \
    --sat-id SAT_001
```

**注意**：Java后端默认同时计算卫星-目标窗口和卫星-地面站窗口，无需额外步骤。

### 数据文件管理

```bash
# 下载/更新Orekit数据文件
./scripts/download_orekit_data.sh

# 验证数据完整性
./scripts/download_orekit_data.sh --verify

# 强制重新下载
./scripts/download_orekit_data.sh -f
```

### 场景配置

```bash
# 验证场景文件格式
python -c "from utils.entity.validator import ScenarioValidator; \
           ScenarioValidator().validate_file('scenarios/my_scenario.json')"

# 使用CLI工具（如已安装）
python -m utils.entity.cli validate scenarios/my_scenario.json
```

---

## 算法对比实验

### 批量运行

```bash
# 运行基准测试（多次重复以获得统计显著性）
python -m experiments.runner \
    --scenario scenarios/benchmark_set.json \
    --algorithms greedy,edd,spt,ga,sa,aco,pso,tabu \
    --repetitions 10 \
    --output results/benchmark/
```

### 实验流程

完整的实验流程包括6个核心步骤：

1. **场景配置** - 定义卫星、载荷、目标、地面站
2. **可见性计算** - 预计算卫星-目标访问窗口
3. **算法执行** - 运行调度算法生成方案
4. **结果验证** - 检查约束满足情况，分析失败原因
5. **性能评估** - 计算需求满足率、完工时间、利用率等指标
6. **批量对比** - 多算法、多场景的系统性比较

详细流程文档见：[docs/README.md](docs/最新的实验流程.md)

---

## 性能指标

| 指标 | 说明 | 计算公式 |
|-----|------|---------|
| **DSR** | 需求满足率 | Scheduled / Total |
| **Makespan** | 完工时间 | max(CompletionTime) |
| **Utilization** | 卫星利用率 | ImagingTime / AvailableTime |
| **Comp Time** | 计算时间 | AlgorithmExecutionTime |
| **Quality** | 综合质量 | 加权组合指标 |

---

## 文档

- [实验流程文档](docs/最新的实验流程.md) - 7大核心实验流程详细说明（含Docker容器化）
- [架构设计文档](docs/原始设计.md) - 系统设计、模块接口、算法框架
- [CLAUDE.md](CLAUDE.md) - 项目架构记忆文件（关键配置、性能基准、使用命令）

---

## 开发状态

- [x] 系统架构设计
- [x] 核心模块实现（卫星/目标/地面站建模）
- [x] **高性能可见性计算**（Java Orekit后端，80秒处理60卫星×1000目标×24h）
- [x] **智能轨道初始化器**（SGP4/J4/HPOP自适应选择，性能提升170倍）
- [x] **姿态预计算缓存**（O(1)查询，调度加速5.1倍）
- [x] **批量约束检查**（Numba向量化，约束检查加速5-10倍）
- [x] **精确姿态机动模型**（刚体动力学，时间最优轨迹）
- [x] **成像中心点距离优化**（非聚类任务偏差降低8.3%）
- [x] 轨道传播与可见性计算（SGP4 / STK HPOP / Orekit）
- [x] 地面站窗口计算（Java后端默认计算）
- [x] 基础调度算法（贪心、EDD、SPT）
- [x] 元启发式算法（GA、SA、ACO、PSO、Tabu）
- [x] 离散事件仿真引擎
- [x] 约束验证与失败原因追踪
- [x] 性能评估指标
- [x] 实验框架与批量测试
- [x] 可视化工具（甘特图、轨迹图）
- [x] 单元测试与集成测试（80%+ 覆盖率）
- [x] EGM2008高精度重力场支持（90x90阶）

---

## 测试

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_api/ -v
python -m pytest tests/test_config/ -v
python -m pytest tests/test_integration/ -v

# 生成覆盖率报告
python -m pytest tests/ --cov=missionplanalgo --cov-report=html
```

### 测试结构

```
tests/
├── test_api/              # API 测试
│   ├── test_basic.py      # 基础功能测试
│   └── test_errors.py     # 错误处理测试
├── test_config/           # 配置测试
│   └── test_validation.py # 配置验证测试
├── test_cli/              # CLI 测试
│   └── test_main.py       # 主命令测试
├── test_server/           # 服务器测试
│   └── test_backends.py   # 任务后端测试
├── test_integration/      # 集成测试
│   └── test_workflow.py   # 端到端工作流测试
└── unit/                  # 单元测试
```

---

## 打包与分发

### 打包为 Wheel

```bash
# 安装构建工具
pip install build

# 构建 wheel 和 sdist
python -m build

# 输出文件
# dist/missionplanalgo-1.0.0-py3-none-any.whl
# dist/missionplanalgo-1.0.0.tar.gz
```

### 离线安装

```bash
# 在联网机器上打包
python -m build

# 复制 dist/ 目录到离线机器
scp -r dist/ user@offline-server:/tmp/

# 离线安装
pip install /tmp/dist/missionplanalgo-1.0.0-py3-none-any.whl --no-index --find-links /tmp/dist/
```

### 依赖管理

```bash
# 导出依赖
pip freeze > requirements.txt

# 离线安装依赖
pip download -r requirements.txt -d packages/
pip install --no-index --find-links packages/ -r requirements.txt
```

---

## 常见问题

### Q: Git LFS 文件拉取失败怎么办？

**A:** 如果使用 `git clone` 后，Java 库文件或输出数据文件显示为指针文件（1-2KB），说明 Git LFS 文件未正确下载。

**解决方案：**

```bash
# 确保已安装并初始化 Git LFS
git lfs install

# 拉取所有 LFS 文件
git lfs fetch
git lfs checkout

# 或克隆时直接拉取 LFS 文件
git lfs clone <repository-url>
```

**常见问题：**

1. **Git LFS 未安装**：运行 `git lfs install` 初始化
2. **网络超时**：使用 SSH 保活参数：
   ```bash
   GIT_SSH_COMMAND="ssh -o ServerAliveInterval=60" git lfs fetch
   ```
3. **配额超限**：GitHub LFS 免费额度为 1GB/月，本项目依赖库约 30MB

### Q: 下载脚本失败怎么办？

**A:** 检查以下几点：

1. **网络连接**：确保能访问外网，特别是 `earth-info.nga.mil` 和 `repo1.maven.org`
2. **磁盘空间**：确保有至少 500MB 可用空间
3. **依赖安装**：确保已安装 `curl` 或 `wget`，以及 `unzip`

如果仍失败，使用手动安装步骤（见上文）。

### Q: EGM2008 和 EGM96 有什么区别？

**A:** EGM2008 是更高精度的地球重力场模型：

| 特性 | EGM96 | EGM2008 | 提升 |
|------|-------|---------|------|
| 最大阶数 | 360 | 2190 | 6倍 |
| 空间分辨率 | ~55km | ~5km | 10倍 |
| 大地水准面精度 | ~1米 | ~0.02米 | 50倍 |

本项目配置使用 EGM2008 90x90 阶，相比 EGM96 36x36 能显著提升轨道传播精度。

### Q: 可以离线使用吗？

**A:** 可以。在一台联网机器上运行下载脚本，然后将整个 `~/orekit-data` 目录复制到离线机器。确保设置相同的环境变量：

```bash
export OREKIT_DATA_DIR=/path/to/orekit-data
```

### Q: 如何更新数据文件？

**A:** IERS 地球自转数据需要定期更新（建议每月）：

```bash
# 强制重新下载所有文件
./scripts/download_orekit_data.sh -f

# 或仅验证现有数据
./scripts/download_orekit_data.sh --verify
```

### Q: 为什么需要Java后端？

**A:** Python的Orekit绑定（orekit-python）存在性能和稳定性问题。我们采用Java实现高性能计算核心，通过JNI/JPype与Python交互：

- **计算速度**：Java HPOP传播比Python快10-100倍
- **内存管理**：Java更稳定的内存管理，避免Python的GC问题
- **并行计算**：Java parallelStream实现60颗卫星并行处理
- **数据共享**：通过JSON文件交换数据，避免跨语言对象转换开销

### Q: 如何处理历元时间与场景时间差距大的情况？

**A:** 系统内置智能轨道初始化器，自动选择最优策略：

```
历元距场景时间 < 3天 → 直接使用历元，HPOP传播
历元距场景时间 > 3天 → J4解析外推（避免HPOP长期误差累积）
TLE数据 → SGP4外推到场景开始
```

这解决了历元（如J2000）距场景（如2024年）24年的问题，性能从226分钟提升到80秒。

### Q: Python 包如何离线安装？

**A:** 在有网络的机器上打包，然后离线安装：

```bash
# 1. 在联网机器上打包
python -m build

# 2. 下载依赖包
pip wheel . -w dist/packages/

# 3. 复制到离线机器并安装
pip install dist/missionplanalgo-*.whl --no-index --find-links dist/packages/
```

### Q: 如何验证安装成功？

**A:** 运行以下命令验证：

```bash
# 验证 CLI
mpa --version

# 验证 Python API
python -c "import missionplanalgo as mpa; print(mpa.__version__)"

# 运行测试
python -m pytest tests/test_api/ tests/test_config/ -v
```

### Q: 卫星物理参数如何配置？

**A:** 在场景配置文件中为每颗卫星指定：

```json
{
  "satellites": [{
    "id": "SAT_001",
    "mass": 100.0,           // kg，默认：光学100，SAR150
    "dragArea": 5.0,         // m²，默认：光学5，SAR8
    "reflectivity": 1.5,     // 无单位，默认：光学1.5，SAR1.3
    "dragCoefficient": 2.2   // 无单位，默认：2.2
  }]
}
```

缺失参数将使用默认值。

### Q: 如何启用姿态预计算缓存？

**A:** 在调度器配置中启用：

```python
config = {
    'enable_attitude_precache': True,      # 启用姿态预计算缓存
    'orbit_json_path': 'java/output/frequency_scenario/orbits.json.gz'
}
scheduler = GreedyScheduler(config)
```

**适用场景**：
- ✅ 内存充足（>1GB可用）
- ✅ 重复调度相同场景
- ✅ 实时性要求高

**不适用场景**：
- ❌ 内存受限环境
- ❌ 一次性调度任务
- ❌ 动态变化场景

### Q: 批量约束检查如何工作？

**A:** 所有调度器默认使用批量约束检查器，无需额外配置：

```python
# 自动使用批量检查器
scheduler = GreedyScheduler()  # 默认启用

# 手动批量检查
from scheduler.constraints import BatchSlewConstraintChecker
checker = BatchSlewConstraintChecker(mission)
results = checker.check_slew_feasibility_batch(candidates)
```

### Q: 成像中心点距离优化是什么？

**A:** 在任务调度评分中引入成像中心点与目标坐标的距离因子，使非聚类任务的成像中心更靠近目标坐标。

**启用方式**：
```python
config = {
    'enable_center_distance_score': True,   # 启用优化
    'center_distance_weight': 15.0,         # 评分权重（分/度）
}
scheduler = GreedyScheduler(config)
```

**优化效果**：
- 平均偏差降低8.3%（6.51° → 5.97°）
- 最小偏差降低95.8%（3.80° → 0.16°）
- 9.5%的任务实现高精度成像（偏差 < 2°）

**注意事项**：
- 仅对非聚类任务启用（聚类任务覆盖多目标）
- 权重范围建议：10-25（默认15）
- 过高权重可能牺牲任务覆盖率

### Q: 为什么地面站窗口计算从Java后端输出？

**A:** 为了保证地面站窗口使用与目标窗口相同的高精度HPOP轨道数据：

- Java后端默认同时计算卫星-目标窗口和卫星-地面站窗口
- 地面站窗口标识：`GS:` 前缀（如 `GS:GS-BEIJING`）
- 统一使用EGM2008 90x90重力场模型
- 调度器通过 `load_window_cache_from_json()` 自动加载

---

## 许可证

MIT License
