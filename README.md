# 卫星星座任务规划算法研究平台

面向大规模遥感卫星星座的任务规划算法研究与仿真平台，支撑异构卫星、多类观测目标、多成像模式的调度优化研究。

---

## 核心特性

- **大规模星座支持**：50颗异构卫星（光学1/2型、SAR-1/2型）
- **多类型目标**：点群目标、大区域目标
- **多成像模式**：聚束/滑动聚束/条带（SAR）、推扫/框幅（光学）
- **地面站资源调度**：多地理位置、多天线约束
- **算法即插即用**：统一接口支持贪心、EDD、SPT、GA、SA、ACO、PSO、禁忌搜索等
- **双后端可见性计算**：STK HPOP（高精度）+ Orekit（自主可控）
- **星载边缘计算支持**：在轨数据预处理，减少下传压力
- **完整实验流程**：场景配置 → 可见性计算 → 算法执行 → 结果验证 → 性能评估 → 批量对比

---

## 技术栈

- **语言**: Python 3.10+
- **轨道计算**: SGP4 / STK HPOP / Orekit
- **算法库**: NumPy, SciPy, 自研元启发式框架
- **可视化**: Matplotlib, Plotly
- **数据**: JSON/YAML 配置, HDF5 结果存储
- **测试**: pytest, 覆盖率 80%+

---

## 项目结构

```
missionPlanAlgo/
├── core/                    # 核心层
│   ├── models/             # 实体模型（卫星、目标、地面站、任务）
│   ├── orbit/              # 轨道模块（传播器、可见性计算）
│   ├── resources/          # 资源管理（卫星池、地面站池）
│   ├── decomposer/         # 目标分解（条带、网格）
│   ├── processing/         # 在轨处理管理
│   ├── network/            # 星间链路网络与路由
│   ├── dynamic_scheduler/  # 事件驱动和滚动时域调度
│   ├── validators/         # 约束验证
│   └── telecommand/        # 指令序列生成
├── payload/                # 载荷模块（光学/SAR成像器、成像模式）
├── scheduler/              # 调度算法
│   ├── greedy/            # 启发式算法（Greedy、EDD、SPT）
│   └── metaheuristic/     # 元启发式算法（GA、SA、ACO、PSO、Tabu）
├── simulator/             # 离散事件仿真引擎
│   ├── state_tracker.py   # 实时状态追踪
│   ├── thermal_model.py   # 热控模型
│   ├── schedule_validator.py  # 方案验证
│   └── eclipse_calculator.py  # 地影计算
├── evaluation/            # 性能指标与结果分析
├── experiments/           # 实验管理与基准测试
├── visualization/         # 甘特图、星下点轨迹、覆盖图
├── scenarios/             # JSON/YAML 场景配置文件
├── docs/                  # 架构设计文档与实验流程文档
└── tests/                 # 单元测试与集成测试（80%+ 覆盖率）
```

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

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd missionPlanAlgo

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 运行示例

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

# 加载任务场景
mission = Mission.load("scenarios/point_group_50sats.json")

# 选择调度算法
scheduler = GreedyScheduler({"name": "Greedy-EDD"})
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

---

## 开发状态

- [x] 系统架构设计
- [x] 核心模块实现（卫星/目标/地面站建模）
- [x] 轨道传播与可见性计算
- [x] 基础调度算法（贪心、EDD、SPT）
- [x] 元启发式算法（GA、SA、ACO、PSO、Tabu）
- [x] 离散事件仿真引擎
- [x] 约束验证与失败原因追踪
- [x] 性能评估指标
- [x] 实验框架与批量测试
- [x] 可视化工具（甘特图、轨迹图）
- [x] 单元测试与集成测试（80%+ 覆盖率）

---

## 许可证

MIT License
