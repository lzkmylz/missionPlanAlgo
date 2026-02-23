# 卫星星座任务规划算法研究平台

基于SpaceSim模块化设计思想的大规模遥感卫星星座任务规划算法研究与实验平台。

## 项目概述

本项目面向大规模遥感卫星星座任务规划算法研究，支持：

- **50颗异构卫星**：光学1型、光学2型、SAR-1型、SAR-2型
- **两类目标**：点群目标、大区域目标
- **多成像模式**：聚束、滑动聚束、条带（SAR）；推扫、框幅（光学）
- **数传站资源**：多地理位置、多天线、单天线单任务约束
- **算法对比**：贪心、EDD、SPT、GA、SA、ACO、PSO、禁忌搜索 + 自定义新算法

## 架构设计

```
missionPlanAlgo/
├── core/                    # 核心层（装备建模+轨道）
│   ├── models/             # 实体模型定义
│   │   ├── satellite.py    # 卫星模型
│   │   ├── target.py       # 目标模型
│   │   ├── ground_station.py # 地面站模型
│   │   └── mission.py      # 任务场景
│   ├── orbit/              # 轨道模块
│   │   ├── propagator/     # 轨道传播器
│   │   └── visibility/     # 可见性分析
│   └── resources/          # 资源管理
│
├── scheduler/              # 调度模块（研究核心）
│   ├── base_scheduler.py   # 调度器基类
│   ├── greedy/             # 启发式算法
│   └── metaheuristic/      # 元启发式算法
│
├── simulator/              # 仿真引擎
├── evaluation/             # 评估模块
├── visualization/          # 可视化
└── experiments/            # 实验管理
```

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 安装项目（开发模式）
pip install -e .
```

## 快速开始

```bash
# 使用贪心算法运行点群场景
python main.py --scenario scenarios/point_group_scenario.yaml --algorithm Greedy

# 使用遗传算法并生成可视化
python main.py --scenario scenarios/point_group_scenario.yaml --algorithm GA --visualize

# 查看帮助
python main.py --help
```

## 性能指标

平台支持以下性能指标：

| 指标名称 | 符号 | 计算方式 | 用途 |
|---------|------|---------|------|
| 需求满足率 | DSR | 成功调度任务数/总任务数 | 衡量算法完成任务能力 |
| 全部完成用时 | Makespan | 最后一个任务完成时间 | 衡量整体效率 |
| 平均观测间隔 | ARI | 同一目标多次观测的平均间隔 | 衡量重访性能 |
| 数据回传用时 | DDT | 观测完成到数据下传完成时间 | 衡量时效性 |
| 算法求解用时 | CT | 算法运行时间 | 衡量计算效率 |
| 解质量 | SQ | 与理论下界或最优解的差距 | 衡量解的最优性 |

## 核心特性

### 1. 即插即用算法接口

所有调度算法必须继承 `BaseScheduler` 基类，实现统一的 `schedule()` 接口：

```python
class MyScheduler(BaseScheduler):
    def schedule(self) -> ScheduleResult:
        # 实现调度逻辑
        pass
```

### 2. 时间窗口缓存

可见性计算在实验初始化阶段预计算并缓存到内存，算法迭代时只做O(1)的内存访问：

| 方案 | 单次访问延迟 | 10万次迭代耗时 |
|------|-------------|---------------|
| SGP4实时计算 | 10-50ms | 数十分钟 |
| **内存缓存** | **<1μs** | **<1秒** |

### 3. 双后端可见性计算

支持SGP4和STK HPOP两种轨道传播后端，自动检测并切换。

## 文档

- [架构设计文档](docs/原始设计.md) - 详细设计文档
- [API文档](docs/api.md) - API参考
- [使用指南](docs/guide.md) - 详细使用说明

## 参考

1. 魏承, 乔彬, 刘天喜, 等. 航天器系统仿真软件SpaceSim设计与应用[J]. 宇航学报, 2024, 45(11): 1724-1731
2. SpaceSim官方文档: https://spacesim-ori.readthedocs.io/

## License

MIT License
