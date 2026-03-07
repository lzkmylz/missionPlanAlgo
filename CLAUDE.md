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
    └─ 精化扫描 (1秒步长)
    │
    ▼
输出文件
    ├─ visibility_windows.json (318,312窗口)
    ├─ satellites.json (轨道数据)
    └─ ground_stations.json
```

---

## 关键配置参数

### 时间阈值
| 参数 | 值 | 说明 |
|------|-----|------|
| DIRECT_HPOP_THRESHOLD_DAYS | 3.0 | 历元距场景<3天直接HPOP，>3天用J4外推 |
| 粗扫描步长 | 5.0s | 快速定位窗口 |
| 精化扫描步长 | 1.0s | 精确窗口边界 |
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

## 使用命令

```bash
# 编译
cd java && make build

# 运行大规模场景测试
cd java && java -cp "classes:lib/*" orekit.visibility.LargeScaleFrequencyTest

# 场景文件路径
./scenarios/large_scale_frequency.json

# 输出目录
./output/frequency_scenario/
```

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

## 历史关键修复

### 2026-03-06: 智能轨道初始化器
**问题**: 历元(J2000)距场景时间(2024)24年，HPOP传播耗时226+分钟
**解决**: 实现SmartOrbitInitializer
- TLE → SGP4外推
- 历元>3天 → J4解析外推
- 历元<3天 → Direct HPOP
**结果**: 性能提升170倍 (226分钟 → 80秒)
