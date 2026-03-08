# Scripts 使用指南

本文档介绍 `scripts` 目录下的所有脚本及其使用方法。

---

## 快速开始

### 1. 生成场景

```bash
# 生成基础场景
python scripts/generate_scenario.py

# 生成带频次约束的场景
python scripts/generate_scenario.py --frequency

# 指定输出路径
python scripts/generate_scenario.py -o scenarios/my_scenario.json
```

### 2. 计算可见性

```bash
# 批量计算模式 (推荐)
python scripts/compute_visibility.py -s scenarios/large_scale_frequency.json

# 逐对计算模式 (较慢但更详细)
python scripts/compute_visibility.py -s scenario.json --mode pairwise

# 自定义扫描步长
python scripts/compute_visibility.py -s scenario.json --coarse-step 10 --fine-step 2
```

### 3. 运行调度

```bash
# 单一算法模式 (默认启用频次和数传)
python scripts/run_scheduler.py -c cache.json -s scenario.json -a greedy

# 多算法对比
python scripts/run_scheduler.py -c cache.json -s scenario.json --mode compare -a greedy,ga,edd

# 禁用频次需求
python scripts/run_scheduler.py -c cache.json -s scenario.json --no-frequency

# 禁用数传规划
python scripts/run_scheduler.py -c cache.json -s scenario.json --no-downlink

# GA算法指定参数
python scripts/run_scheduler.py -c cache.json -s scenario.json -a ga --generations 200
```

### 4. 基准测试

```bash
# 测试所有算法
python scripts/benchmark.py

# 快速测试 (仅基础算法: greedy, edd, ga)
python scripts/benchmark.py --quick

# 只测试特定算法
python scripts/benchmark.py --algorithms greedy edd

# 指定GA参数
python scripts/benchmark.py --ga-generations 200 --ga-population 100

# 禁用数传规划
python scripts/benchmark.py --no-downlink
```

### 5. 查看调度结果

```bash
# 查看算法结果
python scripts/view_schedule_tasks.py --algorithm edd

# 筛选特定目标
python scripts/view_schedule_tasks.py --algorithm edd --target TGT-0001

# 筛选特定卫星
python scripts/view_schedule_tasks.py --algorithm edd --satellite OPT-01

# 显示统计信息
python scripts/view_schedule_tasks.py --algorithm edd --stats

# 导出到CSV
python scripts/view_schedule_tasks.py --algorithm edd --output tasks.csv
```

---

## 脚本说明

### 核心脚本

| 脚本 | 用途 | 状态 |
|------|------|------|
| `generate_scenario.py` | 生成实验场景 | 活跃 |
| `compute_visibility.py` | 计算可见性窗口 | 活跃 |
| `run_scheduler.py` | 运行调度算法 | 活跃 |
| `benchmark.py` | 批量测试算法性能 | 活跃 |
| `view_schedule_tasks.py` | 查看调度结果详情 | 活跃 |

### 配置和工具

| 脚本 | 用途 | 状态 |
|------|------|------|
| `utils.py` | 公共工具函数 | 活跃 |
| `config.py` | 统一配置管理 | 活跃 |


---

## 完整工作流示例

```bash
# 1. 生成带频次约束的场景
python scripts/generate_scenario.py --frequency -o scenarios/frequency_scenario.json

# 2. 计算可见性窗口
python scripts/compute_visibility.py \
    -s scenarios/frequency_scenario.json \
    -o results/visibility_cache.json

# 3. 运行多算法对比
python scripts/run_scheduler.py \
    -c results/visibility_cache.json \
    -s scenarios/frequency_scenario.json \
    --mode compare \
    -a greedy,edd,ga \
    -o results/comparison.json

# 4. 运行完整基准测试
python scripts/benchmark.py \
    --scenario scenarios/frequency_scenario.json \
    --cache results/visibility_cache.json \
    --output results/benchmark \
    --algorithms all

# 5. 查看最佳算法的结果
python scripts/view_schedule_tasks.py --algorithm ga --stats
```

---

## 配置模块使用

其他脚本可以通过 `scripts.config` 模块获取配置：

```python
from scripts.config import get_algorithm_config, get_algorithm_name

# 获取GA算法配置
config = get_algorithm_config(
    algorithm='ga',
    enable_downlink=True,
    enable_frequency=True,
    seed=42
)

# 获取算法显示名称
name = get_algorithm_name('ga')  # 返回: "GA (遗传算法)"
```

---

## 工具模块使用

```python
from scripts.utils import (
    load_window_cache_from_json,
    setup_logging,
    save_results,
    format_metrics_table
)

# 加载窗口缓存
cache = load_window_cache_from_json('cache.json', mission)

# 设置日志
logger = setup_logging()

# 保存结果
save_results(data, 'output.json')

# 格式化结果表格
table = format_metrics_table(results)
print(table)
```

---

## 重构说明

本次重构整合了原 `deprecated/` 目录下的 13 个脚本，将其功能合并到统一的脚本中：

| 原脚本 | 新脚本 | 说明 |
|--------|--------|------|
| `compute_large_scale_visibility.py` | `compute_visibility.py` | 功能合并 |
| `compute_large_scale_visibility_parallel.py` | `compute_visibility.py` | 功能合并 |
| `generate_large_scale_scenario.py` | `generate_scenario.py` | 功能合并 |
| `generate_scenario_with_frequency.py` | `generate_scenario.py` | 功能合并 |
| `run_all_schedulers.py` | `run_scheduler.py` | 功能合并 |
| `run_frequency_comparison.py` | `run_scheduler.py` | 功能合并 |
| `run_large_scale_comparison.py` | `run_scheduler.py` | 功能合并 |
| `run_scheduler_with_cache.py` | `run_scheduler.py` | 功能合并 |
| `run_scheduler_with_frequency.py` | `run_scheduler.py` | 功能合并 |
| `benchmark_all_algorithms.py` | `benchmark.py` | 功能合并 |
| `benchmark_all_algorithms_complete.py` | `benchmark.py` | 功能合并 |
| `run_unified_scheduler.py` | `run_scheduler.py` | 功能合并 |
| `test_performance_simple.py` | - | 删除 |
| `test_visibility_performance.py` | - | 删除 |
| `update_scenario_with_observation_requirements.py` | - | 删除 |
