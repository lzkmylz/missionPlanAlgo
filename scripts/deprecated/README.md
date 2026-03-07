# 已废弃脚本说明

此目录包含已被合并的脚本，保留作为备份参考。

## 合并情况

### 调度脚本 (5个 → 1个)
- `run_scheduler_with_cache.py` → 合并到 `../run_scheduler.py`
- `run_scheduler_with_frequency.py` → 合并到 `../run_scheduler.py`
- `run_all_schedulers.py` → 合并到 `../run_scheduler.py`
- `run_frequency_comparison.py` → 合并到 `../run_scheduler.py`
- `run_large_scale_comparison.py` → 合并到 `../run_scheduler.py`

**新脚本:** `../run_scheduler.py`

使用方式:
```bash
# 单一算法
python scripts/run_scheduler.py -c cache.json -s scenario.json -a greedy

# 多算法对比
python scripts/run_scheduler.py -c cache.json -s scenario.json --mode compare -a greedy,ga,edd

# 启用频次和数传
python scripts/run_scheduler.py -c cache.json -s scenario.json --frequency --downlink
```

---

### 可见性计算脚本 (3个 → 1个)
- `compute_large_scale_visibility.py` → 合并到 `../compute_visibility.py`
- `compute_large_scale_visibility_parallel.py` → 合并到 `../compute_visibility.py`

**新脚本:** `../compute_visibility.py`

使用方式:
```bash
# 批量计算模式 (默认)
python scripts/compute_visibility.py -s scenario.json

# 逐对计算模式
python scripts/compute_visibility.py -s scenario.json --mode pairwise
```

---

### 场景生成脚本 (2个 → 1个)
- `generate_large_scale_scenario.py` → 合并到 `../generate_scenario.py`
- `generate_scenario_with_frequency.py` → 合并到 `../generate_scenario.py`

**新脚本:** `../generate_scenario.py`

使用方式:
```bash
# 基础场景
python scripts/generate_scenario.py

# 带频次约束的场景
python scripts/generate_scenario.py --frequency
```

---

### 其他废弃脚本
- `update_scenario_with_observation_requirements.py` - 功能过时
- `generate_sample_schedule.py` - 功能被统一调度器覆盖
- `test_performance_simple.py` - 功能被对比脚本覆盖
- `test_visibility_performance.py` - 功能被compute_visibility覆盖

---

## 新增公共模块

### `../utils.py`
提取了重复代码:
- `load_window_cache_from_json()` - 支持新旧格式
- `SCHEDULER_REGISTRY` - 统一算法注册表
- `setup_logging()` - 统一日志配置
- `save_results()` - 统一结果保存

---

## 脚本数量变化

| 类别 | 合并前 | 合并后 | 减少 |
|------|--------|--------|------|
| 调度脚本 | 6个 | 1个 | 5个 (83%) |
| 可见性脚本 | 3个 | 1个 | 2个 (67%) |
| 场景生成 | 2个 | 1个 | 1个 (50%) |
| 其他 | 4个 | 0个 | 4个 (100%) |
| **总计** | **15个** | **3个** | **12个 (80%)** |

---

*废弃日期: 2026-03-07*
