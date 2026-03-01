# 大规模场景算法对比实验报告
**场景**: scenarios/large_scale_experiment.json
**生成时间**: 2026-03-01T08:42:53.471052
**重复次数**: 1

## 算法对比

| 算法 | 任务完成数 | 需求满足率 | 卫星利用率 | 计算时间 |
|------|-----------|-----------|-----------|---------|
| Greedy-MaxVal | 450 | 45.0% | 2.7% | 4.59s |
| GA | 450 | 45.0% | 3.7% | 749.22s |
| SA | 450 | 45.0% | 3.5% | 20.56s |
| FCFS | 435 | 43.5% | 2.7% | 3.17s |
| Greedy-EDF | 435 | 43.5% | 2.7% | 3.38s |

## 详细统计

### Greedy-MaxVal

- scheduled_count: 450.0000 (±0.0000)
- demand_satisfaction_rate: 0.4500 (±0.0000)
- makespan_hours: 10.4173 (±0.0000)
- satellite_utilization: 0.0274 (±0.0000)
- computation_time: 4.5914 (±0.0000)
- solution_quality: 0.4964 (±0.0000)

### GA

- scheduled_count: 450.0000 (±0.0000)
- demand_satisfaction_rate: 0.4500 (±0.0000)
- makespan_hours: 23.9862 (±0.0000)
- satellite_utilization: 0.0367 (±0.0000)
- computation_time: 749.2169 (±0.0000)
- solution_quality: 0.2702 (±0.0000)

### SA

- scheduled_count: 450.0000 (±0.0000)
- demand_satisfaction_rate: 0.4500 (±0.0000)
- makespan_hours: 23.9036 (±0.0000)
- satellite_utilization: 0.0351 (±0.0000)
- computation_time: 20.5582 (±0.0000)
- solution_quality: 0.2716 (±0.0000)

### FCFS

- scheduled_count: 435.0000 (±0.0000)
- demand_satisfaction_rate: 0.4350 (±0.0000)
- makespan_hours: 23.9882 (±0.0000)
- satellite_utilization: 0.0266 (±0.0000)
- computation_time: 3.1750 (±0.0000)
- solution_quality: 0.2612 (±0.0000)

### Greedy-EDF

- scheduled_count: 435.0000 (±0.0000)
- demand_satisfaction_rate: 0.4350 (±0.0000)
- makespan_hours: 23.9882 (±0.0000)
- satellite_utilization: 0.0266 (±0.0000)
- computation_time: 3.3805 (±0.0000)
- solution_quality: 0.2612 (±0.0000)

