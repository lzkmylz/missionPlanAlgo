# Clustering Metrics Module

Phase 5 implementation - Quality Metrics for Target Clustering

## Overview

This module provides comprehensive quality evaluation metrics for cluster-aware scheduling, enabling comparison between clustering-based and traditional individual target scheduling approaches.

## Features

### 1. Efficiency Metrics (`ClusteringEfficiencyMetrics`)

Measures the efficiency gains from clustering:

- **Task Reduction Ratio**: Percentage of tasks saved by clustering (0-1)
- **Task Reduction Count**: Absolute number of tasks saved
- **Time Savings**: Estimated time saved in seconds
- **Average Targets per Task**: Mean targets covered per cluster task
- **Max Targets in Single Task**: Largest cluster size
- **Cluster Utilization Ratio**: Percentage of targets in clusters (0-1)

### 2. Coverage Metrics (`ClusteringCoverageMetrics`)

Measures the coverage effectiveness:

- **Target Coverage Ratio**: Percentage of targets covered (0-1)
- **Targets Covered/Total**: Absolute numbers
- **High Priority Coverage**: Coverage of priority >= 8 targets (0-1)
- **High Priority Covered/Total**: Absolute numbers
- **Area Coverage**: Total area covered in km²

### 3. Quality Scoring (`ClusteringQualityScore`)

Comprehensive quality scoring with weighted components:

- **Overall Score**: Weighted average (0-100)
- **Efficiency Score**: Based on task reduction (0-100)
- **Coverage Score**: Based on target coverage (0-100)
- **Priority Score**: Based on high-priority coverage (0-100)
- **Balance Score**: Based on cluster distribution (0-100)

Scoring weights:
- Efficiency: 30%
- Coverage: 25%
- Priority: 30%
- Balance: 15%

### 4. Comparison Analysis

Compares clustering vs traditional scheduling:

- Traditional task count
- Clustering task count
- Improvement ratio
- Time saved (minutes)
- Fuel saved estimate (kg)

### 5. Visualization Support (`ClusteringVisualizer`)

Prepares data for visualization tools:

- **Cluster Map Data**: Clusters with centers, targets, colors
- **Coverage Heatmap Data**: Points with intensity values
- **Efficiency Chart Data**: Comparison data for charts

## Usage

```python
from scheduler.clustering_greedy_scheduler import ClusteringGreedyScheduler
from scheduler.metrics.clustering_metrics import (
    ClusteringMetricsCollector,
    ClusteringVisualizer,
)

# Run scheduler
scheduler = ClusteringGreedyScheduler(config)
result = scheduler.schedule()

# Collect metrics
collector = ClusteringMetricsCollector(scheduler)

efficiency = collector.collect_efficiency_metrics()
coverage = collector.collect_coverage_metrics()
score = collector.calculate_quality_score()
comparison = collector.compare_with_traditional()
report = collector.generate_report()

# Prepare visualization data
visualizer = ClusteringVisualizer(scheduler)
map_data = visualizer.prepare_cluster_map_data()
heatmap_data = visualizer.prepare_coverage_heatmap_data()
chart_data = visualizer.prepare_efficiency_chart_data()
```

## Test Coverage

- **Unit Tests**: 33 tests covering all metrics, edge cases
- **Integration Tests**: 5 tests verifying scheduler integration
- **Total Tests**: 38 tests
- **Method Coverage**: 92.3% (12/13 methods)

### Test Files

- `/tests/unit/scheduler/metrics/test_clustering_metrics.py`
- `/tests/integration/test_clustering_metrics_integration.py`

## Implementation Details

### File Structure

```
scheduler/metrics/
├── __init__.py              # Module exports
├── clustering_metrics.py    # Main implementation
└── README.md               # This file
```

### Key Classes

1. **ClusteringMetricsCollector**: Collects and calculates all metrics
2. **ClusteringVisualizer**: Prepares data for visualization
3. **ClusteringEfficiencyMetrics**: Dataclass for efficiency metrics
4. **ClusteringCoverageMetrics**: Dataclass for coverage metrics
5. **ClusteringQualityScore**: Dataclass for quality scores

### Scoring Formulas

**Efficiency Score**:
```
score = task_reduction_ratio * 100
if avg_targets_per_task > 2:
    score += 10
```

**Coverage Score**:
```
score = target_coverage_ratio * 100
if high_priority_coverage > 0.9:
    score += 5
```

**Priority Score**:
```
score = high_priority_coverage * 100
if high_priority_coverage < 0.8:
    score -= 10
if high_priority_coverage == 1.0:
    score += 5
```

**Balance Score**:
Based on cluster size distribution (optimal: 2-8 targets per cluster)

**Overall Score**:
```
overall = efficiency*0.30 + coverage*0.25 + priority*0.30 + balance*0.15
```

## Example Output

```
Efficiency Metrics:
  Task reduction ratio: 50.00%
  Task reduction count: 5
  Time savings: 750 seconds
  Avg targets per task: 3.5
  Max targets in single task: 4

Coverage Metrics:
  Target coverage ratio: 70.00%
  High priority coverage: 40.00%
  Area coverage: 7.0 km²

Quality Score:
  Overall: 59.5
  Efficiency: 60.0
  Coverage: 70.0
  Priority: 30.0
  Balance: 100.0

Comparison with Traditional:
  Traditional tasks: 10
  Clustering tasks: 5
  Improvement: 50.00%
  Time saved: 15.0 minutes
  Fuel saved: 6.250 kg
```
