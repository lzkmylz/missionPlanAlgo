# 轨道预计算优化方案

## 目标
将Java Orekit的轨道预计算结果以Parquet格式保存，供Python调度器直接加载，消除重复HPOP计算。

## 技术选型

### 引力场模型
- **使用 EGM2008 36x36**（已在orekit-data中可用）
- 数据文件: `/home/lz/orekit-data/potential/egm-format/EGM2008_to2190_TideFree.ascii.gz`

### 数据格式
- **Parquet**（列式存储，高压缩比，快速查询）
- 预期大小: 60颗卫星 × 86401秒 × 9列 ≈ 50-80MB（压缩后）

## Schema设计

```
satellite_id: string      # 卫星标识
 timestamp: int32          # 相对于场景开始的秒数 [0, 86400]
 pos_x: float64            # ECEF X (米)
 pos_y: float64            # ECEF Y (米)
 pos_z: float64            # ECEF Z (米)
 vel_x: float64            # ECEF Vx (米/秒)
 vel_y: float64            # ECEF Vy (米/秒)
 vel_z: float64            # ECEF Vz (米/秒)
 lat: float64              # 地心纬度 (度)
 lon: float64              # 地心经度 (度)
 alt: float64              # 海拔高度 (米)
```

## 实施步骤

### Phase 1: Java端导出功能

#### 1.1 添加Parquet依赖
**文件**: `java/Makefile`
- 下载Apache Parquet Java库
- 添加到classpath

#### 1.2 创建OrbitDataExporter类
**文件**: `java/src/orekit/visibility/OrbitDataExporter.java`

主要方法签名：
```java
public class OrbitDataExporter {
    public void exportToParquet(
        Map<String, List<OrbitState>> orbitCache,
        String outputPath
    ) throws IOException;
}
```

#### 1.3 在OrbitStateCache中添加导出方法
**文件**: `java/src/orekit/visibility/OrbitStateCache.java`

```java
public void exportToParquet(String outputPath) throws IOException {
    OrbitDataExporter exporter = new OrbitDataExporter();
    exporter.exportToParquet(cache, outputPath);
}
```

#### 1.4 集成到LargeScaleFrequencyTest
**文件**: `java/src/orekit/visibility/LargeScaleFrequencyTest.java`

在`persistData`方法后添加轨道数据导出调用。

### Phase 2: Python端加载功能

#### 2.1 添加PyArrow依赖
**文件**: `requirements.txt`
```
pyarrow>=14.0.0
```

#### 2.2 创建OrbitDataLoader类
**文件**: `core/dynamics/orbit_data_loader.py`

```python
import pyarrow.parquet as pq

class OrbitDataLoader:
    """轨道数据加载器"""
    def load_from_parquet(self, filepath: str) -> Dict[str, SatelliteOrbitCache]:
        table = pq.read_table(filepath)
        # 转换为SatelliteOrbitCache字典
```

#### 2.3 扩展OrekitBatchPropagator
**文件**: `core/dynamics/orbit_batch_propagator.py`

添加方法：
```python
def load_precomputed_orbits(self, parquet_path: str) -> bool:
    """从Parquet文件加载预计算轨道数据"""
    
def has_precomputed_data(self, satellite_id: str) -> bool:
    """检查是否有预计算的轨道数据"""
```

#### 2.4 修改precompute_satellite_orbit
添加预计算数据优先检查逻辑。

### Phase 3: 配置和集成

#### 3.1 添加配置选项
**文件**: `core/config.py`

```python
ORBIT_PRECOMPUTE_CONFIG = {
    'enabled': True,
    'parquet_path': 'java/output/frequency_scenario/orbits.parquet',
    'fallback_to_hpop': True,
}
```

#### 3.2 统一调度器集成
**文件**: `scheduler/unified_scheduler.py`

在初始化时自动加载预计算轨道数据。

## 性能预期

| 指标 | 当前 | 优化后 | 提升 |
|------|------|--------|------|
| Python调度初始化 | 10-15分钟 | < 5秒 | **99%** |
| 总调度流程 | 15-20分钟 | 30-60秒 | **95%** |
| 磁盘空间 | 36MB | ~130MB | 增加但可接受 |
| 内存使用 | ~50MB | ~250MB | 增加但可接受 |

## 验证计划

1. **单元测试**: Java导出、Python加载、数据一致性
2. **集成测试**: 端到端流程、性能对比
3. **回归测试**: 所有现有测试通过

## 时间表

- Phase 1 (Java): 1天
- Phase 2 (Python): 1天
- Phase 3 (集成): 0.5天
- **总计: 2.5天**
