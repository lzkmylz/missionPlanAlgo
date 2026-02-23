"""
完整数据库模式定义

实现Chapter 10: 数据持久化与存储设计
包含21个表的完整定义

注意: 此文件现在只是 core.models.schema 包的兼容性包装器
所有实际定义已迁移到 core/models/schema/ 目录下的各个模块
"""

# 从新的包结构导入所有内容，保持向后兼容
from core.models.schema import (
    # 基础类
    ColumnType,
    ConstraintType,
    TableCategory,
    ColumnDefinition,
    IndexDefinition,
    TableDefinition,
    # 配置层表 (5个)
    SatellitesTable,
    GroundStationsTable,
    AntennasTable,
    TargetsTable,
    ScenariosTable,
    # 场景实例层表 (3个)
    ScenarioSatellitesTable,
    ScenarioTargetsTable,
    ScenarioGroundStationsTable,
    # 目标分解层表 (2个)
    DecompositionConfigsTable,
    SubTargetsTable,
    # 实验层表 (3个)
    ExperimentsTable,
    ExperimentRunsTable,
    AlgorithmParamsTable,
    # 结果层表 (3个)
    ScheduleTasksTable,
    PerformanceMetricsTable,
    TaskSequencesTable,
    # 状态层表 (2个)
    SatelliteStateSnapshotsTable,
    ConstraintViolationsTable,
    # 网络层表 (4个)
    ISLWindowsTable,
    DataRoutingPathsTable,
    UplinkCommandsTable,
    RelaySatellitesTable,
    # 数据库模式管理器
    DatabaseSchema,
)

# 导出所有名称
__all__ = [
    # 基础类
    "ColumnType",
    "ConstraintType",
    "TableCategory",
    "ColumnDefinition",
    "IndexDefinition",
    "TableDefinition",
    # 配置层表
    "SatellitesTable",
    "GroundStationsTable",
    "AntennasTable",
    "TargetsTable",
    "ScenariosTable",
    # 场景层表
    "ScenarioSatellitesTable",
    "ScenarioTargetsTable",
    "ScenarioGroundStationsTable",
    # 分解层表
    "DecompositionConfigsTable",
    "SubTargetsTable",
    # 实验层表
    "ExperimentsTable",
    "ExperimentRunsTable",
    "AlgorithmParamsTable",
    # 结果层表
    "ScheduleTasksTable",
    "PerformanceMetricsTable",
    "TaskSequencesTable",
    # 状态层表
    "SatelliteStateSnapshotsTable",
    "ConstraintViolationsTable",
    # 网络层表
    "ISLWindowsTable",
    "DataRoutingPathsTable",
    "UplinkCommandsTable",
    "RelaySatellitesTable",
    # 管理器
    "DatabaseSchema",
]
