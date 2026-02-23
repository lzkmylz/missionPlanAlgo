"""
数据库模式定义包

完整数据库模式定义，包含21个表的定义
按功能域组织为多个子模块
"""

# 基础定义
from core.models.schema.base import (
    ColumnType,
    ConstraintType,
    TableCategory,
    ColumnDefinition,
    IndexDefinition,
    TableDefinition,
)

# 配置层表 (5个)
from core.models.schema.config_tables import (
    SatellitesTable,
    GroundStationsTable,
    AntennasTable,
    TargetsTable,
    ScenariosTable,
)

# 场景实例层表 (3个)
from core.models.schema.scenario_tables import (
    ScenarioSatellitesTable,
    ScenarioTargetsTable,
    ScenarioGroundStationsTable,
)

# 目标分解层表 (2个)
from core.models.schema.decomposition_tables import (
    DecompositionConfigsTable,
    SubTargetsTable,
)

# 实验层表 (3个)
from core.models.schema.experiment_tables import (
    ExperimentsTable,
    ExperimentRunsTable,
    AlgorithmParamsTable,
)

# 结果层表 (3个)
from core.models.schema.result_tables import (
    ScheduleTasksTable,
    PerformanceMetricsTable,
    TaskSequencesTable,
)

# 状态层表 (2个)
from core.models.schema.state_tables import (
    SatelliteStateSnapshotsTable,
    ConstraintViolationsTable,
)

# 网络层表 (4个)
from core.models.schema.network_tables import (
    ISLWindowsTable,
    DataRoutingPathsTable,
    UplinkCommandsTable,
    RelaySatellitesTable,
)

from typing import Dict, List, Optional


class DatabaseSchema:
    """
    完整数据库模式管理器

    管理所有22个表的定义
    """

    def __init__(self):
        self.tables: Dict[str, TableDefinition] = {}
        self._init_tables()

    def _init_tables(self):
        """初始化所有表定义"""
        # 配置层 (5表)
        self._add_table(SatellitesTable())
        self._add_table(GroundStationsTable())
        self._add_table(AntennasTable())
        self._add_table(TargetsTable())
        self._add_table(ScenariosTable())

        # 场景实例层 (3表)
        self._add_table(ScenarioSatellitesTable())
        self._add_table(ScenarioTargetsTable())
        self._add_table(ScenarioGroundStationsTable())

        # 目标分解层 (2表)
        self._add_table(DecompositionConfigsTable())
        self._add_table(SubTargetsTable())

        # 实验层 (3表)
        self._add_table(ExperimentsTable())
        self._add_table(ExperimentRunsTable())
        self._add_table(AlgorithmParamsTable())

        # 结果层 (3表)
        self._add_table(ScheduleTasksTable())
        self._add_table(PerformanceMetricsTable())
        self._add_table(TaskSequencesTable())

        # 状态层 (2表)
        self._add_table(SatelliteStateSnapshotsTable())
        self._add_table(ConstraintViolationsTable())

        # 网络层 (4表 - 包含UplinkCommandsTable)
        self._add_table(ISLWindowsTable())
        self._add_table(DataRoutingPathsTable())
        self._add_table(UplinkCommandsTable())
        self._add_table(RelaySatellitesTable())

    def _add_table(self, table: TableDefinition):
        """添加表定义"""
        self.tables[table.name] = table

    def get_table(self, name: str) -> Optional[TableDefinition]:
        """获取表定义"""
        return self.tables.get(name)

    def get_tables_by_category(self, category: TableCategory) -> List[TableDefinition]:
        """按类别获取表"""
        return [t for t in self.tables.values() if t.category == category]

    def generate_create_table_sql(self, table_name: str) -> str:
        """生成建表SQL"""
        table = self.get_table(table_name)
        if table:
            return table.to_sql()
        return ""

    def generate_all_sql(self) -> List[str]:
        """生成所有建表SQL"""
        return [table.to_sql() for table in self.tables.values()]


# 为向后兼容，导出所有名称
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
