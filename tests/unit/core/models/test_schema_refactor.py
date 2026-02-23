"""
测试 database_schema 拆分后的向后兼容性

验证所有原有导入路径仍然有效
"""

import pytest
from typing import List


class TestBackwardCompatibility:
    """测试向后兼容的导入路径"""

    def test_import_all_from_original_path(self):
        """测试从原始路径导入所有类"""
        from core.models.database_schema import (
            # Enums
            ColumnType,
            ConstraintType,
            TableCategory,
            # Base classes
            ColumnDefinition,
            IndexDefinition,
            TableDefinition,
            # Config tables
            SatellitesTable,
            GroundStationsTable,
            AntennasTable,
            TargetsTable,
            ScenariosTable,
            # Scenario tables
            ScenarioSatellitesTable,
            ScenarioTargetsTable,
            ScenarioGroundStationsTable,
            # Decomposition tables
            DecompositionConfigsTable,
            SubTargetsTable,
            # Experiment tables
            ExperimentsTable,
            ExperimentRunsTable,
            AlgorithmParamsTable,
            # Result tables
            ScheduleTasksTable,
            PerformanceMetricsTable,
            TaskSequencesTable,
            # State tables
            SatelliteStateSnapshotsTable,
            ConstraintViolationsTable,
            # Network tables
            ISLWindowsTable,
            DataRoutingPathsTable,
            UplinkCommandsTable,
            RelaySatellitesTable,
            # Manager
            DatabaseSchema,
        )

        # 验证所有导入都是有效的类/枚举
        from enum import Enum
        assert issubclass(ColumnType, Enum)
        assert issubclass(ConstraintType, Enum)
        assert issubclass(TableCategory, Enum)
        assert issubclass(DatabaseSchema, object)

    def test_column_type_enum_values(self):
        """测试 ColumnType 枚举值"""
        from core.models.database_schema import ColumnType

        assert ColumnType.VARCHAR.value == "varchar"
        assert ColumnType.INT.value == "int"
        assert ColumnType.JSON.value == "json"

    def test_constraint_type_enum_values(self):
        """测试 ConstraintType 枚举值"""
        from core.models.database_schema import ConstraintType

        assert ConstraintType.PRIMARY_KEY.value == "primary_key"
        assert ConstraintType.FOREIGN_KEY.value == "foreign_key"
        assert ConstraintType.UNIQUE.value == "unique"

    def test_table_category_enum_values(self):
        """测试 TableCategory 枚举值"""
        from core.models.database_schema import TableCategory

        assert TableCategory.CONFIG.value == "config"
        assert TableCategory.SCENARIO.value == "scenario"
        assert TableCategory.RESULT.value == "result"

    def test_column_definition_creation(self):
        """测试 ColumnDefinition 创建"""
        from core.models.database_schema import ColumnDefinition, ColumnType

        col = ColumnDefinition(
            name="test_id",
            col_type=ColumnType.INT,
            nullable=False,
            comment="测试列"
        )
        assert col.name == "test_id"
        assert col.col_type == ColumnType.INT
        assert col.nullable is False

    def test_index_definition_creation(self):
        """测试 IndexDefinition 创建"""
        from core.models.database_schema import IndexDefinition, ConstraintType

        idx = IndexDefinition(
            name="pk_test",
            constraint_type=ConstraintType.PRIMARY_KEY,
            columns=["id"]
        )
        assert idx.name == "pk_test"
        assert idx.constraint_type == ConstraintType.PRIMARY_KEY
        assert idx.columns == ["id"]

    def test_table_definition_creation(self):
        """测试 TableDefinition 子类创建"""
        from core.models.database_schema import SatellitesTable

        table = SatellitesTable()
        assert table.name == "satellites"
        assert len(table.columns) > 0
        assert len(table.constraints) > 0

    def test_database_schema_manager(self):
        """测试 DatabaseSchema 管理器"""
        from core.models.database_schema import DatabaseSchema, TableCategory

        schema = DatabaseSchema()

        # 验证所有表都被加载
        assert len(schema.tables) == 22  # 总共22个表 (5+3+2+3+3+2+4)

        # 验证可以获取单个表
        satellites = schema.get_table("satellites")
        assert satellites is not None
        assert satellites.name == "satellites"

        # 验证可以按类别获取表
        config_tables = schema.get_tables_by_category(TableCategory.CONFIG)
        assert len(config_tables) == 5  # 配置层5个表

    def test_sql_generation(self):
        """测试 SQL 生成功能"""
        from core.models.database_schema import DatabaseSchema

        schema = DatabaseSchema()

        # 测试单表 SQL
        sql = schema.generate_create_table_sql("satellites")
        assert "CREATE TABLE" in sql
        assert "satellites" in sql

        # 测试所有表 SQL
        all_sql = schema.generate_all_sql()
        assert len(all_sql) == 22
        assert all("CREATE TABLE" in s for s in all_sql)


class TestNewPackageStructure:
    """测试新的包结构导入"""

    def test_import_from_base(self):
        """测试从 base 模块导入基础类"""
        from enum import Enum
        from core.models.schema.base import (
            ColumnType,
            ConstraintType,
            TableCategory,
            ColumnDefinition,
            IndexDefinition,
            TableDefinition,
        )

        assert issubclass(ColumnType, Enum)
        assert issubclass(TableDefinition, object)

    def test_import_from_config_tables(self):
        """测试从 config_tables 导入配置表"""
        from core.models.schema.config_tables import (
            SatellitesTable,
            GroundStationsTable,
            AntennasTable,
            TargetsTable,
            ScenariosTable,
        )

        table = SatellitesTable()
        assert table.name == "satellites"

    def test_import_from_scenario_tables(self):
        """测试从 scenario_tables 导入场景表"""
        from core.models.schema.scenario_tables import (
            ScenarioSatellitesTable,
            ScenarioTargetsTable,
            ScenarioGroundStationsTable,
        )

        table = ScenarioSatellitesTable()
        assert table.name == "scenario_satellites"

    def test_import_from_decomposition_tables(self):
        """测试从 decomposition_tables 导入分解表"""
        from core.models.schema.decomposition_tables import (
            DecompositionConfigsTable,
            SubTargetsTable,
        )

        table = SubTargetsTable()
        assert table.name == "sub_targets"

    def test_import_from_experiment_tables(self):
        """测试从 experiment_tables 导入实验表"""
        from core.models.schema.experiment_tables import (
            ExperimentsTable,
            ExperimentRunsTable,
            AlgorithmParamsTable,
        )

        table = ExperimentsTable()
        assert table.name == "experiments"

    def test_import_from_result_tables(self):
        """测试从 result_tables 导入结果表"""
        from core.models.schema.result_tables import (
            ScheduleTasksTable,
            PerformanceMetricsTable,
            TaskSequencesTable,
        )

        table = ScheduleTasksTable()
        assert table.name == "schedule_tasks"

    def test_import_from_state_tables(self):
        """测试从 state_tables 导入状态表"""
        from core.models.schema.state_tables import (
            SatelliteStateSnapshotsTable,
            ConstraintViolationsTable,
        )

        table = ConstraintViolationsTable()
        assert table.name == "constraint_violations"

    def test_import_from_network_tables(self):
        """测试从 network_tables 导入网络表"""
        from core.models.schema.network_tables import (
            ISLWindowsTable,
            DataRoutingPathsTable,
            UplinkCommandsTable,
            RelaySatellitesTable,
        )

        table = ISLWindowsTable()
        assert table.name == "isl_windows"

    def test_import_from_schema_package(self):
        """测试从 schema 包导入所有内容"""
        from core.models.schema import (
            DatabaseSchema,
            SatellitesTable,
            ColumnType,
            TableCategory,
        )

        schema = DatabaseSchema()
        assert len(schema.tables) == 22


class TestTableCategories:
    """测试各类表的正确分类"""

    def test_config_tables_category(self):
        """测试配置表分类"""
        from core.models.database_schema import (
            SatellitesTable, GroundStationsTable, AntennasTable,
            TargetsTable, ScenariosTable, TableCategory
        )

        tables = [
            SatellitesTable(),
            GroundStationsTable(),
            AntennasTable(),
            TargetsTable(),
            ScenariosTable(),
        ]

        for table in tables:
            assert table.category == TableCategory.CONFIG, f"{table.name} should be CONFIG"

    def test_scenario_tables_category(self):
        """测试场景表分类"""
        from core.models.database_schema import (
            ScenarioSatellitesTable, ScenarioTargetsTable,
            ScenarioGroundStationsTable, TableCategory
        )

        tables = [
            ScenarioSatellitesTable(),
            ScenarioTargetsTable(),
            ScenarioGroundStationsTable(),
        ]

        for table in tables:
            assert table.category == TableCategory.SCENARIO, f"{table.name} should be SCENARIO"

    def test_result_tables_category(self):
        """测试结果表分类"""
        from core.models.database_schema import (
            ScheduleTasksTable, PerformanceMetricsTable,
            TaskSequencesTable, TableCategory
        )

        tables = [
            ScheduleTasksTable(),
            PerformanceMetricsTable(),
            TaskSequencesTable(),
        ]

        for table in tables:
            assert table.category == TableCategory.RESULT, f"{table.name} should be RESULT"


class TestEdgeCases:
    """测试边界情况"""

    def test_get_nonexistent_table(self):
        """测试获取不存在的表"""
        from core.models.database_schema import DatabaseSchema

        schema = DatabaseSchema()
        result = schema.get_table("nonexistent_table")
        assert result is None

    def test_column_to_sql(self):
        """测试列 SQL 生成"""
        from core.models.database_schema import ColumnDefinition, ColumnType

        col = ColumnDefinition(
            name="id",
            col_type=ColumnType.INT,
            nullable=False,
            auto_increment=True
        )
        sql = col.to_sql()
        assert "`id`" in sql
        assert "INT" in sql
        assert "NOT NULL" in sql
        assert "AUTO_INCREMENT" in sql

    def test_index_to_sql(self):
        """测试索引 SQL 生成"""
        from core.models.database_schema import IndexDefinition, ConstraintType

        idx = IndexDefinition(
            name="pk_test",
            constraint_type=ConstraintType.PRIMARY_KEY,
            columns=["id", "name"]
        )
        sql = idx.to_sql()
        assert "PRIMARY KEY" in sql
        assert "`id`" in sql
        assert "`name`" in sql

    def test_foreign_key_to_sql(self):
        """测试外键 SQL 生成"""
        from core.models.database_schema import IndexDefinition, ConstraintType

        fk = IndexDefinition(
            name="fk_test",
            constraint_type=ConstraintType.FOREIGN_KEY,
            columns=["user_id"],
            reference_table="users",
            reference_columns=["id"],
            on_delete="CASCADE"
        )
        sql = fk.to_sql()
        assert "FOREIGN KEY" in sql
        assert "REFERENCES `users`" in sql
        assert "ON DELETE CASCADE" in sql
