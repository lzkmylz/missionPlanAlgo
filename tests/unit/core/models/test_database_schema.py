"""
数据库模式单元测试

测试H1: 完整数据库模式 (Chapter 10)
验证21个表定义
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from core.models.database_schema import (
    DatabaseSchema,
    TableDefinition,
    ColumnDefinition,
    ColumnType,
    ConstraintType,
    IndexDefinition,
    TableCategory,
    # 表定义类
    SatellitesTable,
    GroundStationsTable,
    AntennasTable,
    TargetsTable,
    ScenariosTable,
    ScenarioSatellitesTable,
    ScenarioTargetsTable,
    ScenarioGroundStationsTable,
    DecompositionConfigsTable,
    SubTargetsTable,
    ExperimentsTable,
    ExperimentRunsTable,
    AlgorithmParamsTable,
    ScheduleTasksTable,
    PerformanceMetricsTable,
    TaskSequencesTable,
    SatelliteStateSnapshotsTable,
    ConstraintViolationsTable,
    ISLWindowsTable,
    DataRoutingPathsTable,
    UplinkCommandsTable,
    RelaySatellitesTable,
)


class TestColumnType:
    """测试列类型枚举"""

    def test_column_type_values(self):
        """测试列类型值"""
        assert ColumnType.VARCHAR.value == "varchar"
        assert ColumnType.INT.value == "int"
        assert ColumnType.BIGINT.value == "bigint"
        assert ColumnType.DECIMAL.value == "decimal"
        assert ColumnType.DATETIME.value == "datetime"
        assert ColumnType.TIMESTAMP.value == "timestamp"
        assert ColumnType.BOOLEAN.value == "boolean"
        assert ColumnType.JSON.value == "json"
        assert ColumnType.ENUM.value == "enum"
        assert ColumnType.TEXT.value == "text"


class TestConstraintType:
    """测试约束类型枚举"""

    def test_constraint_type_values(self):
        """测试约束类型值"""
        assert ConstraintType.PRIMARY_KEY.value == "primary_key"
        assert ConstraintType.FOREIGN_KEY.value == "foreign_key"
        assert ConstraintType.UNIQUE.value == "unique"
        assert ConstraintType.NOT_NULL.value == "not_null"
        assert ConstraintType.INDEX.value == "index"


class TestTableCategory:
    """测试表类别枚举"""

    def test_table_category_values(self):
        """测试表类别值"""
        assert TableCategory.CONFIG.value == "config"
        assert TableCategory.SCENARIO.value == "scenario"
        assert TableCategory.DECOMPOSITION.value == "decomposition"
        assert TableCategory.EXPERIMENT.value == "experiment"
        assert TableCategory.RESULT.value == "result"
        assert TableCategory.STATE.value == "state"
        assert TableCategory.NETWORK.value == "network"


class TestColumnDefinition:
    """测试列定义"""

    def test_column_definition_creation(self):
        """测试创建列定义"""
        col = ColumnDefinition(
            name="id",
            col_type=ColumnType.VARCHAR,
            length=32,
            nullable=False,
            default=None,
            comment="主键ID"
        )
        assert col.name == "id"
        assert col.col_type == ColumnType.VARCHAR
        assert col.length == 32
        assert col.nullable is False
        assert col.comment == "主键ID"

    def test_column_to_sql(self):
        """测试生成SQL"""
        col = ColumnDefinition(
            name="id",
            col_type=ColumnType.VARCHAR,
            length=32,
            nullable=False,
            default=None,
            comment="主键ID"
        )
        sql = col.to_sql()
        assert "id" in sql
        assert "VARCHAR(32)" in sql
        assert "NOT NULL" in sql


class TestSatellitesTable:
    """测试卫星表"""

    def test_table_structure(self):
        """测试表结构"""
        table = SatellitesTable()
        assert table.name == "satellites"
        assert table.category == TableCategory.CONFIG
        assert len(table.columns) >= 10

        # 检查必需列
        column_names = [col.name for col in table.columns]
        assert "id" in column_names
        assert "name" in column_names
        assert "sat_type" in column_names
        assert "orbit_type" in column_names
        assert "altitude" in column_names
        assert "inclination" in column_names

    def test_primary_key(self):
        """测试主键约束"""
        table = SatellitesTable()
        pk_constraints = [c for c in table.constraints if c.constraint_type == ConstraintType.PRIMARY_KEY]
        assert len(pk_constraints) == 1
        assert "id" in pk_constraints[0].columns


class TestAntennasTable:
    """测试天线表 - H1关键缺失表"""

    def test_table_structure(self):
        """测试表结构"""
        table = AntennasTable()
        assert table.name == "antennas"
        assert table.category == TableCategory.CONFIG

        # 检查必需列
        column_names = [col.name for col in table.columns]
        assert "id" in column_names
        assert "ground_station_id" in column_names
        assert "name" in column_names
        assert "elevation_min" in column_names
        assert "data_rate" in column_names

    def test_foreign_key(self):
        """测试外键约束"""
        table = AntennasTable()
        fk_constraints = [c for c in table.constraints if c.constraint_type == ConstraintType.FOREIGN_KEY]
        assert len(fk_constraints) >= 1
        # 检查是否关联到ground_stations
        fk = fk_constraints[0]
        assert fk.reference_table == "ground_stations"


class TestDecompositionTables:
    """测试分解相关表 - H1关键缺失表"""

    def test_decomposition_configs_table(self):
        """测试分解配置表"""
        table = DecompositionConfigsTable()
        assert table.name == "decomposition_configs"
        assert table.category == TableCategory.DECOMPOSITION

        column_names = [col.name for col in table.columns]
        assert "id" in column_names
        assert "target_id" in column_names
        assert "scenario_id" in column_names
        assert "strategy" in column_names

    def test_sub_targets_table(self):
        """测试子目标表"""
        table = SubTargetsTable()
        assert table.name == "sub_targets"
        assert table.category == TableCategory.DECOMPOSITION

        column_names = [col.name for col in table.columns]
        assert "id" in column_names
        assert "parent_target_id" in column_names
        assert "scenario_id" in column_names
        assert "sub_target_code" in column_names


class TestSatelliteStateSnapshotsTable:
    """测试卫星状态快照表 - H1关键缺失表"""

    def test_table_structure(self):
        """测试表结构"""
        table = SatelliteStateSnapshotsTable()
        assert table.name == "satellite_state_snapshots"
        assert table.category == TableCategory.STATE

        # 检查必需列
        column_names = [col.name for col in table.columns]
        assert "id" in column_names
        assert "run_id" in column_names
        assert "satellite_id" in column_names
        assert "snapshot_time" in column_names
        assert "power_level" in column_names
        assert "storage_level" in column_names
        assert "is_eclipse" in column_names


class TestNetworkTables:
    """测试网络相关表"""

    def test_isl_windows_table(self):
        """测试ISL窗口表"""
        table = ISLWindowsTable()
        assert table.name == "isl_windows"
        assert table.category == TableCategory.NETWORK

        column_names = [col.name for col in table.columns]
        assert "satellite_a_id" in column_names
        assert "satellite_b_id" in column_names
        assert "start_time" in column_names
        assert "link_quality" in column_names

    def test_data_routing_paths_table(self):
        """测试数据路由路径表"""
        table = DataRoutingPathsTable()
        assert table.name == "data_routing_paths"
        assert table.category == TableCategory.NETWORK

        column_names = [col.name for col in table.columns]
        assert "task_id" in column_names
        assert "route_hops" in column_names
        assert "total_latency" in column_names

    def test_uplink_commands_table(self):
        """测试指令上行记录表"""
        table = UplinkCommandsTable()
        assert table.name == "uplink_commands"
        assert table.category == TableCategory.NETWORK

        column_names = [col.name for col in table.columns]
        assert "satellite_id" in column_names
        assert "command_type" in column_names
        assert "uplink_time" in column_names

    def test_relay_satellites_table(self):
        """测试中继星配置表"""
        table = RelaySatellitesTable()
        assert table.name == "relay_satellites"
        assert table.category == TableCategory.NETWORK

        column_names = [col.name for col in table.columns]
        assert "id" in column_names
        assert "uplink_capacity" in column_names
        assert "downlink_capacity" in column_names
        assert "coverage_zones" in column_names


class TestDatabaseSchema:
    """测试完整数据库模式"""

    def test_all_tables_defined(self):
        """测试所有22个表已定义"""
        schema = DatabaseSchema()
        assert len(schema.tables) == 22

    def test_table_categories(self):
        """测试表分类正确"""
        schema = DatabaseSchema()

        categories = {}
        for table in schema.tables.values():
            cat = table.category
            categories[cat] = categories.get(cat, 0) + 1

        # 验证各类别表数量
        assert categories.get(TableCategory.CONFIG, 0) == 5  # satellites, ground_stations, antennas, targets, scenarios
        assert categories.get(TableCategory.SCENARIO, 0) == 3  # scenario_satellites, scenario_targets, scenario_ground_stations
        assert categories.get(TableCategory.DECOMPOSITION, 0) == 2  # decomposition_configs, sub_targets
        assert categories.get(TableCategory.EXPERIMENT, 0) == 3  # experiments, experiment_runs, algorithm_params
        assert categories.get(TableCategory.RESULT, 0) == 3  # schedule_tasks, performance_metrics, task_sequences
        assert categories.get(TableCategory.STATE, 0) == 2  # satellite_state_snapshots, constraint_violations
        assert categories.get(TableCategory.NETWORK, 0) == 4  # isl_windows, data_routing_paths, uplink_commands, relay_satellites

    def test_get_table(self):
        """测试获取表定义"""
        schema = DatabaseSchema()

        table = schema.get_table("satellites")
        assert table is not None
        assert table.name == "satellites"

        # 获取不存在的表
        assert schema.get_table("non_existent") is None

    def test_get_tables_by_category(self):
        """测试按类别获取表"""
        schema = DatabaseSchema()

        config_tables = schema.get_tables_by_category(TableCategory.CONFIG)
        assert len(config_tables) == 5

        for table in config_tables:
            assert table.category == TableCategory.CONFIG

    def test_generate_create_table_sql(self):
        """测试生成建表SQL"""
        schema = DatabaseSchema()

        sql = schema.generate_create_table_sql("satellites")
        assert "CREATE TABLE" in sql
        assert "satellites" in sql

    def test_generate_all_sql(self):
        """测试生成所有建表SQL"""
        schema = DatabaseSchema()

        all_sql = schema.generate_all_sql()
        assert len(all_sql) == 22

        # 验证包含所有表
        table_names = set()
        for sql in all_sql:
            for line in sql.split('\n'):
                if 'CREATE TABLE' in line:
                    # 提取表名
                    parts = line.split()
                    if len(parts) >= 3:
                        table_names.add(parts[2].strip('`'))

        assert "satellites" in table_names
        assert "antennas" in table_names
        assert "sub_targets" in table_names
        assert "satellite_state_snapshots" in table_names


class TestTableDefinition:
    """测试表定义基类"""

    def test_table_definition_creation(self):
        """测试创建表定义"""
        columns = [
            ColumnDefinition("id", ColumnType.INT, nullable=False),
            ColumnDefinition("name", ColumnType.VARCHAR, length=64, nullable=False),
        ]

        table = TableDefinition(
            name="test_table",
            category=TableCategory.CONFIG,
            columns=columns,
            comment="测试表"
        )

        assert table.name == "test_table"
        assert table.category == TableCategory.CONFIG
        assert len(table.columns) == 2
        assert table.comment == "测试表"

    def test_to_sql(self):
        """测试生成SQL"""
        columns = [
            ColumnDefinition("id", ColumnType.INT, nullable=False),
            ColumnDefinition("name", ColumnType.VARCHAR, length=64, nullable=False),
        ]

        constraints = [
            IndexDefinition(
                name="pk_id",
                constraint_type=ConstraintType.PRIMARY_KEY,
                columns=["id"]
            )
        ]

        table = TableDefinition(
            name="test_table",
            category=TableCategory.CONFIG,
            columns=columns,
            constraints=constraints,
            comment="测试表"
        )

        sql = table.to_sql()
        assert "CREATE TABLE" in sql
        assert "test_table" in sql
        assert "id" in sql
        assert "name" in sql
        assert "PRIMARY KEY" in sql


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
