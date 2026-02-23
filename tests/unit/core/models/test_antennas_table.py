"""
测试AntennasTable天线配置表

M5: 天线配置表测试
"""

import pytest
from datetime import datetime

from core.models.database_schema import (
    AntennasTable,
    DatabaseSchema,
    ColumnType,
    ConstraintType,
    TableCategory
)


class TestAntennasTable:
    """测试AntennasTable定义"""

    def test_table_exists_in_schema(self):
        """测试antennas表存在于数据库模式中"""
        schema = DatabaseSchema()

        table = schema.get_table("antennas")
        assert table is not None, "antennas table should exist in schema"

    def test_table_category(self):
        """测试表分类正确"""
        table = AntennasTable()

        assert table.category == TableCategory.CONFIG

    def test_table_name(self):
        """测试表名正确"""
        table = AntennasTable()

        assert table.name == "antennas"

    def test_required_columns_exist(self):
        """测试必需的列存在"""
        table = AntennasTable()

        column_names = [col.name for col in table.columns]

        required_columns = [
            "id",
            "ground_station_id",
            "name",
            "elevation_min",
            "elevation_max",
            "data_rate",
            "slew_rate"
        ]

        for col in required_columns:
            assert col in column_names, f"Required column '{col}' should exist"

    def test_id_column_definition(self):
        """测试ID列定义"""
        table = AntennasTable()

        id_col = next(col for col in table.columns if col.name == "id")

        assert id_col.col_type == ColumnType.VARCHAR
        assert id_col.length == 32
        assert id_col.nullable == False

    def test_ground_station_id_column(self):
        """测试地面站ID列"""
        table = AntennasTable()

        gs_col = next(col for col in table.columns if col.name == "ground_station_id")

        assert gs_col.col_type == ColumnType.VARCHAR
        assert gs_col.length == 32
        assert gs_col.nullable == False

    def test_elevation_columns(self):
        """测试仰角列"""
        table = AntennasTable()

        min_col = next(col for col in table.columns if col.name == "elevation_min")
        max_col = next(col for col in table.columns if col.name == "elevation_max")

        assert min_col.col_type == ColumnType.DECIMAL
        assert min_col.precision == 5
        assert min_col.scale == 2
        assert min_col.default == 5.0

        assert max_col.col_type == ColumnType.DECIMAL
        assert max_col.precision == 5
        assert max_col.scale == 2
        assert max_col.default == 90.0

    def test_data_rate_column(self):
        """测试数据速率列"""
        table = AntennasTable()

        rate_col = next(col for col in table.columns if col.name == "data_rate")

        assert rate_col.col_type == ColumnType.INT

    def test_slew_rate_column(self):
        """测试转动速率列"""
        table = AntennasTable()

        slew_col = next(col for col in table.columns if col.name == "slew_rate")

        assert slew_col.col_type == ColumnType.DECIMAL
        assert slew_col.precision == 5
        assert slew_col.scale == 2

    def test_primary_key_constraint(self):
        """测试主键约束"""
        table = AntennasTable()

        pk_constraints = [c for c in table.constraints
                         if c.constraint_type == ConstraintType.PRIMARY_KEY]

        assert len(pk_constraints) == 1
        assert "id" in pk_constraints[0].columns

    def test_foreign_key_constraint(self):
        """测试外键约束"""
        table = AntennasTable()

        fk_constraints = [c for c in table.constraints
                         if c.constraint_type == ConstraintType.FOREIGN_KEY]

        assert len(fk_constraints) >= 1

        # 检查指向ground_stations的外键
        gs_fk = next((c for c in fk_constraints
                     if c.reference_table == "ground_stations"), None)
        assert gs_fk is not None
        assert "ground_station_id" in gs_fk.columns

    def test_index_on_station(self):
        """测试地面站索引"""
        table = AntennasTable()

        index_constraints = [c for c in table.constraints
                            if c.constraint_type == ConstraintType.INDEX]

        station_index = next((c for c in index_constraints
                             if "ground_station_id" in c.columns), None)
        assert station_index is not None

    def test_sql_generation(self):
        """测试SQL生成"""
        table = AntennasTable()

        sql = table.to_sql()

        assert "CREATE TABLE" in sql
        assert "antennas" in sql
        assert "id" in sql
        assert "ground_station_id" in sql

    def test_multiple_antennas_per_station(self):
        """测试支持多天线地面站"""
        # 通过外键设计，一个地面站可以有多个天线
        table = AntennasTable()

        # 验证外键允许重复（即一个ground_station_id可以对应多条记录）
        gs_fk = next(c for c in table.constraints
                    if c.constraint_type == ConstraintType.FOREIGN_KEY
                    and c.reference_table == "ground_stations")

        # 外键不应该有唯一约束
        unique_constraints = [c for c in table.constraints
                            if c.constraint_type == ConstraintType.UNIQUE]

        # 检查没有限制ground_station_id的唯一约束
        for uc in unique_constraints:
            assert "ground_station_id" not in uc.columns or len(uc.columns) > 1


class TestAntennasTableIntegration:
    """测试AntennasTable与整体模式的集成"""

    def test_schema_includes_antennas(self):
        """测试完整模式包含antennas表"""
        schema = DatabaseSchema()

        all_tables = schema.generate_all_sql()

        antennas_sql = None
        for sql in all_tables:
            if "antennas" in sql:
                antennas_sql = sql
                break

        assert antennas_sql is not None

    def test_table_references_ground_stations(self):
        """测试表正确引用ground_stations"""
        schema = DatabaseSchema()

        antennas_table = schema.get_table("antennas")
        ground_stations_table = schema.get_table("ground_stations")

        assert antennas_table is not None
        assert ground_stations_table is not None

        # 验证外键引用存在
        fk_constraints = [c for c in antennas_table.constraints
                         if c.constraint_type == ConstraintType.FOREIGN_KEY]

        gs_fk = next((c for c in fk_constraints
                     if c.reference_table == "ground_stations"), None)
        assert gs_fk is not None
        assert gs_fk.reference_columns == ["id"]

    def test_config_category_tables(self):
        """测试配置层表包含antennas"""
        schema = DatabaseSchema()

        config_tables = schema.get_tables_by_category(TableCategory.CONFIG)
        config_table_names = [t.name for t in config_tables]

        assert "antennas" in config_table_names
        assert "ground_stations" in config_table_names
        assert "satellites" in config_table_names
