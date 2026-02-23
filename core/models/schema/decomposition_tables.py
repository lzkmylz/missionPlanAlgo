"""
目标分解层表定义

包含: DecompositionConfigs, SubTargets
"""

from core.models.schema.base import (
    TableDefinition, ColumnDefinition, IndexDefinition,
    ColumnType, ConstraintType, TableCategory
)


class DecompositionConfigsTable(TableDefinition):
    """分解配置表 - H1关键缺失表"""

    def __init__(self):
        super().__init__(
            name="decomposition_configs",
            category=TableCategory.DECOMPOSITION,
            comment="分解配置表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("target_id", ColumnType.VARCHAR, length=32, nullable=False, comment="目标ID"),
                ColumnDefinition("scenario_id", ColumnType.INT, nullable=False, comment="所属场景"),
                ColumnDefinition("strategy", ColumnType.ENUM, nullable=False, enum_values=["grid", "strip"], comment="分解策略"),
                ColumnDefinition("resolution", ColumnType.DECIMAL, precision=6, scale=2, comment="所需分辨率(米)"),
                ColumnDefinition("sat_type", ColumnType.VARCHAR, length=16, comment="针对的卫星类型"),
                ColumnDefinition("strip_direction", ColumnType.DECIMAL, precision=6, scale=2, comment="条带方向(度)"),
                ColumnDefinition("strip_overlap", ColumnType.DECIMAL, precision=4, scale=2, default=0.1, comment="条带重叠率 0-1"),
                ColumnDefinition("grid_size", ColumnType.DECIMAL, precision=8, scale=2, comment="网格大小(米)"),
                ColumnDefinition("created_at", ColumnType.TIMESTAMP, default="CURRENT_TIMESTAMP", comment="创建时间"),
            ],
            constraints=[
                IndexDefinition("pk_decomposition_configs", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_dc_target", ConstraintType.FOREIGN_KEY, ["target_id"], "targets", ["id"]),
                IndexDefinition("fk_dc_scenario", ConstraintType.FOREIGN_KEY, ["scenario_id"], "scenarios", ["id"]),
                IndexDefinition("uk_target_scenario", ConstraintType.UNIQUE, ["target_id", "scenario_id", "sat_type"]),
            ]
        )


class SubTargetsTable(TableDefinition):
    """分解后的子任务表 - H1关键缺失表"""

    def __init__(self):
        super().__init__(
            name="sub_targets",
            category=TableCategory.DECOMPOSITION,
            comment="分解后的子任务表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("parent_target_id", ColumnType.VARCHAR, length=32, nullable=False, comment="父目标ID"),
                ColumnDefinition("scenario_id", ColumnType.INT, nullable=False, comment="所属场景"),
                ColumnDefinition("sub_target_code", ColumnType.VARCHAR, length=64, nullable=False, comment="子任务编码"),
                ColumnDefinition("sub_type", ColumnType.ENUM, nullable=False, enum_values=["grid_cell", "strip"], comment="子任务类型"),
                ColumnDefinition("center_lon", ColumnType.DECIMAL, precision=9, scale=6, comment="中心经度"),
                ColumnDefinition("center_lat", ColumnType.DECIMAL, precision=8, scale=6, comment="中心纬度"),
                ColumnDefinition("vertices", ColumnType.JSON, comment="顶点坐标"),
                ColumnDefinition("area_sqkm", ColumnType.DECIMAL, precision=10, scale=2, comment="面积(平方公里)"),
                ColumnDefinition("strip_length", ColumnType.DECIMAL, precision=10, scale=2, comment="条带长度(米)"),
                ColumnDefinition("required_mode", ColumnType.VARCHAR, length=32, comment="推荐的成像模式"),
                ColumnDefinition("estimated_duration", ColumnType.INT, comment="预估成像时长(秒)"),
                ColumnDefinition("status", ColumnType.ENUM, default="pending", enum_values=["pending", "scheduled", "completed", "failed"], comment="状态"),
                ColumnDefinition("created_at", ColumnType.TIMESTAMP, default="CURRENT_TIMESTAMP", comment="创建时间"),
            ],
            constraints=[
                IndexDefinition("pk_sub_targets", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_st_parent", ConstraintType.FOREIGN_KEY, ["parent_target_id"], "targets", ["id"]),
                IndexDefinition("fk_st_scenario", ConstraintType.FOREIGN_KEY, ["scenario_id"], "scenarios", ["id"]),
                IndexDefinition("idx_parent", ConstraintType.INDEX, ["parent_target_id"]),
                IndexDefinition("idx_scenario", ConstraintType.INDEX, ["scenario_id"]),
                IndexDefinition("idx_status", ConstraintType.INDEX, ["status"]),
            ]
        )
