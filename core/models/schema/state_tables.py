"""
状态层表定义

包含: SatelliteStateSnapshots, ConstraintViolations
"""

from core.models.schema.base import (
    TableDefinition, ColumnDefinition, IndexDefinition,
    ColumnType, ConstraintType, TableCategory
)


class SatelliteStateSnapshotsTable(TableDefinition):
    """卫星状态快照表 - H1关键缺失表"""

    def __init__(self):
        super().__init__(
            name="satellite_state_snapshots",
            category=TableCategory.STATE,
            comment="卫星状态快照表",
            columns=[
                ColumnDefinition("id", ColumnType.BIGINT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("run_id", ColumnType.INT, nullable=False, comment="所属运行ID"),
                ColumnDefinition("satellite_id", ColumnType.VARCHAR, length=32, nullable=False, comment="卫星ID"),
                ColumnDefinition("snapshot_time", ColumnType.TIMESTAMP, nullable=False, comment="快照时间"),
                ColumnDefinition("power_level", ColumnType.DECIMAL, precision=10, scale=2, comment="电量(Wh)"),
                ColumnDefinition("storage_level", ColumnType.DECIMAL, precision=10, scale=2, comment="存储(GB)"),
                ColumnDefinition("is_eclipse", ColumnType.BOOLEAN, comment="是否地影"),
                ColumnDefinition("current_activity", ColumnType.VARCHAR, length=32, comment="当前活动"),
                ColumnDefinition("current_task_id", ColumnType.BIGINT, comment="当前执行任务ID"),
            ],
            constraints=[
                IndexDefinition("pk_satellite_state_snapshots", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_snapshot_run", ConstraintType.FOREIGN_KEY, ["run_id"], "experiment_runs", ["id"], "CASCADE"),
                IndexDefinition("idx_run_satellite", ConstraintType.INDEX, ["run_id", "satellite_id"]),
                IndexDefinition("idx_time", ConstraintType.INDEX, ["snapshot_time"]),
            ]
        )


class ConstraintViolationsTable(TableDefinition):
    """约束违反记录表"""

    def __init__(self):
        super().__init__(
            name="constraint_violations",
            category=TableCategory.STATE,
            comment="约束违反记录表",
            columns=[
                ColumnDefinition("id", ColumnType.BIGINT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("run_id", ColumnType.INT, nullable=False, comment="所属运行ID"),
                ColumnDefinition("violation_type", ColumnType.ENUM, nullable=False, enum_values=["power", "storage", "time", "thermal"], comment="违反类型"),
                ColumnDefinition("satellite_id", ColumnType.VARCHAR, length=32, comment="涉及卫星"),
                ColumnDefinition("violation_time", ColumnType.TIMESTAMP, comment="违反发生时间"),
                ColumnDefinition("expected_value", ColumnType.DECIMAL, precision=10, scale=2, comment="预期值"),
                ColumnDefinition("limit_value", ColumnType.DECIMAL, precision=10, scale=2, comment="限制值"),
                ColumnDefinition("description", ColumnType.TEXT, comment="详细描述"),
            ],
            constraints=[
                IndexDefinition("pk_constraint_violations", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_violation_run", ConstraintType.FOREIGN_KEY, ["run_id"], "experiment_runs", ["id"], "CASCADE"),
                IndexDefinition("idx_run", ConstraintType.INDEX, ["run_id"]),
            ]
        )
