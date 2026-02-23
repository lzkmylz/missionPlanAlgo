"""
结果层表定义

包含: ScheduleTasks, PerformanceMetrics, TaskSequences
"""

from core.models.schema.base import (
    TableDefinition, ColumnDefinition, IndexDefinition,
    ColumnType, ConstraintType, TableCategory
)


class ScheduleTasksTable(TableDefinition):
    """调度任务详情表"""

    def __init__(self):
        super().__init__(
            name="schedule_tasks",
            category=TableCategory.RESULT,
            comment="调度任务详情表",
            columns=[
                ColumnDefinition("id", ColumnType.BIGINT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("run_id", ColumnType.INT, nullable=False, comment="所属运行ID"),
                ColumnDefinition("sub_target_id", ColumnType.INT, comment="子任务ID"),
                ColumnDefinition("target_id", ColumnType.VARCHAR, length=32, comment="原始目标ID"),
                ColumnDefinition("satellite_id", ColumnType.VARCHAR, length=32, nullable=False, comment="执行卫星"),
                ColumnDefinition("antenna_id", ColumnType.VARCHAR, length=32, comment="数传天线"),
                ColumnDefinition("imaging_start", ColumnType.TIMESTAMP, nullable=False, comment="成像开始时间"),
                ColumnDefinition("imaging_end", ColumnType.TIMESTAMP, nullable=False, comment="成像结束时间"),
                ColumnDefinition("imaging_mode", ColumnType.VARCHAR, length=32, comment="使用的成像模式"),
                ColumnDefinition("slew_start", ColumnType.TIMESTAMP, comment="姿态调整开始"),
                ColumnDefinition("slew_end", ColumnType.TIMESTAMP, comment="姿态调整结束"),
                ColumnDefinition("slew_angle", ColumnType.DECIMAL, precision=6, scale=2, comment="侧摆角"),
                ColumnDefinition("downlink_start", ColumnType.TIMESTAMP, comment="数传开始"),
                ColumnDefinition("downlink_end", ColumnType.TIMESTAMP, comment="数传结束"),
                ColumnDefinition("downlink_station_id", ColumnType.VARCHAR, length=32, comment="数传地面站"),
                ColumnDefinition("storage_before", ColumnType.DECIMAL, precision=5, scale=2, comment="成像前存储使用(GB)"),
                ColumnDefinition("storage_after", ColumnType.DECIMAL, precision=5, scale=2, comment="成像后存储使用(GB)"),
                ColumnDefinition("sequence_number", ColumnType.INT, comment="在该卫星上的执行序号"),
            ],
            constraints=[
                IndexDefinition("pk_schedule_tasks", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_task_run", ConstraintType.FOREIGN_KEY, ["run_id"], "experiment_runs", ["id"], "CASCADE"),
                IndexDefinition("idx_run", ConstraintType.INDEX, ["run_id"]),
                IndexDefinition("idx_satellite", ConstraintType.INDEX, ["satellite_id"]),
                IndexDefinition("idx_time", ConstraintType.INDEX, ["imaging_start", "imaging_end"]),
            ]
        )


class PerformanceMetricsTable(TableDefinition):
    """性能指标表"""

    def __init__(self):
        super().__init__(
            name="performance_metrics",
            category=TableCategory.RESULT,
            comment="性能指标表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("run_id", ColumnType.INT, nullable=False, comment="所属运行ID"),
                ColumnDefinition("total_tasks", ColumnType.INT, comment="总任务数"),
                ColumnDefinition("scheduled_tasks", ColumnType.INT, comment="成功调度任务数"),
                ColumnDefinition("unscheduled_tasks", ColumnType.INT, comment="未调度任务数"),
                ColumnDefinition("demand_satisfaction_rate", ColumnType.DECIMAL, precision=5, scale=4, comment="需求满足率"),
                ColumnDefinition("makespan", ColumnType.DECIMAL, precision=12, scale=2, comment="总完成用时(秒)"),
                ColumnDefinition("avg_revisit_time", ColumnType.DECIMAL, precision=12, scale=2, comment="平均观测间隔(秒)"),
                ColumnDefinition("max_revisit_time", ColumnType.DECIMAL, precision=12, scale=2, comment="最大观测间隔(秒)"),
                ColumnDefinition("data_delivery_time", ColumnType.DECIMAL, precision=12, scale=2, comment="数据回传用时(秒)"),
                ColumnDefinition("computation_time", ColumnType.DECIMAL, precision=10, scale=3, comment="算法求解用时(秒)"),
                ColumnDefinition("solution_quality", ColumnType.DECIMAL, precision=5, scale=4, comment="解质量"),
                ColumnDefinition("avg_satellite_utilization", ColumnType.DECIMAL, precision=5, scale=4, comment="卫星平均利用率"),
                ColumnDefinition("max_satellite_load", ColumnType.INT, comment="最大负载卫星的任务数"),
                ColumnDefinition("ground_station_utilization", ColumnType.DECIMAL, precision=5, scale=4, comment="地面站利用率"),
                ColumnDefinition("load_balance_std", ColumnType.DECIMAL, precision=10, scale=4, comment="卫星负载标准差"),
            ],
            constraints=[
                IndexDefinition("pk_performance_metrics", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_metric_run", ConstraintType.FOREIGN_KEY, ["run_id"], "experiment_runs", ["id"], "CASCADE"),
                IndexDefinition("uk_run", ConstraintType.UNIQUE, ["run_id"]),
            ]
        )


class TaskSequencesTable(TableDefinition):
    """任务执行序列表"""

    def __init__(self):
        super().__init__(
            name="task_sequences",
            category=TableCategory.RESULT,
            comment="任务执行序列表",
            columns=[
                ColumnDefinition("id", ColumnType.BIGINT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("run_id", ColumnType.INT, nullable=False, comment="所属运行ID"),
                ColumnDefinition("satellite_id", ColumnType.VARCHAR, length=32, nullable=False, comment="卫星ID"),
                ColumnDefinition("task_sequence", ColumnType.JSON, nullable=False, comment="任务ID序列"),
                ColumnDefinition("total_idle_time", ColumnType.INT, comment="总空闲时间(秒)"),
                ColumnDefinition("idle_segments", ColumnType.INT, comment="空闲段数"),
            ],
            constraints=[
                IndexDefinition("pk_task_sequences", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_seq_run", ConstraintType.FOREIGN_KEY, ["run_id"], "experiment_runs", ["id"], "CASCADE"),
                IndexDefinition("uk_run_satellite", ConstraintType.UNIQUE, ["run_id", "satellite_id"]),
            ]
        )
