"""
实验层表定义

包含: Experiments, ExperimentRuns, AlgorithmParams
"""

from core.models.schema.base import (
    TableDefinition, ColumnDefinition, IndexDefinition,
    ColumnType, ConstraintType, TableCategory
)


class ExperimentsTable(TableDefinition):
    """实验记录表"""

    def __init__(self):
        super().__init__(
            name="experiments",
            category=TableCategory.EXPERIMENT,
            comment="实验记录表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("name", ColumnType.VARCHAR, length=128, nullable=False, comment="实验名称"),
                ColumnDefinition("scenario_id", ColumnType.INT, nullable=False, comment="使用的场景ID"),
                ColumnDefinition("description", ColumnType.TEXT, comment="实验描述"),
                ColumnDefinition("algorithms", ColumnType.JSON, nullable=False, comment="参与的算法列表"),
                ColumnDefinition("repetitions", ColumnType.INT, default=10, comment="每种算法重复运行次数"),
                ColumnDefinition("status", ColumnType.ENUM, default="pending", enum_values=["pending", "running", "completed", "failed"], comment="状态"),
                ColumnDefinition("started_at", ColumnType.TIMESTAMP, comment="开始时间"),
                ColumnDefinition("completed_at", ColumnType.TIMESTAMP, comment="完成时间"),
                ColumnDefinition("best_algorithm", ColumnType.VARCHAR, length=64, comment="最佳算法"),
                ColumnDefinition("best_makespan", ColumnType.DECIMAL, precision=12, scale=2, comment="最优完成时间"),
            ],
            constraints=[
                IndexDefinition("pk_experiments", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_exp_scenario", ConstraintType.FOREIGN_KEY, ["scenario_id"], "scenarios", ["id"]),
                IndexDefinition("idx_status", ConstraintType.INDEX, ["status"]),
                IndexDefinition("idx_scenario", ConstraintType.INDEX, ["scenario_id"]),
            ]
        )


class ExperimentRunsTable(TableDefinition):
    """单次运行记录表"""

    def __init__(self):
        super().__init__(
            name="experiment_runs",
            category=TableCategory.EXPERIMENT,
            comment="单次运行记录表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("experiment_id", ColumnType.INT, nullable=False, comment="所属实验ID"),
                ColumnDefinition("run_number", ColumnType.INT, nullable=False, comment="运行序号"),
                ColumnDefinition("algorithm_name", ColumnType.VARCHAR, length=64, nullable=False, comment="算法名称"),
                ColumnDefinition("algorithm_version", ColumnType.VARCHAR, length=32, comment="算法版本"),
                ColumnDefinition("status", ColumnType.ENUM, default="running", enum_values=["running", "success", "failed", "timeout"], comment="状态"),
                ColumnDefinition("started_at", ColumnType.TIMESTAMP, comment="开始时间"),
                ColumnDefinition("completed_at", ColumnType.TIMESTAMP, comment="完成时间"),
                ColumnDefinition("demand_satisfaction_rate", ColumnType.DECIMAL, precision=5, scale=4, comment="需求满足率 0-1"),
                ColumnDefinition("makespan", ColumnType.DECIMAL, precision=12, scale=2, comment="全部完成用时(秒)"),
                ColumnDefinition("avg_revisit_time", ColumnType.DECIMAL, precision=12, scale=2, comment="平均观测间隔(秒)"),
                ColumnDefinition("data_delivery_time", ColumnType.DECIMAL, precision=12, scale=2, comment="数据回传用时(秒)"),
                ColumnDefinition("computation_time", ColumnType.DECIMAL, precision=10, scale=3, comment="算法求解用时(秒)"),
                ColumnDefinition("solution_quality", ColumnType.DECIMAL, precision=5, scale=4, comment="解质量 0-1"),
            ],
            constraints=[
                IndexDefinition("pk_experiment_runs", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_run_experiment", ConstraintType.FOREIGN_KEY, ["experiment_id"], "experiments", ["id"], "CASCADE"),
                IndexDefinition("uk_exp_run", ConstraintType.UNIQUE, ["experiment_id", "run_number", "algorithm_name"]),
                IndexDefinition("idx_algorithm", ConstraintType.INDEX, ["algorithm_name"]),
            ]
        )


class AlgorithmParamsTable(TableDefinition):
    """算法参数表"""

    def __init__(self):
        super().__init__(
            name="algorithm_params",
            category=TableCategory.EXPERIMENT,
            comment="算法参数表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("run_id", ColumnType.INT, nullable=False, comment="所属运行ID"),
                ColumnDefinition("param_name", ColumnType.VARCHAR, length=64, nullable=False, comment="参数名"),
                ColumnDefinition("param_value", ColumnType.VARCHAR, length=256, comment="参数值"),
            ],
            constraints=[
                IndexDefinition("pk_algorithm_params", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_param_run", ConstraintType.FOREIGN_KEY, ["run_id"], "experiment_runs", ["id"], "CASCADE"),
                IndexDefinition("uk_run_param", ConstraintType.UNIQUE, ["run_id", "param_name"]),
            ]
        )
