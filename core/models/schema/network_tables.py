"""
网络层表定义

包含: ISLWindows, DataRoutingPaths, UplinkCommands, RelaySatellites
"""

from core.models.schema.base import (
    TableDefinition, ColumnDefinition, IndexDefinition,
    ColumnType, ConstraintType, TableCategory
)


class ISLWindowsTable(TableDefinition):
    """星间链路窗口表"""

    def __init__(self):
        super().__init__(
            name="isl_windows",
            category=TableCategory.NETWORK,
            comment="星间链路窗口表",
            columns=[
                ColumnDefinition("id", ColumnType.BIGINT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("scenario_id", ColumnType.INT, nullable=False, comment="场景ID"),
                ColumnDefinition("satellite_a_id", ColumnType.VARCHAR, length=32, nullable=False, comment="卫星A"),
                ColumnDefinition("satellite_b_id", ColumnType.VARCHAR, length=32, nullable=False, comment="卫星B"),
                ColumnDefinition("start_time", ColumnType.TIMESTAMP, nullable=False, comment="窗口开始"),
                ColumnDefinition("end_time", ColumnType.TIMESTAMP, nullable=False, comment="窗口结束"),
                ColumnDefinition("link_quality", ColumnType.DECIMAL, precision=3, scale=2, comment="链路质量 0-1"),
                ColumnDefinition("max_data_rate", ColumnType.DECIMAL, precision=10, scale=2, comment="最大数据速率(Mbps)"),
                ColumnDefinition("distance", ColumnType.DECIMAL, precision=10, scale=2, comment="星间距离(km)"),
            ],
            constraints=[
                IndexDefinition("pk_isl_windows", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_iw_scenario", ConstraintType.FOREIGN_KEY, ["scenario_id"], "scenarios", ["id"], "CASCADE"),
                IndexDefinition("idx_satellites", ConstraintType.INDEX, ["satellite_a_id", "satellite_b_id"]),
                IndexDefinition("idx_time", ConstraintType.INDEX, ["start_time", "end_time"]),
            ]
        )


class DataRoutingPathsTable(TableDefinition):
    """数据回传路径表"""

    def __init__(self):
        super().__init__(
            name="data_routing_paths",
            category=TableCategory.NETWORK,
            comment="数据回传路径表",
            columns=[
                ColumnDefinition("id", ColumnType.BIGINT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("run_id", ColumnType.INT, nullable=False, comment="运行ID"),
                ColumnDefinition("task_id", ColumnType.VARCHAR, length=32, nullable=False, comment="任务ID"),
                ColumnDefinition("source_satellite", ColumnType.VARCHAR, length=32, comment="源卫星"),
                ColumnDefinition("destination_gs", ColumnType.VARCHAR, length=32, comment="目标地面站"),
                ColumnDefinition("route_hops", ColumnType.JSON, comment="路径跳数"),
                ColumnDefinition("total_latency", ColumnType.DECIMAL, precision=10, scale=3, comment="总延迟(秒)"),
                ColumnDefinition("used_relay", ColumnType.BOOLEAN, default=False, comment="是否使用中继"),
                ColumnDefinition("relay_satellite_id", ColumnType.VARCHAR, length=32, comment="中继星ID"),
            ],
            constraints=[
                IndexDefinition("pk_data_routing_paths", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_drp_run", ConstraintType.FOREIGN_KEY, ["run_id"], "experiment_runs", ["id"], "CASCADE"),
                IndexDefinition("idx_run_task", ConstraintType.INDEX, ["run_id", "task_id"]),
            ]
        )


class UplinkCommandsTable(TableDefinition):
    """指令上行记录表"""

    def __init__(self):
        super().__init__(
            name="uplink_commands",
            category=TableCategory.NETWORK,
            comment="指令上行记录表",
            columns=[
                ColumnDefinition("id", ColumnType.BIGINT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("run_id", ColumnType.INT, nullable=False, comment="运行ID"),
                ColumnDefinition("satellite_id", ColumnType.VARCHAR, length=32, nullable=False, comment="卫星ID"),
                ColumnDefinition("command_type", ColumnType.VARCHAR, length=64, comment="指令类型"),
                ColumnDefinition("uplink_time", ColumnType.TIMESTAMP, comment="实际上行时间"),
                ColumnDefinition("scheduled_task_id", ColumnType.VARCHAR, length=32, comment="对应的调度任务"),
                ColumnDefinition("link_type", ColumnType.ENUM, enum_values=["direct", "relay"], comment="上行链路类型"),
                ColumnDefinition("ground_station_id", ColumnType.VARCHAR, length=32, comment="地面站ID"),
                ColumnDefinition("relay_satellite_id", ColumnType.VARCHAR, length=32, comment="中继星ID"),
            ],
            constraints=[
                IndexDefinition("pk_uplink_commands", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_uc_run", ConstraintType.FOREIGN_KEY, ["run_id"], "experiment_runs", ["id"], "CASCADE"),
                IndexDefinition("idx_run_satellite", ConstraintType.INDEX, ["run_id", "satellite_id"]),
            ]
        )


class RelaySatellitesTable(TableDefinition):
    """中继星配置表"""

    def __init__(self):
        super().__init__(
            name="relay_satellites",
            category=TableCategory.NETWORK,
            comment="中继星配置表",
            columns=[
                ColumnDefinition("id", ColumnType.VARCHAR, length=32, nullable=False, comment="中继星ID"),
                ColumnDefinition("name", ColumnType.VARCHAR, length=64, comment="名称"),
                ColumnDefinition("orbit_type", ColumnType.VARCHAR, length=32, comment="轨道类型"),
                ColumnDefinition("longitude", ColumnType.DECIMAL, precision=9, scale=6, comment="定点经度"),
                ColumnDefinition("uplink_capacity", ColumnType.DECIMAL, precision=10, scale=2, comment="上行容量(Mbps)"),
                ColumnDefinition("downlink_capacity", ColumnType.DECIMAL, precision=10, scale=2, comment="下行容量(Mbps)"),
                ColumnDefinition("coverage_zones", ColumnType.JSON, comment="覆盖区域列表"),
            ],
            constraints=[
                IndexDefinition("pk_relay_satellites", ConstraintType.PRIMARY_KEY, ["id"]),
            ]
        )
