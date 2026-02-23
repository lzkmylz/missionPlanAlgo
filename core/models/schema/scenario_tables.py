"""
场景实例层表定义

包含: ScenarioSatellites, ScenarioTargets, ScenarioGroundStations
"""

from core.models.schema.base import (
    TableDefinition, ColumnDefinition, IndexDefinition,
    ColumnType, ConstraintType, TableCategory
)


class ScenarioSatellitesTable(TableDefinition):
    """场景-卫星关联表"""

    def __init__(self):
        super().__init__(
            name="scenario_satellites",
            category=TableCategory.SCENARIO,
            comment="场景卫星关联表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("scenario_id", ColumnType.INT, nullable=False, comment="场景ID"),
                ColumnDefinition("satellite_id", ColumnType.VARCHAR, length=32, nullable=False, comment="卫星ID"),
                ColumnDefinition("initial_storage_used", ColumnType.DECIMAL, precision=5, scale=2, default=0, comment="初始已用存储(GB)"),
                ColumnDefinition("initial_power_used", ColumnType.DECIMAL, precision=5, scale=2, default=0, comment="初始已用电量(Wh)"),
            ],
            constraints=[
                IndexDefinition("pk_scenario_satellites", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_ss_scenario", ConstraintType.FOREIGN_KEY, ["scenario_id"], "scenarios", ["id"], "CASCADE"),
                IndexDefinition("fk_ss_satellite", ConstraintType.FOREIGN_KEY, ["satellite_id"], "satellites", ["id"]),
                IndexDefinition("uk_scenario_satellite", ConstraintType.UNIQUE, ["scenario_id", "satellite_id"]),
            ]
        )


class ScenarioTargetsTable(TableDefinition):
    """场景-目标关联表"""

    def __init__(self):
        super().__init__(
            name="scenario_targets",
            category=TableCategory.SCENARIO,
            comment="场景目标关联表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("scenario_id", ColumnType.INT, nullable=False, comment="场景ID"),
                ColumnDefinition("target_id", ColumnType.VARCHAR, length=32, nullable=False, comment="目标ID"),
                ColumnDefinition("adjusted_priority", ColumnType.INT, comment="场景调整后的优先级"),
                ColumnDefinition("adjusted_observations", ColumnType.INT, comment="场景调整后的观测次数"),
            ],
            constraints=[
                IndexDefinition("pk_scenario_targets", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_st_scenario", ConstraintType.FOREIGN_KEY, ["scenario_id"], "scenarios", ["id"], "CASCADE"),
                IndexDefinition("fk_st_target", ConstraintType.FOREIGN_KEY, ["target_id"], "targets", ["id"]),
                IndexDefinition("uk_scenario_target", ConstraintType.UNIQUE, ["scenario_id", "target_id"]),
            ]
        )


class ScenarioGroundStationsTable(TableDefinition):
    """场景-地面站关联表"""

    def __init__(self):
        super().__init__(
            name="scenario_ground_stations",
            category=TableCategory.SCENARIO,
            comment="场景地面站关联表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="ID"),
                ColumnDefinition("scenario_id", ColumnType.INT, nullable=False, comment="场景ID"),
                ColumnDefinition("ground_station_id", ColumnType.VARCHAR, length=32, nullable=False, comment="地面站ID"),
            ],
            constraints=[
                IndexDefinition("pk_scenario_ground_stations", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_sgs_scenario", ConstraintType.FOREIGN_KEY, ["scenario_id"], "scenarios", ["id"], "CASCADE"),
                IndexDefinition("fk_sgs_gs", ConstraintType.FOREIGN_KEY, ["ground_station_id"], "ground_stations", ["id"]),
                IndexDefinition("uk_scenario_gs", ConstraintType.UNIQUE, ["scenario_id", "ground_station_id"]),
            ]
        )
