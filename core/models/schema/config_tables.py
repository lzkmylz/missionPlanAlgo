"""
配置层表定义

包含: Satellites, GroundStations, Antennas, Targets, Scenarios
"""

from core.models.schema.base import (
    TableDefinition, ColumnDefinition, IndexDefinition,
    ColumnType, ConstraintType, TableCategory
)


class SatellitesTable(TableDefinition):
    """卫星基础配置表"""

    def __init__(self):
        super().__init__(
            name="satellites",
            category=TableCategory.CONFIG,
            comment="卫星基础配置表",
            columns=[
                ColumnDefinition("id", ColumnType.VARCHAR, length=32, nullable=False, comment="卫星ID"),
                ColumnDefinition("name", ColumnType.VARCHAR, length=64, nullable=False, comment="卫星名称"),
                ColumnDefinition("sat_type", ColumnType.ENUM, nullable=False, enum_values=["optical_1", "optical_2", "sar_1", "sar_2"], comment="卫星类型"),
                ColumnDefinition("orbit_type", ColumnType.ENUM, nullable=False, default="SSO", enum_values=["SSO", "LEO", "MEO", "GEO"], comment="轨道类型"),
                ColumnDefinition("altitude", ColumnType.INT, comment="轨道高度(米)"),
                ColumnDefinition("inclination", ColumnType.DECIMAL, precision=6, scale=2, comment="轨道倾角(度)"),
                ColumnDefinition("orbit_params", ColumnType.JSON, comment="完整轨道参数JSON"),
                ColumnDefinition("max_off_nadir", ColumnType.DECIMAL, precision=5, scale=2, comment="最大侧摆角(度)"),
                ColumnDefinition("agility", ColumnType.JSON, comment="姿态机动能力"),
                ColumnDefinition("storage_capacity", ColumnType.INT, comment="存储容量(GB)"),
                ColumnDefinition("power_capacity", ColumnType.INT, comment="电量(Wh)"),
                ColumnDefinition("data_rate", ColumnType.INT, comment="数传速率(Mbps)"),
                ColumnDefinition("supported_modes", ColumnType.JSON, comment="支持的成像模式列表"),
                ColumnDefinition("created_at", ColumnType.TIMESTAMP, default="CURRENT_TIMESTAMP", comment="创建时间"),
                ColumnDefinition("updated_at", ColumnType.TIMESTAMP, default="CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP", comment="更新时间"),
            ],
            constraints=[
                IndexDefinition("pk_satellites", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("idx_sat_type", ConstraintType.INDEX, ["sat_type"]),
            ]
        )


class GroundStationsTable(TableDefinition):
    """地面站配置表"""

    def __init__(self):
        super().__init__(
            name="ground_stations",
            category=TableCategory.CONFIG,
            comment="地面站配置表",
            columns=[
                ColumnDefinition("id", ColumnType.VARCHAR, length=32, nullable=False, comment="地面站ID"),
                ColumnDefinition("name", ColumnType.VARCHAR, length=64, nullable=False, comment="地面站名称"),
                ColumnDefinition("longitude", ColumnType.DECIMAL, precision=9, scale=6, nullable=False, comment="经度"),
                ColumnDefinition("latitude", ColumnType.DECIMAL, precision=8, scale=6, nullable=False, comment="纬度"),
                ColumnDefinition("altitude", ColumnType.DECIMAL, precision=8, scale=2, default=0, comment="海拔高度(米)"),
                ColumnDefinition("created_at", ColumnType.TIMESTAMP, default="CURRENT_TIMESTAMP", comment="创建时间"),
            ],
            constraints=[
                IndexDefinition("pk_ground_stations", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("idx_location", ConstraintType.INDEX, ["latitude", "longitude"]),
            ]
        )


class AntennasTable(TableDefinition):
    """天线配置表 - H1关键缺失表"""

    def __init__(self):
        super().__init__(
            name="antennas",
            category=TableCategory.CONFIG,
            comment="天线配置表",
            columns=[
                ColumnDefinition("id", ColumnType.VARCHAR, length=32, nullable=False, comment="天线ID"),
                ColumnDefinition("ground_station_id", ColumnType.VARCHAR, length=32, nullable=False, comment="所属地面站"),
                ColumnDefinition("name", ColumnType.VARCHAR, length=64, nullable=False, comment="天线名称"),
                ColumnDefinition("elevation_min", ColumnType.DECIMAL, precision=5, scale=2, default=5.0, comment="最小仰角(度)"),
                ColumnDefinition("elevation_max", ColumnType.DECIMAL, precision=5, scale=2, default=90.0, comment="最大仰角(度)"),
                ColumnDefinition("data_rate", ColumnType.INT, comment="数据传输速率(Mbps)"),
                ColumnDefinition("slew_rate", ColumnType.DECIMAL, precision=5, scale=2, comment="天线转动速率(度/s)"),
            ],
            constraints=[
                IndexDefinition("pk_antennas", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("fk_antennas_gs", ConstraintType.FOREIGN_KEY, ["ground_station_id"], "ground_stations", ["id"], "CASCADE"),
                IndexDefinition("idx_station", ConstraintType.INDEX, ["ground_station_id"]),
            ]
        )


class TargetsTable(TableDefinition):
    """目标基础配置表"""

    def __init__(self):
        super().__init__(
            name="targets",
            category=TableCategory.CONFIG,
            comment="目标基础配置表",
            columns=[
                ColumnDefinition("id", ColumnType.VARCHAR, length=32, nullable=False, comment="目标ID"),
                ColumnDefinition("name", ColumnType.VARCHAR, length=128, comment="目标名称或描述"),
                ColumnDefinition("target_type", ColumnType.ENUM, nullable=False, enum_values=["point", "area"], comment="目标类型"),
                ColumnDefinition("longitude", ColumnType.DECIMAL, precision=9, scale=6, comment="经度（点目标）"),
                ColumnDefinition("latitude", ColumnType.DECIMAL, precision=8, scale=6, comment="纬度（点目标）"),
                ColumnDefinition("area_vertices", ColumnType.JSON, comment="区域顶点坐标列表"),
                ColumnDefinition("priority", ColumnType.INT, default=1, comment="优先级 1-10"),
                ColumnDefinition("required_observations", ColumnType.INT, default=1, comment="需要观测次数"),
                ColumnDefinition("time_window_start", ColumnType.TIMESTAMP, comment="观测时间窗口开始"),
                ColumnDefinition("time_window_end", ColumnType.TIMESTAMP, comment="观测时间窗口结束"),
                ColumnDefinition("immediate_downlink", ColumnType.BOOLEAN, default=False, comment="是否立即回传"),
                ColumnDefinition("resolution_required", ColumnType.DECIMAL, precision=6, scale=2, comment="分辨率要求(米)"),
                ColumnDefinition("created_at", ColumnType.TIMESTAMP, default="CURRENT_TIMESTAMP", comment="创建时间"),
            ],
            constraints=[
                IndexDefinition("pk_targets", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("idx_type", ConstraintType.INDEX, ["target_type"]),
                IndexDefinition("idx_location", ConstraintType.INDEX, ["latitude", "longitude"]),
                IndexDefinition("idx_priority", ConstraintType.INDEX, ["priority"]),
            ]
        )


class ScenariosTable(TableDefinition):
    """场景版本管理表"""

    def __init__(self):
        super().__init__(
            name="scenarios",
            category=TableCategory.CONFIG,
            comment="场景版本管理表",
            columns=[
                ColumnDefinition("id", ColumnType.INT, auto_increment=True, nullable=False, comment="场景ID"),
                ColumnDefinition("name", ColumnType.VARCHAR, length=128, nullable=False, comment="场景名称"),
                ColumnDefinition("version", ColumnType.INT, default=1, comment="版本号"),
                ColumnDefinition("description", ColumnType.TEXT, comment="场景描述"),
                ColumnDefinition("start_time", ColumnType.TIMESTAMP, nullable=False, comment="规划周期开始"),
                ColumnDefinition("end_time", ColumnType.TIMESTAMP, nullable=False, comment="规划周期结束"),
                ColumnDefinition("satellite_count", ColumnType.INT, comment="卫星数量"),
                ColumnDefinition("target_count", ColumnType.INT, comment="目标数量"),
                ColumnDefinition("config_json", ColumnType.JSON, comment="完整场景配置JSON"),
                ColumnDefinition("created_at", ColumnType.TIMESTAMP, default="CURRENT_TIMESTAMP", comment="创建时间"),
                ColumnDefinition("is_active", ColumnType.BOOLEAN, default=True, comment="是否当前使用版本"),
            ],
            constraints=[
                IndexDefinition("pk_scenarios", ConstraintType.PRIMARY_KEY, ["id"]),
                IndexDefinition("uk_name_version", ConstraintType.UNIQUE, ["name", "version"]),
                IndexDefinition("idx_active", ConstraintType.INDEX, ["is_active"]),
            ]
        )
