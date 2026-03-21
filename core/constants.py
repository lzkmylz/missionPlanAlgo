"""
核心常量定义模块

统一管理所有物理常数、默认值和配置参数，避免魔法数字。
"""

from datetime import datetime, timezone

# =============================================================================
# 物理常数
# =============================================================================

# 地球半径 (米) - WGS84标准
EARTH_RADIUS_M = 6371000.0

# 地球半径 (公里) - WGS84标准
EARTH_RADIUS_KM = 6371.0

# 地球赤道半径 (公里) - WGS84标准
EARTH_EQUATORIAL_RADIUS_KM = 6378.137

# 地球引力常数 (m^3/s^2)
EARTH_GM = 3.986004418e14

# 标准大气压 (Pa)
STANDARD_ATMOSPHERE_PA = 101325.0

# 大气标高 (米) - 用于大气密度计算
ATMOSPHERE_SCALE_HEIGHT_M = 8500.0

# 地球J2项系数（扁率）
EARTH_J2 = 1.08263e-3

# 地球自转角速度（rad/s）
EARTH_ROTATION_RATE = 7.2921158553e-5

# =============================================================================
# 时间常数
# =============================================================================

# J2000纪元 (Julian Date)
J2000_JULIAN_DATE = 2451545.0

# J2000纪元 (datetime)
J2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# 参考时间 - 用于简化轨道计算 (datetime)
REFERENCE_EPOCH = datetime(2024, 1, 1, tzinfo=timezone.utc)

# 最早的TLE参考时间 (datetime)
TLE_REFERENCE_EPOCH = datetime(1950, 1, 1, tzinfo=timezone.utc)

# 一天的秒数
SECONDS_PER_DAY = 86400.0

# 一小时的秒数
SECONDS_PER_HOUR = 3600.0

# 一分钟的秒数
SECONDS_PER_MINUTE = 60.0

# GMST计算常数
GMST_CONSTANT_1 = 280.46061837
GMST_CONSTANT_2 = 360.98564736629
GMST_CONSTANT_3 = 18.697374558
GMST_CONSTANT_4 = 24.06570982441908

# =============================================================================
# 轨道默认值
# =============================================================================

# 默认轨道高度 (米) - 500km SSO
DEFAULT_ORBIT_ALTITUDE_M = 500000.0

# 默认轨道倾角 (度) - SSO典型值
DEFAULT_ORBIT_INCLINATION_DEG = 97.4

# 默认半长轴 (米) - 对应500km高度
DEFAULT_SEMI_MAJOR_AXIS_M = 6871000.0

# 默认轨道偏心率
DEFAULT_ORBIT_ECCENTRICITY = 0.0

# 默认升交点赤经 (度)
DEFAULT_RAAN_DEG = 0.0

# 默认近地点幅角 (度)
DEFAULT_ARG_OF_PERIGEE_DEG = 0.0

# 默认平近点角 (度)
DEFAULT_MEAN_ANOMALY_DEG = 0.0

# =============================================================================
# 卫星能力默认值
# =============================================================================

# 默认存储容量 (GB)
DEFAULT_STORAGE_CAPACITY_GB = 500.0

# 默认功率容量 (Wh)
DEFAULT_POWER_CAPACITY_WH = 2000.0

# 默认数据率 (Mbps)
DEFAULT_DATA_RATE_MBPS = 300.0

# 默认分辨率 (米)
DEFAULT_RESOLUTION_M = 10.0

# 默认幅宽 (米)
DEFAULT_SWATH_WIDTH_M = 10000.0

# =============================================================================
# FOV (视场) 默认值
# =============================================================================

# 默认FOV类型
DEFAULT_FOV_TYPE = 'cone'

# 默认圆锥视场半角 (度)
DEFAULT_FOV_HALF_ANGLE_DEG = 0.5

# 默认矩形视场半角 (度)
DEFAULT_FOV_HALF_ANGLE_X_DEG = 0.5  # X方向（沿轨迹）
DEFAULT_FOV_HALF_ANGLE_Y_DEG = 0.35  # Y方向（垂直轨迹）

# =============================================================================
# SAR FOV (视场) 默认值
# =============================================================================

# SAR默认距离向视场半角 (度) - cross-track方向，控制幅宽
# 典型SAR卫星在500km高度，2.5度半角对应约30km幅宽
DEFAULT_SAR_RANGE_HALF_ANGLE_DEG = 2.5

# SAR默认方位向视场半角 (度) - along-track方向，控制场景长度
# 典型值约1.0度
DEFAULT_SAR_AZIMUTH_HALF_ANGLE_DEG = 1.0

# SAR方位向排除角 (度) - 沿飞行方向的观测盲区
# 由于天线波束形状限制，沿轨迹方向存在一定的盲区
DEFAULT_SAR_AZIMUTH_EXCLUSION_ANGLE_DEG = 0.5

# 默认最大滚转角 (度) - 绕X轴旋转，控制侧摆
DEFAULT_MAX_ROLL_ANGLE_DEG = 30.0

# 默认最大俯仰角 (度) - 绕Y轴旋转，控制前后斜视
DEFAULT_MAX_PITCH_ANGLE_DEG = 20.0

# 卫星类型特定滚转角限制
OPTICAL_MAX_ROLL_ANGLE_DEG = 35.0  # 光学卫星最大滚转角 ±35°
SAR_MAX_ROLL_ANGLE_DEG = 45.0      # SAR卫星最大滚转角 ±45°

# 卫星类型特定俯仰角限制
OPTICAL_MAX_PITCH_ANGLE_DEG = 20.0  # 光学卫星最大俯仰角 ±20°
SAR_MAX_PITCH_ANGLE_DEG = 30.0      # SAR卫星最大俯仰角 ±30°

# 向后兼容: max_off_nadir 已废弃，使用 max_roll_angle 替代
DEFAULT_MAX_OFF_NADIR_DEG = DEFAULT_MAX_ROLL_ANGLE_DEG

# 光学卫星功率容量 (Wh)
OPTICAL_1_POWER_CAPACITY_WH = 2000.0
OPTICAL_2_POWER_CAPACITY_WH = 2500.0

# SAR卫星功率容量 (Wh)
SAR_1_POWER_CAPACITY_WH = 3000.0
SAR_2_POWER_CAPACITY_WH = 4000.0

# 光学卫星存储容量 (GB)
OPTICAL_1_STORAGE_CAPACITY_GB = 500.0
OPTICAL_2_STORAGE_CAPACITY_GB = 800.0

# SAR卫星存储容量 (GB)
SAR_1_STORAGE_CAPACITY_GB = 1000.0
SAR_2_STORAGE_CAPACITY_GB = 1500.0

# SAR卫星默认幅宽 (米)
SAR_1_SWATH_WIDTH_M = 10000.0
SAR_2_SWATH_WIDTH_M = 20000.0

# =============================================================================
# 成像参数默认值
# =============================================================================

# 光学卫星成像时长约束 (秒)
OPTICAL_IMAGING_MIN_DURATION_S = 6.0
OPTICAL_IMAGING_MAX_DURATION_S = 12.0

# SAR-1成像时长约束 (秒)
SAR_1_SPOTLIGHT_MIN_DURATION_S = 10.0
SAR_1_SPOTLIGHT_MAX_DURATION_S = 20.0
SAR_1_SLIDING_SPOTLIGHT_MIN_DURATION_S = 10.0
SAR_1_SLIDING_SPOTLIGHT_MAX_DURATION_S = 25.0
SAR_1_STRIPMAP_MIN_DURATION_S = 15.0
SAR_1_STRIPMAP_MAX_DURATION_S = 40.0

# SAR-2成像时长约束 (秒)
SAR_2_SPOTLIGHT_MIN_DURATION_S = 10.0
SAR_2_SPOTLIGHT_MAX_DURATION_S = 25.0
SAR_2_SLIDING_SPOTLIGHT_MIN_DURATION_S = 10.0
SAR_2_SLIDING_SPOTLIGHT_MAX_DURATION_S = 30.0
SAR_2_STRIPMAP_MIN_DURATION_S = 15.0
SAR_2_STRIPMAP_MAX_DURATION_S = 56.0

# SAR TOPSAR模式时长约束 (秒) — SAR-1/SAR-2通用
SAR_TOPSAR_MIN_DURATION_S = 5.0
SAR_TOPSAR_MAX_DURATION_S = 60.0

# SAR ScanSAR模式时长约束 (秒) — SAR-1/SAR-2通用
SAR_SCANSAR_MIN_DURATION_S = 5.0
SAR_SCANSAR_MAX_DURATION_S = 30.0

# 典型LEO卫星轨道速度 (m/s)，用于SAR物理计算的默认值
SAR_DEFAULT_V_SAT_MS = 7500.0

# =============================================================================
# 成像模式特定功耗和数据率默认值
# =============================================================================

# 光学 - 被动推扫模式
OPTICAL_PUSH_BROOM_POWER_W = 150.0           # 功耗 (瓦特)
OPTICAL_PUSH_BROOM_DATA_RATE_MBPS = 200.0    # 数据率 (Mbps)

# SAR - 条带模式 (Stripmap)
SAR_STRIPMAP_POWER_W = 300.0                 # 功耗 (瓦特)
SAR_STRIPMAP_DATA_RATE_MBPS = 400.0          # 数据率 (Mbps)

# SAR - 聚束模式 (Spotlight)
SAR_SPOTLIGHT_POWER_W = 500.0                # 功耗 (瓦特)
SAR_SPOTLIGHT_DATA_RATE_MBPS = 800.0         # 数据率 (Mbps)

# SAR - 扫描模式 (Scan)
SAR_SCAN_POWER_W = 400.0                     # 功耗 (瓦特)
SAR_SCAN_DATA_RATE_MBPS = 600.0              # 数据率 (Mbps)

# SAR - 滑动聚束模式 (Sliding Spotlight)
SAR_SLIDING_SPOTLIGHT_POWER_W = 450.0        # 功耗 (瓦特)
SAR_SLIDING_SPOTLIGHT_DATA_RATE_MBPS = 700.0 # 数据率 (Mbps)

# =============================================================================
# 姿态机动默认值
# =============================================================================

# 默认最大角速度 (度/秒) - 标量限制（向后兼容）
DEFAULT_MAX_SLEW_RATE_DEG_S = 3.0

# 默认角加速度 (度/秒²) - 标量限制（向后兼容）
DEFAULT_SLEW_ACCELERATION_DEG_S2 = 1.5

# 默认稳定时间 (秒)
DEFAULT_SETTLING_TIME_S = 5.0

# =============================================================================
# 分轴姿态机动默认值（新增）
# =============================================================================

# 光学卫星滚转角速度限制 (度/秒)
OPTICAL_MAX_ROLL_RATE_DEG_S = 3.0
# 光学卫星俯仰角速度限制 (度/秒)
OPTICAL_MAX_PITCH_RATE_DEG_S = 2.0

# SAR卫星滚转角速度限制 (度/秒)
SAR_MAX_ROLL_RATE_DEG_S = 4.0
# SAR卫星俯仰角速度限制 (度/秒)
SAR_MAX_PITCH_RATE_DEG_S = 3.0

# 光学卫星滚转角加速度限制 (度/秒²)
OPTICAL_MAX_ROLL_ACCEL_DEG_S2 = 1.5
# 光学卫星俯仰角加速度限制 (度/秒²)
OPTICAL_MAX_PITCH_ACCEL_DEG_S2 = 1.0

# SAR卫星滚转角加速度限制 (度/秒²)
SAR_MAX_ROLL_ACCEL_DEG_S2 = 2.0
# SAR卫星俯仰角加速度限制 (度/秒²)
SAR_MAX_PITCH_ACCEL_DEG_S2 = 1.5

# 默认分轴角速度限制 (度/秒)
DEFAULT_MAX_ROLL_RATE_DEG_S = 3.0
DEFAULT_MAX_PITCH_RATE_DEG_S = 2.0

# 默认分轴角加速度限制 (度/秒²)
DEFAULT_MAX_ROLL_ACCEL_DEG_S2 = 1.5
DEFAULT_MAX_PITCH_ACCEL_DEG_S2 = 1.0

# =============================================================================
# 网络通信默认值
# =============================================================================

# 激光星间链路默认数据率 (Mbps)
DEFAULT_LASER_DATA_RATE_MBPS = 10000.0

# 射频星间链路默认数据率 (Mbps)
DEFAULT_RF_DATA_RATE_MBPS = 1000.0

# 默认最大星间链路距离 (公里)
DEFAULT_MAX_ISL_DISTANCE_KM = 5000.0

# 默认地面站通信距离 (公里)
DEFAULT_GS_MAX_RANGE_KM = 2000.0

# 默认上载通信距离 (公里)
DEFAULT_UPLINK_RANGE_KM = 1000.0

# 默认卫星功率水平 (W)
DEFAULT_SATELLITE_POWER_LEVEL_W = 1800

# 卫星功率生成最大值 (W)
DEFAULT_MAX_POWER_GENERATION_W = 1000.0

# 动量轮卸载间隔 (秒) - 4小时
DEFAULT_MOMENTUM_DUMP_INTERVAL_S = 14400.0

# =============================================================================
# 轨道传播默认值
# =============================================================================

# 默认最大积分步长 (秒)
DEFAULT_MAX_PROPAGATION_STEP_S = 1000.0

# 默认最小积分步长 (秒)
DEFAULT_MIN_PROPAGATION_STEP_S = 0.001

# 积分器位置容差 (米)
DEFAULT_POSITION_TOLERANCE_M = 10.0

# 默认粗扫描步长 (秒)
DEFAULT_COARSE_SCAN_STEP_S = 5.0

# 默认精扫描步长 (秒)
DEFAULT_FINE_SCAN_STEP_S = 1.0

# 姿态轨迹规划点乘阈值
ATTITUDE_DOT_THRESHOLD = 0.9995

# =============================================================================
# 数据处理常数
# =============================================================================

# 每GB数据的比特数 (1 GB = 8 Gb = 8000 Mb)
BITS_PER_GB = 8000

# 每GB数据的千比特数
KBITS_PER_GB = 1000

# 字节到MB的转换
BYTES_PER_MB = 1024 * 1024

# 字节到KB的转换
BYTES_PER_KB = 1024

# =============================================================================
# 太阳位置计算常数
# =============================================================================

# 太阳平均经度常数
SUN_MEAN_LONGITUDE_CONST_1 = 280.460
SUN_MEAN_LONGITUDE_CONST_2 = 0.9856474

# 太阳平近点角常数
SUN_MEAN_ANOMALY_CONST_1 = 357.528
SUN_MEAN_ANOMALY_CONST_2 = 0.9856003

# 太阳距离常数 (AU)
SUN_DISTANCE_CONST_1 = 1.00014
SUN_DISTANCE_CONST_2 = 0.01671
SUN_DISTANCE_CONST_3 = 0.00014

# 黄赤交角常数
OBLIQUITY_CONST = 23.439
OBLIQUITY_RATE = 0.0000004

# =============================================================================
# 单位转换
# =============================================================================

# 米到公里的转换
METERS_TO_KM = 0.001

# 公里到米的转换
KM_TO_METERS = 1000.0

# 度到弧度的转换
DEGREES_TO_RADIANS = 3.141592653589793 / 180.0

# 弧度到度的转换
RADIANS_TO_DEGREES = 180.0 / 3.141592653589793

# =============================================================================
# 地面站天线默认值
# =============================================================================

# 默认链路建立时间（秒）- 指向 + 捕获 + 同步
DEFAULT_ACQUISITION_TIME_SECONDS = 15.0

# 默认最小任务切换时间（秒）- 天线重新指向、系统切换
DEFAULT_MIN_SWITCH_TIME_SECONDS = 30.0

# 快速天线切换时间（秒）- 同卫星连续任务
FAST_SWITCH_TIME_SECONDS = 10.0

# 标准天线切换时间（秒）- 不同卫星间切换
STANDARD_SWITCH_TIME_SECONDS = 60.0
