package orekit.visibility.model;

/**
 * 卫星配置数据类
 *
 * 包含卫星轨道参数和计算配置。
 */
public class SatelliteConfig {
    private final String id;
    private final String tleLine1;
    private final String tleLine2;
    private final double minElevation;
    private final double sensorFov;
    private final double maxRollAngle;      // 最大滚转角（度）
    private final double maxPitchAngle;     // 最大俯仰角（度）
    private final String satelliteType;     // 卫星类型: "OPTICAL" 或 "SAR"

    /**
     * 创建卫星配置（带姿态限制）
     *
     * @param id 卫星ID
     * @param tleLine1 TLE第一行
     * @param tleLine2 TLE第二行
     * @param minElevation 最小仰角（度）
     * @param sensorFov 传感器视场角（度）
     * @param maxRollAngle 最大滚转角（度）
     * @param maxPitchAngle 最大俯仰角（度）
     * @param satelliteType 卫星类型: "OPTICAL" 或 "SAR"
     */
    public SatelliteConfig(String id, String tleLine1, String tleLine2,
                          double minElevation, double sensorFov,
                          double maxRollAngle, double maxPitchAngle,
                          String satelliteType) {
        this.id = id;
        this.tleLine1 = tleLine1;
        this.tleLine2 = tleLine2;
        this.minElevation = minElevation;
        this.sensorFov = sensorFov;
        this.maxRollAngle = maxRollAngle;
        this.maxPitchAngle = maxPitchAngle;
        this.satelliteType = satelliteType;
    }

    /**
     * 创建卫星配置（使用默认姿态限制）
     */
    public SatelliteConfig(String id, String tleLine1, String tleLine2,
                          double minElevation, double sensorFov) {
        this(id, tleLine1, tleLine2, minElevation, sensorFov, 35.0, 20.0, "OPTICAL");
    }

    /**
     * 创建卫星配置（使用默认值）
     */
    public SatelliteConfig(String id, String tleLine1, String tleLine2) {
        this(id, tleLine1, tleLine2, 5.0, 0.0);
    }

    public String getId() { return id; }
    public String getTleLine1() { return tleLine1; }
    public String getTleLine2() { return tleLine2; }
    public double getMinElevation() { return minElevation; }
    public double getSensorFov() { return sensorFov; }
    public double getMaxRollAngle() { return maxRollAngle; }
    public double getMaxPitchAngle() { return maxPitchAngle; }
    public String getSatelliteType() { return satelliteType; }
}
