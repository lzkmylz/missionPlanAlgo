package orekit.visibility.model;

/**
 * 目标配置数据类
 *
 * 包含目标地理位置和观测要求。
 */
public class TargetConfig {
    private final String id;
    private final double longitude;  // 经度（度）
    private final double latitude;   // 纬度（度）
    private final double altitude;   // 海拔（米）
    private final int minObservationDuration;  // 最小观测时长（秒）
    private final int priority;      // 优先级（1-10）

    /**
     * 创建目标配置
     *
     * @param id 目标ID
     * @param longitude 经度（度）
     * @param latitude 纬度（度）
     * @param altitude 海拔（米）
     * @param minObservationDuration 最小观测时长（秒）
     * @param priority 优先级
     */
    public TargetConfig(String id, double longitude, double latitude,
                       double altitude, int minObservationDuration, int priority) {
        this.id = id;
        this.longitude = longitude;
        this.latitude = latitude;
        this.altitude = altitude;
        this.minObservationDuration = minObservationDuration;
        this.priority = priority;
    }

    /**
     * 创建目标配置（使用默认值）
     */
    public TargetConfig(String id, double longitude, double latitude) {
        this(id, longitude, latitude, 0.0, 60, 5);
    }

    public String getId() { return id; }
    public double getLongitude() { return longitude; }
    public double getLatitude() { return latitude; }
    public double getAltitude() { return altitude; }
    public int getMinObservationDuration() { return minObservationDuration; }
    public int getPriority() { return priority; }
}
