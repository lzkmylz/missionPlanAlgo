package orekit.visibility.model;

/**
 * 地面站配置类
 *
 * 存储地面站位置和通信参数。
 */
public class GroundStationConfig {
    private final String id;
    private final double longitude;
    private final double latitude;
    private final double altitude;
    private final double minElevation;
    private final double maxRange;

    /**
     * 创建地面站配置
     *
     * @param id 地面站ID
     * @param longitude 经度（度）
     * @param latitude 纬度（度）
     * @param altitude 海拔高度（米）
     * @param minElevation 最小仰角（度）
     * @param maxRange 最大通信距离（米）
     */
    public GroundStationConfig(String id, double longitude, double latitude,
                               double altitude, double minElevation, double maxRange) {
        this.id = id;
        this.longitude = longitude;
        this.latitude = latitude;
        this.altitude = altitude;
        this.minElevation = minElevation;
        this.maxRange = maxRange;
    }

    public String getId() { return id; }
    public double getLongitude() { return longitude; }
    public double getLatitude() { return latitude; }
    public double getAltitude() { return altitude; }
    public double getMinElevation() { return minElevation; }
    public double getMaxRange() { return maxRange; }

    @Override
    public String toString() {
        return String.format("GroundStation[%s: %.2f°, %.2f°]",
            id, longitude, latitude);
    }
}
