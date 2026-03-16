package orekit.visibility.model;

/**
 * 中继卫星配置类
 *
 * 存储GEO中继卫星的配置参数，用于数据中继服务。
 * 支持天链式中继卫星星座配置。
 */
public class RelaySatelliteConfig {
    private final String id;
    private final String name;
    private final String orbitType;
    private final double longitude;           // 定点经度（度）
    private final double minElevation;        // 最小仰角（度）
    private final double maxRange;            // 最大通信距离（米）
    private final double uplinkCapacity;      // 上行容量（Mbps）
    private final double downlinkCapacity;    // 下行容量（Mbps）

    /**
     * 创建中继卫星配置
     *
     * @param id 中继卫星ID
     * @param name 名称
     * @param orbitType 轨道类型（通常为GEO）
     * @param longitude 定点经度（度）
     * @param minElevation 最小仰角（度）
     * @param maxRange 最大通信距离（米）
     * @param uplinkCapacity 上行容量（Mbps）
     * @param downlinkCapacity 下行容量（Mbps）
     */
    public RelaySatelliteConfig(String id, String name, String orbitType,
                                 double longitude, double minElevation, double maxRange,
                                 double uplinkCapacity, double downlinkCapacity) {
        this.id = id;
        this.name = name;
        this.orbitType = orbitType;
        this.longitude = longitude;
        this.minElevation = minElevation;
        this.maxRange = maxRange;
        this.uplinkCapacity = uplinkCapacity;
        this.downlinkCapacity = downlinkCapacity;
    }

    /**
     * 创建中继卫星配置（使用默认值）
     *
     * @param id 中继卫星ID
     * @param name 名称
     * @param longitude 定点经度（度）
     */
    public RelaySatelliteConfig(String id, String name, double longitude) {
        this(id, name, "GEO", longitude, 5.0, 45000000.0, 450.0, 450.0);
    }

    public String getId() { return id; }
    public String getName() { return name; }
    public String getOrbitType() { return orbitType; }
    public double getLongitude() { return longitude; }
    public double getMinElevation() { return minElevation; }
    public double getMaxRange() { return maxRange; }
    public double getUplinkCapacity() { return uplinkCapacity; }
    public double getDownlinkCapacity() { return downlinkCapacity; }

    /**
     * 获取有效传输容量（上下行较小值）
     */
    public double getEffectiveCapacity() {
        return Math.min(uplinkCapacity, downlinkCapacity);
    }

    @Override
    public String toString() {
        return String.format("RelaySatellite[%s: %s @ %.1f°]",
            id, name, longitude);
    }
}
