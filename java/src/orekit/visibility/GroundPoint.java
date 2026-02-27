package orekit.visibility;

import java.io.Serializable;

/**
 * 地面点参数
 *
 * 表示目标或地面站的位置信息
 */
public class GroundPoint implements Serializable {

    private static final long serialVersionUID = 1L;

    private String id;
    private String name;
    private double longitude;   // 度
    private double latitude;    // 度
    private double altitude;    // 米
    private double minElevation; // 最小仰角（度，仅地面站使用）

    public GroundPoint() {
        this.minElevation = 0.0;
    }

    public GroundPoint(String id, String name, double longitude,
                       double latitude, double altitude) {
        this.id = id;
        this.name = name;
        this.longitude = longitude;
        this.latitude = latitude;
        this.altitude = altitude;
        this.minElevation = 0.0;
    }

    // Getters and Setters
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public double getLongitude() {
        return longitude;
    }

    public void setLongitude(double longitude) {
        this.longitude = longitude;
    }

    public double getLatitude() {
        return latitude;
    }

    public void setLatitude(double latitude) {
        this.latitude = latitude;
    }

    public double getAltitude() {
        return altitude;
    }

    public void setAltitude(double altitude) {
        this.altitude = altitude;
    }

    public double getMinElevation() {
        return minElevation;
    }

    public void setMinElevation(double minElevation) {
        this.minElevation = minElevation;
    }

    @Override
    public String toString() {
        return "GroundPoint{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", longitude=" + longitude +
                ", latitude=" + latitude +
                ", altitude=" + altitude +
                '}';
    }
}
