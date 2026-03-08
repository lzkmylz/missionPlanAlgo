package orekit.visibility;

import java.io.Serializable;

/**
 * 卫星参数
 *
 * 从Python端传入的卫星轨道参数
 */
public class SatelliteParameters implements Serializable {

    private static final long serialVersionUID = 1L;

    private String id;
    private String name;
    private String orbitType;
    private double semiMajorAxis;
    private double eccentricity;
    private double inclination;
    private double raan;
    private double argOfPerigee;
    private double meanAnomaly;
    private double altitude;
    private String epoch;

    // 物理参数（用于HPOP/J4传播）
    private double mass = 100.0;              // kg，默认光学卫星质量
    private double dragArea = 5.0;            // m²，默认光学卫星阻力面积
    private double reflectivity = 1.5;        // 反射系数
    private double dragCoefficient = 2.2;     // 阻力系数

    public SatelliteParameters() {
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

    public String getOrbitType() {
        return orbitType;
    }

    public void setOrbitType(String orbitType) {
        this.orbitType = orbitType;
    }

    public double getSemiMajorAxis() {
        return semiMajorAxis;
    }

    public void setSemiMajorAxis(double semiMajorAxis) {
        this.semiMajorAxis = semiMajorAxis;
    }

    public double getEccentricity() {
        return eccentricity;
    }

    public void setEccentricity(double eccentricity) {
        this.eccentricity = eccentricity;
    }

    public double getInclination() {
        return inclination;
    }

    public void setInclination(double inclination) {
        this.inclination = inclination;
    }

    public double getRaan() {
        return raan;
    }

    public void setRaan(double raan) {
        this.raan = raan;
    }

    public double getArgOfPerigee() {
        return argOfPerigee;
    }

    public void setArgOfPerigee(double argOfPerigee) {
        this.argOfPerigee = argOfPerigee;
    }

    public double getMeanAnomaly() {
        return meanAnomaly;
    }

    public void setMeanAnomaly(double meanAnomaly) {
        this.meanAnomaly = meanAnomaly;
    }

    public double getAltitude() {
        return altitude;
    }

    public void setAltitude(double altitude) {
        this.altitude = altitude;
    }

    public String getEpoch() {
        return epoch;
    }

    public void setEpoch(String epoch) {
        this.epoch = epoch;
    }

    public double getMass() {
        return mass;
    }

    public void setMass(double mass) {
        this.mass = mass;
    }

    public double getDragArea() {
        return dragArea;
    }

    public void setDragArea(double dragArea) {
        this.dragArea = dragArea;
    }

    public double getReflectivity() {
        return reflectivity;
    }

    public void setReflectivity(double reflectivity) {
        this.reflectivity = reflectivity;
    }

    public double getDragCoefficient() {
        return dragCoefficient;
    }

    public void setDragCoefficient(double dragCoefficient) {
        this.dragCoefficient = dragCoefficient;
    }

    @Override
    public String toString() {
        return "SatelliteParameters{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", altitude=" + altitude +
                '}';
    }
}
