package orekit.visibility.model;

import org.orekit.time.AbsoluteDate;
import java.util.List;
import java.util.ArrayList;

/**
 * 可见窗口数据类
 *
 * 存储卫星-目标可见窗口的完整信息，包括姿态采样数据。
 */
public class VisibilityWindow {

    /**
     * 姿态采样点数据类
     * 存储窗口内特定时间点的姿态角
     */
    public static class AttitudeSample {
        public final double timestamp;  // 相对于窗口开始的秒数
        public final double roll;       // 滚转角（度）
        public final double pitch;      // 俯仰角（度）

        public AttitudeSample(double timestamp, double roll, double pitch) {
            this.timestamp = timestamp;
            this.roll = roll;
            this.pitch = pitch;
        }

        @Override
        public String toString() {
            return String.format("AttitudeSample[t=%.1f, r=%.2f, p=%.2f]", timestamp, roll, pitch);
        }
    }
    private final String satelliteId;
    private final String targetId;
    private final AbsoluteDate startTime;
    private final AbsoluteDate endTime;
    private final double durationSeconds;
    private final double maxElevation;
    private final AbsoluteDate maxElevationTime;
    private final double entryAzimuth;
    private final double exitAzimuth;
    private final double qualityScore;
    private final String confidence;
    private final List<AttitudeSample> attitudeSamples;  // 姿态采样数据
    private final boolean attitudeFeasible;               // 是否通过姿态约束检查

    // ISL元数据（非ISL窗口时均为null/0）
    private String islLinkType = null;
    private double islDataRateMbps = 0.0;
    private double islLinkMarginDb = 0.0;
    private double islDistanceKm = 0.0;
    private double islRelativeVelocityKmS = 0.0;
    private double islAtpSetupTimeS = 0.0;

    /**
     * 创建可见窗口（带姿态数据）
     */
    public VisibilityWindow(String satelliteId, String targetId,
                           AbsoluteDate startTime, AbsoluteDate endTime,
                           double durationSeconds, double maxElevation,
                           AbsoluteDate maxElevationTime, double entryAzimuth,
                           double exitAzimuth, double qualityScore,
                           String confidence, List<AttitudeSample> attitudeSamples,
                           boolean attitudeFeasible) {
        this.satelliteId = satelliteId;
        this.targetId = targetId;
        this.startTime = startTime;
        this.endTime = endTime;
        this.durationSeconds = durationSeconds;
        this.maxElevation = maxElevation;
        this.maxElevationTime = maxElevationTime;
        this.entryAzimuth = entryAzimuth;
        this.exitAzimuth = exitAzimuth;
        this.qualityScore = qualityScore;
        this.confidence = confidence;
        this.attitudeSamples = attitudeSamples != null ? attitudeSamples : new ArrayList<>();
        this.attitudeFeasible = attitudeFeasible;
    }

    /**
     * 创建可见窗口
     */
    public VisibilityWindow(String satelliteId, String targetId,
                           AbsoluteDate startTime, AbsoluteDate endTime,
                           double durationSeconds, double maxElevation,
                           AbsoluteDate maxElevationTime, double entryAzimuth,
                           double exitAzimuth, double qualityScore,
                           String confidence) {
        this(satelliteId, targetId, startTime, endTime, durationSeconds, maxElevation,
             maxElevationTime, entryAzimuth, exitAzimuth, qualityScore, confidence,
             new ArrayList<>(), true);
    }

    // 简化的构造函数
    public VisibilityWindow(String satelliteId, String targetId,
                           AbsoluteDate startTime, AbsoluteDate endTime,
                           double maxElevation) {
        this(satelliteId, targetId, startTime, endTime,
             endTime.durationFrom(startTime), maxElevation,
             startTime, 0.0, 0.0, 0.5, "HIGH",
             new ArrayList<>(), true);
    }

    public String getSatelliteId() { return satelliteId; }
    public String getTargetId() { return targetId; }
    public AbsoluteDate getStartTime() { return startTime; }
    public AbsoluteDate getEndTime() { return endTime; }
    public double getDurationSeconds() { return durationSeconds; }
    public double getMaxElevation() { return maxElevation; }
    public AbsoluteDate getMaxElevationTime() { return maxElevationTime; }
    public double getEntryAzimuth() { return entryAzimuth; }
    public double getExitAzimuth() { return exitAzimuth; }
    public double getQualityScore() { return qualityScore; }
    public String getConfidence() { return confidence; }
    public List<AttitudeSample> getAttitudeSamples() { return attitudeSamples; }
    public boolean isAttitudeFeasible() { return attitudeFeasible; }

    // ===== ISL元数据 =====

    /**
     * 设置ISL链路元数据（仅ISL窗口调用）。
     */
    public void setISLMetadata(String linkType, double dataRateMbps, double linkMarginDb,
                               double distanceKm, double relVelKmS, double atpSetupTimeS) {
        this.islLinkType = linkType;
        this.islDataRateMbps = dataRateMbps;
        this.islLinkMarginDb = linkMarginDb;
        this.islDistanceKm = distanceKm;
        this.islRelativeVelocityKmS = relVelKmS;
        this.islAtpSetupTimeS = atpSetupTimeS;
    }

    /** 是否为ISL窗口 */
    public boolean isISLWindow() { return islLinkType != null; }
    public String getIslLinkType() { return islLinkType; }
    public double getIslDataRateMbps() { return islDataRateMbps; }
    public double getIslLinkMarginDb() { return islLinkMarginDb; }
    public double getIslDistanceKm() { return islDistanceKm; }
    public double getIslRelativeVelocityKmS() { return islRelativeVelocityKmS; }
    public double getIslAtpSetupTimeS() { return islAtpSetupTimeS; }

    @Override
    public String toString() {
        return String.format("VisibilityWindow[%s-%s: %s to %s, max_el=%.1f°]",
            satelliteId, targetId, startTime, endTime, maxElevation);
    }
}
