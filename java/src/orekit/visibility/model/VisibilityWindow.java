package orekit.visibility.model;

import org.orekit.time.AbsoluteDate;

/**
 * 可见窗口数据类
 *
 * 存储卫星-目标可见窗口的完整信息。
 */
public class VisibilityWindow {
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

    /**
     * 创建可见窗口
     */
    public VisibilityWindow(String satelliteId, String targetId,
                           AbsoluteDate startTime, AbsoluteDate endTime,
                           double durationSeconds, double maxElevation,
                           AbsoluteDate maxElevationTime, double entryAzimuth,
                           double exitAzimuth, double qualityScore,
                           String confidence) {
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
    }

    // 简化的构造函数
    public VisibilityWindow(String satelliteId, String targetId,
                           AbsoluteDate startTime, AbsoluteDate endTime,
                           double maxElevation) {
        this(satelliteId, targetId, startTime, endTime,
             endTime.durationFrom(startTime), maxElevation,
             startTime, 0.0, 0.0, 0.5, "HIGH");
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

    @Override
    public String toString() {
        return String.format("VisibilityWindow[%s-%s: %s to %s, max_el=%.1f°]",
            satelliteId, targetId, startTime, endTime, maxElevation);
    }
}
