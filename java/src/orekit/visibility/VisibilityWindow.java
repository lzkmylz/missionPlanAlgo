package orekit.visibility;

import org.orekit.time.AbsoluteDate;
import java.io.Serializable;

/**
 * 可见窗口数据类
 *
 * 表示卫星与地面点之间的可见时间窗口
 */
public class VisibilityWindow implements Serializable {

    private static final long serialVersionUID = 1L;

    private final String satelliteId;
    private final String pointId;
    private final boolean isTarget;  // true=目标, false=地面站
    private final AbsoluteDate startTime;
    private final AbsoluteDate endTime;
    private final double maxElevation;
    private final double durationSeconds;

    public VisibilityWindow(String satelliteId, String pointId, boolean isTarget,
                           AbsoluteDate startTime, AbsoluteDate endTime,
                           double maxElevation) {
        this.satelliteId = satelliteId;
        this.pointId = pointId;
        this.isTarget = isTarget;
        this.startTime = startTime;
        this.endTime = endTime;
        this.maxElevation = maxElevation;
        this.durationSeconds = endTime.durationFrom(startTime);
    }

    // Getters
    public String getSatelliteId() {
        return satelliteId;
    }

    public String getPointId() {
        return pointId;
    }

    public boolean isTarget() {
        return isTarget;
    }

    public AbsoluteDate getStartTime() {
        return startTime;
    }

    public AbsoluteDate getEndTime() {
        return endTime;
    }

    public double getMaxElevation() {
        return maxElevation;
    }

    public double getDurationSeconds() {
        return durationSeconds;
    }

    /**
     * 获取用于分组的key
     */
    public String getKey() {
        return satelliteId + ":" + pointId;
    }

    /**
     * 创建精化边界后的新窗口
     */
    public VisibilityWindow withRefinedBoundaries(AbsoluteDate newStart,
                                                   AbsoluteDate newEnd) {
        return new VisibilityWindow(satelliteId, pointId, isTarget,
                                   newStart, newEnd, maxElevation);
    }

    @Override
    public String toString() {
        return String.format("VisibilityWindow[%s -> %s, %s to %s, %.1f°]",
            satelliteId, pointId, startTime, endTime, maxElevation);
    }
}
