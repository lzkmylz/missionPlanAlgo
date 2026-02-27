package orekit.visibility;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * 批量计算结果
 *
 * 包含所有卫星-目标/地面站的可见窗口
 */
public class BatchResult implements Serializable {

    private static final long serialVersionUID = 1L;

    private Map<String, List<VisibilityWindow>> targetWindows;
    private Map<String, List<VisibilityWindow>> groundStationWindows;
    private ComputationStats stats;

    public BatchResult(Map<String, List<VisibilityWindow>> targetWindows,
                      Map<String, List<VisibilityWindow>> groundStationWindows,
                      ComputationStats stats) {
        this.targetWindows = targetWindows;
        this.groundStationWindows = groundStationWindows;
        this.stats = stats;
    }

    // Getters
    public Map<String, List<VisibilityWindow>> getTargetWindows() {
        return targetWindows;
    }

    public Map<String, List<VisibilityWindow>> getGroundStationWindows() {
        return groundStationWindows;
    }

    public ComputationStats getStats() {
        return stats;
    }

    /**
     * 获取总窗口数
     */
    public int getTotalWindowCount() {
        int count = targetWindows.values().stream().mapToInt(List::size).sum();
        count += groundStationWindows.values().stream().mapToInt(List::size).sum();
        return count;
    }
}
