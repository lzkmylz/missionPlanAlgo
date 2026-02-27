package orekit.visibility.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 批量计算结果类
 *
 * 存储所有卫星-目标对的可见窗口计算结果。
 */
public class BatchResult {
    // (satelliteId, targetId) -> List<VisibilityWindow>
    private final Map<String, List<VisibilityWindow>> windows;
    private final List<ComputationError> errors;
    private final ComputationStatistics statistics;

    public BatchResult() {
        this.windows = new HashMap<>();
        this.errors = new ArrayList<>();
        this.statistics = new ComputationStatistics();
    }

    /**
     * 添加可见窗口
     */
    public void addWindow(String satelliteId, String targetId, VisibilityWindow window) {
        String key = makeKey(satelliteId, targetId);
        windows.computeIfAbsent(key, k -> new ArrayList<>()).add(window);
        statistics.incrementWindowCount();
    }

    /**
     * 添加多个可见窗口
     */
    public void addWindows(String satelliteId, String targetId, List<VisibilityWindow> windowList) {
        String key = makeKey(satelliteId, targetId);
        windows.computeIfAbsent(key, k -> new ArrayList<>()).addAll(windowList);
        statistics.addToWindowCount(windowList.size());
    }

    /**
     * 添加错误信息
     */
    public void addError(String satelliteId, String targetId, String errorType, String errorMessage) {
        errors.add(new ComputationError(satelliteId, targetId, errorType, errorMessage));
        statistics.incrementErrorCount();
    }

    /**
     * 获取指定卫星-目标对的窗口
     */
    public List<VisibilityWindow> getWindows(String satelliteId, String targetId) {
        return windows.getOrDefault(makeKey(satelliteId, targetId), new ArrayList<>());
    }

    /**
     * 获取所有窗口
     */
    public Map<String, List<VisibilityWindow>> getAllWindows() {
        return new HashMap<>(windows);
    }

    /**
     * 获取所有错误
     */
    public List<ComputationError> getErrors() {
        return new ArrayList<>(errors);
    }

    public ComputationStatistics getStatistics() {
        return statistics;
    }

    private String makeKey(String satelliteId, String targetId) {
        return satelliteId + "-" + targetId;
    }

    /**
     * 计算统计信息内部类
     */
    public static class ComputationStatistics {
        private int totalPairs = 0;
        private int pairsWithWindows = 0;
        private int totalWindows = 0;
        private int errorCount = 0;
        private long computationTimeMs = 0;
        private int coarseScanPoints = 0;
        private int fineScanPoints = 0;

        public void setTotalPairs(int totalPairs) { this.totalPairs = totalPairs; }
        public void incrementPairsWithWindows() { this.pairsWithWindows++; }
        public void incrementWindowCount() { this.totalWindows++; }
        public void addToWindowCount(int count) { this.totalWindows += count; }
        public void incrementErrorCount() { this.errorCount++; }
        public void setComputationTimeMs(long time) { this.computationTimeMs = time; }
        public void setCoarseScanPoints(int points) { this.coarseScanPoints = points; }
        public void setFineScanPoints(int points) { this.fineScanPoints = points; }

        public int getTotalPairs() { return totalPairs; }
        public int getPairsWithWindows() { return pairsWithWindows; }
        public int getTotalWindows() { return totalWindows; }
        public int getErrorCount() { return errorCount; }
        public long getComputationTimeMs() { return computationTimeMs; }
        public int getCoarseScanPoints() { return coarseScanPoints; }
        public int getFineScanPoints() { return fineScanPoints; }
    }

    /**
     * 计算错误信息内部类
     */
    public static class ComputationError {
        private final String satelliteId;
        private final String targetId;
        private final String errorType;
        private final String errorMessage;

        public ComputationError(String satelliteId, String targetId,
                               String errorType, String errorMessage) {
            this.satelliteId = satelliteId;
            this.targetId = targetId;
            this.errorType = errorType;
            this.errorMessage = errorMessage;
        }

        public String getSatelliteId() { return satelliteId; }
        public String getTargetId() { return targetId; }
        public String getErrorType() { return errorType; }
        public String getErrorMessage() { return errorMessage; }
    }
}
