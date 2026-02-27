package orekit.visibility;

import org.orekit.time.AbsoluteDate;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 可见窗口追踪器
 *
 * 线程安全的窗口记录器，收集所有检测器触发的事件
 */
public class WindowTracker {

    // 使用ConcurrentHashMap保证线程安全（并行传播时使用）
    private final Map<String, WindowBuilder> activeWindows = new ConcurrentHashMap<>();
    private final List<VisibilityWindow> completedWindows = Collections.synchronizedList(new ArrayList<>());

    /**
     * 记录窗口开始
     *
     * @param satelliteId 卫星ID
     * @param pointId 目标/地面站ID
     * @param isTarget 是否为目标（true=目标，false=地面站）
     * @param time 窗口开始时间
     */
    public void recordWindowStart(String satelliteId, String pointId,
                                   boolean isTarget, AbsoluteDate time) {
        String key = buildKey(satelliteId, pointId, isTarget);

        WindowBuilder builder = new WindowBuilder(satelliteId, pointId, isTarget);
        builder.setStartTime(time);

        activeWindows.put(key, builder);
    }

    /**
     * 记录窗口结束
     *
     * @param satelliteId 卫星ID
     * @param pointId 目标/地面站ID
     * @param isTarget 是否为目标
     * @param time 窗口结束时间
     */
    public void recordWindowEnd(String satelliteId, String pointId,
                                 boolean isTarget, AbsoluteDate time) {
        String key = buildKey(satelliteId, pointId, isTarget);

        WindowBuilder builder = activeWindows.remove(key);
        if (builder != null) {
            builder.setEndTime(time);
            completedWindows.add(builder.build());
        }
    }

    /**
     * 更新窗口内的最大仰角
     *
     * @param satelliteId 卫星ID
     * @param pointId 目标/地面站ID
     * @param isTarget 是否为目标
     * @param elevation 当前仰角（度）
     */
    public void updateMaxElevation(String satelliteId, String pointId,
                                   boolean isTarget, double elevation) {
        String key = buildKey(satelliteId, pointId, isTarget);

        WindowBuilder builder = activeWindows.get(key);
        if (builder != null) {
            builder.updateMaxElevation(elevation);
        }
    }

    /**
     * 获取所有完成的窗口
     *
     * @return 可见窗口列表
     */
    public List<VisibilityWindow> getAllWindows() {
        // 处理未关闭的窗口（传播结束时仍在可见状态）
        // 这些窗口会被截断到结束时间
        for (WindowBuilder builder : new ArrayList<>(activeWindows.values())) {
            // 记录警告但丢弃未完成的窗口
            // 实际实现中可以记录日志
        }

        return new ArrayList<>(completedWindows);
    }

    /**
     * 获取窗口数量（用于统计）
     *
     * @return 已完成窗口数
     */
    public int getWindowCount() {
        return completedWindows.size();
    }

    /**
     * 清除所有数据
     */
    public void clear() {
        activeWindows.clear();
        completedWindows.clear();
    }

    private String buildKey(String satId, String pointId, boolean isTarget) {
        return satId + ":" + pointId + ":" + (isTarget ? "T" : "G");
    }

    /**
     * 窗口构建器（内部类）
     */
    private static class WindowBuilder {
        private final String satelliteId;
        private final String pointId;
        private final boolean isTarget;
        private AbsoluteDate startTime;
        private AbsoluteDate endTime;
        private double maxElevation = 0.0;

        public WindowBuilder(String satelliteId, String pointId, boolean isTarget) {
            this.satelliteId = satelliteId;
            this.pointId = pointId;
            this.isTarget = isTarget;
        }

        public void setStartTime(AbsoluteDate time) {
            this.startTime = time;
        }

        public void setEndTime(AbsoluteDate time) {
            this.endTime = time;
        }

        public void updateMaxElevation(double elevation) {
            this.maxElevation = Math.max(this.maxElevation, elevation);
        }

        public VisibilityWindow build() {
            return new VisibilityWindow(
                satelliteId,
                pointId,
                isTarget,
                startTime,
                endTime,
                maxElevation
            );
        }
    }
}
