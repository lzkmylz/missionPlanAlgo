package orekit.visibility;

import org.hipparchus.ode.events.Action;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.events.EventDetector;
import org.orekit.propagation.events.handlers.EventHandler;

/**
 * 仰角检测事件处理器
 *
 * 处理Orekit的仰角检测器触发的事件，记录窗口开始和结束
 */
public class ElevationHandler implements EventHandler {

    private final String satelliteId;
    private final String pointId;
    private final boolean isTarget;
    private final WindowTracker tracker;

    /**
     * 创建仰角事件处理器
     *
     * @param satelliteId 卫星ID
     * @param pointId 目标/地面站ID
     * @param isTarget 是否为目标（true=目标，false=地面站）
     * @param tracker 窗口追踪器
     */
    public ElevationHandler(String satelliteId, String pointId,
                           boolean isTarget, WindowTracker tracker) {
        this.satelliteId = satelliteId;
        this.pointId = pointId;
        this.isTarget = isTarget;
        this.tracker = tracker;
    }

    @Override
    public Action eventOccurred(SpacecraftState s, EventDetector detector,
                                boolean increasing) {
        if (increasing) {
            // 仰角从零增加到阈值以上：窗口开始
            tracker.recordWindowStart(satelliteId, pointId, isTarget, s.getDate());
            return Action.CONTINUE;
        } else {
            // 仰角从阈值以上减小到零：窗口结束
            tracker.recordWindowEnd(satelliteId, pointId, isTarget, s.getDate());
            return Action.CONTINUE;
        }
    }

    @Override
    public SpacecraftState resetState(EventDetector detector, SpacecraftState oldState) {
        // 不需要重置状态
        return oldState;
    }
}
