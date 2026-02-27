package orekit.visibility;

import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import java.util.*;

/**
 * Python调用入口
 *
 * 提供JPype可以直接调用的静态方法
 */
public class PythonBridge {

    private static final BatchVisibilityCalculator calculator = new BatchVisibilityCalculator();

    /**
     * Python调用的批量计算方法
     *
     * @param satellites 卫星列表
     * @param targets 目标列表
     * @param groundStations 地面站列表
     * @param startTimeIso 开始时间（ISO 8601格式）
     * @param endTimeIso 结束时间（ISO 8601格式）
     * @param config 计算配置
     * @return 结果Map
     */
    public static Map<String, Object> computeVisibilityBatch(
            List<SatelliteParameters> satellites,
            List<GroundPoint> targets,
            List<GroundPoint> groundStations,
            String startTimeIso,
            String endTimeIso,
            ComputationConfig config) {

        try {
            // 解析时间 - Orekit 12使用字符串+时间尺度构造函数
            AbsoluteDate startTime = new AbsoluteDate(startTimeIso, TimeScalesFactory.getUTC());
            AbsoluteDate endTime = new AbsoluteDate(endTimeIso, TimeScalesFactory.getUTC());

            // 执行计算
            BatchResult result = calculator.computeAllWindows(
                satellites, targets, groundStations,
                startTime, endTime, config
            );

            // 转换为Python友好的Map格式
            return convertToPythonMap(result);

        } catch (Exception e) {
            // 返回错误信息
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", true);
            errorResult.put("errorMessage", e.getMessage());
            errorResult.put("errorType", e.getClass().getName());
            return errorResult;
        }
    }

    /**
     * 转换为Python友好的Map
     */
    private static Map<String, Object> convertToPythonMap(BatchResult result) {
        Map<String, Object> map = new HashMap<>();

        // 转换目标窗口
        List<Map<String, Object>> targetWindowsList = new ArrayList<>();
        for (Map.Entry<String, List<VisibilityWindow>> entry :
             result.getTargetWindows().entrySet()) {
            for (VisibilityWindow w : entry.getValue()) {
                Map<String, Object> windowMap = new HashMap<>();
                windowMap.put("satelliteId", w.getSatelliteId());
                windowMap.put("targetId", w.getPointId());
                windowMap.put("startTime", w.getStartTime().toString());
                windowMap.put("endTime", w.getEndTime().toString());
                windowMap.put("maxElevation", w.getMaxElevation());
                windowMap.put("durationSeconds", w.getDurationSeconds());
                targetWindowsList.add(windowMap);
            }
        }
        map.put("targetWindows", targetWindowsList);

        // 转换地面站窗口
        List<Map<String, Object>> gsWindowsList = new ArrayList<>();
        for (Map.Entry<String, List<VisibilityWindow>> entry :
             result.getGroundStationWindows().entrySet()) {
            for (VisibilityWindow w : entry.getValue()) {
                Map<String, Object> windowMap = new HashMap<>();
                windowMap.put("satelliteId", w.getSatelliteId());
                windowMap.put("targetId", w.getPointId());
                windowMap.put("startTime", w.getStartTime().toString());
                windowMap.put("endTime", w.getEndTime().toString());
                windowMap.put("maxElevation", w.getMaxElevation());
                windowMap.put("durationSeconds", w.getDurationSeconds());
                gsWindowsList.add(windowMap);
            }
        }
        map.put("groundStationWindows", gsWindowsList);

        // 统计信息
        ComputationStats stats = result.getStats();
        Map<String, Object> statsMap = new HashMap<>();
        statsMap.put("computationTimeMs", stats.getComputationTimeMs());
        statsMap.put("nWindows", stats.getNWindowsFound());
        statsMap.put("memoryUsageMb", stats.getMemoryUsageMb());
        map.put("stats", statsMap);

        return map;
    }
}
