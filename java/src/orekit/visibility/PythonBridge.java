package orekit.visibility;

// 注意：BatchResult有两个版本
// - orekit.visibility.BatchResult (用于BatchVisibilityCalculator)
// - orekit.visibility.model.BatchResult (用于OptimizedVisibilityCalculator)
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;

import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;

import java.util.*;

/**
 * Python调用入口
 *
 * 提供JPype可以直接调用的静态方法
 */
public class PythonBridge {

    // 延迟初始化：避免类加载时就创建计算器（可能触发Orekit异常）
    private static BatchVisibilityCalculator calculator = null;
    private static OptimizedVisibilityCalculator optimizedCalculator = null;

    /**
     * 获取 BatchVisibilityCalculator 实例（延迟初始化）
     */
    private static synchronized BatchVisibilityCalculator getCalculator() {
        if (calculator == null) {
            calculator = new BatchVisibilityCalculator();
        }
        return calculator;
    }

    /**
     * Python调用的批量计算方法（原始版本）
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
            // 解析时间
            AbsoluteDate startTime = new AbsoluteDate(startTimeIso, TimeScalesFactory.getUTC());
            AbsoluteDate endTime = new AbsoluteDate(endTimeIso, TimeScalesFactory.getUTC());

            // 执行计算 (使用orekit.visibility.BatchResult，不是model包中的)
            orekit.visibility.BatchResult result = getCalculator().computeAllWindows(
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
     * 批量计算所有可见性窗口（优化版本）
     *
     * 使用轨道预计算缓存和并行计算，大幅提升性能。
     *
     * @param satelliteJson 卫星配置JSON数组字符串
     * @param targetJson 目标配置JSON数组字符串
     * @param startTimeIso 开始时间（ISO 8601格式）
     * @param endTimeIso 结束时间（ISO 8601格式）
     * @param coarseStep 粗扫描步长（秒）
     * @param fineStep 精化步长（秒）
     * @return 结果JSON字符串
     */
    public static String computeAllWindowsOptimized(
            String satelliteJson,
            String targetJson,
            String startTimeIso,
            String endTimeIso,
            double coarseStep,
            double fineStep) {

        try {
            // 解析JSON
            List<SatelliteConfig> satellites = parseSatellitesFromJson(satelliteJson);
            List<TargetConfig> targets = parseTargetsFromJson(targetJson);

            // 解析时间
            AbsoluteDate startTime = new AbsoluteDate(startTimeIso, TimeScalesFactory.getUTC());
            AbsoluteDate endTime = new AbsoluteDate(endTimeIso, TimeScalesFactory.getUTC());

            // 使用优化计算器
            if (optimizedCalculator == null) {
                optimizedCalculator = new OptimizedVisibilityCalculator();
            }

            // 使用model包中的BatchResult (OptimizedVisibilityCalculator返回的)
            orekit.visibility.model.BatchResult result = optimizedCalculator.computeAllVisibilityWindows(
                satellites, targets, startTime, endTime, coarseStep, fineStep
            );

            // 转换为JSON返回
            return convertToJson(result);

        } catch (Exception e) {
            return "{\"error\":true,\"message\":\"" + escapeJson(e.getMessage()) + "\"}";
        }
    }

    /**
     * 解析卫星配置JSON
     */
    private static List<SatelliteConfig> parseSatellitesFromJson(String json) {
        List<SatelliteConfig> satellites = new ArrayList<>();

        // 简单的JSON解析（假设格式为JSON数组）
        // 实际项目中应使用Jackson或Gson库
        json = json.trim();
        if (json.startsWith("[")) json = json.substring(1);
        if (json.endsWith("]")) json = json.substring(0, json.length() - 1);

        if (json.isEmpty()) return satellites;

        // 分割对象
        String[] objects = json.split("\\},\\s*\\{");
        for (String obj : objects) {
            obj = obj.replace("{", "").replace("}", "").trim();
            if (obj.isEmpty()) continue;

            String id = "";
            String tleLine1 = "";
            String tleLine2 = "";
            double minElevation = 5.0;

            String[] fields = obj.split(",");
            for (String field : fields) {
                String[] kv = field.split(":");
                if (kv.length == 2) {
                    String key = kv[0].trim().replace("\"", "");
                    String value = kv[1].trim().replace("\"", "");

                    switch (key) {
                        case "id":
                            id = value;
                            break;
                        case "tle_line1":
                        case "tleLine1":
                            tleLine1 = value;
                            break;
                        case "tle_line2":
                        case "tleLine2":
                            tleLine2 = value;
                            break;
                        case "min_elevation":
                        case "minElevation":
                            try {
                                minElevation = Double.parseDouble(value);
                            } catch (NumberFormatException ignored) {}
                            break;
                    }
                }
            }

            satellites.add(new SatelliteConfig(id, tleLine1, tleLine2, minElevation, 0.0));
        }

        return satellites;
    }

    /**
     * 解析目标配置JSON
     */
    private static List<TargetConfig> parseTargetsFromJson(String json) {
        List<TargetConfig> targets = new ArrayList<>();

        json = json.trim();
        if (json.startsWith("[")) json = json.substring(1);
        if (json.endsWith("]")) json = json.substring(0, json.length() - 1);

        if (json.isEmpty()) return targets;

        String[] objects = json.split("\\},\\s*\\{");
        for (String obj : objects) {
            obj = obj.replace("{", "").replace("}", "").trim();
            if (obj.isEmpty()) continue;

            String id = "";
            double longitude = 0.0;
            double latitude = 0.0;
            double altitude = 0.0;
            int minDuration = 60;

            String[] fields = obj.split(",");
            for (String field : fields) {
                String[] kv = field.split(":");
                if (kv.length == 2) {
                    String key = kv[0].trim().replace("\"", "");
                    String value = kv[1].trim().replace("\"", "");

                    try {
                        switch (key) {
                            case "id":
                                id = value;
                                break;
                            case "longitude":
                            case "lon":
                                longitude = Double.parseDouble(value);
                                break;
                            case "latitude":
                            case "lat":
                                latitude = Double.parseDouble(value);
                                break;
                            case "altitude":
                            case "alt":
                                altitude = Double.parseDouble(value);
                                break;
                            case "min_duration":
                            case "minDuration":
                                minDuration = Integer.parseInt(value);
                                break;
                        }
                    } catch (NumberFormatException ignored) {}
                }
            }

            targets.add(new TargetConfig(id, longitude, latitude, altitude, minDuration, 5));
        }

        return targets;
    }

    /**
     * 将结果转换为JSON字符串 (使用model包中的类)
     */
    private static String convertToJson(orekit.visibility.model.BatchResult result) {
        StringBuilder sb = new StringBuilder();
        sb.append("{");

        // windows
        sb.append("\"windows\":[");
        boolean first = true;
        for (List<orekit.visibility.model.VisibilityWindow> windowList : result.getAllWindows().values()) {
            for (orekit.visibility.model.VisibilityWindow w : windowList) {
                if (!first) sb.append(",");
                first = false;
                sb.append(windowToJson(w));
            }
        }
        sb.append("],");

        // stats
        orekit.visibility.model.BatchResult.ComputationStatistics stats = result.getStatistics();
        sb.append("\"stats\":{");
        sb.append("\"computationTimeMs\":").append(stats.getComputationTimeMs()).append(",");
        sb.append("\"totalPairs\":").append(stats.getTotalPairs()).append(",");
        sb.append("\"totalWindows\":").append(stats.getTotalWindows()).append(",");
        sb.append("\"errorCount\":").append(stats.getErrorCount());
        sb.append("}");

        sb.append("}");
        return sb.toString();
    }

    /**
     * 将单个窗口转换为JSON (使用model包中的VisibilityWindow)
     */
    private static String windowToJson(orekit.visibility.model.VisibilityWindow w) {
        return "{" +
            "\"satelliteId\":\"" + escapeJson(w.getSatelliteId()) + "\"," +
            "\"targetId\":\"" + escapeJson(w.getTargetId()) + "\"," +
            "\"startTime\":\"" + w.getStartTime().toString() + "\"," +
            "\"endTime\":\"" + w.getEndTime().toString() + "\"," +
            "\"durationSeconds\":" + w.getDurationSeconds() + "," +
            "\"maxElevation\":" + w.getMaxElevation() + "," +
            "\"qualityScore\":" + w.getQualityScore() + "," +
            "\"confidence\":\"" + w.getConfidence() + "\"" +
            "}";
    }

    /**
     * 转义JSON字符串
     */
    private static String escapeJson(String s) {
        if (s == null) return "";
        return s.replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t");
    }

    /**
     * 转换为Python友好的Map（原始版本兼容）
     * 使用 orekit.visibility.BatchResult (不是model包中的)
     */
    private static Map<String, Object> convertToPythonMap(orekit.visibility.BatchResult result) {
        Map<String, Object> map = new HashMap<>();

        // 转换窗口 (使用 orekit.visibility.VisibilityWindow)
        List<Map<String, Object>> windowsList = new ArrayList<>();
        for (Map.Entry<String, List<orekit.visibility.VisibilityWindow>> entry : result.getTargetWindows().entrySet()) {
            for (orekit.visibility.VisibilityWindow w : entry.getValue()) {
                Map<String, Object> windowMap = new HashMap<>();
                windowMap.put("satelliteId", w.getSatelliteId());
                windowMap.put("targetId", w.getPointId());  // 使用 getPointId 而不是 getTargetId
                windowMap.put("startTime", w.getStartTime().toString());
                windowMap.put("endTime", w.getEndTime().toString());
                windowMap.put("maxElevation", w.getMaxElevation());
                windowMap.put("durationSeconds", w.getDurationSeconds());
                windowsList.add(windowMap);
            }
        }
        // 添加地面站窗口
        for (Map.Entry<String, List<orekit.visibility.VisibilityWindow>> entry : result.getGroundStationWindows().entrySet()) {
            for (orekit.visibility.VisibilityWindow w : entry.getValue()) {
                Map<String, Object> windowMap = new HashMap<>();
                windowMap.put("satelliteId", w.getSatelliteId());
                windowMap.put("targetId", w.getPointId());  // 使用 getPointId 而不是 getTargetId
                windowMap.put("startTime", w.getStartTime().toString());
                windowMap.put("endTime", w.getEndTime().toString());
                windowMap.put("maxElevation", w.getMaxElevation());
                windowMap.put("durationSeconds", w.getDurationSeconds());
                windowsList.add(windowMap);
            }
        }
        map.put("windows", windowsList);

        // 统计信息 (使用正确的getter方法名)
        orekit.visibility.ComputationStats stats = result.getStats();
        Map<String, Object> statsMap = new HashMap<>();
        statsMap.put("computationTimeMs", stats.getComputationTimeMs());
        statsMap.put("nSatellites", stats.getNSatellites());
        statsMap.put("nTargets", stats.getNTargets());
        statsMap.put("nGroundStations", stats.getNGroundStations());
        statsMap.put("nWindowsFound", stats.getNWindowsFound());
        map.put("stats", statsMap);

        return map;
    }

    /**
     * 批量计算可见性窗口并导出轨道数据
     *
     * 使用 OptimizedVisibilityCalculator 进行计算，并在计算完成后导出轨道数据到指定路径。
     * 这是 computeVisibilityBatch 的增强版本，支持轨道数据持久化。
     *
     * @param satellites 卫星列表
     * @param targets 目标列表
     * @param groundStations 地面站列表
     * @param startTimeIso 开始时间（ISO 8601格式）
     * @param endTimeIso 结束时间（ISO 8601格式）
     * @param config 计算配置
     * @param orbitOutputPath 轨道数据输出路径（可选，为null则不导出）
     * @return 结果Map，包含 windows, groundStationWindows, stats，以及 orbitExportStatus
     */
    public static Map<String, Object> computeVisibilityBatchWithOrbitExport(
            List<SatelliteParameters> satellites,
            List<GroundPoint> targets,
            List<GroundPoint> groundStations,
            String startTimeIso,
            String endTimeIso,
            ComputationConfig config,
            String orbitOutputPath) {

        try {
            // 解析时间
            AbsoluteDate startTime = new AbsoluteDate(startTimeIso, TimeScalesFactory.getUTC());
            AbsoluteDate endTime = new AbsoluteDate(endTimeIso, TimeScalesFactory.getUTC());

            // 转换卫星参数到 ExtendedSatelliteConfig
            List<JsonScenarioLoader.ExtendedSatelliteConfig> satConfigs = new ArrayList<>();
            for (SatelliteParameters sat : satellites) {
                // 解析epoch字符串为AbsoluteDate
                AbsoluteDate epoch;
                if (sat.getEpoch() != null && !sat.getEpoch().isEmpty()) {
                    epoch = new AbsoluteDate(sat.getEpoch(), TimeScalesFactory.getUTC());
                } else {
                    epoch = startTime; // 默认使用场景开始时间
                }

                // 根据orbitType确定卫星类型和默认物理参数
                String satType = sat.getOrbitType();
                double mass = sat.getMass();
                double dragArea = sat.getDragArea();
                double reflectivity = sat.getReflectivity();
                double dragCoefficient = sat.getDragCoefficient();

                // 如果没有设置物理参数，根据卫星类型使用默认值
                if (mass <= 0) {
                    mass = "SAR".equalsIgnoreCase(satType) ? 150.0 : 100.0;
                }
                if (dragArea <= 0) {
                    dragArea = "SAR".equalsIgnoreCase(satType) ? 8.0 : 5.0;
                }
                if (reflectivity <= 0) {
                    reflectivity = "SAR".equalsIgnoreCase(satType) ? 1.3 : 1.5;
                }
                if (dragCoefficient <= 0) {
                    dragCoefficient = 2.2;
                }

                satConfigs.add(new JsonScenarioLoader.ExtendedSatelliteConfig(
                    sat.getId(),
                    satType != null ? satType : "SSO",
                    sat.getSemiMajorAxis(),
                    sat.getEccentricity(),
                    sat.getInclination(),
                    sat.getRaan(),
                    sat.getArgOfPerigee(),
                    sat.getMeanAnomaly(),
                    5.0,   // minElevation
                    45.0,  // maxOffNadir
                    epoch,
                    mass,
                    dragArea,
                    reflectivity,
                    dragCoefficient
                ));
            }

            // 转换目标参数到 TargetConfig
            List<orekit.visibility.model.TargetConfig> targetConfigs = new ArrayList<>();
            for (GroundPoint target : targets) {
                targetConfigs.add(new orekit.visibility.model.TargetConfig(
                    target.getId(),
                    target.getLongitude(),
                    target.getLatitude(),
                    target.getAltitude(),
                    60, // minDuration
                    5   // priority
                ));
            }

            // 转换地面站参数
            List<orekit.visibility.model.GroundStationConfig> gsConfigs = new ArrayList<>();
            for (GroundPoint gs : groundStations) {
                gsConfigs.add(new orekit.visibility.model.GroundStationConfig(
                    gs.getId(),
                    gs.getLongitude(),
                    gs.getLatitude(),
                    gs.getAltitude(),
                    gs.getMinElevation(),
                    1000000.0 // maxRange (1000km)
                ));
            }

            // 使用 OptimizedVisibilityCalculator 进行计算
            if (optimizedCalculator == null) {
                optimizedCalculator = new OptimizedVisibilityCalculator();
            }

            orekit.visibility.model.BatchResult result = optimizedCalculator.computeAllVisibilityWindows(
                (List) satConfigs,  // ExtendedSatelliteConfig extends SatelliteConfig
                targetConfigs,
                gsConfigs,
                startTime,
                endTime,
                config.getCoarseStep(),
                config.getFineStep()
            );

            // 转换为Python友好的Map格式
            Map<String, Object> map = new HashMap<>();

            // 转换窗口（从统一的getAllWindows中获取，通过key区分目标和地面站）
            List<Map<String, Object>> targetWindowsList = new ArrayList<>();
            List<Map<String, Object>> gsWindowsList = new ArrayList<>();

            for (Map.Entry<String, List<orekit.visibility.model.VisibilityWindow>> entry : result.getAllWindows().entrySet()) {
                for (orekit.visibility.model.VisibilityWindow w : entry.getValue()) {
                    Map<String, Object> windowMap = new HashMap<>();
                    windowMap.put("satelliteId", w.getSatelliteId());
                    windowMap.put("targetId", w.getTargetId());
                    windowMap.put("startTime", w.getStartTime().toString());
                    windowMap.put("endTime", w.getEndTime().toString());
                    windowMap.put("maxElevation", w.getMaxElevation());
                    windowMap.put("durationSeconds", w.getDurationSeconds());

                    // 通过targetId前缀区分是目标还是地面站
                    if (w.getTargetId().startsWith("GS:")) {
                        gsWindowsList.add(windowMap);
                    } else {
                        targetWindowsList.add(windowMap);
                    }
                }
            }
            map.put("targetWindows", targetWindowsList);
            map.put("groundStationWindows", gsWindowsList);

            // 统计信息
            Map<String, Object> statsMap = new HashMap<>();
            statsMap.put("computationTimeMs", result.getStatistics().getComputationTimeMs());
            statsMap.put("nSatellites", satellites.size());
            statsMap.put("nTargets", targets.size());
            statsMap.put("nGroundStations", groundStations.size());
            statsMap.put("nWindowsFound", result.getStatistics().getTotalWindows());
            map.put("stats", statsMap);

            // 导出轨道数据（如果指定了输出路径）
            if (orbitOutputPath != null && !orbitOutputPath.isEmpty()) {
                try {
                    OrbitStateCache orbitCache = optimizedCalculator.getOrbitCache();
                    OrbitDataExporter exporter = new OrbitDataExporter();
                    exporter.exportToJson(orbitCache.getCache(), orbitOutputPath);

                    Map<String, Object> exportStatus = new HashMap<>();
                    exportStatus.put("success", true);
                    exportStatus.put("path", orbitOutputPath);
                    map.put("orbitExportStatus", exportStatus);
                } catch (Exception e) {
                    Map<String, Object> exportStatus = new HashMap<>();
                    exportStatus.put("success", false);
                    exportStatus.put("error", e.getMessage());
                    map.put("orbitExportStatus", exportStatus);
                }
            }

            return map;

        } catch (Exception e) {
            // 返回错误信息
            Map<String, Object> errorResult = new HashMap<>();
            errorResult.put("error", true);
            errorResult.put("errorMessage", e.getMessage());
            errorResult.put("errorType", e.getClass().getName());
            return errorResult;
        }
    }
}
