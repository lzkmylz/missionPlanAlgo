package orekit.visibility;

import orekit.visibility.model.BatchResult;
import orekit.visibility.model.GroundStationConfig;
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;
import orekit.visibility.model.VisibilityWindow;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * 大规模场景可见性计算与数据持久化测试
 *
 * 场景: 100颗卫星 × 1000个目标 × 1天
 * 功能:
 * 1. 计算可见性窗口
 * 2. 频次需求分析
 * 3. 卫星位置数据持久化
 * 4. 任务规划数据输出
 */
public class LargeScaleScenarioTest {

    private static final String OUTPUT_DIR = "output";
    private static final String ORBIT_DATA_FILE = OUTPUT_DIR + "/satellite_orbits.json";
    private static final String VISIBILITY_FILE = OUTPUT_DIR + "/visibility_windows.json";
    private static final String COVERAGE_FILE = OUTPUT_DIR + "/coverage_analysis.json";
    private static final String PLANNING_INPUT_FILE = OUTPUT_DIR + "/planning_input.json";

    // 频次需求配置
    private static final int REQUIRED_OBSERVATIONS_PER_DAY = 3;  // 每个目标每天需要观测3次
    private static final double MIN_GAP_BETWEEN_OBSERVATIONS = 7200; // 两次观测最小间隔2小时

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("大规模场景可见性计算与数据持久化");
        System.out.println("========================================");

        try {
            // 初始化Orekit
            initializeOrekit();

            // 创建输出目录
            createOutputDirectory();

            // 配置大规模场景
            System.out.println("\n[1/5] 配置大规模场景...");
            ScenarioConfig scenario = createLargeScaleScenario();
            System.out.println("  卫星数量: " + scenario.satellites.size());
            System.out.println("  目标数量: " + scenario.targets.size());
            System.out.println("  地面站数量: " + scenario.groundStations.size());
            System.out.println("  时间跨度: 1天");
            System.out.println("  目标计算对: " + scenario.satellites.size() * scenario.targets.size());
            System.out.println("  地面站计算对: " + scenario.satellites.size() * scenario.groundStations.size());

            // 执行可见性计算
            System.out.println("\n[2/5] 执行可见性计算...");
            long calcStart = System.nanoTime();

            // 计算目标可见性
            OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();
            BatchResult result = calculator.computeAllVisibilityWindows(
                scenario.satellites,
                scenario.targets,
                scenario.startTime,
                scenario.endTime,
                scenario.coarseStep,
                scenario.fineStep
            );

            // 计算地面站可见性（将地面站转为目标进行计算）
            List<TargetConfig> gsAsTargets = convertGroundStationsToTargets(scenario.groundStations);
            BatchResult gsResult = calculator.computeAllVisibilityWindows(
                scenario.satellites,
                gsAsTargets,
                scenario.startTime,
                scenario.endTime,
                scenario.coarseStep,
                scenario.fineStep
            );

            long calcTime = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - calcStart);
            OrbitStateCache cache = calculator.getOrbitCache();
            System.out.println("  计算完成: " + calcTime + " ms (" + String.format("%.2f", calcTime/1000.0) + "秒)");
            System.out.println("  目标窗口: " + result.getStatistics().getTotalWindows());
            System.out.println("  地面站窗口: " + gsResult.getStatistics().getTotalWindows());

            // 频次需求分析
            System.out.println("\n[3/5] 频次需求分析...");
            CoverageAnalysis coverage = analyzeCoverage(result, scenario);
            int totalTargetsWithWindows = 0;
            for (int count : coverage.targetObservationCounts.values()) {
                if (count > 0) totalTargetsWithWindows++;
            }
            System.out.println("  有可见窗口的目标: " + totalTargetsWithWindows + "/" + scenario.targets.size());
            System.out.println("  满足频次需求(≥3次,间隔≥2h): " + coverage.satisfiedTargets + "/" + scenario.targets.size());
            System.out.println("  覆盖率: " + String.format("%.2f%%", coverage.coveragePercent));
            System.out.println("  平均每目标观测次数: " + String.format("%.1f", (double) result.getStatistics().getTotalWindows() / scenario.targets.size()));

            // 持久化卫星轨道数据
            System.out.println("\n[4/5] 持久化卫星轨道数据...");
            persistOrbitData(scenario);
            System.out.println("  文件: " + ORBIT_DATA_FILE);

            // 生成任务规划输入数据
            System.out.println("\n[5/5] 生成任务规划数据...");
            generatePlanningData(result, scenario, coverage);
            System.out.println("  可见性窗口: " + VISIBILITY_FILE);
            System.out.println("  覆盖分析: " + COVERAGE_FILE);
            System.out.println("  规划输入: " + PLANNING_INPUT_FILE);

            // 输出统计信息
            printStatistics(result, scenario, calcTime, cache);

            // 输出地面站信息
            System.out.println("\n地面站网络:");
            for (GroundStationConfig gs : scenario.groundStations) {
                System.out.println("  " + gs);
            }

            System.out.println("\n========================================");
            System.out.println("大规模场景计算完成!");
            System.out.println("输出目录: " + OUTPUT_DIR);
            System.out.println("========================================");

        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * 场景配置
     */
    static class ScenarioConfig {
        List<SatelliteConfig> satellites;
        List<TargetConfig> targets;
        List<GroundStationConfig> groundStations;
        AbsoluteDate startTime;
        AbsoluteDate endTime;
        double coarseStep;
        double fineStep;
    }

    /**
     * 覆盖分析结果
     */
    static class CoverageAnalysis {
        int satisfiedTargets;
        int totalTargets;
        double coveragePercent;
        Map<String, Integer> targetObservationCounts;
        List<String> unsatisfiedTargets;
    }

    /**
     * 创建大规模场景配置
     * 60星 × 1000+目标 × 多个地面站 × 1天
     */
    private static ScenarioConfig createLargeScaleScenario() {
        ScenarioConfig config = new ScenarioConfig();

        // 创建60颗卫星（模拟Walker星座 60/6/1）
        config.satellites = createSatelliteConstellation(60);

        // 创建1200个目标（全球均匀分布）
        config.targets = createGlobalTargets(1200);

        // 创建地面站网络
        config.groundStations = createGroundStationNetwork();

        // 时间配置
        config.startTime = AbsoluteDate.J2000_EPOCH;
        config.endTime = config.startTime.shiftedBy(24 * 3600); // 1天

        // 步长配置
        config.coarseStep = 5.0;  // 5秒粗扫描
        config.fineStep = 1.0;    // 1秒精化

        return config;
    }

    /**
     * 创建卫星星座（Walker Delta配置）
     * 60/6/1 配置: 6个轨道面，每面10颗卫星
     */
    private static List<SatelliteConfig> createSatelliteConstellation(int count) {
        List<SatelliteConfig> satellites = new ArrayList<>();

        // Walker Delta 60/6/1 配置
        int numPlanes = 6;
        int satsPerPlane = 10;
        double phaseFactor = 1.0;

        for (int plane = 0; plane < numPlanes; plane++) {
            double raan = plane * 360.0 / numPlanes;

            for (int sat = 0; sat < satsPerPlane; sat++) {
                int satNum = plane * satsPerPlane + sat + 1;
                double meanAnomaly = sat * 360.0 / satsPerPlane +
                                     plane * phaseFactor * 360.0 / count;

                // 从ID编码轨道相位信息
                String satId = String.format("SAT-%02d-%02d", plane + 1, sat + 1);

                satellites.add(new SatelliteConfig(
                    satId,
                    "",  // TLE line 1
                    "",  // TLE line 2
                    5.0, // min elevation 5 degrees
                    30.0 // sensor FOV 30 degrees
                ));
            }
        }

        return satellites;
    }

    /**
     * 创建全球分布目标
     */
    private static List<TargetConfig> createGlobalTargets(int count) {
        List<TargetConfig> targets = new ArrayList<>();

        // 使用Fibonacci lattice均匀分布
        double goldenRatio = (1 + Math.sqrt(5)) / 2;

        for (int i = 0; i < count; i++) {
            double theta = 2 * Math.PI * i / goldenRatio;
            double phi = Math.acos(1 - 2 * (i + 0.5) / count);

            double latitude = Math.toDegrees(phi - Math.PI / 2);
            double longitude = Math.toDegrees(theta % (2 * Math.PI));
            if (longitude > 180) longitude -= 360;

            String targetId = String.format("TGT-%04d", i + 1);

            // 优先级根据地理位置分布（模拟高价值区域）
            int priority = calculatePriority(latitude, longitude);

            targets.add(new TargetConfig(
                targetId,
                longitude,
                latitude,
                0.0,    // altitude
                60,     // min observation duration 60 seconds
                priority
            ));
        }

        return targets;
    }

    /**
     * 计算目标优先级（模拟）
     */
    private static int calculatePriority(double latitude, double longitude) {
        // 模拟高优先级区域（人口密集区、冲突地区等）
        // 例如：北美、欧洲、东亚等区域优先级较高

        // 简化模型：赤道附近低纬度优先级较高
        double absLat = Math.abs(latitude);

        if (absLat < 30) return 10;      // 热带地区
        else if (absLat < 60) return 7;  // 温带地区
        else return 5;                    // 高纬度地区
    }

    /**
     * 将地面站转换为目标配置（用于可见性计算）
     */
    private static List<TargetConfig> convertGroundStationsToTargets(List<GroundStationConfig> stations) {
        List<TargetConfig> targets = new ArrayList<>();
        for (GroundStationConfig gs : stations) {
            targets.add(new TargetConfig(
                gs.getId(),
                gs.getLongitude(),
                gs.getLatitude(),
                gs.getAltitude(),
                30,  // 最小观测时间30秒（通信链路建立）
                10   // 地面站最高优先级
            ));
        }
        return targets;
    }

    /**
     * 创建全球地面站网络
     */
    private static List<GroundStationConfig> createGroundStationNetwork() {
        List<GroundStationConfig> stations = new ArrayList<>();

        // 全球主要地面站分布
        stations.add(new GroundStationConfig("GS-BEIJING", 116.4, 39.9, 0, 5.0, 2500000));
        stations.add(new GroundStationConfig("GS-SHANGHAI", 121.5, 31.2, 0, 5.0, 2500000));
        stations.add(new GroundStationConfig("GS-XINJIANG", 87.6, 43.8, 0, 5.0, 2500000));
        stations.add(new GroundStationConfig("GS-SANYA", 109.5, 18.3, 0, 5.0, 2500000));
        stations.add(new GroundStationConfig("GS-KASHI", 76.0, 39.5, 0, 5.0, 2500000));
        stations.add(new GroundStationConfig("GS-HARBIN", 126.6, 45.8, 0, 5.0, 2500000));
        stations.add(new GroundStationConfig("GS-KUNMING", 102.7, 25.0, 0, 5.0, 2500000));
        stations.add(new GroundStationConfig("GS-XIAN", 108.9, 34.3, 0, 5.0, 2500000));
        stations.add(new GroundStationConfig("GS-NANJING", 118.8, 32.1, 0, 5.0, 2500000));
        stations.add(new GroundStationConfig("GS-GUANGZHOU", 113.3, 23.1, 0, 5.0, 2500000));

        return stations;
    }

    /**
     * 执行可见性计算
     */
    private static BatchResult computeVisibility(ScenarioConfig scenario) throws Exception {
        OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();

        return calculator.computeAllVisibilityWindows(
            scenario.satellites,
            scenario.targets,
            scenario.startTime,
            scenario.endTime,
            scenario.coarseStep,
            scenario.fineStep
        );
    }

    /**
     * 分析覆盖率和频次需求
     */
    private static CoverageAnalysis analyzeCoverage(BatchResult result, ScenarioConfig scenario) {
        CoverageAnalysis analysis = new CoverageAnalysis();
        analysis.totalTargets = scenario.targets.size();
        analysis.targetObservationCounts = new HashMap<>();
        analysis.unsatisfiedTargets = new ArrayList<>();

        // 统计每个目标的观测次数
        for (TargetConfig target : scenario.targets) {
            int count = 0;
            List<Long> observationTimes = new ArrayList<>();

            for (SatelliteConfig sat : scenario.satellites) {
                String key = sat.getId() + "_" + target.getId();
                List<VisibilityWindow> windows = result.getWindows(sat.getId(), target.getId());

                for (VisibilityWindow window : windows) {
                    count++;
                    observationTimes.add((long)(window.getStartTime().durationFrom(scenario.startTime) * 1000));
                }
            }

            // 检查频次需求（至少3次，间隔至少2小时）
            boolean satisfied = checkFrequencyRequirement(observationTimes);

            analysis.targetObservationCounts.put(target.getId(), count);

            if (satisfied) {
                analysis.satisfiedTargets++;
            } else {
                analysis.unsatisfiedTargets.add(target.getId());
            }
        }

        analysis.coveragePercent = 100.0 * analysis.satisfiedTargets / analysis.totalTargets;

        return analysis;
    }

    /**
     * 检查频次需求
     */
    private static boolean checkFrequencyRequirement(List<Long> observationTimes) {
        if (observationTimes.size() < REQUIRED_OBSERVATIONS_PER_DAY) {
            return false;
        }

        // 按时间排序
        Collections.sort(observationTimes);

        // 检查间隔
        int validObservations = 1;
        long lastTime = observationTimes.get(0);

        for (int i = 1; i < observationTimes.size(); i++) {
            long gap = (observationTimes.get(i) - lastTime) / 1000; // 转换为秒
            if (gap >= MIN_GAP_BETWEEN_OBSERVATIONS) {
                validObservations++;
                lastTime = observationTimes.get(i);

                if (validObservations >= REQUIRED_OBSERVATIONS_PER_DAY) {
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * 持久化卫星轨道数据
     */
    private static void persistOrbitData(ScenarioConfig scenario) throws IOException {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"metadata\": {\n");
        json.append("    \"satelliteCount\": ").append(scenario.satellites.size()).append(",\n");
        json.append("    \"timeStep\": 5.0,\n");
        json.append("    \"startTime\": \"").append(scenario.startTime.toString()).append("\",\n");
        json.append("    \"endTime\": \"").append(scenario.endTime.toString()).append("\"\n");
        json.append("  },\n");
        json.append("  \"satellites\": [\n");

        // 获取轨道缓存数据（复用主计算的缓存，避免重复计算）
        OrbitStateCache cache = new OrbitStateCache();

        for (int i = 0; i < scenario.satellites.size(); i++) {
            SatelliteConfig sat = scenario.satellites.get(i);
            json.append("    {\n");
            json.append("      \"id\": \"").append(sat.getId()).append("\",\n");
            json.append("      \"tleLine1\": \"").append(sat.getTleLine1()).append("\",\n");
            json.append("      \"tleLine2\": \"").append(sat.getTleLine2()).append("\",\n");
            json.append("      \"minElevation\": ").append(sat.getMinElevation()).append("\n");
            json.append("    }");
            if (i < scenario.satellites.size() - 1) json.append(",");
            json.append("\n");
        }

        json.append("  ]\n");
        json.append("}\n");

        Files.write(Paths.get(ORBIT_DATA_FILE), json.toString().getBytes());
    }

    /**
     * 生成任务规划输入数据
     */
    private static void generatePlanningData(BatchResult result, ScenarioConfig scenario,
                                            CoverageAnalysis coverage) throws IOException {
        // 1. 可见性窗口数据
        StringBuilder visJson = new StringBuilder();
        visJson.append("{\n");
        visJson.append("  \"windows\": [\n");

        boolean first = true;
        for (Map.Entry<String, List<VisibilityWindow>> entry : result.getAllWindows().entrySet()) {
            for (VisibilityWindow window : entry.getValue()) {
                if (!first) visJson.append(",\n");
                first = false;

                visJson.append("    {\n");
                visJson.append("      \"satelliteId\": \"").append(window.getSatelliteId()).append("\",\n");
                visJson.append("      \"targetId\": \"").append(window.getTargetId()).append("\",\n");
                visJson.append("      \"startTime\": \"").append(window.getStartTime().toString()).append("\",\n");
                visJson.append("      \"endTime\": \"").append(window.getEndTime().toString()).append("\",\n");
                visJson.append("      \"duration\": ").append(window.getDurationSeconds()).append(",\n");
                visJson.append("      \"maxElevation\": ").append(window.getMaxElevation()).append(",\n");
                visJson.append("      \"quality\": \"").append(window.getConfidence()).append("\"\n");
                visJson.append("    }");
            }
        }

        visJson.append("\n  ]\n");
        visJson.append("}\n");

        Files.write(Paths.get(VISIBILITY_FILE), visJson.toString().getBytes());

        // 2. 覆盖分析数据
        StringBuilder covJson = new StringBuilder();
        covJson.append("{\n");
        covJson.append("  \"summary\": {\n");
        covJson.append("    \"totalTargets\": ").append(coverage.totalTargets).append(",\n");
        covJson.append("    \"satisfiedTargets\": ").append(coverage.satisfiedTargets).append(",\n");
        covJson.append("    \"coveragePercent\": ").append(String.format("%.2f", coverage.coveragePercent)).append(",\n");
        covJson.append("    \"requiredObservations\": ").append(REQUIRED_OBSERVATIONS_PER_DAY).append(",\n");
        covJson.append("    \"minGapSeconds\": ").append(MIN_GAP_BETWEEN_OBSERVATIONS).append("\n");
        covJson.append("  },\n");
        covJson.append("  \"targetObservations\": {\n");

        boolean firstTarget = true;
        for (Map.Entry<String, Integer> entry : coverage.targetObservationCounts.entrySet()) {
            if (!firstTarget) covJson.append(",\n");
            firstTarget = false;
            covJson.append("    \"").append(entry.getKey()).append("\": ").append(entry.getValue());
        }

        covJson.append("\n  },\n");
        covJson.append("  \"unsatisfiedTargets\": [\n");

        for (int i = 0; i < coverage.unsatisfiedTargets.size(); i++) {
            if (i > 0) covJson.append(",\n");
            covJson.append("    \"").append(coverage.unsatisfiedTargets.get(i)).append("\"");
        }

        covJson.append("\n  ]\n");
        covJson.append("}\n");

        Files.write(Paths.get(COVERAGE_FILE), covJson.toString().getBytes());

        // 3. 任务规划优化输入
        StringBuilder planJson = new StringBuilder();
        planJson.append("{\n");
        planJson.append("  \"planningParameters\": {\n");
        planJson.append("    \"planningHorizon\": \"1 day\",\n");
        planJson.append("    \"optimizationObjective\": \"maximize_coverage_with_priority\",\n");
        planJson.append("    \"constraints\": [\n");
        planJson.append("      \"satellite_agility\",\n");
        planJson.append("      \"ground_station_contact\",\n");
        planJson.append("      \"memory_capacity\",\n");
        planJson.append("      \"power_budget\"\n");
        planJson.append("    ]\n");
        planJson.append("  },\n");
        planJson.append("  \"satellites\": ");
        planJson.append(scenario.satellites.size()).append(",\n");
        planJson.append("  \"targets\": ");
        planJson.append(scenario.targets.size()).append(",\n");
        planJson.append("  \"dataFiles\": {\n");
        planJson.append("    \"orbitData\": \"").append(ORBIT_DATA_FILE).append("\",\n");
        planJson.append("    \"visibilityWindows\": \"").append(VISIBILITY_FILE).append("\",\n");
        planJson.append("    \"coverageAnalysis\": \"").append(COVERAGE_FILE).append("\"\n");
        planJson.append("  }\n");
        planJson.append("}\n");

        Files.write(Paths.get(PLANNING_INPUT_FILE), planJson.toString().getBytes());
    }

    /**
     * 输出统计信息
     */
    private static void printStatistics(BatchResult result, ScenarioConfig scenario, long calcTime, OrbitStateCache cache) {
        System.out.println("\n========================================");
        System.out.println("统计信息");
        System.out.println("========================================");
        System.out.println("计算性能:");
        System.out.println("  总耗时: " + String.format("%.2f", calcTime / 1000.0) + " 秒");
        System.out.println("  计算对: " + scenario.satellites.size() * scenario.targets.size());
        System.out.println("  每对耗时: " + String.format("%.3f", (double) calcTime /
            (scenario.satellites.size() * scenario.targets.size())) + " ms");
        System.out.println("\n可见性:");
        System.out.println("  总窗口数: " + result.getStatistics().getTotalWindows());
        System.out.println("  平均每目标窗口: " + String.format("%.2f",
            (double) result.getStatistics().getTotalWindows() / scenario.targets.size()));
        System.out.println("\n缓存使用:");
        System.out.println("  内存使用: " + (cache.getMemoryUsage() / (1024 * 1024)) + " MB");
    }

    /**
     * 初始化Orekit
     */
    private static void initializeOrekit() throws Exception {
        String dataPath = System.getenv("OREKIT_DATA_PATH");
        if (dataPath == null) {
            dataPath = System.getProperty("user.home") + "/orekit-data";
        }

        File dataDir = new File(dataPath);
        if (dataDir.exists()) {
            DataContext.getDefault().getDataProvidersManager().addProvider(
                new DirectoryCrawler(dataDir)
            );
        }
    }

    /**
     * 创建输出目录
     */
    private static void createOutputDirectory() throws IOException {
        Files.createDirectories(Paths.get(OUTPUT_DIR));
    }
}
