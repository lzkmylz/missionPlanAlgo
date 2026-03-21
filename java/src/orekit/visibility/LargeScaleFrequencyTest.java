package orekit.visibility;

import orekit.visibility.model.BatchResult;
import orekit.visibility.model.GroundStationConfig;
import orekit.visibility.model.RelaySatelliteConfig;
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;
import orekit.visibility.model.VisibilityWindow;
import orekit.visibility.JsonScenarioLoader.ObservationRequirement;
import java.util.Map;
import orekit.visibility.RelayVisibilityCalculator;
import orekit.visibility.ISLVisibilityCalculator;
import orekit.visibility.model.ISLSatellitePairConfig;

import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.time.AbsoluteDate;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * 大规模频次约束场景测试
 *
 * 使用 ./scenarios/large_scale_frequency.json 配置文件
 * 60颗卫星(30光学+30SAR) vs 1000目标，含频次约束
 *
 * 用法:
 *   # 使用默认配置
 *   java -cp "classes:lib/*" orekit.visibility.LargeScaleFrequencyTest
 *
 *   # 指定场景文件和输出目录
 *   java -cp "classes:lib/*" orekit.visibility.LargeScaleFrequencyTest \
 *        --scenario ../scenarios/my_scenario.json \
 *        --output output/my_results \
 *        --orbit-output output/my_results/orbits.json.gz
 */
public class LargeScaleFrequencyTest {

    // 默认配置
    private static final String DEFAULT_SCENARIO_FILE = "../scenarios/large_scale_frequency.json";
    private static final String DEFAULT_OUTPUT_DIR = "output/frequency_scenario";
    private static final String DEFAULT_ORBIT_FILENAME = "orbits.json.gz";

    // 运行时配置
    private String scenarioFile;
    private String outputDir;
    private String orbitOutputPath;
    private double coarseStep = 5.0;  // 粗扫描步长(秒)
    private double fineStep = 1.0;    // 精化步长(秒)

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("大规模频次约束场景测试");
        System.out.println("========================================");

        LargeScaleFrequencyTest test = new LargeScaleFrequencyTest();

        // 解析命令行参数
        if (!test.parseArguments(args)) {
            printUsage();
            System.exit(1);
        }

        try {
            test.run();
        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * 解析命令行参数
     */
    private boolean parseArguments(String[] args) {
        // 默认值
        this.scenarioFile = DEFAULT_SCENARIO_FILE;
        this.outputDir = DEFAULT_OUTPUT_DIR;

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];

            switch (arg) {
                case "--scenario":
                case "-s":
                    if (i + 1 >= args.length) {
                        System.err.println("错误: " + arg + " 需要参数");
                        return false;
                    }
                    this.scenarioFile = args[++i];
                    break;

                case "--output":
                case "-o":
                    if (i + 1 >= args.length) {
                        System.err.println("错误: " + arg + " 需要参数");
                        return false;
                    }
                    this.outputDir = args[++i];
                    break;

                case "--orbit-output":
                    if (i + 1 >= args.length) {
                        System.err.println("错误: " + arg + " 需要参数");
                        return false;
                    }
                    this.orbitOutputPath = args[++i];
                    break;

                case "--coarse-step":
                    if (i + 1 >= args.length) {
                        System.err.println("错误: " + arg + " 需要参数");
                        return false;
                    }
                    try {
                        this.coarseStep = Double.parseDouble(args[++i]);
                    } catch (NumberFormatException e) {
                        System.err.println("错误: 粗扫描步长必须是数字");
                        return false;
                    }
                    break;

                case "--fine-step":
                    if (i + 1 >= args.length) {
                        System.err.println("错误: " + arg + " 需要参数");
                        return false;
                    }
                    try {
                        this.fineStep = Double.parseDouble(args[++i]);
                    } catch (NumberFormatException e) {
                        System.err.println("错误: 精化步长必须是数字");
                        return false;
                    }
                    break;

                case "--help":
                case "-h":
                    printUsage();
                    return false;

                default:
                    System.err.println("错误: 未知参数 " + arg);
                    return false;
            }
        }

        // 如果没有指定轨道输出路径，使用输出目录下的默认文件名
        if (this.orbitOutputPath == null) {
            this.orbitOutputPath = this.outputDir + "/" + DEFAULT_ORBIT_FILENAME;
        }

        return true;
    }

    /**
     * 打印使用说明
     */
    private static void printUsage() {
        System.out.println("\n用法:");
        System.out.println("  java -cp \"classes:lib/*\" orekit.visibility.LargeScaleFrequencyTest [选项]\n");
        System.out.println("选项:");
        System.out.println("  -s, --scenario <路径>       场景配置文件路径 (默认: " + DEFAULT_SCENARIO_FILE + ")");
        System.out.println("  -o, --output <目录>         输出目录 (默认: " + DEFAULT_OUTPUT_DIR + ")");
        System.out.println("      --orbit-output <路径>   轨道数据输出路径 (默认: <输出目录>/" + DEFAULT_ORBIT_FILENAME + ")");
        System.out.println("      --coarse-step <秒>      粗扫描步长 (默认: 5.0秒)");
        System.out.println("      --fine-step <秒>        精化步长 (默认: 1.0秒)");
        System.out.println("  -h, --help                  显示此帮助\n");
        System.out.println("示例:");
        System.out.println("  # 使用默认配置");
        System.out.println("  java -cp \"classes:lib/*\" orekit.visibility.LargeScaleFrequencyTest\n");
        System.out.println("  # 指定场景和输出目录");
        System.out.println("  java -cp \"classes:lib/*\" orekit.visibility.LargeScaleFrequencyTest \\");
        System.out.println("    --scenario ../scenarios/custom.json \\");
        System.out.println("    --output output/custom_results\n");
    }

    /**
     * 执行测试
     */
    public void run() throws Exception {
        // 初始化Orekit
        initializeOrekit();

        // 创建输出目录
        Files.createDirectories(Paths.get(outputDir));

        System.out.println("\n配置信息:");
        System.out.println("  场景文件: " + scenarioFile);
        System.out.println("  输出目录: " + outputDir);
        System.out.println("  轨道输出: " + orbitOutputPath);
        System.out.println("  粗扫描步长: " + coarseStep + "秒");
        System.out.println("  精化步长: " + fineStep + "秒");

        // 加载场景
        System.out.println("\n[1/4] 加载场景配置...");
        JsonScenarioLoader loader = new JsonScenarioLoader(scenarioFile);
        loader.load();

            System.out.println("  场景: " + loader.getName());
            System.out.println("  描述: " + loader.getDescription());

            List<SatelliteConfig> satellites = loader.loadSatellites();
            List<TargetConfig> targets = loader.loadTargets();
            List<GroundStationConfig> groundStations = loader.loadGroundStations();
            List<ObservationRequirement> requirements = loader.loadObservationRequirements();

            AbsoluteDate startTime = loader.getStartTime();
            AbsoluteDate endTime = loader.getEndTime();

            System.out.println("  卫星数量: " + satellites.size());
            System.out.println("  目标数量: " + targets.size());
            System.out.println("  地面站数量: " + groundStations.size());
            System.out.println("  频次需求: " + requirements.size() + "个目标有频次约束");
            System.out.println("  时间跨度: 24小时");

            // 加载中继卫星配置
            List<RelaySatelliteConfig> relaySatellites = loader.loadRelaySatellites();
            System.out.println("  中继卫星数量: " + relaySatellites.size());

        // 执行可见性计算（包含卫星-目标 + 卫星-地面站）
        System.out.println("\n[2/4] 执行可见性计算...");
        System.out.println("  计算卫星-目标窗口: " + satellites.size() + "卫星 x " + targets.size() + "目标");
        System.out.println("  计算卫星-地面站窗口: " + satellites.size() + "卫星 x " + groundStations.size() + "地面站");
        long calcStart = System.nanoTime();

        OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();
        BatchResult result = calculator.computeAllVisibilityWindows(
            satellites,
            targets,
            groundStations,
            startTime,
            endTime,
            coarseStep,
            fineStep
        );

            long calcTime = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - calcStart);
            System.out.println("  计算完成: " + String.format("%.2f", calcTime/1000.0) + "秒");
            System.out.println("  发现窗口: " + result.getStatistics().getTotalWindows());

            // 计算卫星-中继卫星窗口（如果有中继卫星）
            if (!relaySatellites.isEmpty()) {
                System.out.println("\n[2.5/4] 计算中继卫星可见窗口...");
                System.out.println("  计算卫星-中继窗口: " + satellites.size() + "卫星 x " + relaySatellites.size() + "中继");
                long relayStart = System.nanoTime();

                RelayVisibilityCalculator relayCalculator = new RelayVisibilityCalculator(calculator.getOrbitCache());
                // 使用两阶段扫描策略：粗扫描(5秒步长) + 精扫描(1秒步长)
                // 与地面站窗口计算策略保持一致，确保窗口边界精度
                double relayCoarseStep = 5.0;   // 粗扫描步长
                double relayFineStep = 1.0;     // 精扫描步长（与轨道数据步长一致）
                Map<String, List<VisibilityWindow>> relayWindows = relayCalculator.computeRelayVisibilityWindows(
                    satellites,
                    relaySatellites,
                    startTime,
                    endTime,
                    relayCoarseStep,  // 粗扫描步长
                    relayFineStep     // 精扫描步长
                );

                // 将中继窗口添加到结果中
                int totalRelayWindows = 0;
                for (Map.Entry<String, List<VisibilityWindow>> entry : relayWindows.entrySet()) {
                    String key = entry.getKey();  // 格式: satId_RELAY:relayId
                    List<VisibilityWindow> windows = entry.getValue();
                    totalRelayWindows += windows.size();

                    // 解析key获取satelliteId和targetId
                    String[] parts = key.split("_", 2);
                    if (parts.length == 2) {
                        String satId = parts[0];
                        String relayId = parts[1];  // 已经是RELAY:xxx格式
                        result.addWindows(satId, relayId, windows);
                    }
                }

                long relayTime = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - relayStart);
                System.out.println("  中继窗口计算完成: " + String.format("%.2f", relayTime/1000.0) + "秒");
                System.out.println("  发现中继窗口: " + totalRelayWindows);
            }

            // 计算ISL可见性窗口（如果有配置ISL的卫星对）
            List<ISLSatellitePairConfig> islPairs = loader.loadISLPairs();
            if (!islPairs.isEmpty()) {
                System.out.println("\n[2.7/4] 计算ISL星间链路可见窗口...");
                System.out.println("  ISL卫星对数量: " + islPairs.size());
                long islStart = System.nanoTime();

                ISLVisibilityCalculator islCalculator = new ISLVisibilityCalculator(calculator.getOrbitCache());
                double islCoarseStep = 5.0;
                double islFineStep = 1.0;
                Map<String, List<VisibilityWindow>> islWindows = islCalculator.computeISLVisibilityWindows(
                    islPairs, startTime, endTime, islCoarseStep, islFineStep
                );

                // 将ISL窗口添加到结果中
                int totalISLWindows = 0;
                for (Map.Entry<String, List<VisibilityWindow>> entry : islWindows.entrySet()) {
                    String key = entry.getKey();  // 格式: "satAId_ISL:satBId"
                    List<VisibilityWindow> windows = entry.getValue();
                    totalISLWindows += windows.size();

                    // 解析key获取satelliteId和"ISL:satBId"
                    String[] parts = key.split("_", 2);
                    if (parts.length == 2) {
                        result.addWindows(parts[0], parts[1], windows);
                    }
                }

                long islTime = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - islStart);
                System.out.println("  ISL窗口计算完成: " + String.format("%.2f", islTime/1000.0) + "秒");
                System.out.println("  发现ISL窗口: " + totalISLWindows);
            }

            // 频次需求分析
            System.out.println("\n[3/4] 频次需求分析...");
            analyzeFrequencyRequirements(result, requirements, startTime);

            // 持久化数据
            System.out.println("\n[4/4] 持久化数据...");
            persistData(result, satellites, targets, groundStations, requirements);

        // 导出轨道数据到JSON+GZIP（供Python端使用）
        System.out.println("\n[4/4+] 导出轨道数据到JSON+GZIP...");
        try {
            OrbitStateCache orbitCache = calculator.getOrbitCache();
            OrbitDataExporter exporter = new OrbitDataExporter();
            exporter.exportToJson(orbitCache.getCache(), orbitOutputPath);
            System.out.println("  轨道数据已导出: " + orbitOutputPath);
        } catch (Exception e) {
            System.err.println("  警告: 导出轨道数据失败: " + e.getMessage());
            e.printStackTrace();
        }

        // 输出统计
        printStatistics(result, satellites, targets, calcTime);

        System.out.println("\n========================================");
        System.out.println("场景计算完成!");
        System.out.println("输出目录: " + outputDir);
        System.out.println("轨道数据: " + orbitOutputPath);
        System.out.println("========================================");
    }

    /**
     * 分析频次需求满足情况
     */
    private static void analyzeFrequencyRequirements(BatchResult result,
                                                      List<ObservationRequirement> requirements,
                                                      AbsoluteDate startTime) {
        int satisfied = 0;
        int partiallySatisfied = 0;
        int notSatisfied = 0;

        Map<String, Integer> targetWindowCounts = new HashMap<>();

        // 统计每个目标的窗口数
        for (Map.Entry<String, List<VisibilityWindow>> entry : result.getAllWindows().entrySet()) {
            for (VisibilityWindow window : entry.getValue()) {
                String targetId = window.getTargetId();
                targetWindowCounts.put(targetId, targetWindowCounts.getOrDefault(targetId, 0) + 1);
            }
        }

        // 检查每个频次需求
        for (ObservationRequirement req : requirements) {
            int actualCount = targetWindowCounts.getOrDefault(req.targetId, 0);

            if (actualCount >= req.requiredCount) {
                satisfied++;
            } else if (actualCount > 0) {
                partiallySatisfied++;
            } else {
                notSatisfied++;
            }
        }

        System.out.println("  频次需求满足情况:");
        System.out.println("    完全满足: " + satisfied + "/" + requirements.size());
        System.out.println("    部分满足: " + partiallySatisfied + "/" + requirements.size());
        System.out.println("    未满足: " + notSatisfied + "/" + requirements.size());
    }

    /**
     * 持久化数据
     */
    private void persistData(BatchResult result,
                             List<SatelliteConfig> satellites,
                             List<TargetConfig> targets,
                             List<GroundStationConfig> groundStations,
                             List<ObservationRequirement> requirements) throws IOException {
        // 1. 可见性窗口（带姿态数据）
        StringBuilder visJson = new StringBuilder();
        visJson.append("{\n  \"windows\": [\n");

        boolean first = true;
        for (Map.Entry<String, List<VisibilityWindow>> entry : result.getAllWindows().entrySet()) {
            for (VisibilityWindow window : entry.getValue()) {
                if (!first) visJson.append(",\n");
                first = false;

                visJson.append("    {\"sat\":\"").append(window.getSatelliteId()).append("\",");
                visJson.append("\"tgt\":\"").append(window.getTargetId()).append("\",");
                visJson.append("\"start\":\"").append(window.getStartTime().toString()).append("\",");
                visJson.append("\"end\":\"").append(window.getEndTime().toString()).append("\",");
                visJson.append("\"dur\":").append(window.getDurationSeconds()).append(",");
                visJson.append("\"el\":").append(String.format("%.1f", window.getMaxElevation())).append(",");

                // ISL元数据（仅ISL窗口输出）
                if (window.isISLWindow()) {
                    visJson.append("\"isl_link_type\":\"").append(window.getIslLinkType()).append("\",");
                    visJson.append("\"isl_data_rate_mbps\":").append(String.format("%.1f", window.getIslDataRateMbps())).append(",");
                    visJson.append("\"isl_link_margin_db\":").append(String.format("%.2f", window.getIslLinkMarginDb())).append(",");
                    visJson.append("\"isl_distance_km\":").append(String.format("%.1f", window.getIslDistanceKm())).append(",");
                    visJson.append("\"isl_relative_velocity_km_s\":").append(String.format("%.3f", window.getIslRelativeVelocityKmS())).append(",");
                    visJson.append("\"isl_atp_setup_time_s\":").append(String.format("%.1f", window.getIslAtpSetupTimeS())).append(",");
                }

                visJson.append("\"attitude_feasible\":").append(window.isAttitudeFeasible()).append(",");

                // 添加姿态采样数据
                visJson.append("\"attitude_samples\":[");
                List<VisibilityWindow.AttitudeSample> samples = window.getAttitudeSamples();
                for (int i = 0; i < samples.size(); i++) {
                    VisibilityWindow.AttitudeSample sample = samples.get(i);
                    if (i > 0) visJson.append(",");
                    visJson.append(String.format("{\"t\":%.1f,\"r\":%.2f,\"p\":%.2f}",
                        sample.timestamp, sample.roll, sample.pitch));
                }
                visJson.append("]}");
            }
        }
        visJson.append("\n  ]\n}\n");
        Files.write(Paths.get(outputDir + "/visibility_windows.json"), visJson.toString().getBytes());

        // 2. 卫星配置
        StringBuilder satJson = new StringBuilder();
        satJson.append("{\n  \"satellites\": [\n");
        for (int i = 0; i < satellites.size(); i++) {
            SatelliteConfig sat = satellites.get(i);
            satJson.append("    {\"id\":\"").append(sat.getId()).append("\"}");
            if (i < satellites.size() - 1) satJson.append(",");
            satJson.append("\n");
        }
        satJson.append("  ]\n}\n");
        Files.write(Paths.get(outputDir + "/satellites.json"), satJson.toString().getBytes());

        // 3. 地面站配置
        StringBuilder gsJson = new StringBuilder();
        gsJson.append("{\n  \"ground_stations\": [\n");
        for (int i = 0; i < groundStations.size(); i++) {
            GroundStationConfig gs = groundStations.get(i);
            gsJson.append("    {\"id\":\"").append(gs.getId()).append("\",");
            gsJson.append("\"lon\":").append(gs.getLongitude()).append(",");
            gsJson.append("\"lat\":").append(gs.getLatitude()).append("}");
            if (i < groundStations.size() - 1) gsJson.append(",");
            gsJson.append("\n");
        }
        gsJson.append("  ]\n}\n");
        Files.write(Paths.get(outputDir + "/ground_stations.json"), gsJson.toString().getBytes());

        // 统计各类窗口数量
        int totalWindows = result.getStatistics().getTotalWindows();
        int gsWindowCount = 0;
        int relayWindowCount = 0;
        int islWindowCount = 0;
        for (Map.Entry<String, List<VisibilityWindow>> entry : result.getAllWindows().entrySet()) {
            String key = entry.getKey();
            int count = entry.getValue().size();
            // BatchResult.makeKey() 使用 "-" 分隔符，即 "satId-GS:xxx" 或 "satId-ISL:xxx"
            if (key.contains("-GS:")) {
                gsWindowCount += count;
            } else if (key.contains("-RELAY:")) {
                relayWindowCount += count;
            } else if (key.contains("-ISL:")) {
                islWindowCount += count;
            }
        }
        int targetWindowCount = totalWindows - gsWindowCount - relayWindowCount - islWindowCount;

        System.out.println("  已保存: visibility_windows.json (" + totalWindows + "窗口, 目标:" + targetWindowCount
            + ", 地面站:" + gsWindowCount + ", 中继:" + relayWindowCount + ", ISL:" + islWindowCount + ")");
        System.out.println("  已保存: satellites.json (" + satellites.size() + "卫星)");
        System.out.println("  已保存: ground_stations.json (" + groundStations.size() + "地面站)");
    }

    // Getter方法，供其他类使用
    public String getScenarioFile() {
        return scenarioFile;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public String getOrbitOutputPath() {
        return orbitOutputPath;
    }

    /**
     * 输出统计信息
     */
    private static void printStatistics(BatchResult result,
                                         List<SatelliteConfig> satellites,
                                         List<TargetConfig> targets,
                                         long calcTime) {
        System.out.println("\n========================================");
        System.out.println("统计信息");
        System.out.println("========================================");
        System.out.println("计算性能:");
        System.out.println("  总耗时: " + String.format("%.2f", calcTime / 1000.0) + " 秒");
        System.out.println("  计算对: " + satellites.size() * targets.size());
        System.out.println("  每对耗时: " + String.format("%.3f", (double) calcTime /
            (satellites.size() * targets.size())) + " ms");
        System.out.println("\n可见性:");
        System.out.println("  总窗口数: " + result.getStatistics().getTotalWindows());
        System.out.println("  平均每目标: " + String.format("%.1f",
            (double) result.getStatistics().getTotalWindows() / targets.size()));
        System.out.println("  平均每卫星: " + String.format("%.1f",
            (double) result.getStatistics().getTotalWindows() / satellites.size()));
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
}
