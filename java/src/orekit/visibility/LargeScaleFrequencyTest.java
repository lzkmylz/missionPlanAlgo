package orekit.visibility;

import orekit.visibility.model.BatchResult;
import orekit.visibility.model.GroundStationConfig;
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;
import orekit.visibility.model.VisibilityWindow;
import orekit.visibility.JsonScenarioLoader.ObservationRequirement;

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
 */
public class LargeScaleFrequencyTest {

    private static final String SCENARIO_FILE = "../scenarios/large_scale_frequency.json";
    private static final String OUTPUT_DIR = "output/frequency_scenario";

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("大规模频次约束场景测试");
        System.out.println("========================================");

        try {
            // 初始化Orekit
            initializeOrekit();

            // 创建输出目录
            Files.createDirectories(Paths.get(OUTPUT_DIR));

            // 加载场景
            System.out.println("\n[1/4] 加载场景配置...");
            JsonScenarioLoader loader = new JsonScenarioLoader(SCENARIO_FILE);
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

            // 执行可见性计算
            System.out.println("\n[2/4] 执行可见性计算...");
            long calcStart = System.nanoTime();

            OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();
            BatchResult result = calculator.computeAllVisibilityWindows(
                satellites,
                targets,
                startTime,
                endTime,
                5.0,  // 粗扫描步长5秒
                1.0   // 精化步长1秒
            );

            long calcTime = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - calcStart);
            System.out.println("  计算完成: " + String.format("%.2f", calcTime/1000.0) + "秒");
            System.out.println("  发现窗口: " + result.getStatistics().getTotalWindows());

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
                String jsonPath = OUTPUT_DIR + "/orbits.json.gz";
                exporter.exportToJson(orbitCache.getCache(), jsonPath);
                System.out.println("  轨道数据已导出: " + jsonPath);
            } catch (Exception e) {
                System.err.println("  警告: 导出轨道数据失败: " + e.getMessage());
                e.printStackTrace();
            }

            // 输出统计
            printStatistics(result, satellites, targets, calcTime);

            System.out.println("\n========================================");
            System.out.println("场景计算完成!");
            System.out.println("输出目录: " + OUTPUT_DIR);
            System.out.println("========================================");

        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
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
    private static void persistData(BatchResult result,
                                     List<SatelliteConfig> satellites,
                                     List<TargetConfig> targets,
                                     List<GroundStationConfig> groundStations,
                                     List<ObservationRequirement> requirements) throws IOException {
        // 1. 可见性窗口
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
                visJson.append("\"el\":").append(String.format("%.1f", window.getMaxElevation())).append("}");
            }
        }
        visJson.append("\n  ]\n}\n");
        Files.write(Paths.get(OUTPUT_DIR + "/visibility_windows.json"), visJson.toString().getBytes());

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
        Files.write(Paths.get(OUTPUT_DIR + "/satellites.json"), satJson.toString().getBytes());

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
        Files.write(Paths.get(OUTPUT_DIR + "/ground_stations.json"), gsJson.toString().getBytes());

        System.out.println("  已保存: visibility_windows.json (" + result.getStatistics().getTotalWindows() + "窗口)");
        System.out.println("  已保存: satellites.json (" + satellites.size() + "卫星)");
        System.out.println("  已保存: ground_stations.json (" + groundStations.size() + "地面站)");
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
