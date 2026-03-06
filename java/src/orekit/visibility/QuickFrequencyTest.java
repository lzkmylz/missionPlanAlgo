package orekit.visibility;

import orekit.visibility.model.BatchResult;
import orekit.visibility.model.GroundStationConfig;
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;
import orekit.visibility.model.VisibilityWindow;

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
 * 快速频次约束场景测试（简化版）
 *
 * 使用 ./scenarios/large_scale_frequency.json 配置文件
 * 但只使用前20颗卫星以加快测试速度
 */
public class QuickFrequencyTest {

    private static final String SCENARIO_FILE = "../scenarios/large_scale_frequency.json";
    private static final String OUTPUT_DIR = "output/quick_frequency";

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("快速频次约束场景测试（20颗卫星）");
        System.out.println("========================================");

        try {
            // 初始化Orekit
            initializeOrekit();

            // 创建输出目录
            Files.createDirectories(Paths.get(OUTPUT_DIR));

            // 加载场景
            System.out.println("\n[1/3] 加载场景配置...");
            JsonScenarioLoader loader = new JsonScenarioLoader(SCENARIO_FILE);
            loader.load();

            System.out.println("  场景: " + loader.getName());

            // 只使用前20颗卫星
            List<SatelliteConfig> allSatellites = loader.loadSatellites();
            List<SatelliteConfig> satellites = allSatellites.subList(0, Math.min(20, allSatellites.size()));
            List<TargetConfig> targets = loader.loadTargets();
            List<GroundStationConfig> groundStations = loader.loadGroundStations();

            AbsoluteDate startTime = loader.getStartTime();
            AbsoluteDate endTime = loader.getEndTime();

            System.out.println("  卫星数量: " + satellites.size() + " (从" + allSatellites.size() + "颗中选取)");
            System.out.println("  目标数量: " + targets.size());
            System.out.println("  地面站数量: " + groundStations.size());
            System.out.println("  时间跨度: 24小时");

            // 执行可见性计算
            System.out.println("\n[2/3] 执行可见性计算...");
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

            // 输出统计
            printStatistics(result, satellites, targets, calcTime);

            // 持久化数据
            System.out.println("\n[3/3] 持久化数据...");
            persistData(result);

            System.out.println("\n========================================");
            System.out.println("快速测试完成!");
            System.out.println("输出目录: " + OUTPUT_DIR);
            System.out.println("========================================");

        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * 持久化数据
     */
    private static void persistData(BatchResult result) throws IOException {
        // 可见性窗口（简化格式）
        StringBuilder json = new StringBuilder();
        json.append("{\n  \"windows\": [\n");

        int count = 0;
        int maxWindows = 10000; // 只保存前10000个窗口

        for (Map.Entry<String, List<VisibilityWindow>> entry : result.getAllWindows().entrySet()) {
            for (VisibilityWindow window : entry.getValue()) {
                if (count >= maxWindows) break;
                if (count > 0) json.append(",\n");

                json.append("    {\"sat\":\"").append(window.getSatelliteId()).append("\",");
                json.append("\"tgt\":\"").append(window.getTargetId()).append("\",");
                json.append("\"start\":\"").append(window.getStartTime().toString()).append("\",");
                json.append("\"dur\":").append(window.getDurationSeconds()).append("}");
                count++;
            }
            if (count >= maxWindows) break;
        }
        json.append("\n  ]\n}\n");
        Files.write(Paths.get(OUTPUT_DIR + "/visibility_windows.json"), json.toString().getBytes());
        System.out.println("  已保存: visibility_windows.json (" + count + "个窗口)");
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
