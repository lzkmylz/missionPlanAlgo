package orekit.visibility;

import orekit.visibility.model.BatchResult;
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;
import orekit.visibility.model.VisibilityWindow;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * 小规模验证测试
 * 10星 × 100目标 × 1小时
 */
public class SmallScaleTest {
    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("小规模验证测试 (10星 × 100目标 × 1小时)");
        System.out.println("========================================");

        try {
            // 初始化Orekit
            initializeOrekit();

            // 创建场景
            List<SatelliteConfig> satellites = createSatellites(10);
            List<TargetConfig> targets = createTargets(100);
            AbsoluteDate startTime = AbsoluteDate.J2000_EPOCH;
            AbsoluteDate endTime = startTime.shiftedBy(3600); // 1小时

            System.out.println("卫星数量: " + satellites.size());
            System.out.println("目标数量: " + targets.size());
            System.out.println("时间跨度: 1小时");
            System.out.println("计算对: " + satellites.size() * targets.size());

            // 执行计算
            System.out.println("\n开始计算...");
            long startNs = System.nanoTime();

            OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();
            BatchResult result = calculator.computeAllVisibilityWindows(
                satellites, targets, startTime, endTime, 5.0, 1.0
            );

            long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNs);

            // 输出结果
            System.out.println("\n========================================");
            System.out.println("计算完成!");
            System.out.println("========================================");
            System.out.println("耗时: " + elapsedMs + " ms (" + String.format("%.2f", elapsedMs/1000.0) + "秒)");
            System.out.println("总窗口数: " + result.getStatistics().getTotalWindows());
            System.out.println("平均每目标: " + String.format("%.2f", (double)result.getStatistics().getTotalWindows()/targets.size()));

            // 显示部分窗口
            System.out.println("\n前5个可见窗口:");
            int count = 0;
            for (Map.Entry<String, List<VisibilityWindow>> entry : result.getAllWindows().entrySet()) {
                for (VisibilityWindow window : entry.getValue()) {
                    if (count++ >= 5) break;
                    System.out.println("  " + window);
                }
                if (count >= 5) break;
            }

            // 创建输出目录测试
            Files.createDirectories(Paths.get("output"));
            System.out.println("\n输出目录创建成功: output/");

        } catch (Exception e) {
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static List<SatelliteConfig> createSatellites(int count) {
        List<SatelliteConfig> satellites = new ArrayList<>();
        for (int i = 1; i <= count; i++) {
            satellites.add(new SatelliteConfig(
                "SAT-" + String.format("%02d", i),
                "", "", 5.0, 30.0
            ));
        }
        return satellites;
    }

    private static List<TargetConfig> createTargets(int count) {
        List<TargetConfig> targets = new ArrayList<>();
        for (int i = 1; i <= count; i++) {
            double longitude = -180 + (360.0 * (i - 1) / count);
            double latitude = -90 + (180.0 * (i - 1) / count);
            targets.add(new TargetConfig(
                "TGT-" + String.format("%03d", i),
                longitude, latitude, 0.0, 60, 5
            ));
        }
        return targets;
    }

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
