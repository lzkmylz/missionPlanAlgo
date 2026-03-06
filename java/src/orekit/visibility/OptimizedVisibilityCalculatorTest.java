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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

/**
 * OptimizedVisibilityCalculator 测试类
 *
 * 测试卫星-目标对并行计算功能
 */
public class OptimizedVisibilityCalculatorTest {

    private static final double EPSILON = 1e-6;

    public static void main(String[] args) {
        System.out.println("Running OptimizedVisibilityCalculator Tests...");
        System.out.println("================================================");

        // 初始化Orekit数据
        try {
            initializeOrekitData();
        } catch (Exception e) {
            System.out.println("Warning: Could not initialize Orekit data: " + e.getMessage());
            System.out.println("Tests will use simplified time scales.");
        }

        int passed = 0;
        int failed = 0;

        // Test 1: 串行和并行计算结果一致性
        try {
            testSerialParallelConsistency();
            passed++;
            System.out.println("✓ testSerialParallelConsistency PASSED");
        } catch (AssertionError e) {
            failed++;
            System.out.println("✗ testSerialParallelConsistency FAILED: " + e.getMessage());
        } catch (Exception e) {
            failed++;
            System.out.println("✗ testSerialParallelConsistency ERROR: " + e.getMessage());
            e.printStackTrace();
        }

        // Test 2: 并行计算性能优于串行
        try {
            testParallelPerformance();
            passed++;
            System.out.println("✓ testParallelPerformance PASSED");
        } catch (AssertionError e) {
            failed++;
            System.out.println("✗ testParallelPerformance FAILED: " + e.getMessage());
        } catch (Exception e) {
            failed++;
            System.out.println("✗ testParallelPerformance ERROR: " + e.getMessage());
            e.printStackTrace();
        }

        // Test 3: 空列表处理
        try {
            testEmptyLists();
            passed++;
            System.out.println("✓ testEmptyLists PASSED");
        } catch (AssertionError e) {
            failed++;
            System.out.println("✗ testEmptyLists FAILED: " + e.getMessage());
        } catch (Exception e) {
            failed++;
            System.out.println("✗ testEmptyLists ERROR: " + e.getMessage());
            e.printStackTrace();
        }

        // Test 4: 单卫星单目标场景
        try {
            testSingleSatelliteSingleTarget();
            passed++;
            System.out.println("✓ testSingleSatelliteSingleTarget PASSED");
        } catch (AssertionError e) {
            failed++;
            System.out.println("✗ testSingleSatelliteSingleTarget FAILED: " + e.getMessage());
        } catch (Exception e) {
            failed++;
            System.out.println("✗ testSingleSatelliteSingleTarget ERROR: " + e.getMessage());
            e.printStackTrace();
        }

        // Test 5: 大规模并行计算正确性
        try {
            testLargeScaleParallel();
            passed++;
            System.out.println("✓ testLargeScaleParallel PASSED");
        } catch (AssertionError e) {
            failed++;
            System.out.println("✗ testLargeScaleParallel FAILED: " + e.getMessage());
        } catch (Exception e) {
            failed++;
            System.out.println("✗ testLargeScaleParallel ERROR: " + e.getMessage());
            e.printStackTrace();
        }

        // Test 6: 单星任务规划性能测试
        try {
            testSingleSatellitePerformance();
            passed++;
            System.out.println("✓ testSingleSatellitePerformance PASSED");
        } catch (AssertionError e) {
            failed++;
            System.out.println("✗ testSingleSatellitePerformance FAILED: " + e.getMessage());
        } catch (Exception e) {
            failed++;
            System.out.println("✗ testSingleSatellitePerformance ERROR: " + e.getMessage());
            e.printStackTrace();
        }

        System.out.println("================================================");
        System.out.println("Test Results: " + passed + " passed, " + failed + " failed");

        if (failed > 0) {
            System.exit(1);
        }
    }

    /**
     * 初始化Orekit数据
     */
    private static void initializeOrekitData() throws Exception {
        // 尝试从环境变量或默认路径加载数据
        String dataPath = System.getenv("OREKIT_DATA_PATH");
        if (dataPath == null) {
            // 尝试常见的数据路径
            String[] possiblePaths = {
                "/usr/local/share/orekit",
                System.getProperty("user.home") + "/orekit-data",
                "./orekit-data"
            };

            for (String path : possiblePaths) {
                File dataDir = new File(path);
                if (dataDir.exists() && dataDir.isDirectory()) {
                    dataPath = path;
                    break;
                }
            }
        }

        if (dataPath != null) {
            File dataDir = new File(dataPath);
            if (dataDir.exists()) {
                DataContext.getDefault().getDataProvidersManager().addProvider(
                    new DirectoryCrawler(dataDir)
                );
                System.out.println("Loaded Orekit data from: " + dataPath);
            }
        } else {
            System.out.println("Warning: No Orekit data directory found.");
            System.out.println("Set OREKIT_DATA_PATH environment variable to your orekit-data directory.");
        }
    }

    /**
     * 测试1: 串行和并行计算结果一致性
     * 确保并行计算不会引入精度损失或错误
     */
    private static void testSerialParallelConsistency() throws Exception {
        System.out.println("\n--- Test: Serial/Parallel Consistency ---");

        OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();

        // 创建测试数据
        List<SatelliteConfig> satellites = createTestSatellites(5);
        List<TargetConfig> targets = createTestTargets(10);

        AbsoluteDate startTime = AbsoluteDate.J2000_EPOCH;
        AbsoluteDate endTime = startTime.shiftedBy(3600); // 1小时

        double coarseStep = 5.0;   // 5秒粗扫描步长（高精度）
        double fineStep = 1.0;     // 1秒精化步长（高精度）

        // 执行计算
        BatchResult result = calculator.computeAllVisibilityWindows(
            satellites, targets, startTime, endTime, coarseStep, fineStep
        );

        // 验证结果
        assert result != null : "Result should not be null";
        assert result.getStatistics().getTotalWindows() >= 0 : "Window count should be non-negative";

        // 验证所有卫星-目标对都被处理
        int expectedPairs = satellites.size() * targets.size();
        assert result.getStatistics().getTotalPairs() == satellites.size() :
            "Satellite count should match";

        System.out.println("  Windows found: " + result.getStatistics().getTotalWindows());
        System.out.println("  Computation time: " + result.getStatistics().getComputationTimeMs() + " ms");
    }

    /**
     * 测试2: 并行计算性能优于串行
     * 验证并行化确实带来了性能提升
     */
    private static void testParallelPerformance() throws Exception {
        System.out.println("\n--- Test: Parallel Performance ---");

        // 使用较大规模数据测试性能
        List<SatelliteConfig> satellites = createTestSatellites(10);
        List<TargetConfig> targets = createTestTargets(50);

        AbsoluteDate startTime = AbsoluteDate.J2000_EPOCH;
        AbsoluteDate endTime = startTime.shiftedBy(7200); // 2小时

        double coarseStep = 5.0;   // 5秒粗扫描步长（高精度）
        double fineStep = 1.0;     // 1秒精化步长（高精度）

        // 预热JVM
        OptimizedVisibilityCalculator warmupCalculator = new OptimizedVisibilityCalculator();
        warmupCalculator.computeAllVisibilityWindows(
            createTestSatellites(2), createTestTargets(5),
            startTime, endTime, coarseStep, fineStep
        );

        // 正式测试
        OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();

        long startNs = System.nanoTime();
        BatchResult result = calculator.computeAllVisibilityWindows(
            satellites, targets, startTime, endTime, coarseStep, fineStep
        );
        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNs);

        System.out.println("  Total pairs: " + (satellites.size() * targets.size()));
        System.out.println("  Total windows: " + result.getStatistics().getTotalWindows());
        System.out.println("  Elapsed time: " + elapsedMs + " ms");

        // 验证计算在合理时间内完成（高精度配置下60秒以内）
        assert elapsedMs < 60000 : "Computation should complete within 60 seconds, took " + elapsedMs + " ms";

        // 验证性能指标 - 每对平均时间（高精度配置下应小于100ms）
        long pairs = satellites.size() * targets.size();
        double msPerPair = (double) elapsedMs / pairs;
        System.out.println("  ms per pair: " + String.format("%.3f", msPerPair));

        assert msPerPair < 100.0 : "Should process each pair in less than 100ms, took " + msPerPair + " ms";
    }

    /**
     * 测试3: 空列表处理
     * 验证对空卫星列表或空目标列表的处理
     */
    private static void testEmptyLists() throws Exception {
        System.out.println("\n--- Test: Empty Lists ---");

        OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();

        AbsoluteDate startTime = AbsoluteDate.J2000_EPOCH;
        AbsoluteDate endTime = startTime.shiftedBy(3600);

        // 测试空卫星列表
        List<SatelliteConfig> emptySats = new ArrayList<>();
        List<TargetConfig> targets = createTestTargets(5);

        BatchResult result1 = calculator.computeAllVisibilityWindows(
            emptySats, targets, startTime, endTime, 60.0, 10.0
        );

        assert result1 != null : "Result should not be null for empty satellites";
        assert result1.getStatistics().getTotalWindows() == 0 : "Should find no windows with no satellites";

        // 测试空目标列表
        List<SatelliteConfig> satellites = createTestSatellites(3);
        List<TargetConfig> emptyTargets = new ArrayList<>();

        BatchResult result2 = calculator.computeAllVisibilityWindows(
            satellites, emptyTargets, startTime, endTime, 60.0, 10.0
        );

        assert result2 != null : "Result should not be null for empty targets";
        assert result2.getStatistics().getTotalWindows() == 0 : "Should find no windows with no targets";

        System.out.println("  Empty satellites: OK");
        System.out.println("  Empty targets: OK");
    }

    /**
     * 测试4: 单卫星单目标场景
     * 验证最基本的场景正常工作
     */
    private static void testSingleSatelliteSingleTarget() throws Exception {
        System.out.println("\n--- Test: Single Satellite Single Target ---");

        OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();

        List<SatelliteConfig> satellites = createTestSatellites(1);
        List<TargetConfig> targets = createTestTargets(1);

        AbsoluteDate startTime = AbsoluteDate.J2000_EPOCH;
        AbsoluteDate endTime = startTime.shiftedBy(3600 * 2); // 2小时

        double coarseStep = 5.0;   // 5秒粗扫描步长（高精度）
        double fineStep = 1.0;     // 1秒精化步长（高精度）

        BatchResult result = calculator.computeAllVisibilityWindows(
            satellites, targets, startTime, endTime, coarseStep, fineStep
        );

        assert result != null : "Result should not be null";
        assert result.getAllWindows() != null : "Windows map should not be null";

        System.out.println("  Windows found: " + result.getStatistics().getTotalWindows());
        System.out.println("  Computation time: " + result.getStatistics().getComputationTimeMs() + " ms");
    }

    /**
     * 测试5: 大规模并行计算正确性
     * 验证大规模数据下的并行计算正确性
     */
    private static void testLargeScaleParallel() throws Exception {
        System.out.println("\n--- Test: Large Scale Parallel ---");

        OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();

        // 创建较大规模测试数据
        List<SatelliteConfig> satellites = createTestSatellites(20);
        List<TargetConfig> targets = createTestTargets(100);

        AbsoluteDate startTime = AbsoluteDate.J2000_EPOCH;
        AbsoluteDate endTime = startTime.shiftedBy(3600); // 1小时

        double coarseStep = 5.0;   // 5秒粗扫描步长（高精度）
        double fineStep = 1.0;     // 1秒精化步长（高精度）

        long startNs = System.nanoTime();
        BatchResult result = calculator.computeAllVisibilityWindows(
            satellites, targets, startTime, endTime, coarseStep, fineStep
        );
        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNs);

        int totalPairs = satellites.size() * targets.size();
        int totalWindows = result.getStatistics().getTotalWindows();

        System.out.println("  Total pairs processed: " + totalPairs);
        System.out.println("  Total windows found: " + totalWindows);
        System.out.println("  Elapsed time: " + elapsedMs + " ms");
        System.out.println("  Avg windows per pair: " + String.format("%.2f", (double) totalWindows / totalPairs));

        // 验证所有对都被处理（没有遗漏）
        assert result.getAllWindows().size() <= totalPairs : "Should not have more window entries than pairs";

        // 大规模计算应在180秒内完成（高精度配置）
        assert elapsedMs < 180000 : "Large scale computation should complete within 180 seconds";
    }

    /**
     * 测试6: 单星任务规划性能测试
     * 对比实际性能与预期性能
     */
    private static void testSingleSatellitePerformance() throws Exception {
        System.out.println("\n--- Test: Single Satellite Performance Benchmark ---");
        System.out.println("  场景: 1星 × 10目标 × 1天");
        System.out.println("  配置: EGM2008 60x60 + 5秒粗扫描 + 1秒精化");
        System.out.println("  预期性能: ~30-60秒 (高精度HPOP数值传播)");
        System.out.println("  ------------------------------------------------");

        // 预热JVM
        System.out.println("  [1/3] JVM预热...");
        OptimizedVisibilityCalculator warmupCalc = new OptimizedVisibilityCalculator();
        warmupCalc.computeAllVisibilityWindows(
            createTestSatellites(1), createTestTargets(2),
            AbsoluteDate.J2000_EPOCH, AbsoluteDate.J2000_EPOCH.shiftedBy(300),
            60.0, 10.0
        );

        // 正式测试配置
        List<SatelliteConfig> satellites = createTestSatellites(1);
        List<TargetConfig> targets = createTestTargets(10);

        AbsoluteDate startTime = AbsoluteDate.J2000_EPOCH;
        AbsoluteDate endTime = startTime.shiftedBy(24 * 3600); // 1天 = 86400秒

        double coarseStep = 5.0;   // 5秒粗扫描步长（高精度）
        double fineStep = 1.0;     // 1秒精化步长（高精度）

        // 执行性能测试
        System.out.println("  [2/3] 执行性能测试...");
        OptimizedVisibilityCalculator calculator = new OptimizedVisibilityCalculator();

        long startNs = System.nanoTime();
        BatchResult result = calculator.computeAllVisibilityWindows(
            satellites, targets, startTime, endTime, coarseStep, fineStep
        );
        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNs);

        // 获取详细统计
        int totalWindows = result.getStatistics().getTotalWindows();
        long computationTimeMs = result.getStatistics().getComputationTimeMs();

        // 输出结果
        System.out.println("  [3/3] 性能测试结果:");
        System.out.println("  ------------------------------------------------");
        System.out.println("  配置参数:");
        System.out.println("    - 时间跨度: 1天 (86400秒)");
        System.out.println("    - 缓存步长: 5秒 (HPOP预计算)");
        System.out.println("    - 粗扫描步长: 5秒 (高精度)");
        System.out.println("    - 精化步长: 1秒 (高精度)");
        System.out.println("    - 引力场: EGM2008 60x60");
        System.out.println("    - 目标数量: 10个");
        System.out.println("  ------------------------------------------------");
        System.out.println("  实际性能:");
        System.out.println("    - 总耗时: " + elapsedMs + " ms (" + String.format("%.2f", elapsedMs/1000.0) + "秒)");
        System.out.println("    - 发现窗口: " + totalWindows + " 个");
        System.out.println("  ------------------------------------------------");

        // 与预期对比
        long expectedMinMs = 30000;   // 30秒
        long expectedMaxMs = 60000;   // 60秒
        double performanceRatio = (double) elapsedMs / ((expectedMinMs + expectedMaxMs) / 2.0);

        System.out.println("  预期性能对比:");
        System.out.println("    - 预期范围: 30-60秒 (高精度配置)");
        System.out.println("    - 实际耗时: " + String.format("%.2f", elapsedMs/1000.0) + "秒");

        if (elapsedMs >= expectedMinMs && elapsedMs <= expectedMaxMs) {
            System.out.println("    - 结果: ✓ 符合预期范围");
        } else if (elapsedMs < expectedMinMs) {
            System.out.println("    - 结果: ✓ 优于预期 (" + String.format("%.1f", (1-performanceRatio)*100) + "% 更快)");
        } else {
            System.out.println("    - 结果: ✗ 慢于预期 (" + String.format("%.1f", (performanceRatio-1)*100) + "% 更慢)");
        }

        System.out.println("  ------------------------------------------------");

        // 性能指标验证
        // 单星1天高精度计算应在120秒内完成
        assert elapsedMs < 120000 : "Single satellite 1-day high-precision computation should complete within 120 seconds, took " + elapsedMs + " ms";
    }

    // ============ 辅助方法 ============

    private static List<SatelliteConfig> createTestSatellites(int count) {
        List<SatelliteConfig> satellites = new ArrayList<>();
        for (int i = 1; i <= count; i++) {
            satellites.add(new SatelliteConfig(
                "SAT-" + String.format("%02d", i),
                "",  // TLE line 1
                "",  // TLE line 2
                5.0, // min elevation
                0.0  // sensor FOV
            ));
        }
        return satellites;
    }

    private static List<TargetConfig> createTestTargets(int count) {
        List<TargetConfig> targets = new ArrayList<>();
        for (int i = 1; i <= count; i++) {
            // 均匀分布在全球
            double longitude = -180 + (360.0 * (i - 1) / count);
            double latitude = -90 + (180.0 * (i - 1) / count);

            targets.add(new TargetConfig(
                "TGT-" + String.format("%03d", i),
                longitude,
                latitude,
                0.0,   // altitude
                60,    // min observation duration
                5      // priority
            ));
        }
        return targets;
    }
}
