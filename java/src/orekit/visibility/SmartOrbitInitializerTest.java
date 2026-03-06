package orekit.visibility;

import orekit.visibility.model.SatelliteConfig;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.propagation.SpacecraftState;

/**
 * SmartOrbitInitializer测试类
 *
 * 测试智能轨道初始化逻辑：
 * 1. TLE数据使用SGP4外推
 * 2. 六根数+历元近(<3天)直接HPOP
 * 3. 六根数+历元远(>3天)使用J4外推到场景开始
 */
public class SmartOrbitInitializerTest {

    public static void main(String[] args) throws Exception {
        System.out.println("========================================");
        System.out.println("SmartOrbitInitializer 测试");
        System.out.println("========================================\n");

        // 初始化Orekit
        try {
            org.orekit.data.DataContext.getDefault().getDataProvidersManager()
                .addProvider(new org.orekit.data.ZipJarCrawler("orekit-data.zip"));
        } catch (Exception e) {
            System.out.println("Warning: Could not load orekit-data.zip, using default");
        }

        SmartOrbitInitializerTest test = new SmartOrbitInitializerTest();

        // 运行测试
        test.testTLEPropagation();
        test.testElementsNearEpoch();
        test.testElementsFarEpoch();
        test.testScenarioTimeCalculation();

        System.out.println("\n========================================");
        System.out.println("所有测试通过!");
        System.out.println("========================================");
    }

    /**
     * 测试1: TLE数据使用SGP4外推
     */
    public void testTLEPropagation() throws Exception {
        System.out.println("[Test 1] TLE数据应使用SGP4外推到场景开始时间");

        // 创建一个TLE卫星（国际空间站示例）
        String tleLine1 = "1 25544U 98067A   24075.50000000  .00020000  00000-0  28000-4 0  9999";
        String tleLine2 = "2 25544  51.6400  86.0000 0005000  90.0000 270.0000 15.50000000    10";

        SatelliteConfig sat = new SatelliteConfig("ISS-TEST", tleLine1, tleLine2, 5.0, 30.0);

        // 场景时间（比TLE历元晚几天）
        AbsoluteDate scenarioStart = new AbsoluteDate(
            "2024-03-15T00:00:00.000", TimeScalesFactory.getUTC());

        // 执行初始化
        SmartOrbitInitializer initializer = new SmartOrbitInitializer();
        SpacecraftState state = initializer.getInitialStateAtScenarioStart(sat, scenarioStart);

        // 验证：状态时间应该是场景开始时间
        assert state.getDate().equals(scenarioStart) :
            "TLE外推后的时间应该等于场景开始时间";

        // 验证：位置有效（非零）
        double posNorm = state.getPVCoordinates().getPosition().getNorm();
        assert posNorm > 6000000 && posNorm < 8000000 :
            "LEO卫星位置应该在6000-8000km之间，实际: " + posNorm;

        System.out.println("  ✓ TLE SGP4外推成功");
        System.out.println("    位置范数: " + String.format("%.2f", posNorm/1000) + " km");
    }

    /**
     * 测试2: 六根数+历元近(<3天)直接返回
     */
    public void testElementsNearEpoch() throws Exception {
        System.out.println("\n[Test 2] 历元距场景<3天应直接返回历元轨道");

        // 创建一个历元接近场景的卫星配置
        AbsoluteDate epoch = new AbsoluteDate(
            "2024-03-14T00:00:00.000", TimeScalesFactory.getUTC());  // 1天前
        AbsoluteDate scenarioStart = new AbsoluteDate(
            "2024-03-15T00:00:00.000", TimeScalesFactory.getUTC());

        JsonScenarioLoader.ExtendedSatelliteConfig sat =
            new JsonScenarioLoader.ExtendedSatelliteConfig(
                "SAT-NEAR", "optical",
                6871000.0, 0.001, 55.0, 0.0, 0.0, 0.0,  // 轨道参数
                5.0, 30.0, epoch,                        // 约束和历元
                100.0, 5.0, 1.5, 2.2                     // 物理参数
            );

        SmartOrbitInitializer initializer = new SmartOrbitInitializer();
        SpacecraftState state = initializer.getInitialStateAtScenarioStart(sat, scenarioStart);

        // 验证：历元近的情况下，返回的应该是历元时刻的状态
        // HPOP会从历元传播到场景结束
        assert state != null : "状态不应为空";

        double posNorm = state.getPVCoordinates().getPosition().getNorm();
        assert posNorm > 6000000 && posNorm < 8000000 :
            "位置应该在合理范围，实际: " + posNorm;

        System.out.println("  ✓ 历元近的情况处理正确");
        System.out.println("    位置范数: " + String.format("%.2f", posNorm/1000) + " km");
    }

    /**
     * 测试3: 六根数+历元远(>3天)使用J4外推
     */
    public void testElementsFarEpoch() throws Exception {
        System.out.println("\n[Test 3] 历元距场景>3天应使用J4外推到场景开始");

        // 创建一个历元远离场景的卫星配置（J2000）
        AbsoluteDate epoch = org.orekit.time.AbsoluteDate.J2000_EPOCH;  // 2000年
        AbsoluteDate scenarioStart = new AbsoluteDate(
            "2024-03-15T00:00:00.000", TimeScalesFactory.getUTC());  // 24年后

        JsonScenarioLoader.ExtendedSatelliteConfig sat =
            new JsonScenarioLoader.ExtendedSatelliteConfig(
                "SAT-FAR", "optical",
                6871000.0, 0.001, 55.0, 0.0, 0.0, 0.0,  // 轨道参数
                5.0, 30.0, epoch,                        // 约束和历元
                100.0, 5.0, 1.5, 2.2                     // 物理参数
            );

        SmartOrbitInitializer initializer = new SmartOrbitInitializer();

        long startTime = System.currentTimeMillis();
        SpacecraftState state = initializer.getInitialStateAtScenarioStart(sat, scenarioStart);
        long duration = System.currentTimeMillis() - startTime;

        // 验证：外推应该很快完成（<1秒，因为是解析方法）
        assert duration < 1000 :
            "J4外推应该很快完成，实际耗时: " + duration + "ms";

        // 验证：状态时间应该是场景开始时间
        assert state.getDate().equals(scenarioStart) :
            "J4外推后的时间应该等于场景开始时间";

        // 验证：位置有效
        double posNorm = state.getPVCoordinates().getPosition().getNorm();
        assert posNorm > 6000000 && posNorm < 8000000 :
            "位置应该在合理范围，实际: " + posNorm;

        System.out.println("  ✓ J4外推成功");
        System.out.println("    耗时: " + duration + " ms");
        System.out.println("    位置范数: " + String.format("%.2f", posNorm/1000) + " km");
    }

    /**
     * 测试4: 场景时间计算准确性
     */
    public void testScenarioTimeCalculation() throws Exception {
        System.out.println("\n[Test 4] 场景时间计算准确性");

        AbsoluteDate epoch = new AbsoluteDate(
            "2024-03-10T00:00:00.000", TimeScalesFactory.getUTC());
        AbsoluteDate scenarioStart = new AbsoluteDate(
            "2024-03-15T00:00:00.000", TimeScalesFactory.getUTC());

        // 计算时间差
        double days = scenarioStart.durationFrom(epoch) / 86400.0;

        assert Math.abs(days - 5.0) < 0.001 :
            "时间差应该是5天，实际: " + days;

        System.out.println("  ✓ 时间计算正确: " + String.format("%.2f", days) + " 天");
    }
}
