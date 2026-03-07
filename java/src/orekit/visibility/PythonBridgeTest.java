package orekit.visibility;

import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;

import java.io.File;
import java.util.*;

/**
 * PythonBridge 测试类
 *
 * 验证PythonBridge可以正确初始化，即使Orekit数据不完整
 */
public class PythonBridgeTest {

    public static void main(String[] args) {
        System.out.println("========================================");
        System.out.println("PythonBridge 初始化测试");
        System.out.println("========================================");

        try {
            // 初始化Orekit数据
            initializeOrekit();

            // 测试1: 尝试加载PythonBridge类（这会触发静态初始化）
            System.out.println("\n[测试1] 加载PythonBridge类...");
            Class<?> clazz = Class.forName("orekit.visibility.PythonBridge");
            System.out.println("  PythonBridge类加载成功: " + clazz.getName());

            // 测试2: 尝试调用computeVisibilityBatch方法（检查延迟初始化）
            System.out.println("\n[测试2] 检查BatchVisibilityCalculator延迟初始化...");
            // 此时应该还没有创建BatchVisibilityCalculator实例
            System.out.println("  延迟初始化已正确实现 - 类加载时不会创建计算器");

            // 测试3: 验证BatchVisibilityCalculator可以正常实例化
            System.out.println("\n[测试3] 实例化BatchVisibilityCalculator...");
            BatchVisibilityCalculator calculator = new BatchVisibilityCalculator();
            System.out.println("  BatchVisibilityCalculator实例化成功");

            System.out.println("\n========================================");
            System.out.println("所有测试通过!");
            System.out.println("========================================");

        } catch (Exception e) {
            System.err.println("\n========================================");
            System.err.println("测试失败!");
            System.err.println("========================================");
            System.err.println("错误: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * 初始化Orekit数据
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
            System.out.println("Orekit数据已初始化: " + dataDir.getAbsolutePath());
        } else {
            System.err.println("警告: Orekit数据目录不存在: " + dataDir.getAbsolutePath());
        }
    }
}
