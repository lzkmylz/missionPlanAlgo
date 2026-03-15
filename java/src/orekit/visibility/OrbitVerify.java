package orekit.visibility;

import orekit.visibility.model.SatelliteConfig;
import org.orekit.data.DataContext;
import org.orekit.data.DirectoryCrawler;
import java.io.File;
import java.util.*;

public class OrbitVerify {
    public static void main(String[] args) throws Exception {
        String dataPath = System.getenv().getOrDefault("OREKIT_DATA_PATH",
            System.getProperty("user.home") + "/orekit-data");
        DataContext.getDefault().getDataProvidersManager().addProvider(
            new DirectoryCrawler(new File(dataPath))
        );

        JsonScenarioLoader loader = new JsonScenarioLoader("../scenarios/generated_scenario.json");
        loader.load();

        List<SatelliteConfig> sats = loader.loadSatellites();

        System.out.println("================================================================================");
        System.out.println("                        全部60颗卫星轨道参数");
        System.out.println("================================================================================");
        System.out.printf("%-8s %-6s %-10s %-8s %-8s %-8s %-10s %-10s %-10s%n",
            "卫星ID", "类型", "半长轴(km)", "倾角(°)", "RAAN(°)", "MA(°)", "滚转限制", "俯仰限制", "轨道面");
        System.out.println("--------------------------------------------------------------------------------");

        // 按轨道面分组统计
        Map<Integer, List<String>> planeGroups = new TreeMap<>();

        for (SatelliteConfig sat : sats) {
            String satId = sat.getId();
            String satType = satId.startsWith("OPT") ? "光学" : "SAR ";
            double maxRoll = sat.getMaxRollAngle();
            double maxPitch = sat.getMaxPitchAngle();

            if (sat instanceof JsonScenarioLoader.ExtendedSatelliteConfig) {
                JsonScenarioLoader.ExtendedSatelliteConfig es =
                    (JsonScenarioLoader.ExtendedSatelliteConfig) sat;

                double a = es.semiMajorAxis / 1000.0; // 转换为km
                double i = es.inclination;  // 已经是度数
                double raan = es.raan;      // 已经是度数
                double ma = es.meanAnomaly; // 已经是度数

                // 计算轨道面编号 (基于RAAN)
                int planeNum = (int) Math.round(raan / 72.0) + 1;
                if (planeNum > 5) planeNum = 5;

                planeGroups.computeIfAbsent(planeNum, k -> new ArrayList<>()).add(satId);

                System.out.printf("%-8s %-6s %-10.1f %-8.1f %-8.1f %-8.1f %-10.1f %-10.1f %-10d%n",
                    satId, satType, a, i, raan, ma, maxRoll, maxPitch, planeNum);
            } else {
                System.out.printf("%-8s %-6s %-10s %-8s %-8s %-8s %-10.1f %-10.1f %-10s%n",
                    satId, satType, "N/A", "N/A", "N/A", "N/A", maxRoll, maxPitch, "N/A");
            }
        }

        System.out.println("================================================================================");
        System.out.println("                           轨道面分布统计");
        System.out.println("================================================================================");

        for (Map.Entry<Integer, List<String>> entry : planeGroups.entrySet()) {
            int planeNum = entry.getKey();
            List<String> members = entry.getValue();
            double raan = (planeNum - 1) * 72.0;

            System.out.printf("轨道面 %d (RAAN=%.0f°): %d颗卫星%n", planeNum, raan, members.size());
            System.out.println("  成员: " + String.join(", ", members));
        }

        System.out.println("================================================================================");
        System.out.println("                           卫星类型统计");
        System.out.println("================================================================================");

        long opticalCount = sats.stream().filter(s -> s.getId().startsWith("OPT")).count();
        long sarCount = sats.stream().filter(s -> s.getId().startsWith("SAR")).count();

        System.out.println("光学卫星: " + opticalCount + "颗");
        System.out.println("SAR卫星:  " + sarCount + "颗");
        System.out.println("总计:     " + sats.size() + "颗");

        System.out.println("================================================================================");
        System.out.println("                           姿态限制配置");
        System.out.println("================================================================================");

        SatelliteConfig firstOptical = sats.stream()
            .filter(s -> s.getId().startsWith("OPT"))
            .findFirst().orElse(null);
        SatelliteConfig firstSar = sats.stream()
            .filter(s -> s.getId().startsWith("SAR"))
            .findFirst().orElse(null);

        if (firstOptical != null) {
            System.out.println("光学卫星姿态限制:");
            System.out.println("  最大滚转角: ±" + firstOptical.getMaxRollAngle() + "°");
            System.out.println("  最大俯仰角: ±" + firstOptical.getMaxPitchAngle() + "°");
        }

        if (firstSar != null) {
            System.out.println("SAR卫星姿态限制:");
            System.out.println("  最大滚转角: ±" + firstSar.getMaxRollAngle() + "°");
            System.out.println("  最大俯仰角: ±" + firstSar.getMaxPitchAngle() + "°");
        }

        System.out.println("================================================================================");
    }
}
