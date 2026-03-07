package orekit.visibility;

import java.io.*;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

/**
 * 轨道数据导出器
 *
 * 将OrbitStateCache中的轨道数据导出为JSON格式，供Python端加载使用。
 * 使用GZIP压缩，实现高压缩比。
 */
public class OrbitDataExporter {

    /**
     * 导出轨道数据到JSON文件（GZIP压缩）
     *
     * @param orbitCache 轨道缓存 (satellite_id -> List<OrbitState>)
     * @param outputPath 输出文件路径
     * @throws IOException 当写入失败时抛出
     */
    public void exportToJson(
            Map<String, List<OrbitStateCache.OrbitState>> orbitCache,
            String outputPath) throws IOException {

        System.out.println("导出轨道数据到JSON文件: " + outputPath);
        System.out.println("  卫星数量: " + orbitCache.size());

        // 计算总记录数
        long totalRecords = orbitCache.values().stream()
                .mapToLong(List::size)
                .sum();
        System.out.println("  总记录数: " + totalRecords);

        // 确保输出目录存在
        File outputFile = new File(outputPath);
        File parentDir = outputFile.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        // 删除已存在的文件
        if (outputFile.exists()) {
            outputFile.delete();
        }

        // 使用GZIP压缩写入
        try (Writer writer = new OutputStreamWriter(
                new GZIPOutputStream(new FileOutputStream(outputPath)), "UTF-8")) {

            // 写入JSON头
            writer.write("{\n");
            writer.write("  \"metadata\": {\n");
            writer.write("    \"satellite_count\": " + orbitCache.size() + ",\n");
            writer.write("    \"total_records\": " + totalRecords + "\n");
            writer.write("  },\n");
            writer.write("  \"orbits\": [\n");

            long recordCount = 0;
            long lastProgress = 0;
            boolean firstRecord = true;

            // 遍历所有卫星
            for (Map.Entry<String, List<OrbitStateCache.OrbitState>> entry : orbitCache.entrySet()) {
                String satId = entry.getKey();
                List<OrbitStateCache.OrbitState> states = entry.getValue();

                // 写入该卫星的所有状态
                for (OrbitStateCache.OrbitState state : states) {
                    if (!firstRecord) {
                        writer.write(",\n");
                    }
                    firstRecord = false;

                    // 写入单条记录
                    writer.write("    {\n");
                    writer.write(String.format("      \"satellite_id\": \"%s\",\n", satId));
                    writer.write(String.format("      \"timestamp\": %d,\n", (int) state.timestamp));
                    writer.write(String.format("      \"pos_x\": %.6f,\n", state.x));
                    writer.write(String.format("      \"pos_y\": %.6f,\n", state.y));
                    writer.write(String.format("      \"pos_z\": %.6f,\n", state.z));
                    writer.write(String.format("      \"vel_x\": %.6f,\n", state.vx));
                    writer.write(String.format("      \"vel_y\": %.6f,\n", state.vy));
                    writer.write(String.format("      \"vel_z\": %.6f,\n", state.vz));
                    writer.write(String.format("      \"lat\": %.6f,\n", state.latitude));
                    writer.write(String.format("      \"lon\": %.6f,\n", state.longitude));
                    writer.write(String.format("      \"alt\": %.6f\n", state.altitude));
                    writer.write("    }");

                    recordCount++;

                    // 每10万条记录显示进度
                    if (recordCount - lastProgress >= 100000) {
                        System.out.printf("  进度: %d/%d (%.1f%%)%n",
                                recordCount, totalRecords,
                                100.0 * recordCount / totalRecords);
                        lastProgress = recordCount;
                    }
                }
            }

            // 写入JSON尾
            writer.write("\n  ]\n");
            writer.write("}\n");

            System.out.println("  成功导出 " + recordCount + " 条记录");
        }

        // 显示文件大小
        long fileSize = outputFile.length();
        System.out.printf("  文件大小: %.2f MB (GZIP压缩)%n", fileSize / (1024.0 * 1024.0));
        System.out.printf("  压缩率: %.2fx%n",
                (totalRecords * 200.0) / fileSize); // 估算原始JSON大小
    }

    /**
     * 导出轨道数据（简化接口）
     *
     * 从OrbitStateCache直接导出
     */
    public void exportFromCache(OrbitStateCache cache, String outputPath) throws IOException {
        // 通过反射获取缓存数据
        java.lang.reflect.Field cacheField;
        try {
            cacheField = OrbitStateCache.class.getDeclaredField("cache");
            cacheField.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<String, List<OrbitStateCache.OrbitState>> data =
                    (Map<String, List<OrbitStateCache.OrbitState>>) cacheField.get(cache);
            exportToJson(data, outputPath);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new IOException("无法访问OrbitStateCache内部数据", e);
        }
    }
}
