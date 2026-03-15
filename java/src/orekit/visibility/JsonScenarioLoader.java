package orekit.visibility;

import orekit.visibility.model.GroundStationConfig;
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;
import org.json.JSONArray;
import org.json.JSONObject;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.DateTimeComponents;
import org.orekit.time.TimeScalesFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * JSON场景文件加载器
 *
 * 从large_scale_frequency.json加载场景配置
 */
public class JsonScenarioLoader {

    private final String jsonFilePath;
    private JSONObject scenarioData;

    public JsonScenarioLoader(String jsonFilePath) {
        this.jsonFilePath = jsonFilePath;
    }

    /**
     * 加载JSON文件
     */
    public void load() throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(jsonFilePath)));
        this.scenarioData = new JSONObject(content);
    }

    /**
     * 获取场景名称
     */
    public String getName() {
        return scenarioData.getString("name");
    }

    /**
     * 获取场景描述
     */
    public String getDescription() {
        return scenarioData.getString("description");
    }

    /**
     * 获取开始时间
     */
    public AbsoluteDate getStartTime() {
        JSONObject duration = scenarioData.getJSONObject("duration");
        String startStr = duration.getString("start");
        return parseDateTime(startStr);
    }

    /**
     * 获取结束时间
     */
    public AbsoluteDate getEndTime() {
        JSONObject duration = scenarioData.getJSONObject("duration");
        String endStr = duration.getString("end");
        return parseDateTime(endStr);
    }

    /**
     * 加载卫星配置
     */
    public List<SatelliteConfig> loadSatellites() {
        List<SatelliteConfig> satellites = new ArrayList<>();
        JSONArray satArray = scenarioData.getJSONArray("satellites");

        for (int i = 0; i < satArray.length(); i++) {
            JSONObject satJson = satArray.getJSONObject(i);

            String id = satJson.getString("id");
            String satType = satJson.getString("sat_type");

            // 获取能力参数
            JSONObject capabilities = satJson.getJSONObject("capabilities");
            double minElevation = 5.0; // 默认最小仰角

            // 读取姿态限制参数
            double maxRollAngle = capabilities.optDouble("max_roll_angle", 35.0);
            double maxPitchAngle = capabilities.optDouble("max_pitch_angle",
                    "sar".equalsIgnoreCase(satType) ? 30.0 : 20.0);

            // 构建卫星配置（使用轨道六根数存储在 SatelliteConfig 中）
            JSONObject orbit = satJson.getJSONObject("orbit");
            double semiMajorAxis = orbit.getDouble("semi_major_axis");
            double eccentricity = orbit.getDouble("eccentricity");
            double inclination = orbit.getDouble("inclination");
            double raan = orbit.getDouble("raan");
            double argOfPerigee = orbit.getDouble("arg_of_perigee");
            double meanAnomaly = orbit.getDouble("mean_anomaly");

            // 获取轨道历元（优先使用orbit中的epoch，否则使用场景开始时间）
            AbsoluteDate epoch;
            if (orbit.has("epoch")) {
                String epochStr = orbit.getString("epoch");
                epoch = new AbsoluteDate(epochStr, TimeScalesFactory.getUTC());
            } else {
                epoch = getStartTime();
            }

            // 读取物理参数（有默认值）
            double mass, dragArea, reflectivity, dragCoefficient;
            if (satJson.has("physical_params")) {
                JSONObject phys = satJson.getJSONObject("physical_params");
                mass = phys.optDouble("mass", -1);
                dragArea = phys.optDouble("drag_area", -1);
                reflectivity = phys.optDouble("reflectivity", -1);
                dragCoefficient = phys.optDouble("drag_coefficient", -1);
            } else {
                mass = -1;
                dragArea = -1;
                reflectivity = -1;
                dragCoefficient = -1;
            }

            // 根据卫星类型设置默认值
            if ("sar".equalsIgnoreCase(satType)) {
                if (mass < 0) mass = 150.0;           // SAR卫星更重
                if (dragArea < 0) dragArea = 8.0;     // SAR天线面积更大
                if (reflectivity < 0) reflectivity = 1.3;
                if (dragCoefficient < 0) dragCoefficient = 2.2;
            } else {
                // 默认光学卫星参数
                if (mass < 0) mass = 100.0;
                if (dragArea < 0) dragArea = 5.0;
                if (reflectivity < 0) reflectivity = 1.5;
                if (dragCoefficient < 0) dragCoefficient = 2.2;
            }

            // 创建卫星配置（扩展以支持轨道参数和物理参数）
            satellites.add(new ExtendedSatelliteConfig(
                id, satType, semiMajorAxis, eccentricity,
                inclination, raan, argOfPerigee, meanAnomaly,
                minElevation, maxRollAngle, maxPitchAngle, epoch,
                mass, dragArea, reflectivity, dragCoefficient
            ));
        }

        return satellites;
    }

    /**
     * 加载目标配置
     */
    public List<TargetConfig> loadTargets() {
        List<TargetConfig> targets = new ArrayList<>();
        JSONArray targetArray = scenarioData.getJSONArray("targets");

        for (int i = 0; i < targetArray.length(); i++) {
            JSONObject tgtJson = targetArray.getJSONObject(i);

            String id = tgtJson.getString("id");
            JSONArray location = tgtJson.getJSONArray("location");
            double lon = location.getDouble(0);
            double lat = location.getDouble(1);
            int priority = tgtJson.getInt("priority");

            // 默认最小观测时间60秒
            int minDuration = 60;

            targets.add(new TargetConfig(id, lon, lat, 0.0, minDuration, priority));
        }

        return targets;
    }

    /**
     * 加载地面站配置
     */
    public List<GroundStationConfig> loadGroundStations() {
        List<GroundStationConfig> stations = new ArrayList<>();

        if (!scenarioData.has("ground_stations")) {
            return stations;
        }

        JSONArray gsArray = scenarioData.getJSONArray("ground_stations");

        for (int i = 0; i < gsArray.length(); i++) {
            JSONObject gsJson = gsArray.getJSONObject(i);

            String id = gsJson.getString("id");
            JSONArray location = gsJson.getJSONArray("location");
            double lon = location.getDouble(0);
            double lat = location.getDouble(1);
            double alt = location.getDouble(2);
            double minElevation = gsJson.getDouble("min_elevation");
            double dataRate = gsJson.optDouble("data_rate", 500.0);
            // 最大通信距离2,500km
            double maxRange = 2500000;

            stations.add(new GroundStationConfig(id, lon, lat, alt, minElevation, maxRange));
        }

        return stations;
    }

    /**
     * 加载频次需求
     */
    public List<ObservationRequirement> loadObservationRequirements() {
        List<ObservationRequirement> requirements = new ArrayList<>();

        if (!scenarioData.has("observation_requirements")) {
            return requirements;
        }

        JSONArray reqArray = scenarioData.getJSONArray("observation_requirements");

        for (int i = 0; i < reqArray.length(); i++) {
            JSONObject reqJson = reqArray.getJSONObject(i);

            String targetId = reqJson.getString("target_id");
            int requiredCount = reqJson.getInt("required_count");
            double minInterval = reqJson.optDouble("min_interval_seconds", 7200);
            double maxInterval = reqJson.optDouble("max_interval_seconds", 86400);

            requirements.add(new ObservationRequirement(targetId, requiredCount,
                minInterval, maxInterval));
        }

        return requirements;
    }

    /**
     * 解析ISO 8601日期时间字符串
     */
    private AbsoluteDate parseDateTime(String dateTimeStr) {
        try {
            // 移除Z后缀并解析
            if (dateTimeStr.endsWith("Z")) {
                dateTimeStr = dateTimeStr.substring(0, dateTimeStr.length() - 1);
            }

            // 使用Orekit的DateTimeComponents解析
            DateTimeComponents components = DateTimeComponents.parseDateTime(dateTimeStr);
            return new AbsoluteDate(components, org.orekit.time.TimeScalesFactory.getUTC());
        } catch (Exception e) {
            // 解析失败返回J2000
            System.err.println("Warning: Failed to parse date: " + dateTimeStr + ", using J2000");
            return AbsoluteDate.J2000_EPOCH;
        }
    }

    /**
     * 观测需求类
     */
    public static class ObservationRequirement {
        public final String targetId;
        public final int requiredCount;
        public final double minIntervalSeconds;
        public final double maxIntervalSeconds;

        public ObservationRequirement(String targetId, int requiredCount,
                                      double minIntervalSeconds, double maxIntervalSeconds) {
            this.targetId = targetId;
            this.requiredCount = requiredCount;
            this.minIntervalSeconds = minIntervalSeconds;
            this.maxIntervalSeconds = maxIntervalSeconds;
        }
    }

    /**
     * 扩展卫星配置类，包含完整轨道参数和物理参数
     */
    public static class ExtendedSatelliteConfig extends SatelliteConfig {
        public final String satType;
        public final double semiMajorAxis;
        public final double eccentricity;
        public final double inclination;
        public final double raan;
        public final double argOfPerigee;
        public final double meanAnomaly;
        public final AbsoluteDate epoch;  // 轨道历元时间

        // 物理参数（用于J4解析传播）
        public final double mass;              // kg
        public final double dragArea;          // m²
        public final double reflectivity;      // 反射系数
        public final double dragCoefficient;   // 阻力系数

        public ExtendedSatelliteConfig(String id, String satType,
                                       double semiMajorAxis, double eccentricity,
                                       double inclination, double raan,
                                       double argOfPerigee, double meanAnomaly,
                                       double minElevation, double maxRollAngle, double maxPitchAngle,
                                       AbsoluteDate epoch,
                                       double mass, double dragArea,
                                       double reflectivity, double dragCoefficient) {
            super(id, "", "", minElevation, 0.0, maxRollAngle, maxPitchAngle, satType);
            this.satType = satType;
            this.semiMajorAxis = semiMajorAxis;
            this.eccentricity = eccentricity;
            this.inclination = inclination;
            this.raan = raan;
            this.argOfPerigee = argOfPerigee;
            this.meanAnomaly = meanAnomaly;
            this.epoch = epoch;
            this.mass = mass;
            this.dragArea = dragArea;
            this.reflectivity = reflectivity;
            this.dragCoefficient = dragCoefficient;
        }
    }
}
