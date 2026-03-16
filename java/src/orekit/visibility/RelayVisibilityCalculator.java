package orekit.visibility;

import orekit.visibility.model.RelaySatelliteConfig;
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.VisibilityWindow;

import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.Transform;
import org.orekit.models.earth.ReferenceEllipsoid;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.PositionAngle;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.time.TimeScale;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;

import org.hipparchus.geometry.euclidean.threed.Vector3D;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 中继卫星可见性计算器
 *
 * 使用Orekit计算低轨卫星(LEO)与中继卫星(GEO)之间的可见性窗口。
 * GEO中继卫星位置固定（定点经度），使用简化的轨道模型。
 *
 * 计算原理：
 * 1. GEO卫星位置由其定点经度确定（地球静止轨道）
 * 2. 计算LEO卫星与GEO卫星之间的几何距离和仰角
 * 3. 当LEO卫星相对于GEO卫星的仰角超过阈值且距离在范围内时可见
 */
public class RelayVisibilityCalculator {

    private final OrbitStateCache orbitCache;
    private final Frame itrfFrame;
    private final Frame eme2000Frame;
    private final OneAxisEllipsoid earth;
    private final TimeScale utc;

    // 地球半径和GEO轨道半径
    private static final double EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS;
    private static final double GEO_ALTITUDE = 35786000.0; // GEO高度（米）
    private static final double GEO_RADIUS = EARTH_RADIUS + GEO_ALTITUDE;

    // SLF4J日志
    private static final Logger logger = LoggerFactory.getLogger(RelayVisibilityCalculator.class);

    public RelayVisibilityCalculator() {
        this(new OrbitStateCache());
    }

    public RelayVisibilityCalculator(OrbitStateCache orbitCache) {
        this.orbitCache = orbitCache;

        try {
            this.utc = TimeScalesFactory.getUTC();
            this.eme2000Frame = FramesFactory.getEME2000();
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize time scales", e);
        }

        // 尝试获取ITRF框架
        Frame tempItrf;
        OneAxisEllipsoid tempEarth;
        try {
            tempItrf = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
            tempEarth = ReferenceEllipsoid.getIers2010(tempItrf);
        } catch (Exception e) {
            System.err.println("Warning: IERS data not available, using simplified Earth model");
            try {
                tempItrf = FramesFactory.getGCRF();
                tempEarth = new OneAxisEllipsoid(
                    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                    Constants.WGS84_EARTH_FLATTENING,
                    tempItrf
                );
            } catch (Exception ex) {
                throw new RuntimeException("Failed to initialize frames", ex);
            }
        }
        this.itrfFrame = tempItrf;
        this.earth = tempEarth;
    }

    /**
     * 批量计算所有卫星-中继卫星对的可见窗口
     *
     * @param satellites 卫星配置列表（LEO）
     * @param relays 中继卫星配置列表（GEO）
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @param timeStep 时间步长（秒）
     * @return Map<satId_relayId, List<VisibilityWindow>>
     */
    public Map<String, List<VisibilityWindow>> computeRelayVisibilityWindows(
            List<SatelliteConfig> satellites,
            List<RelaySatelliteConfig> relays,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double timeStep) {

        long startNs = System.nanoTime();

        // 检查空列表
        if (satellites == null || satellites.isEmpty()) {
            throw new IllegalArgumentException("Satellite list is empty");
        }
        if (relays == null || relays.isEmpty()) {
            return new ConcurrentHashMap<>();
        }

        // 确保轨道缓存已预计算
        if (!orbitCache.hasSatellite(satellites.get(0).getId())) {
            throw new IllegalStateException(
                "Orbit cache not precomputed. Call orbitCache.precomputeAllOrbits() first."
            );
        }

        // 预计算所有GEO中继卫星的位置（固定）
        Map<String, Vector3D> relayPositions = new ConcurrentHashMap<>();
        for (RelaySatelliteConfig relay : relays) {
            Vector3D position = calculateGeoPosition(relay.getLongitude());
            relayPositions.put(relay.getId(), position);
        }

        // 创建所有卫星-中继对
        List<SatRelayPair> pairs = new ArrayList<>();
        for (SatelliteConfig sat : satellites) {
            for (RelaySatelliteConfig relay : relays) {
                pairs.add(new SatRelayPair(sat, relay));
            }
        }

        // 并行计算所有对的可见窗口
        Map<String, List<VisibilityWindow>> relayWindows = new ConcurrentHashMap<>();

        int totalWindows = pairs.parallelStream()
            .mapToInt(pair -> {
                List<VisibilityWindow> windows = computeRelayWindowsUsingCache(
                    pair.sat, pair.relay, relayPositions.get(pair.relay.getId()),
                    startTime, endTime, timeStep
                );

                if (!windows.isEmpty()) {
                    // 使用RELAY:前缀标识中继卫星目标
                    String key = pair.sat.getId() + "_RELAY:" + pair.relay.getId();
                    relayWindows.computeIfAbsent(key, k -> new ArrayList<>()).addAll(windows);
                }

                return windows.size();
            })
            .sum();

        long elapsedMs = (System.nanoTime() - startNs) / 1_000_000;
        logger.info("Computed {} relay visibility windows in {} ms", totalWindows, elapsedMs);

        return relayWindows;
    }

    /**
     * 计算GEO卫星在ITRF框架中的位置
     *
     * @param longitudeDeg 定点经度（度）
     * @return GEO卫星位置向量（米，ITRF框架）
     */
    private Vector3D calculateGeoPosition(double longitudeDeg) {
        double longitudeRad = Math.toRadians(longitudeDeg);

        // GEO卫星位置：在赤道平面上，固定经度
        double x = GEO_RADIUS * Math.cos(longitudeRad);
        double y = GEO_RADIUS * Math.sin(longitudeRad);
        double z = 0.0; // 在赤道平面上

        return new Vector3D(x, y, z);
    }

    /**
     * 使用缓存的轨道状态计算单个卫星-中继对的可见窗口
     */
    private List<VisibilityWindow> computeRelayWindowsUsingCache(
            SatelliteConfig sat,
            RelaySatelliteConfig relay,
            Vector3D relayPosition,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double timeStep) {

        List<VisibilityWindow> windows = new ArrayList<>();

        // 中继卫星参数
        double minElevationRad = Math.toRadians(relay.getMinElevation());
        double maxRange = relay.getMaxRange();

        // 计算总时长
        double duration = endTime.durationFrom(startTime);

        // 扫描轨道缓存，检查每个时间点的可见性
        AbsoluteDate windowStart = null;
        AbsoluteDate maxElevationTime = null;
        double maxElevationInWindow = 0.0;
        boolean wasVisible = false;

        for (double t = 0; t <= duration; t += timeStep) {
            OrbitStateCache.OrbitState state = orbitCache.getStateAtTime(sat.getId(), t);
            if (state == null) continue;

            // 计算LEO卫星与GEO中继卫星之间的几何关系
            VisibilityResult visibility = calculateVisibility(
                state, relayPosition, minElevationRad, maxRange
            );

            if (visibility.isVisible != wasVisible) {
                if (wasVisible && windowStart != null) {
                    // 窗口结束，创建窗口对象
                    AbsoluteDate windowEnd = startTime.shiftedBy(t);
                    VisibilityWindow window = new VisibilityWindow(
                        windowStart,
                        windowEnd,
                        Math.toDegrees(maxElevationInWindow),
                        maxElevationTime != null ? maxElevationTime : windowStart
                    );
                    windows.add(window);
                } else {
                    // 窗口开始
                    windowStart = startTime.shiftedBy(t);
                    maxElevationInWindow = 0.0;
                    maxElevationTime = null;
                }
                wasVisible = visibility.isVisible;
            }

            // 跟踪窗口内的最大仰角
            if (visibility.isVisible && visibility.elevation > maxElevationInWindow) {
                maxElevationInWindow = visibility.elevation;
                maxElevationTime = startTime.shiftedBy(t);
            }
        }

        // 处理最后一个窗口
        if (wasVisible && windowStart != null) {
            AbsoluteDate windowEnd = endTime;
            VisibilityWindow window = new VisibilityWindow(
                windowStart,
                windowEnd,
                Math.toDegrees(maxElevationInWindow),
                maxElevationTime != null ? maxElevationTime : windowStart
            );
            windows.add(window);
        }

        return windows;
    }

    /**
     * 计算LEO卫星与GEO中继卫星的可见性
     *
     * @param leoState LEO卫星状态
     * @param relayPosition GEO中继卫星位置
     * @param minElevationRad 最小仰角（弧度）
     * @param maxRange 最大通信距离（米）
     * @return 可见性结果
     */
    private VisibilityResult calculateVisibility(
            OrbitStateCache.OrbitState leoState,
            Vector3D relayPosition,
            double minElevationRad,
            double maxRange) {

        // LEO卫星位置
        Vector3D leoPosition = new Vector3D(leoState.px, leoState.py, leoState.pz);

        // 计算距离
        double distance = leoPosition.distance(relayPosition);
        if (distance > maxRange) {
            return new VisibilityResult(false, 0.0);
        }

        // 检查地球遮挡（视线是否被地球遮挡）
        if (isEarthBlocking(leoPosition, relayPosition)) {
            return new VisibilityResult(false, 0.0);
        }

        // 计算仰角（从GEO看向LEO的仰角）
        double elevation = calculateElevation(relayPosition, leoPosition);

        boolean isVisible = elevation >= minElevationRad;
        return new VisibilityResult(isVisible, elevation);
    }

    /**
     * 检查视线是否被地球遮挡
     */
    private boolean isEarthBlocking(Vector3D from, Vector3D to) {
        // 计算视线与地球的最小距离
        Vector3D direction = to.subtract(from);
        double t = -from.dotProduct(direction) / direction.getNormSq();

        // 如果t不在[0,1]范围内，视线在延长线上
        if (t < 0 || t > 1) {
            return false;
        }

        // 计算最近点
        Vector3D closestPoint = from.add(direction.multiply(t));
        double closestDistance = closestPoint.getNorm();

        // 如果最近点距离小于地球半径，则被遮挡
        return closestDistance < EARTH_RADIUS * 0.99; // 0.99容差
    }

    /**
     * 计算从GEO位置看向LEO位置的仰角
     *
     * @param geoPosition GEO位置
     * @param leoPosition LEO位置
     * @return 仰角（弧度）
     */
    private double calculateElevation(Vector3D geoPosition, Vector3D leoPosition) {
        // 从GEO指向LEO的向量
        Vector3D toLeo = leoPosition.subtract(geoPosition);

        // GEO位置的径向向外向量（从地心指向GEO）
        Vector3D geoRadial = geoPosition.normalize();

        // 计算仰角：90° - 与径向向量的夹角
        double cosAngle = geoRadial.dotProduct(toLeo.normalize());
        double angleFromRadial = Math.acos(Math.max(-1.0, Math.min(1.0, cosAngle)));

        // 仰角 = 90° - 与径向的夹角
        return Math.PI / 2.0 - angleFromRadial;
    }

    /**
     * 内部类：卫星-中继对
     */
    private static class SatRelayPair {
        final SatelliteConfig sat;
        final RelaySatelliteConfig relay;

        SatRelayPair(SatelliteConfig sat, RelaySatelliteConfig relay) {
            this.sat = sat;
            this.relay = relay;
        }
    }

    /**
     * 内部类：可见性结果
     */
    private static class VisibilityResult {
        final boolean isVisible;
        final double elevation;

        VisibilityResult(boolean isVisible, double elevation) {
            this.isVisible = isVisible;
            this.elevation = elevation;
        }
    }
}
