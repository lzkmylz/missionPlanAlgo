package orekit.visibility;

import orekit.visibility.model.BatchResult;
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;
import orekit.visibility.model.VisibilityWindow;

import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.models.earth.ReferenceEllipsoid;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.time.TimeScale;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * 优化版可见性计算器
 *
 * 使用轨道预计算缓存和并行计算，大幅提升大规模场景性能。
 *
 * 优化策略：
 * 1. 预计算所有卫星轨道状态并缓存（HPOP高精度模型）
 * 2. 可见性计算只使用缓存的几何数据，不做轨道传播
 * 3. 卫星-目标对并行计算（Java Parallel Stream）
 * 4. 使用ConcurrentHashMap确保线程安全
 *
 * 性能提升：相比串行计算，预期提升4-8倍（取决于CPU核心数）
 */
public class OptimizedVisibilityCalculator {

    private final OrbitStateCache orbitCache;
    private final Frame itrfFrame;
    private final Frame eme2000Frame;
    private final OneAxisEllipsoid earth;
    private final TimeScale utc;

    // 地球半径
    private static final double EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS;
    private static final double EARTH_FLATTENING = Constants.WGS84_EARTH_FLATTENING;

    public OptimizedVisibilityCalculator() {
        this.orbitCache = new OrbitStateCache();

        try {
            this.utc = TimeScalesFactory.getUTC();
            this.eme2000Frame = FramesFactory.getEME2000();
        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize time scales", e);
        }

        // 尝试获取ITRF框架，如果IERS数据不可用则回退到简化模型
        Frame tempItrf;
        OneAxisEllipsoid tempEarth;
        try {
            tempItrf = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
            tempEarth = ReferenceEllipsoid.getIers2010(tempItrf);
        } catch (Exception e) {
            // IERS数据不可用，使用简化模型
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
     * 批量计算所有卫星-目标对的可见窗口（主入口）
     *
     * @param satellites 卫星配置列表
     * @param targets 目标配置列表
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @param coarseStep 粗扫描步长（秒）
     * @param fineStep 精化步长（秒）
     * @return 批量计算结果
     */
    public BatchResult computeAllVisibilityWindows(
            List<SatelliteConfig> satellites,
            List<TargetConfig> targets,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double coarseStep,
            double fineStep) throws Exception {
        // 调用新版本，地面站列表为空
        return computeAllVisibilityWindows(satellites, targets, null, startTime, endTime, coarseStep, fineStep);
    }

    /**
     * 批量计算所有卫星-目标对和卫星-地面站对的可见窗口（完整版）
     *
     * @param satellites 卫星配置列表
     * @param targets 目标配置列表
     * @param groundStations 地面站配置列表（可为null）
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @param coarseStep 粗扫描步长（秒）
     * @param fineStep 精化步长（秒）
     * @return 批量计算结果
     */
    public BatchResult computeAllVisibilityWindows(
            List<SatelliteConfig> satellites,
            List<TargetConfig> targets,
            List<orekit.visibility.model.GroundStationConfig> groundStations,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double coarseStep,
            double fineStep) throws Exception {

        long startNs = System.nanoTime();

        // Phase 1: 预计算所有卫星轨道（并行）
        // 使用精扫描步长（fineStep）而非粗扫描步长，确保轨道数据精度与可见性计算一致
        orbitCache.precomputeAllOrbits(satellites, startTime, endTime, fineStep);

        // Phase 2: 批量计算所有卫星-目标对的可见性（使用缓存，无需传播）
        // 使用并行流进行卫星-目标对并行计算
        Map<String, List<VisibilityWindow>> targetWindows = new ConcurrentHashMap<>();
        Map<String, List<VisibilityWindow>> gsWindows = new ConcurrentHashMap<>();

        // 创建所有卫星-目标对
        List<SatTargetPair> pairs = new ArrayList<>();
        for (SatelliteConfig sat : satellites) {
            for (TargetConfig target : targets) {
                pairs.add(new SatTargetPair(sat, target));
            }
        }

        // 并行计算所有对的可见窗口
        int totalWindows = pairs.parallelStream()
            .mapToInt(pair -> {
                List<VisibilityWindow> windows = computeWindowsUsingCache(
                    pair.sat, pair.target, startTime, endTime, coarseStep, fineStep
                );

                if (!windows.isEmpty()) {
                    String key = pair.sat.getId() + "_" + pair.target.getId();
                    targetWindows.computeIfAbsent(key, k -> new ArrayList<>()).addAll(windows);
                }

                return windows.size();
            })
            .sum();

        // Phase 3: 计算卫星-地面站可见窗口（如果提供了地面站列表）
        int gsWindowCount = 0;
        if (groundStations != null && !groundStations.isEmpty()) {
            gsWindowCount = computeGroundStationWindows(
                satellites, groundStations, startTime, endTime, coarseStep, fineStep, gsWindows
            );
            totalWindows += gsWindowCount;
        }

        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNs);

        // 构建统计信息
        Map<String, Object> stats = new HashMap<>();
        stats.put("computationTimeMs", elapsedMs);
        stats.put("satelliteCount", satellites.size());
        stats.put("targetCount", targets.size());
        stats.put("groundStationCount", groundStations != null ? groundStations.size() : 0);
        stats.put("totalWindows", totalWindows);
        stats.put("targetWindows", totalWindows - gsWindowCount);
        stats.put("gsWindows", gsWindowCount);
        stats.put("cacheMemoryMB", orbitCache.getMemoryUsage() / (1024 * 1024));

        return new BatchResult(targetWindows, gsWindows, stats);
    }

    /**
     * 计算卫星-地面站可见窗口
     */
    private int computeGroundStationWindows(
            List<SatelliteConfig> satellites,
            List<orekit.visibility.model.GroundStationConfig> groundStations,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double coarseStep,
            double fineStep,
            Map<String, List<VisibilityWindow>> gsWindows) {

        // 创建所有卫星-地面站对
        List<SatGroundStationPair> pairs = new ArrayList<>();
        for (SatelliteConfig sat : satellites) {
            for (orekit.visibility.model.GroundStationConfig gs : groundStations) {
                pairs.add(new SatGroundStationPair(sat, gs));
            }
        }

        // 并行计算所有卫星-地面站对的可见窗口
        return pairs.parallelStream()
            .mapToInt(pair -> {
                List<VisibilityWindow> windows = computeGsWindowsUsingCache(
                    pair.sat, pair.gs, startTime, endTime, coarseStep, fineStep
                );

                if (!windows.isEmpty()) {
                    // 使用GS:前缀标识地面站目标
                    String key = pair.sat.getId() + "_GS:" + pair.gs.getId();
                    gsWindows.computeIfAbsent(key, k -> new ArrayList<>()).addAll(windows);
                }

                return windows.size();
            })
            .sum();
    }

    /**
     * 使用缓存的轨道状态计算单个卫星-目标对的可见窗口
     * 关键优化：不调用propagator.propagate()，只做几何计算
     */
    private List<VisibilityWindow> computeWindowsUsingCache(
            SatelliteConfig sat,
            TargetConfig target,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double coarseStep,
            double fineStep) {

        List<VisibilityWindow> windows = new ArrayList<>();

        // 目标点
        GeodeticPoint targetPoint = new GeodeticPoint(
            Math.toRadians(target.getLatitude()),
            Math.toRadians(target.getLongitude()),
            target.getAltitude()
        );

        // 目标点笛卡尔坐标（缓存，避免重复计算）
        double[] targetCart = geodeticToCartesian(targetPoint);

        // 最小仰角
        double minElevationRad = Math.toRadians(
            Math.max(5.0, sat.getMinElevation())
        );

        // 计算总时长
        double duration = endTime.durationFrom(startTime);

        // 粗扫描：使用缓存的轨道状态
        AbsoluteDate windowStart = null;
        boolean wasVisible = false;
        double maxElevationInWindow = 0.0;
        AbsoluteDate maxElevationTime = null;

        for (double t = 0; t <= duration; t += coarseStep) {
            OrbitStateCache.OrbitState state = orbitCache.getStateAtTime(sat.getId(), t);
            if (state == null) continue;

            // 从缓存状态计算可见性（纯几何计算）
            double elevation = calculateElevationFromState(state, targetCart);
            boolean isVisible = elevation >= minElevationRad;

            if (isVisible != wasVisible) {
                if (wasVisible && windowStart != null) {
                    // 窗口结束，精化边界
                    AbsoluteDate windowEnd = startTime.shiftedBy(t);
                    VisibilityWindow window = refineWindow(
                        sat, target, windowStart, windowEnd,
                        Math.toDegrees(maxElevationInWindow), maxElevationTime,
                        targetPoint, minElevationRad, fineStep, startTime, endTime
                    );
                    if (window != null) {
                        windows.add(window);
                    }
                } else {
                    // 窗口开始
                    windowStart = startTime.shiftedBy(t);
                    maxElevationInWindow = 0.0;
                    maxElevationTime = null;
                }
                wasVisible = isVisible;
            }

            if (isVisible) {
                if (elevation > maxElevationInWindow) {
                    maxElevationInWindow = elevation;
                    maxElevationTime = startTime.shiftedBy(t);
                }
            }
        }

        // 处理最后一个窗口
        if (wasVisible && windowStart != null) {
            VisibilityWindow window = refineWindow(
                sat, target, windowStart, endTime,
                Math.toDegrees(maxElevationInWindow), maxElevationTime,
                targetPoint, minElevationRad, fineStep, startTime, endTime
            );
            if (window != null) {
                windows.add(window);
            }
        }

        return windows;
    }

    /**
     * 精化窗口边界
     */
    private VisibilityWindow refineWindow(
            SatelliteConfig sat,
            TargetConfig target,
            AbsoluteDate coarseStart,
            AbsoluteDate coarseEnd,
            double coarseMaxElevation,
            AbsoluteDate coarseMaxElevationTime,
            GeodeticPoint targetPoint,
            double minElevationRad,
            double fineStep,
            AbsoluteDate missionStart,
            AbsoluteDate missionEnd) {

        double[] targetCart = geodeticToCartesian(targetPoint);

        // 向前搜索精确开始时间
        AbsoluteDate preciseStart = coarseStart;
        while (preciseStart.compareTo(missionStart) > 0) {
            AbsoluteDate prevTime = preciseStart.shiftedBy(-fineStep);
            if (prevTime.compareTo(missionStart) < 0) break;

            double t = prevTime.durationFrom(missionStart);
            OrbitStateCache.OrbitState state = orbitCache.getStateAtTime(sat.getId(), t);
            if (state == null) break;

            double elevation = calculateElevationFromState(state, targetCart);
            if (elevation < minElevationRad) break;

            preciseStart = prevTime;
        }

        // 向后搜索精确结束时间
        AbsoluteDate preciseEnd = coarseEnd;
        while (preciseEnd.compareTo(missionEnd) < 0) {
            AbsoluteDate nextTime = preciseEnd.shiftedBy(fineStep);
            if (nextTime.compareTo(missionEnd) > 0) break;

            double t = nextTime.durationFrom(missionStart);
            OrbitStateCache.OrbitState state = orbitCache.getStateAtTime(sat.getId(), t);
            if (state == null) break;

            double elevation = calculateElevationFromState(state, targetCart);
            if (elevation < minElevationRad) break;

            preciseEnd = nextTime;
        }

        // 重新计算窗口中点的最大仰角
        double midT = preciseStart.durationFrom(missionStart) +
                      preciseEnd.durationFrom(preciseStart) / 2.0;
        OrbitStateCache.OrbitState midState = orbitCache.getStateAtTime(sat.getId(), midT);

        double maxElevation = coarseMaxElevation;
        AbsoluteDate maxElTime = coarseMaxElevationTime;

        if (midState != null) {
            double midEl = Math.toDegrees(calculateElevationFromState(midState, targetCart));
            if (midEl > maxElevation) {
                maxElevation = midEl;
                maxElTime = missionStart.shiftedBy(midT);
            }
        }

        // 计算持续时间
        double duration = preciseEnd.durationFrom(preciseStart);

        // 检查最小持续时间
        if (duration < target.getMinObservationDuration()) {
            return null;
        }

        // 计算质量分数
        double qualityScore = Math.min(1.0, maxElevation / 90.0);

        return new VisibilityWindow(
            sat.getId(),
            target.getId(),
            preciseStart,
            preciseEnd,
            duration,
            maxElevation,
            maxElTime != null ? maxElTime : preciseStart.shiftedBy(duration / 2),
            0.0,  // entryAzimuth - 简化处理
            0.0,  // exitAzimuth - 简化处理
            qualityScore,
            maxElevation > 30.0 ? "HIGH" : (maxElevation > 15.0 ? "MEDIUM" : "LOW")
        );
    }

    /**
     * 从缓存状态计算仰角（纯几何计算）
     */
    private double calculateElevationFromState(
            OrbitStateCache.OrbitState state,
            double[] targetCart) {

        // 从目标指向卫星的向量
        double dx = state.x - targetCart[0];
        double dy = state.y - targetCart[1];
        double dz = state.z - targetCart[2];

        // 目标点的本地垂直方向（从地心指向目标的单位向量）
        double r = Math.sqrt(targetCart[0]*targetCart[0] +
                            targetCart[1]*targetCart[1] +
                            targetCart[2]*targetCart[2]);
        double ux = targetCart[0] / r;
        double uy = targetCart[1] / r;
        double uz = targetCart[2] / r;

        // 计算仰角
        double range = Math.sqrt(dx*dx + dy*dy + dz*dz);
        double cosZenith = (dx*ux + dy*uy + dz*uz) / range;

        // 限制范围避免数值误差
        cosZenith = Math.max(-1.0, Math.min(1.0, cosZenith));

        return Math.PI / 2 - Math.acos(cosZenith);
    }

    /**
     * 地理坐标转笛卡尔坐标
     */
    private double[] geodeticToCartesian(GeodeticPoint point) {
        try {
            org.hipparchus.geometry.euclidean.threed.Vector3D vector =
                earth.transform(point);
            return new double[]{vector.getX(), vector.getY(), vector.getZ()};
        } catch (Exception e) {
            // 简化计算
            double lat = point.getLatitude();
            double lon = point.getLongitude();
            double alt = point.getAltitude();

            double r = EARTH_RADIUS + alt;
            double x = r * Math.cos(lat) * Math.cos(lon);
            double y = r * Math.cos(lat) * Math.sin(lon);
            double z = r * Math.sin(lat);

            return new double[]{x, y, z};
        }
    }

    /**
     * 获取轨道缓存
     */
    public OrbitStateCache getOrbitCache() {
        return orbitCache;
    }

    /**
     * 使用缓存的轨道状态计算单个卫星-地面站对的可见窗口
     */
    private List<VisibilityWindow> computeGsWindowsUsingCache(
            SatelliteConfig sat,
            orekit.visibility.model.GroundStationConfig gs,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double coarseStep,
            double fineStep) {

        List<VisibilityWindow> windows = new ArrayList<>();

        // 地面站点
        GeodeticPoint gsPoint = new GeodeticPoint(
            Math.toRadians(gs.getLatitude()),
            Math.toRadians(gs.getLongitude()),
            gs.getAltitude()
        );

        // 地面站笛卡尔坐标（缓存，避免重复计算）
        double[] gsCart = geodeticToCartesian(gsPoint);

        // 最小仰角（地面站通常有更高的最小仰角要求）
        double minElevationRad = Math.toRadians(
            Math.max(5.0, gs.getMinElevation())
        );

        // 最大通信距离（如果配置）
        double maxRange = gs.getMaxRange() > 0 ? gs.getMaxRange() : Double.MAX_VALUE;

        // 计算总时长
        double duration = endTime.durationFrom(startTime);

        // 粗扫描：使用缓存的轨道状态
        AbsoluteDate windowStart = null;
        boolean wasVisible = false;
        double maxElevationInWindow = 0.0;
        AbsoluteDate maxElevationTime = null;

        for (double t = 0; t <= duration; t += coarseStep) {
            OrbitStateCache.OrbitState state = orbitCache.getStateAtTime(sat.getId(), t);
            if (state == null) continue;

            // 检查距离约束
            double dist = calculateDistance(state, gsCart);
            if (dist > maxRange) {
                if (wasVisible && windowStart != null) {
                    // 窗口结束
                    AbsoluteDate windowEnd = startTime.shiftedBy(t);
                    VisibilityWindow window = createGsVisibilityWindow(
                        sat, gs, windowStart, windowEnd,
                        Math.toDegrees(maxElevationInWindow), maxElevationTime,
                        startTime, endTime
                    );
                    if (window != null) {
                        windows.add(window);
                    }
                    windowStart = null;
                    maxElevationInWindow = 0.0;
                    maxElevationTime = null;
                }
                wasVisible = false;
                continue;
            }

            // 从缓存状态计算可见性（纯几何计算）
            double elevation = calculateElevationFromState(state, gsCart);
            boolean isVisible = elevation >= minElevationRad;

            if (isVisible != wasVisible) {
                if (wasVisible && windowStart != null) {
                    // 窗口结束，精化边界
                    AbsoluteDate windowEnd = startTime.shiftedBy(t);
                    VisibilityWindow window = createGsVisibilityWindow(
                        sat, gs, windowStart, windowEnd,
                        Math.toDegrees(maxElevationInWindow), maxElevationTime,
                        startTime, endTime
                    );
                    if (window != null) {
                        windows.add(window);
                    }
                } else {
                    // 窗口开始
                    windowStart = startTime.shiftedBy(t);
                    maxElevationInWindow = 0.0;
                    maxElevationTime = null;
                }
                wasVisible = isVisible;
            }

            if (isVisible) {
                if (elevation > maxElevationInWindow) {
                    maxElevationInWindow = elevation;
                    maxElevationTime = startTime.shiftedBy(t);
                }
            }
        }

        // 处理最后一个窗口
        if (wasVisible && windowStart != null) {
            VisibilityWindow window = createGsVisibilityWindow(
                sat, gs, windowStart, endTime,
                Math.toDegrees(maxElevationInWindow), maxElevationTime,
                startTime, endTime
            );
            if (window != null) {
                windows.add(window);
            }
        }

        return windows;
    }

    /**
     * 计算两点间距离
     */
    private double calculateDistance(OrbitStateCache.OrbitState state, double[] pointCart) {
        double dx = state.x - pointCart[0];
        double dy = state.y - pointCart[1];
        double dz = state.z - pointCart[2];
        return Math.sqrt(dx*dx + dy*dy + dz*dz);
    }

    /**
     * 创建地面站可见窗口对象
     */
    private VisibilityWindow createGsVisibilityWindow(
            SatelliteConfig sat,
            orekit.visibility.model.GroundStationConfig gs,
            AbsoluteDate start,
            AbsoluteDate end,
            double maxElevation,
            AbsoluteDate maxElTime,
            AbsoluteDate globalStart,
            AbsoluteDate globalEnd) {

        // 确保窗口在全局时间范围内
        AbsoluteDate preciseStart = start.compareTo(globalStart) < 0 ? globalStart : start;
        AbsoluteDate preciseEnd = end.compareTo(globalEnd) > 0 ? globalEnd : end;

        // 计算持续时间
        double duration = preciseEnd.durationFrom(preciseStart);

        // 地面站窗口最小持续时间（通常较短，如30秒）
        double minDuration = 30.0;
        if (duration < minDuration) {
            return null;
        }

        // 计算质量分数
        double qualityScore = Math.min(1.0, maxElevation / 90.0);

        return new VisibilityWindow(
            sat.getId(),
            "GS:" + gs.getId(),  // 使用GS:前缀标识地面站目标
            preciseStart,
            preciseEnd,
            duration,
            maxElevation,
            maxElTime != null ? maxElTime : preciseStart.shiftedBy(duration / 2),
            0.0,  // entryAzimuth
            0.0,  // exitAzimuth
            qualityScore,
            maxElevation > 30.0 ? "HIGH" : (maxElevation > 15.0 ? "MEDIUM" : "LOW")
        );
    }

    /**
     * 卫星-地面站对辅助类
     */
    private static class SatGroundStationPair {
        final SatelliteConfig sat;
        final orekit.visibility.model.GroundStationConfig gs;

        SatGroundStationPair(SatelliteConfig sat, orekit.visibility.model.GroundStationConfig gs) {
            this.sat = sat;
            this.gs = gs;
        }
    }

    /**
     * 卫星-目标对辅助类
     *
     * 用于并行流处理中传递卫星和目标配置
     */
    private static class SatTargetPair {
        final SatelliteConfig sat;
        final TargetConfig target;

        SatTargetPair(SatelliteConfig sat, TargetConfig target) {
            this.sat = sat;
            this.target = target;
        }
    }
}
