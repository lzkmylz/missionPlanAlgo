package orekit.visibility;

import orekit.visibility.model.BatchResult;
import orekit.visibility.model.RelaySatelliteConfig;
import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;
import orekit.visibility.model.VisibilityWindow;
import orekit.visibility.AttitudeCalculator;

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
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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

    // SLF4J日志
    private static final Logger logger = LoggerFactory.getLogger(OptimizedVisibilityCalculator.class);

    // 线程安全的调试计数器
    private static final AtomicInteger debugCount = new AtomicInteger(0);
    private static final int MAX_DEBUG_MESSAGES = 10;

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
                    List<VisibilityWindow> refinedWindows = refineWindow(
                        sat, target, windowStart, windowEnd,
                        Math.toDegrees(maxElevationInWindow), maxElevationTime,
                        targetPoint, minElevationRad, fineStep, startTime, endTime
                    );
                    if (refinedWindows != null) {
                        windows.addAll(refinedWindows);
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
            List<VisibilityWindow> refinedWindows = refineWindow(
                sat, target, windowStart, endTime,
                Math.toDegrees(maxElevationInWindow), maxElevationTime,
                targetPoint, minElevationRad, fineStep, startTime, endTime
            );
            if (refinedWindows != null) {
                windows.addAll(refinedWindows);
            }
        }

        return windows;
    }

    /**
     * 精化窗口边界，并基于姿态约束提取满足条件的子窗口
     * 返回满足姿态约束的所有子窗口列表
     */
    private List<VisibilityWindow> refineWindow(
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

        // Phase 4: 姿态计算和过滤
        // 计算窗口内所有采样点的姿态（1秒步长）
        List<VisibilityWindow.AttitudeSample> attitudeSamples = computeAttitudeSamples(
            sat, target, preciseStart, preciseEnd, missionStart
        );

        // 找出所有姿态约束满足的连续子时段
        List<AttitudeFeasibleInterval> feasibleIntervals = findAttitudeFeasibleIntervals(
            attitudeSamples, sat.getMaxRollAngle(), sat.getMaxPitchAngle(),
            sat.getId(), target.getId(), target.getMinObservationDuration()
        );

        // 如果没有满足约束的子时段，过滤掉此窗口
        if (feasibleIntervals.isEmpty()) {
            return Collections.emptyList();
        }

        // 为每个满足约束的子时段创建一个窗口
        List<VisibilityWindow> resultWindows = new ArrayList<>();

        for (AttitudeFeasibleInterval interval : feasibleIntervals) {
            // 计算子窗口的绝对时间
            AbsoluteDate subStart = preciseStart.shiftedBy(interval.startTime);
            AbsoluteDate subEnd = preciseStart.shiftedBy(interval.endTime);
            double subDuration = interval.endTime - interval.startTime;

            // 计算子窗口内的最大仰角
            double subMaxElevation = maxElevation;
            AbsoluteDate subMaxElTime = maxElTime;

            // 重新计算子窗口中点的仰角
            double subMidT = interval.startTime + (interval.endTime - interval.startTime) / 2.0;
            OrbitStateCache.OrbitState subMidState = orbitCache.getStateAtTime(
                sat.getId(), preciseStart.durationFrom(missionStart) + subMidT);

            if (subMidState != null) {
                double subMidEl = Math.toDegrees(calculateElevationFromState(subMidState, targetCart));
                if (subMidEl > subMaxElevation) {
                    subMaxElevation = subMidEl;
                    subMaxElTime = preciseStart.shiftedBy(subMidT);
                }
            }

            // 计算质量分数
            double qualityScore = Math.min(1.0, subMaxElevation / 90.0);

            // 创建子窗口
            resultWindows.add(new VisibilityWindow(
                sat.getId(),
                target.getId(),
                subStart,
                subEnd,
                subDuration,
                subMaxElevation,
                subMaxElTime != null ? subMaxElTime : subStart.shiftedBy(subDuration / 2),
                0.0,  // entryAzimuth
                0.0,  // exitAzimuth
                qualityScore,
                subMaxElevation > 30.0 ? "HIGH" : (subMaxElevation > 15.0 ? "MEDIUM" : "LOW"),
                interval.samples,
                true  // 姿态可行
            ));
        }

        return resultWindows;
    }

    /**
     * 计算窗口内所有采样点的姿态（1秒步长）
     */
    private List<VisibilityWindow.AttitudeSample> computeAttitudeSamples(
            SatelliteConfig sat,
            TargetConfig target,
            AbsoluteDate windowStart,
            AbsoluteDate windowEnd,
            AbsoluteDate missionStart) {

        List<VisibilityWindow.AttitudeSample> samples = new ArrayList<>();
        double duration = windowEnd.durationFrom(windowStart);

        // 1秒步长采样
        for (double t = 0; t <= duration; t += 1.0) {
            AbsoluteDate currentTime = windowStart.shiftedBy(t);
            double relativeTime = currentTime.durationFrom(missionStart);

            OrbitStateCache.OrbitState state = orbitCache.getStateAtTime(sat.getId(), relativeTime);
            if (state == null) continue;

            // 计算姿态
            AttitudeCalculator.AttitudeAngles attitude = AttitudeCalculator.calculateAttitude(
                new double[]{state.x, state.y, state.z},
                new double[]{state.vx, state.vy, state.vz},
                target.getLatitude(),
                target.getLongitude(),
                target.getAltitude()
            );

            samples.add(new VisibilityWindow.AttitudeSample(t, attitude.roll, attitude.pitch));
        }

        return samples;
    }

    /**
     * 姿态约束时段类 - 记录满足约束的子时段
     */
    public static class AttitudeFeasibleInterval {
        public final double startTime;  // 相对于窗口开始的秒数
        public final double endTime;
        public final List<VisibilityWindow.AttitudeSample> samples;

        public AttitudeFeasibleInterval(double start, double end, List<VisibilityWindow.AttitudeSample> samples) {
            this.startTime = start;
            this.endTime = end;
            this.samples = samples;
        }
    }

    /**
     * 从姿态采样中找出所有满足约束的连续子时段
     * @param samples 完整的姿态采样列表
     * @param maxRoll 最大滚转角
     * @param maxPitch 最大俯仰角
     * @param satId 卫星ID（用于调试）
     * @param targetId 目标ID（用于调试）
     * @param minDuration 最小持续时间（秒）
     * @return 满足约束的子时段列表
     */
    private List<AttitudeFeasibleInterval> findAttitudeFeasibleIntervals(
            List<VisibilityWindow.AttitudeSample> samples,
            double maxRoll,
            double maxPitch,
            String satId,
            String targetId,
            double minDuration) {

        List<AttitudeFeasibleInterval> intervals = new ArrayList<>();

        if (samples.isEmpty()) {
            return intervals;
        }

        // 找到所有满足约束的连续时段
        boolean inFeasibleInterval = false;
        double intervalStart = 0;
        List<VisibilityWindow.AttitudeSample> currentSamples = new ArrayList<>();

        for (VisibilityWindow.AttitudeSample sample : samples) {
            double roll = sample.roll;
            double pitch = sample.pitch;
            boolean feasible = Math.abs(roll) <= maxRoll && Math.abs(pitch) <= maxPitch;

            if (feasible) {
                if (!inFeasibleInterval) {
                    // 开始新的可行区间
                    inFeasibleInterval = true;
                    intervalStart = sample.timestamp;
                    currentSamples = new ArrayList<>();
                }
                currentSamples.add(sample);
            } else {
                if (inFeasibleInterval) {
                    // 结束当前可行区间
                    inFeasibleInterval = false;
                    double intervalEnd = sample.timestamp; // 使用当前时间点作为结束

                    // 检查最小持续时间
                    if (intervalEnd - intervalStart >= minDuration) {
                        intervals.add(new AttitudeFeasibleInterval(
                            intervalStart, intervalEnd, new ArrayList<>(currentSamples)));

                        if (debugCount.getAndIncrement() < MAX_DEBUG_MESSAGES) {
                            logger.debug("[INTERVAL] {} -> {}: found feasible interval {}s to {}s ({}s)",
                                satId, targetId,
                                String.format("%.0f", intervalStart),
                                String.format("%.0f", intervalEnd),
                                String.format("%.0f", intervalEnd - intervalStart));
                        }
                    }
                    currentSamples.clear();
                }

                // 打印被过滤的采样点（用于调试）
                if (debugCount.getAndIncrement() < MAX_DEBUG_MESSAGES) {
                    logger.debug("[FILTERED] {} -> {}: roll={}°, pitch={}° (limit: {}/{}) at t={}s",
                        satId, targetId,
                        String.format("%.1f", roll),
                        String.format("%.1f", pitch),
                        maxRoll, maxPitch,
                        String.format("%.0f", sample.timestamp));
                }
            }
        }

        // 处理最后一个区间
        if (inFeasibleInterval && !currentSamples.isEmpty()) {
            double intervalEnd = samples.get(samples.size() - 1).timestamp;
            if (intervalEnd - intervalStart >= minDuration) {
                intervals.add(new AttitudeFeasibleInterval(
                    intervalStart, intervalEnd, new ArrayList<>(currentSamples)));
            }
        }

        return intervals;
    }

    /**
     * 检查姿态约束（旧方法，保留用于兼容性）
     * 如果所有采样点都满足约束，返回true
     */
    private boolean checkAttitudeConstraints(
            List<VisibilityWindow.AttitudeSample> samples,
            double maxRoll,
            double maxPitch,
            String satId,
            String targetId) {

        // 现在使用新方法，只要有任何可行区间就返回true
        List<AttitudeFeasibleInterval> intervals = findAttitudeFeasibleIntervals(
            samples, maxRoll, maxPitch, satId, targetId, 1.0);
        return !intervals.isEmpty();
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

        // 确保轨道缓存已预计算
        if (!orbitCache.hasSatellite(satellites.get(0).getId())) {
            throw new IllegalStateException(
                "Orbit cache not precomputed. Call precomputeAllOrbits() first."
            );
        }

        // 预计算所有GEO中继卫星的位置（固定）
        Map<String, org.hipparchus.geometry.euclidean.threed.Vector3D> relayPositions =
            new java.util.concurrent.ConcurrentHashMap<>();
        for (RelaySatelliteConfig relay : relays) {
            org.hipparchus.geometry.euclidean.threed.Vector3D position = calculateGeoPosition(relay.getLongitude());
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

        pairs.parallelStream()
            .forEach(pair -> {
                List<VisibilityWindow> windows = computeSingleRelayWindows(
                    pair.sat, pair.relay, relayPositions.get(pair.relay.getId()),
                    startTime, endTime, timeStep
                );

                if (!windows.isEmpty()) {
                    // 使用RELAY:前缀标识中继卫星目标
                    String key = pair.sat.getId() + "_RELAY:" + pair.relay.getId();
                    relayWindows.computeIfAbsent(key, k -> new ArrayList<>()).addAll(windows);
                }
            });

        return relayWindows;
    }

    /**
     * 计算GEO卫星在ITRF框架中的位置
     */
    private org.hipparchus.geometry.euclidean.threed.Vector3D calculateGeoPosition(double longitudeDeg) {
        double longitudeRad = Math.toRadians(longitudeDeg);
        double geoRadius = EARTH_RADIUS + 35786000.0; // GEO高度

        double x = geoRadius * Math.cos(longitudeRad);
        double y = geoRadius * Math.sin(longitudeRad);
        double z = 0.0;

        return new org.hipparchus.geometry.euclidean.threed.Vector3D(x, y, z);
    }

    /**
     * 计算单个卫星-中继对的可见窗口
     */
    private List<VisibilityWindow> computeSingleRelayWindows(
            SatelliteConfig sat,
            RelaySatelliteConfig relay,
            org.hipparchus.geometry.euclidean.threed.Vector3D relayPosition,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double timeStep) {

        List<VisibilityWindow> windows = new ArrayList<>();

        double minElevationRad = Math.toRadians(relay.getMinElevation());
        double maxRange = relay.getMaxRange();
        double duration = endTime.durationFrom(startTime);

        AbsoluteDate windowStart = null;
        AbsoluteDate maxElevationTime = null;
        double maxElevationInWindow = 0.0;
        boolean wasVisible = false;

        for (double t = 0; t <= duration; t += timeStep) {
            OrbitStateCache.OrbitState state = orbitCache.getStateAtTime(sat.getId(), t);
            if (state == null) continue;

            VisibilityResult visibility = calculateRelayVisibility(
                state, relayPosition, minElevationRad, maxRange
            );

            if (visibility.isVisible != wasVisible) {
                if (wasVisible && windowStart != null) {
                    AbsoluteDate windowEnd = startTime.shiftedBy(t);
                    VisibilityWindow window = createRelayVisibilityWindow(
                        sat, relay, windowStart, windowEnd,
                        Math.toDegrees(maxElevationInWindow), maxElevationTime
                    );
                    if (window != null) {
                        windows.add(window);
                    }
                } else {
                    windowStart = startTime.shiftedBy(t);
                    maxElevationInWindow = 0.0;
                    maxElevationTime = null;
                }
                wasVisible = visibility.isVisible;
            }

            if (visibility.isVisible && visibility.elevation > maxElevationInWindow) {
                maxElevationInWindow = visibility.elevation;
                maxElevationTime = startTime.shiftedBy(t);
            }
        }

        if (wasVisible && windowStart != null) {
            VisibilityWindow window = createRelayVisibilityWindow(
                sat, relay, windowStart, endTime,
                Math.toDegrees(maxElevationInWindow), maxElevationTime
            );
            if (window != null) {
                windows.add(window);
            }
        }

        return windows;
    }

    /**
     * 计算LEO卫星与GEO中继卫星的可见性
     */
    private VisibilityResult calculateRelayVisibility(
            OrbitStateCache.OrbitState leoState,
            org.hipparchus.geometry.euclidean.threed.Vector3D relayPosition,
            double minElevationRad,
            double maxRange) {

        org.hipparchus.geometry.euclidean.threed.Vector3D leoPosition =
            new org.hipparchus.geometry.euclidean.threed.Vector3D(leoState.x, leoState.y, leoState.z);

        double distance = leoPosition.distance(relayPosition);
        if (distance > maxRange) {
            return new VisibilityResult(false, 0.0);
        }

        if (isEarthBlocking(leoPosition, relayPosition)) {
            return new VisibilityResult(false, 0.0);
        }

        double elevation = calculateElevationFromGeo(relayPosition, leoPosition);
        return new VisibilityResult(elevation >= minElevationRad, elevation);
    }

    /**
     * 检查视线是否被地球遮挡
     */
    private boolean isEarthBlocking(
            org.hipparchus.geometry.euclidean.threed.Vector3D from,
            org.hipparchus.geometry.euclidean.threed.Vector3D to) {

        org.hipparchus.geometry.euclidean.threed.Vector3D direction = to.subtract(from);
        double t = -from.dotProduct(direction) / direction.getNormSq();

        if (t < 0 || t > 1) {
            return false;
        }

        org.hipparchus.geometry.euclidean.threed.Vector3D closestPoint = from.add(direction.scalarMultiply(t));
        return closestPoint.getNorm() < EARTH_RADIUS * 0.99;
    }

    /**
     * 计算从GEO位置看向LEO位置的仰角
     */
    private double calculateElevationFromGeo(
            org.hipparchus.geometry.euclidean.threed.Vector3D geoPosition,
            org.hipparchus.geometry.euclidean.threed.Vector3D leoPosition) {

        org.hipparchus.geometry.euclidean.threed.Vector3D toLeo = leoPosition.subtract(geoPosition);
        org.hipparchus.geometry.euclidean.threed.Vector3D geoRadial = geoPosition.normalize();

        double cosAngle = geoRadial.dotProduct(toLeo.normalize());
        double angleFromRadial = Math.acos(Math.max(-1.0, Math.min(1.0, cosAngle)));

        return Math.PI / 2.0 - angleFromRadial;
    }

    /**
     * 创建中继卫星可见窗口对象
     */
    private VisibilityWindow createRelayVisibilityWindow(
            SatelliteConfig sat,
            RelaySatelliteConfig relay,
            AbsoluteDate start,
            AbsoluteDate end,
            double maxElevation,
            AbsoluteDate maxElTime) {

        double duration = end.durationFrom(start);
        if (duration < 30.0) { // 最小30秒
            return null;
        }

        double qualityScore = Math.min(1.0, maxElevation / 90.0);

        return new VisibilityWindow(
            sat.getId(),
            "RELAY:" + relay.getId(),
            start,
            end,
            duration,
            maxElevation,
            maxElTime != null ? maxElTime : start.shiftedBy(duration / 2),
            0.0, 0.0,
            qualityScore,
            maxElevation > 30.0 ? "HIGH" : (maxElevation > 15.0 ? "MEDIUM" : "LOW")
        );
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
