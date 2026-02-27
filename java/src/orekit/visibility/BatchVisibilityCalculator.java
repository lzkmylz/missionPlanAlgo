package orekit.visibility;

import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.hipparchus.ode.nonstiff.AdaptiveStepsizeIntegrator;
import org.orekit.bodies.BodyShape;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.errors.OrekitException;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.TopocentricFrame;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.OrbitType;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.events.ElevationDetector;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;

/**
 * 批量可见性计算器
 *
 * 设计目标：单次传播计算所有卫星-目标/地面站的可见窗口
 */
public class BatchVisibilityCalculator {

    private static final Logger logger = Logger.getLogger(BatchVisibilityCalculator.class.getName());

    // 地球物理常数
    private static final double EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS;
    private static final double EARTH_FLATTENING = Constants.WGS84_EARTH_FLATTENING;
    private static final double MU = Constants.WGS84_EARTH_MU;

    // 默认配置
    private static final double DEFAULT_COARSE_STEP = 300.0;  // 5分钟
    private static final double DEFAULT_FINE_STEP = 60.0;     // 1分钟
    private static final double DEFAULT_MIN_ELEVATION = 0.0;  // 地平线以上

    private final TimeScale utc;
    private final Frame eme2000;
    private final Frame itrf;
    private final BodyShape earth;

    public BatchVisibilityCalculator() {
        this.utc = TimeScalesFactory.getUTC();
        this.eme2000 = FramesFactory.getEME2000();
        this.itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
        this.earth = new OneAxisEllipsoid(EARTH_RADIUS, EARTH_FLATTENING, itrf);
    }

    /**
     * 批量计算所有可见窗口（主入口）
     *
     * @param satellites 卫星列表
     * @param targets 目标列表
     * @param groundStations 地面站列表
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @param config 计算配置
     * @return 批量计算结果
     * @throws OrekitException Orekit计算错误
     */
    public BatchResult computeAllWindows(
            List<SatelliteParameters> satellites,
            List<GroundPoint> targets,
            List<GroundPoint> groundStations,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            ComputationConfig config) throws OrekitException {

        long startNs = System.nanoTime();

        // 1. 创建轨道传播器池（每颗卫星一个，复用）
        Map<String, NumericalPropagator> propagatorPool = createPropagatorPool(satellites);

        // 2. 创建窗口追踪器
        WindowTracker windowTracker = new WindowTracker();

        // 3. 为所有卫星-目标/地面站对设置检测器
        setupDetectors(propagatorPool, targets, groundStations,
                      config.getMinElevation(), windowTracker);

        // 4. 批量传播（并行或串行）
        if (config.isUseParallel()) {
            propagateParallel(propagatorPool, startTime, endTime);
        } else {
            propagateSequential(propagatorPool, startTime, endTime);
        }

        // 5. 后处理：精化窗口边界
        List<VisibilityWindow> refinedWindows = refineWindows(
            windowTracker.getAllWindows(), propagatorPool, config.getFineStep()
        );

        // 6. 分类结果
        Map<String, List<VisibilityWindow>> targetWindows = new HashMap<>();
        Map<String, List<VisibilityWindow>> gsWindows = new HashMap<>();

        for (VisibilityWindow w : refinedWindows) {
            if (w.isTarget()) {
                targetWindows.computeIfAbsent(w.getKey(), k -> new ArrayList<>()).add(w);
            } else {
                gsWindows.computeIfAbsent(w.getKey(), k -> new ArrayList<>()).add(w);
            }
        }

        long elapsedNs = System.nanoTime() - startNs;

        return new BatchResult(
            targetWindows,
            gsWindows,
            new ComputationStats(
                TimeUnit.NANOSECONDS.toMillis(elapsedNs),
                satellites.size(),
                targets.size(),
                groundStations.size(),
                refinedWindows.size(),
                Runtime.getRuntime().totalMemory() / (1024 * 1024)
            )
        );
    }

    /**
     * 创建轨道传播器池
     */
    private Map<String, NumericalPropagator> createPropagatorPool(
            List<SatelliteParameters> satellites) throws OrekitException {

        Map<String, NumericalPropagator> pool = new HashMap<>();

        for (SatelliteParameters sat : satellites) {
            // 创建开普勒轨道
            KeplerianOrbit orbit = new KeplerianOrbit(
                sat.getSemiMajorAxis(),
                sat.getEccentricity(),
                Math.toRadians(sat.getInclination()),
                Math.toRadians(sat.getArgOfPerigee()),
                Math.toRadians(sat.getRaan()),
                Math.toRadians(sat.getMeanAnomaly()),
                PositionAngleType.MEAN,
                eme2000,
                new AbsoluteDate(sat.getEpoch(), utc),
                MU
            );

            // 创建数值传播器
            NumericalPropagator propagator = createNumericalPropagator(orbit);
            pool.put(sat.getId(), propagator);
        }

        return pool;
    }

    /**
     * 创建数值传播器
     */
    private NumericalPropagator createNumericalPropagator(KeplerianOrbit orbit) {
        // 使用Dormand-Prince 853积分器（适合高精度轨道传播）
        double minStep = 0.001;
        double maxStep = 1000.0;
        double positionTolerance = 1.0;

        AdaptiveStepsizeIntegrator integrator = new DormandPrince853Integrator(
            minStep, maxStep, positionTolerance, positionTolerance
        );

        NumericalPropagator propagator = new NumericalPropagator(integrator);
        propagator.setInitialState(new SpacecraftState(orbit));
        propagator.setOrbitType(OrbitType.KEPLERIAN);

        return propagator;
    }

    /**
     * 设置仰角检测器
     */
    private void setupDetectors(
            Map<String, NumericalPropagator> propagators,
            List<GroundPoint> targets,
            List<GroundPoint> groundStations,
            double minElevationDeg,
            WindowTracker tracker) throws OrekitException {

        double minElevationRad = Math.toRadians(minElevationDeg);
        double toleranceRad = Math.toRadians(0.01);  // 0.01度容差

        // 为每个传播器设置检测器
        for (Map.Entry<String, NumericalPropagator> entry : propagators.entrySet()) {
            String satId = entry.getKey();
            NumericalPropagator propagator = entry.getValue();

            // 目标检测器
            for (GroundPoint target : targets) {
                TopocentricFrame topo = new TopocentricFrame(
                    earth,
                    new GeodeticPoint(
                        Math.toRadians(target.getLatitude()),
                        Math.toRadians(target.getLongitude()),
                        target.getAltitude()
                    ),
                    target.getId()
                );

                ElevationDetector detector = new ElevationDetector(
                    DEFAULT_COARSE_STEP,
                    toleranceRad,
                    topo
                ).withConstantElevation(minElevationRad)
                 .withHandler(new ElevationHandler(satId, target.getId(), true, tracker));

                propagator.addEventDetector(detector);
            }

            // 地面站检测器
            for (GroundPoint gs : groundStations) {
                TopocentricFrame topo = new TopocentricFrame(
                    earth,
                    new GeodeticPoint(
                        Math.toRadians(gs.getLatitude()),
                        Math.toRadians(gs.getLongitude()),
                        gs.getAltitude()
                    ),
                    gs.getId()
                );

                ElevationDetector detector = new ElevationDetector(
                    DEFAULT_COARSE_STEP,
                    toleranceRad,
                    topo
                ).withConstantElevation(Math.toRadians(gs.getMinElevation()))
                 .withHandler(new ElevationHandler(satId, gs.getId(), false, tracker));

                propagator.addEventDetector(detector);
            }
        }
    }

    /**
     * 串行传播
     */
    private void propagateSequential(
            Map<String, NumericalPropagator> propagators,
            AbsoluteDate startTime,
            AbsoluteDate endTime) throws OrekitException {

        for (NumericalPropagator propagator : propagators.values()) {
            propagator.propagate(startTime, endTime);
        }
    }

    /**
     * 并行传播（每颗卫星独立传播）
     */
    private void propagateParallel(
            Map<String, NumericalPropagator> propagators,
            AbsoluteDate startTime,
            AbsoluteDate endTime) throws OrekitException {

        ExecutorService executor = Executors.newFixedThreadPool(
            Math.min(propagators.size(), Runtime.getRuntime().availableProcessors())
        );

        List<Future<?>> futures = new ArrayList<>();

        for (NumericalPropagator propagator : propagators.values()) {
            futures.add(executor.submit(() -> {
                try {
                    propagator.propagate(startTime, endTime);
                } catch (OrekitException e) {
                    throw new RuntimeException(e);
                }
            }));
        }

        // 等待所有传播完成
        for (Future<?> f : futures) {
            try {
                f.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new RuntimeException("Parallel propagation interrupted", e);
            } catch (ExecutionException e) {
                throw new RuntimeException("Parallel propagation failed", e.getCause());
            }
        }

        executor.shutdown();
    }

    /**
     * 精化窗口边界
     *
     * 使用更小的步长重新计算窗口边界附近，提高精度
     */
    private List<VisibilityWindow> refineWindows(
            List<VisibilityWindow> coarseWindows,
            Map<String, NumericalPropagator> propagators,
            double fineStep) throws OrekitException {

        // 简化实现：实际应该使用更精细的检测
        // 目前直接返回粗扫描结果
        // TODO: 实现精化逻辑

        return coarseWindows;
    }
}
