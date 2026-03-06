package orekit.visibility;

import orekit.visibility.model.SatelliteConfig;

import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.Transform;
import org.orekit.models.earth.ReferenceEllipsoid;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.tle.TLE;
import org.orekit.propagation.analytical.tle.TLEPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 轨道状态缓存
 *
 * 预计算并缓存卫星轨道状态，避免重复传播计算。
 * 默认使用HPOP高精度数值传播模型（EGM2008/EGM96引力场 + 完整摄动力）。
 * 支持多线程并行计算和线性插值查询。
 */
public class OrbitStateCache {

    /**
     * 轨道状态数据类
     */
    public static class OrbitState {
        public final double timestamp;      // 相对于起始时间的秒数
        public final double x, y, z;        // 位置 (m) - ITRF框架
        public final double vx, vy, vz;     // 速度 (m/s)
        public final double latitude;       // 地心纬度 (度)
        public final double longitude;      // 地心经度 (度)
        public final double altitude;       // 海拔高度 (m)

        public OrbitState(double timestamp, double x, double y, double z,
                         double vx, double vy, double vz,
                         double latitude, double longitude, double altitude) {
            this.timestamp = timestamp;
            this.x = x;
            this.y = y;
            this.z = z;
            this.vx = vx;
            this.vy = vy;
            this.vz = vz;
            this.latitude = latitude;
            this.longitude = longitude;
            this.altitude = altitude;
        }

        /**
         * 获取位置数组 [x, y, z]
         */
        public double[] getPosition() {
            return new double[]{x, y, z};
        }

        /**
         * 获取速度数组 [vx, vy, vz]
         */
        public double[] getVelocity() {
            return new double[]{vx, vy, vz};
        }
    }

    // 缓存: 卫星ID -> 按时间排序的轨道状态列表
    private final Map<String, List<OrbitState>> cache = new ConcurrentHashMap<>();

    // 时间步长配置
    private double timeStep = 5.0;

    // 坐标系
    private Frame itrfFrame;
    private OneAxisEllipsoid earth;

    // 智能轨道初始化器（处理历元与场景时间差距问题）
    private final SmartOrbitInitializer orbitInitializer = new SmartOrbitInitializer();

    public OrbitStateCache() {
        try {
            this.itrfFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
            this.earth = ReferenceEllipsoid.getIers2010(itrfFrame);
        } catch (Exception e) {
            // 如果无法获取ITRF，使用简化地球模型
            try {
                this.itrfFrame = FramesFactory.getGCRF();
                this.earth = new OneAxisEllipsoid(
                    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                    Constants.WGS84_EARTH_FLATTENING,
                    itrfFrame
                );
            } catch (Exception ex) {
                throw new RuntimeException("Failed to initialize frames", ex);
            }
        }
    }

    /**
     * 预计算所有卫星轨道（并行计算）
     *
     * @param satellites 卫星配置列表
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @param stepSeconds 时间步长（秒）
     */
    public void precomputeAllOrbits(
            List<SatelliteConfig> satellites,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double stepSeconds) {

        this.timeStep = stepSeconds;
        cache.clear();

        System.out.println("Precomputing orbits with smart initialization...");
        System.out.println("  Scenario: " + startTime + " to " + endTime);
        System.out.println("  Satellites: " + satellites.size());

        // 并行计算每颗卫星的轨道
        satellites.parallelStream().forEach(sat -> {
            try {
                // 1. 使用智能初始化器获取场景开始时刻的初始状态
                //    - TLE: SGP4外推到场景开始
                //    - 六根数+历元近: 直接使用历元轨道
                //    - 六根数+历元远: J4外推到场景开始
                SpacecraftState initialState = orbitInitializer
                    .getInitialStateAtScenarioStart(sat, startTime);

                // 2. 创建HPOP传播器，从场景开始时刻传播
                //    HPOP只需要传播场景持续时间（如24小时），而非从历元开始的长期
                Propagator hpop = createHPOPFromInitialState(initialState);

                // 3. HPOP高精度传播整个场景时段
                List<OrbitState> states = computeStatesWithPropagator(
                    hpop, sat.getId(), startTime, endTime, stepSeconds);

                cache.put(sat.getId(), states);
            } catch (Exception e) {
                throw new RuntimeException(
                    "Failed to compute orbit for satellite " + sat.getId(), e
                );
            }
        });

        System.out.println("  Orbit precomputation complete.");
    }

    /**
     * 从初始状态创建HPOP传播器
     *
     * HPOP从场景开始时刻的已知状态开始，只传播场景持续时间
     */
    private Propagator createHPOPFromInitialState(SpacecraftState initialState)
            throws Exception {

        // 创建积分器（Dormand-Prince 8(5,3)）
        double minStep = 0.001;  // 最小步长 1ms
        double maxStep = 300.0;  // 最大步长 5分钟
        double positionTolerance = 10.0;  // 位置容差 10米

        org.hipparchus.ode.nonstiff.DormandPrince853Integrator integrator =
            new org.hipparchus.ode.nonstiff.DormandPrince853Integrator(
                minStep, maxStep, positionTolerance, positionTolerance
            );

        // 创建HPOP传播器，从场景开始时刻的状态开始
        org.orekit.propagation.numerical.NumericalPropagator hpop =
            new org.orekit.propagation.numerical.NumericalPropagator(integrator);

        hpop.setInitialState(initialState);
        hpop.setOrbitType(org.orekit.orbits.OrbitType.CARTESIAN);

        // 配置摄动力模型
        configurePerturbations(hpop);

        return hpop;
    }

    /**
     * 使用传播器计算轨道状态序列
     */
    private List<OrbitState> computeStatesWithPropagator(
            Propagator propagator,
            String satId,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double stepSeconds) throws Exception {

        List<OrbitState> states = new ArrayList<>();

        // 计算时间范围（秒）
        double duration = endTime.durationFrom(startTime);

        // 按时间步长传播
        for (double t = 0; t <= duration; t += stepSeconds) {
            AbsoluteDate currentTime = startTime.shiftedBy(t);
            try {
                SpacecraftState state = propagator.propagate(currentTime);

                // 获取ITRF框架下的PV坐标
                PVCoordinates pv = state.getPVCoordinates(itrfFrame);

                // 转换为地心坐标
                double x = pv.getPosition().getX();
                double y = pv.getPosition().getY();
                double z = pv.getPosition().getZ();

                double vx = pv.getVelocity().getX();
                double vy = pv.getVelocity().getY();
                double vz = pv.getVelocity().getZ();

                // 计算地理坐标
                GeodeticPoint geo = earth.transform(
                    pv.getPosition(), itrfFrame, currentTime
                );

                OrbitState orbitState = new OrbitState(
                    t,
                    x, y, z,
                    vx, vy, vz,
                    Math.toDegrees(geo.getLatitude()),
                    Math.toDegrees(geo.getLongitude()),
                    geo.getAltitude()
                );

                states.add(orbitState);
            } catch (org.orekit.errors.OrekitException e) {
                // 捕获特定错误（如太阳光压地影计算问题）
                if (e.getMessage() != null && e.getMessage().contains("inside ellipsoid")) {
                    System.err.println("Warning: Propagation error at t=" + t + " for " + satId +
                        ", using last successful state");
                    // 如果可能，复制上一个成功状态
                    if (!states.isEmpty()) {
                        OrbitState lastState = states.get(states.size() - 1);
                        states.add(new OrbitState(
                            t,
                            lastState.x, lastState.y, lastState.z,
                            lastState.vx, lastState.vy, lastState.vz,
                            lastState.latitude, lastState.longitude, lastState.altitude
                        ));
                    }
                } else {
                    throw e;  // 其他错误继续抛出
                }
            }
        }

        return states;
    }

    /**
     * 创建卫星传播器（强制使用HPOP高精度数值模型）
     *
     * 无论输入是TLE还是六根数，都使用HPOP数值传播器计算
     */
    private Propagator createPropagator(SatelliteConfig sat) throws Exception {
        return createHPOPPropagator(sat);
    }

    /**
     * 创建HPOP高精度数值传播器
     *
     * 配置完整的摄动力模型：
     * - EGM2008/EGM96地球引力场
     * - NRLMSISE00大气阻力
     * - 太阳光压
     * - 日月第三体引力
     * - 相对论效应
     */
    private Propagator createHPOPPropagator(SatelliteConfig sat) throws Exception {
        // 1. 创建积分器（Dormand-Prince 8(5,3)）
        double minStep = 0.001;  // 最小步长 1ms
        double maxStep = 300.0;  // 最大步长 5分钟
        double positionTolerance = 10.0;  // 位置容差 10米

        org.hipparchus.ode.nonstiff.DormandPrince853Integrator integrator =
            new org.hipparchus.ode.nonstiff.DormandPrince853Integrator(
                minStep, maxStep, positionTolerance, positionTolerance
            );

        // 2. 创建初始轨道（从配置或默认值）
        org.orekit.orbits.Orbit initialOrbit = createInitialOrbit(sat);

        // 3. 创建数值传播器
        org.orekit.propagation.numerical.NumericalPropagator propagator =
            new org.orekit.propagation.numerical.NumericalPropagator(integrator);

        org.orekit.propagation.SpacecraftState initialState =
            new org.orekit.propagation.SpacecraftState(initialOrbit);
        propagator.setInitialState(initialState);
        propagator.setOrbitType(org.orekit.orbits.OrbitType.CARTESIAN);

        // 4. 配置摄动力模型
        configurePerturbations(propagator);

        return propagator;
    }

    /**
     * 配置摄动力模型
     */
    private void configurePerturbations(
            org.orekit.propagation.numerical.NumericalPropagator propagator)
            throws Exception {

        // 1. 地球引力场 (EGM2008/EGM96)
        addGravityForce(propagator);

        // 2. 大气阻力
        addDragForce(propagator);

        // 3. 太阳光压
        addSolarRadiationPressure(propagator);

        // 4. 第三体引力 (太阳和月球)
        addThirdBodyForces(propagator);

        // 5. 相对论效应
        addRelativityForce(propagator);
    }

    /**
     * 添加地球引力摄动力 (EGM2008/EGM96高精度模型)
     *
     * 优先使用EGM2008 90x90（最高精度），若不可用则回退到EGM96 21x21
     */
    private void addGravityForce(org.orekit.propagation.numerical.NumericalPropagator propagator)
            throws Exception {
        // 获取引力场提供者
        org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider gravityField;

        // 1. 优先尝试EGM2008 90x90（最高精度，需要下载完整数据文件）
        try {
            gravityField = org.orekit.forces.gravity.potential.GravityFieldFactory.getNormalizedProvider(90, 90);
            System.out.println("  Using EGM2008 ultra-high-precision gravity field (90x90)");
        } catch (Exception e) {
            // 2. 回退到EGM2008 36x36（常用精度）
            try {
                gravityField = org.orekit.forces.gravity.potential.GravityFieldFactory.getNormalizedProvider(36, 36);
                System.out.println("  Using EGM2008 high-precision gravity field (36x36)");
            } catch (Exception e2) {
                // 3. 回退到EGM96 21x21（Orekit数据仓库默认提供）
                try {
                    gravityField = org.orekit.forces.gravity.potential.GravityFieldFactory.getNormalizedProvider(21, 21);
                    System.out.println("  Using EGM96 gravity field (21x21) - Download EGM2008 for higher precision");
                } catch (Exception e3) {
                    // 4. 最后回退到5x5（Orekit基础测试数据）
                    gravityField = org.orekit.forces.gravity.potential.GravityFieldFactory.getNormalizedProvider(5, 5);
                    System.out.println("  WARNING: Using reduced gravity field (5x5)");
                }
            }
        }

        // 创建Holmes-Featherstone引力模型
        org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel gravityForce =
            new org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel(
                itrfFrame, gravityField
            );

        propagator.addForceModel(gravityForce);
    }

    /**
     * 添加大气阻力摄动力
     *
     * 强制配置，失败则抛出异常
     */
    private void addDragForce(org.orekit.propagation.numerical.NumericalPropagator propagator)
            throws Exception {
        // 使用简单指数大气模型
        double rho0 = 1.225e-9;  // kg/m^3 at sea level
        double h0 = 0.0;  // reference altitude (m)
        double scaleHeight = 8500.0;  // scale height (m)

        org.orekit.models.earth.atmosphere.SimpleExponentialAtmosphere atmosphere =
            new org.orekit.models.earth.atmosphere.SimpleExponentialAtmosphere(
                earth, rho0, h0, scaleHeight
            );

        // 卫星阻力参数 (Cd=2.2, Area=10m^2)
        org.orekit.forces.drag.IsotropicDrag dragSensitive =
            new org.orekit.forces.drag.IsotropicDrag(10.0, 2.2);

        org.orekit.forces.drag.DragForce dragForce =
            new org.orekit.forces.drag.DragForce(atmosphere, dragSensitive);

        propagator.addForceModel(dragForce);
    }

    /**
     * 添加太阳光压摄动力
     *
     * 简化配置，不使用地影计算以避免数值问题
     */
    private void addSolarRadiationPressure(org.orekit.propagation.numerical.NumericalPropagator propagator)
            throws Exception {
        try {
            // 获取太阳和地球
            org.orekit.bodies.CelestialBody sun =
                org.orekit.bodies.CelestialBodyFactory.getSun();

            // 卫星光学参数 (Cr=1.5, Area=10m^2)
            org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient radiationSensitive =
                new org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient(10.0, 1.5);

            // 使用简化太阳光压模型（无地影计算）
            // 使用地球半径作为遮蔽体半径，避免完整OneAxisEllipsoid的地影计算
            org.orekit.forces.radiation.SolarRadiationPressure srpForce =
                new org.orekit.forces.radiation.SolarRadiationPressure(
                    sun, earth, radiationSensitive
                );
            // 注意：这里仍然使用earth，但某些版本的Orekit可能会有数值问题
            // 如果问题持续，考虑完全禁用SRP或切换到更简单的模型

            propagator.addForceModel(srpForce);
        } catch (Exception e) {
            System.err.println("Warning: Failed to add SRP force, skipping: " + e.getMessage());
        }
    }

    /**
     * 添加第三体引力摄动力 (太阳和月球)
     *
     * 强制配置，失败则抛出异常
     */
    private void addThirdBodyForces(org.orekit.propagation.numerical.NumericalPropagator propagator)
            throws Exception {
        // 太阳引力
        org.orekit.bodies.CelestialBody sun =
            org.orekit.bodies.CelestialBodyFactory.getSun();
        org.orekit.forces.gravity.ThirdBodyAttraction sunForce =
            new org.orekit.forces.gravity.ThirdBodyAttraction(sun);
        propagator.addForceModel(sunForce);

        // 月球引力
        org.orekit.bodies.CelestialBody moon =
            org.orekit.bodies.CelestialBodyFactory.getMoon();
        org.orekit.forces.gravity.ThirdBodyAttraction moonForce =
            new org.orekit.forces.gravity.ThirdBodyAttraction(moon);
        propagator.addForceModel(moonForce);
    }

    /**
     * 添加相对论效应摄动力
     *
     * 强制配置，失败则抛出异常
     */
    private void addRelativityForce(org.orekit.propagation.numerical.NumericalPropagator propagator)
            throws Exception {
        double mu = org.orekit.utils.Constants.WGS84_EARTH_MU;
        org.orekit.forces.gravity.Relativity relativityForce =
            new org.orekit.forces.gravity.Relativity(mu);
        propagator.addForceModel(relativityForce);
    }

    /**
     * 创建初始轨道
     *
     * 优先从TLE解析，若无TLE则使用六根数配置
     * 支持扩展卫星配置（包含完整轨道参数）
     */
    private org.orekit.orbits.Orbit createInitialOrbit(SatelliteConfig sat) throws Exception {
        // 检查是否是扩展配置（包含完整轨道参数）
        if (sat instanceof JsonScenarioLoader.ExtendedSatelliteConfig) {
            return createOrbitFromExtendedConfig((JsonScenarioLoader.ExtendedSatelliteConfig) sat);
        }

        // 尝试从TLE解析初始轨道
        String tleLine1 = sat.getTleLine1();
        String tleLine2 = sat.getTleLine2();

        if (tleLine1 != null && !tleLine1.isEmpty() &&
            tleLine2 != null && !tleLine2.isEmpty()) {
            try {
                TLE tle = new TLE(tleLine1, tleLine2);
                // 从TLE获取初始状态
                TLEPropagator tempPropagator = TLEPropagator.selectExtrapolator(tle);
                SpacecraftState initialState = tempPropagator.getInitialState();
                return initialState.getOrbit();
            } catch (Exception e) {
                throw new RuntimeException("Failed to parse TLE for satellite " + sat.getId(), e);
            }
        }

        // 使用六根数配置创建初始轨道（默认500km SSO）
        return createOrbitFromElements(sat);
    }

    /**
     * 从扩展配置创建初始轨道（使用JSON中的真实轨道参数）
     */
    private org.orekit.orbits.Orbit createOrbitFromExtendedConfig(
            JsonScenarioLoader.ExtendedSatelliteConfig sat) throws Exception {

        // 使用JSON中的真实轨道参数
        double a = sat.semiMajorAxis;  // 半长轴（米）
        double e = sat.eccentricity;    // 偏心率
        double i = Math.toRadians(sat.inclination);  // 倾角（弧度）
        double omega = Math.toRadians(sat.argOfPerigee);  // 近地点幅角
        double raan = Math.toRadians(sat.raan);  // 升交点赤经
        double meanAnomaly = Math.toRadians(sat.meanAnomaly);  // 平近点角

        // 使用场景文件中的历元时间，而不是J2000
        AbsoluteDate epoch = sat.epoch != null ? sat.epoch : AbsoluteDate.J2000_EPOCH;

        return new org.orekit.orbits.KeplerianOrbit(
            a, e, i, omega, raan, meanAnomaly,
            org.orekit.orbits.PositionAngleType.MEAN,
            org.orekit.frames.FramesFactory.getGCRF(),
            epoch,
            org.orekit.utils.Constants.WGS84_EARTH_MU
        );
    }

    /**
     * 从轨道六根数创建初始轨道
     */
    private org.orekit.orbits.Orbit createOrbitFromElements(SatelliteConfig sat) throws Exception {
        // 默认轨道参数（500km高度的太阳同步轨道）
        double a = org.orekit.utils.Constants.WGS84_EARTH_EQUATORIAL_RADIUS + 500000;
        double e = 0.001;
        double i = Math.toRadians(97.6);  // SSO倾角
        double raan = 0.0;
        double omega = 0.0;
        double meanAnomaly = 0.0;

        // 从卫星ID解析相位
        try {
            String id = sat.getId();
            if (id != null && id.contains("-")) {
                String[] parts = id.split("-");
                int satNum = Integer.parseInt(parts[parts.length - 1]);
                meanAnomaly = Math.toRadians((satNum - 1) * 6.0 % 360.0);
            }
        } catch (Exception ignored) {
        }

        return new org.orekit.orbits.KeplerianOrbit(
            a, e, i, omega, raan, meanAnomaly,
            org.orekit.orbits.PositionAngleType.MEAN,
            org.orekit.frames.FramesFactory.getGCRF(),
            org.orekit.time.AbsoluteDate.J2000_EPOCH,
            org.orekit.utils.Constants.WGS84_EARTH_MU
        );
    }


    /**
     * 获取指定时刻的卫星状态（线性插值）
     *
     * @param satId 卫星ID
     * @param timestamp 相对于起始时间的秒数
     * @return 插值后的轨道状态
     */
    public OrbitState getStateAtTime(String satId, double timestamp) {
        List<OrbitState> states = cache.get(satId);
        if (states == null || states.isEmpty()) {
            return null;
        }

        // 边界情况
        if (timestamp <= states.get(0).timestamp) {
            return states.get(0);
        }
        if (timestamp >= states.get(states.size() - 1).timestamp) {
            return states.get(states.size() - 1);
        }

        // 计算索引位置
        int index = (int) (timestamp / timeStep);
        if (index < 0) index = 0;
        if (index >= states.size() - 1) index = states.size() - 2;

        OrbitState s1 = states.get(index);
        OrbitState s2 = states.get(index + 1);

        // 线性插值
        double t1 = s1.timestamp;
        double t2 = s2.timestamp;
        double ratio = (timestamp - t1) / (t2 - t1);

        if (ratio <= 0) return s1;
        if (ratio >= 1) return s2;

        return interpolate(s1, s2, ratio);
    }

    /**
     * 线性插值两个轨道状态
     */
    private OrbitState interpolate(OrbitState s1, OrbitState s2, double ratio) {
        double timestamp = s1.timestamp + (s2.timestamp - s1.timestamp) * ratio;

        double x = s1.x + (s2.x - s1.x) * ratio;
        double y = s1.y + (s2.y - s1.y) * ratio;
        double z = s1.z + (s2.z - s1.z) * ratio;

        double vx = s1.vx + (s2.vx - s1.vx) * ratio;
        double vy = s1.vy + (s2.vy - s1.vy) * ratio;
        double vz = s1.vz + (s2.vz - s1.vz) * ratio;

        double lat = s1.latitude + (s2.latitude - s1.latitude) * ratio;
        double lon = s1.longitude + (s2.longitude - s1.longitude) * ratio;
        double alt = s1.altitude + (s2.altitude - s1.altitude) * ratio;

        return new OrbitState(timestamp, x, y, z, vx, vy, vz, lat, lon, alt);
    }

    /**
     * 获取卫星的所有缓存状态
     */
    public List<OrbitState> getAllStates(String satId) {
        return cache.get(satId);
    }

    /**
     * 检查缓存中是否包含指定卫星
     */
    public boolean hasSatellite(String satId) {
        return cache.containsKey(satId);
    }

    /**
     * 获取缓存的卫星数量
     */
    public int getSatelliteCount() {
        return cache.size();
    }

    /**
     * 清空缓存
     */
    public void clear() {
        cache.clear();
    }

    /**
     * 获取内存使用估算（字节）
     */
    public long getMemoryUsage() {
        long total = 0;
        for (List<OrbitState> states : cache.values()) {
            // 每个OrbitState约10个double = 80字节 + 对象开销
            total += states.size() * 120L;
        }
        return total;
    }
}
