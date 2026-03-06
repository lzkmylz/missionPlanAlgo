package orekit.visibility;

import orekit.visibility.model.SatelliteConfig;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.UnnormalizedSphericalHarmonicsProvider;
import org.orekit.frames.FramesFactory;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.EcksteinHechlerPropagator;
import org.orekit.propagation.analytical.tle.TLE;
import org.orekit.propagation.analytical.tle.TLEPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.Constants;

/**
 * 智能轨道初始化器
 *
 * 根据数据源类型和时间差距智能选择初始化策略：
 * 1. TLE数据：使用SGP4外推到场景开始时间
 * 2. 六根数+历元距场景<3天：直接返回历元轨道（HPOP从历元传播）
 * 3. 六根数+历元距场景>3天：使用J4解析外推到场景开始时间
 *
 * 目标：HPOP始终只传播场景持续时间（如24小时），而非从历元开始的长期传播
 */
public class SmartOrbitInitializer {

    /**
     * 时间阈值：超过此值使用解析外推（天）
     */
    public static final double DIRECT_HPOP_THRESHOLD_DAYS = 3.0;

    /**
     * 获取场景开始时刻的卫星状态
     *
     * @param sat 卫星配置
     * @param scenarioStartTime 场景开始时间
     * @return 场景开始时刻的卫星状态
     */
    public SpacecraftState getInitialStateAtScenarioStart(
            SatelliteConfig sat, AbsoluteDate scenarioStartTime) throws Exception {

        // 判断数据类型
        if (hasTLE(sat)) {
            return getInitialStateFromTLE(sat, scenarioStartTime);
        } else if (sat instanceof JsonScenarioLoader.ExtendedSatelliteConfig) {
            return getInitialStateFromElements(
                (JsonScenarioLoader.ExtendedSatelliteConfig) sat, scenarioStartTime);
        } else {
            throw new IllegalArgumentException(
                "Unknown satellite config type: " + sat.getClass().getName());
        }
    }

    /**
     * TLE数据：使用SGP4外推到场景开始时间
     */
    private SpacecraftState getInitialStateFromTLE(
            SatelliteConfig sat, AbsoluteDate scenarioStartTime) throws Exception {

        TLE tle = new TLE(sat.getTleLine1(), sat.getTleLine2());
        AbsoluteDate epoch = tle.getDate();
        double daysToStart = scenarioStartTime.durationFrom(epoch) / 86400.0;

        TLEPropagator sgp4 = TLEPropagator.selectExtrapolator(tle);
        SpacecraftState stateAtScenarioStart = sgp4.propagate(scenarioStartTime);

        System.out.println("  [TLE] " + sat.getId() +
                          " | Epoch: " + epoch +
                          " | Δt to scenario start: " + String.format("%.2f", daysToStart) + " days" +
                          " | Method: SGP4 extrapolation");

        return stateAtScenarioStart;
    }

    /**
     * 六根数数据：智能选择策略
     */
    private SpacecraftState getInitialStateFromElements(
            JsonScenarioLoader.ExtendedSatelliteConfig sat,
            AbsoluteDate scenarioStartTime) throws Exception {

        AbsoluteDate epoch = sat.epoch;
        if (epoch == null) {
            epoch = AbsoluteDate.J2000_EPOCH;
        }

        double daysToStart = scenarioStartTime.durationFrom(epoch) / 86400.0;

        if (Math.abs(daysToStart) <= DIRECT_HPOP_THRESHOLD_DAYS) {
            // 历元距场景<3天：直接返回历元轨道
            // HPOP会从历元传播到场景结束（最多3天+24小时）
            System.out.println("  [ELEMENTS] " + sat.getId() +
                              " | Epoch close (" + String.format("%.2f", daysToStart) + " days)" +
                              " | Method: Direct HPOP propagation");

            Orbit orbit = createOrbitFromElements(sat, epoch);
            return new SpacecraftState(orbit);
        } else {
            // 历元距场景>3天：使用J4解析外推到场景开始时间
            System.out.println("  [ELEMENTS] " + sat.getId() +
                              " | Epoch far (" + String.format("%.2f", daysToStart) + " days)" +
                              " | Method: J4 analytical propagation to scenario start");

            return propagateWithJ4ToScenarioStart(sat, epoch, scenarioStartTime);
        }
    }

    /**
     * 使用J4解析传播器外推到场景开始时间
     */
    private SpacecraftState propagateWithJ4ToScenarioStart(
            JsonScenarioLoader.ExtendedSatelliteConfig sat,
            AbsoluteDate epoch, AbsoluteDate scenarioStartTime) throws Exception {

        // 创建历元时刻的开普勒轨道
        KeplerianOrbit initialOrbit = new KeplerianOrbit(
            sat.semiMajorAxis, sat.eccentricity,
            Math.toRadians(sat.inclination),
            Math.toRadians(sat.argOfPerigee),
            Math.toRadians(sat.raan),
            Math.toRadians(sat.meanAnomaly),
            PositionAngleType.MEAN,
            FramesFactory.getGCRF(),
            epoch,
            Constants.WGS84_EARTH_MU
        );

        // 获取J4引力场（6x6阶次，EcksteinHechlerPropagator需要(6,0)项）
        UnnormalizedSphericalHarmonicsProvider gravityField =
            GravityFieldFactory.getUnnormalizedProvider(6, 6);

        // 使用Eckstein-Hechler J4解析传播器
        EcksteinHechlerPropagator j4Propagator =
            new EcksteinHechlerPropagator(initialOrbit, sat.mass, gravityField);

        // 传播到场景开始时间
        return j4Propagator.propagate(scenarioStartTime);
    }

    /**
     * 从六根数创建轨道
     */
    private Orbit createOrbitFromElements(
            JsonScenarioLoader.ExtendedSatelliteConfig sat, AbsoluteDate epoch) {

        return new KeplerianOrbit(
            sat.semiMajorAxis, sat.eccentricity,
            Math.toRadians(sat.inclination),
            Math.toRadians(sat.argOfPerigee),
            Math.toRadians(sat.raan),
            Math.toRadians(sat.meanAnomaly),
            PositionAngleType.MEAN,
            FramesFactory.getGCRF(),
            epoch,
            Constants.WGS84_EARTH_MU
        );
    }

    /**
     * 判断卫星配置是否包含TLE数据
     */
    private boolean hasTLE(SatelliteConfig sat) {
        String tleLine1 = sat.getTleLine1();
        String tleLine2 = sat.getTleLine2();
        return tleLine1 != null && !tleLine1.isEmpty() &&
               tleLine2 != null && !tleLine2.isEmpty();
    }
}
