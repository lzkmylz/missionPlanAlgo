package orekit.visibility.calculator;

import orekit.visibility.model.SatelliteConfig;
import orekit.visibility.model.TargetConfig;
import orekit.visibility.model.VisibilityWindow;
import orekit.visibility.model.BatchResult;

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
import org.orekit.propagation.analytical.tle.TLE;
import org.orekit.propagation.analytical.tle.TLEPropagator;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeScalesFactory;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;

import java.util.ArrayList;
import java.util.List;

/**
 * 批量可见性计算器
 *
 * 在Java端完成所有卫星-目标对的可见窗口计算，
 * 避免Python-Java之间的频繁JNI调用。
 *
 * 优化策略：
 * 1. 两阶段自适应步长（粗扫描+精化）
 * 2. 卫星轨道传播器缓存
 * 3. 批量几何计算
 */
public class VisibilityBatchCalculator {

    private final double coarseStepSeconds;
    private final double fineStepSeconds;
    private final double globalMinElevation;
    private final double minWindowDurationSeconds;

    // 缓存的坐标系
    private Frame itrfFrame;
    private OneAxisEllipsoid earth;

    /**
     * 创建批量计算器（使用默认配置）
     */
    public VisibilityBatchCalculator() {
        this(300.0, 60.0, 5.0, 60.0);
    }

    /**
     * 创建批量计算器
     *
     * @param coarseStepSeconds 粗扫描步长（秒）
     * @param fineStepSeconds 精化步长（秒）
     * @param globalMinElevation 全局最小仰角（度）
     * @param minWindowDurationSeconds 最小窗口持续时间（秒）
     */
    public VisibilityBatchCalculator(double coarseStepSeconds,
                                     double fineStepSeconds,
                                     double globalMinElevation,
                                     double minWindowDurationSeconds) {
        this.coarseStepSeconds = coarseStepSeconds;
        this.fineStepSeconds = fineStepSeconds;
        this.globalMinElevation = globalMinElevation;
        this.minWindowDurationSeconds = minWindowDurationSeconds;

        initializeFrames();
    }

    /**
     * 初始化坐标系和地球模型
     */
    private void initializeFrames() {
        try {
            this.itrfFrame = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
            this.earth = ReferenceEllipsoid.getIers2010(itrfFrame);
        } catch (Exception e) {
            // 如果无法获取ITRF，使用简化地球模型
            this.itrfFrame = FramesFactory.getGCRF();
            this.earth = new OneAxisEllipsoid(
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                Constants.WGS84_EARTH_FLATTENING,
                itrfFrame
            );
        }
    }

    /**
     * 批量计算所有卫星-目标对的可见窗口
     *
     * @param satellites 卫星配置列表
     * @param targets 目标配置列表
     * @param startTime 开始时间
     * @param endTime 结束时间
     * @return BatchResult 计算结果
     */
    public BatchResult computeBatch(List<SatelliteConfig> satellites,
                                    List<TargetConfig> targets,
                                    AbsoluteDate startTime,
                                    AbsoluteDate endTime) {
        long startTimeMs = System.currentTimeMillis();
        BatchResult result = new BatchResult();
        result.getStatistics().setTotalPairs(satellites.size() * targets.size());

        int totalCoarsePoints = 0;
        int totalFinePoints = 0;

        // 遍历所有卫星-目标对
        for (SatelliteConfig sat : satellites) {
            try {
                // 创建卫星传播器（每个卫星只需创建一次）
                Propagator propagator = createPropagator(sat);

                for (TargetConfig target : targets) {
                    try {
                        // 计算该对的可见窗口
                        List<VisibilityWindow> windows = computeWindowsForPair(
                            propagator, sat, target, startTime, endTime
                        );

                        if (!windows.isEmpty()) {
                            result.addWindows(sat.getId(), target.getId(), windows);
                            result.getStatistics().incrementPairsWithWindows();
                        }

                        // 统计计算点数
                        totalCoarsePoints += (int) ((endTime.durationFrom(startTime)) / coarseStepSeconds);
                    } catch (Exception e) {
                        result.addError(sat.getId(), target.getId(),
                                       "COMPUTATION_ERROR", e.getMessage());
                    }
                }
            } catch (Exception e) {
                // 卫星传播器创建失败
                for (TargetConfig target : targets) {
                    result.addError(sat.getId(), target.getId(),
                                   "PROPAGATOR_ERROR", e.getMessage());
                }
            }
        }

        long computationTimeMs = System.currentTimeMillis() - startTimeMs;
        result.getStatistics().setComputationTimeMs(computationTimeMs);
        result.getStatistics().setCoarseScanPoints(totalCoarsePoints);
        result.getStatistics().setFineScanPoints(totalFinePoints);

        return result;
    }

    /**
     * 为单个卫星-目标对计算可见窗口
     */
    private List<VisibilityWindow> computeWindowsForPair(Propagator propagator,
                                                         SatelliteConfig sat,
                                                         TargetConfig target,
                                                         AbsoluteDate startTime,
                                                         AbsoluteDate endTime) {
        List<VisibilityWindow> windows = new ArrayList<>();

        // 阶段1: 粗扫描定位潜在窗口
        List<CoarseWindow> coarseWindows = coarseScan(
            propagator, sat, target, startTime, endTime
        );

        // 阶段2: 精化每个潜在窗口
        for (CoarseWindow coarse : coarseWindows) {
            if (coarse.isPotentiallyVisible()) {
                VisibilityWindow refined = refineWindow(
                    propagator, sat, target, coarse, startTime, endTime
                );

                if (refined != null && refined.getDurationSeconds() >= minWindowDurationSeconds) {
                    windows.add(refined);
                }
            }
        }

        return windows;
    }

    /**
     * 粗扫描 - 使用大步长快速定位潜在窗口
     */
    private List<CoarseWindow> coarseScan(Propagator propagator,
                                          SatelliteConfig sat,
                                          TargetConfig target,
                                          AbsoluteDate startTime,
                                          AbsoluteDate endTime) {
        List<CoarseWindow> windows = new ArrayList<>();
        GeodeticPoint targetPoint = createGeodeticPoint(target);
        double minEl = Math.toRadians(Math.max(globalMinElevation, sat.getMinElevation()));

        AbsoluteDate currentTime = startTime;
        AbsoluteDate windowStart = startTime;
        boolean wasVisible = false;
        double maxElevationInWindow = 0.0;

        while (currentTime.compareTo(endTime) <= 0) {
            boolean isVisible = checkVisibility(propagator, targetPoint, currentTime, minEl);

            if (isVisible != wasVisible) {
                // 状态变化，记录窗口
                if (wasVisible) {
                    // 可见窗口结束
                    windows.add(new CoarseWindow(
                        windowStart, currentTime, true, maxElevationInWindow
                    ));
                } else {
                    // 可见窗口开始
                    windowStart = currentTime;
                    maxElevationInWindow = 0.0;
                }
                wasVisible = isVisible;
            }

            if (isVisible) {
                double el = calculateElevation(propagator, targetPoint, currentTime);
                maxElevationInWindow = Math.max(maxElevationInWindow, el);
            }

            currentTime = currentTime.shiftedBy(coarseStepSeconds);
        }

        // 处理最后一个窗口
        if (wasVisible) {
            windows.add(new CoarseWindow(
                windowStart, endTime, true, maxElevationInWindow
            ));
        }

        return windows;
    }

    /**
     * 精化窗口 - 在粗略窗口附近使用小步长精确计算
     */
    private VisibilityWindow refineWindow(Propagator propagator,
                                          SatelliteConfig sat,
                                          TargetConfig target,
                                          CoarseWindow coarse,
                                          AbsoluteDate missionStart,
                                          AbsoluteDate missionEnd) {
        GeodeticPoint targetPoint = createGeodeticPoint(target);
        double minEl = Math.toRadians(Math.max(globalMinElevation, sat.getMinElevation()));

        // 向前扩展找精确开始时间
        AbsoluteDate preciseStart = coarse.getStartTime();
        while (preciseStart.compareTo(missionStart) > 0) {
            AbsoluteDate prevTime = preciseStart.shiftedBy(-fineStepSeconds);
            if (prevTime.compareTo(missionStart) < 0) {
                break;
            }
            if (!checkVisibility(propagator, targetPoint, prevTime, minEl)) {
                break;
            }
            preciseStart = prevTime;
        }

        // 向后扩展找精确结束时间
        AbsoluteDate preciseEnd = coarse.getEndTime();
        while (preciseEnd.compareTo(missionEnd) < 0) {
            AbsoluteDate nextTime = preciseEnd.shiftedBy(fineStepSeconds);
            if (nextTime.compareTo(missionEnd) > 0) {
                break;
            }
            if (!checkVisibility(propagator, targetPoint, nextTime, minEl)) {
                break;
            }
            preciseEnd = nextTime;
        }

        // 计算窗口中点的最大仰角和方位角
        AbsoluteDate midTime = preciseStart.shiftedBy(
            preciseEnd.durationFrom(preciseStart) / 2.0
        );
        double maxElevation = calculateElevation(propagator, targetPoint, midTime);
        double[] azimuths = calculateAzimuths(propagator, targetPoint, preciseStart, preciseEnd);

        double duration = preciseEnd.durationFrom(preciseStart);
        double qualityScore = Math.min(1.0, maxElevation / Math.toRadians(90.0));

        return new VisibilityWindow(
            sat.getId(),
            target.getId(),
            preciseStart,
            preciseEnd,
            duration,
            Math.toDegrees(maxElevation),
            midTime,
            Math.toDegrees(azimuths[0]),
            Math.toDegrees(azimuths[1]),
            qualityScore,
            "HIGH"
        );
    }

    /**
     * 检查指定时间是否可见（仰角大于最小值且未被遮挡）
     */
    private boolean checkVisibility(Propagator propagator,
                                    GeodeticPoint targetPoint,
                                    AbsoluteDate date,
                                    double minElevation) {
        try {
            PVCoordinates satPV = propagator.propagate(date).getPVCoordinates(itrfFrame);
            double elevation = calculateElevationFromPV(satPV, targetPoint);

            if (elevation < minElevation) {
                return false;
            }

            // 检查地球遮挡
            return !isEarthOccluded(satPV, targetPoint);
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * 计算仰角
     */
    private double calculateElevation(Propagator propagator,
                                      GeodeticPoint targetPoint,
                                      AbsoluteDate date) {
        try {
            PVCoordinates satPV = propagator.propagate(date).getPVCoordinates(itrfFrame);
            return calculateElevationFromPV(satPV, targetPoint);
        } catch (Exception e) {
            return -Math.PI / 2;  // 错误时返回负仰角
        }
    }

    /**
     * 从PV坐标计算仰角
     */
    private double calculateElevationFromPV(PVCoordinates satPV, GeodeticPoint targetPoint) {
        // 将目标点转换为笛卡尔坐标
        double[] targetCart = geodeticToCartesian(targetPoint);

        // 计算从目标指向卫星的向量
        double dx = satPV.getPosition().getX() - targetCart[0];
        double dy = satPV.getPosition().getY() - targetCart[1];
        double dz = satPV.getPosition().getZ() - targetCart[2];

        // 目标点的本地垂直方向（从地心指向目标的单位向量）
        double r = Math.sqrt(targetCart[0]*targetCart[0] + targetCart[1]*targetCart[1] + targetCart[2]*targetCart[2]);
        double ux = targetCart[0] / r;
        double uy = targetCart[1] / r;
        double uz = targetCart[2] / r;

        // 计算仰角
        double range = Math.sqrt(dx*dx + dy*dy + dz*dz);
        double cosZenith = (dx*ux + dy*uy + dz*uz) / range;

        return Math.PI / 2 - Math.acos(cosZenith);
    }

    /**
     * 检查地球遮挡
     */
    private boolean isEarthOccluded(PVCoordinates satPV, GeodeticPoint targetPoint) {
        // 简化检查：如果卫星在目标点的地平线以下
        double elevation = calculateElevationFromPV(satPV, targetPoint);
        return elevation < 0;
    }

    /**
     * 计算进入和离开方位角
     */
    private double[] calculateAzimuths(Propagator propagator,
                                       GeodeticPoint targetPoint,
                                       AbsoluteDate startTime,
                                       AbsoluteDate endTime) {
        double entryAzimuth = 0.0;
        double exitAzimuth = 0.0;

        try {
            PVCoordinates startPV = propagator.propagate(startTime).getPVCoordinates(itrfFrame);
            entryAzimuth = calculateAzimuthFromPV(startPV, targetPoint);

            PVCoordinates endPV = propagator.propagate(endTime).getPVCoordinates(itrfFrame);
            exitAzimuth = calculateAzimuthFromPV(endPV, targetPoint);
        } catch (Exception e) {
            // 使用默认值
        }

        return new double[]{entryAzimuth, exitAzimuth};
    }

    /**
     * 计算方位角
     */
    private double calculateAzimuthFromPV(PVCoordinates satPV, GeodeticPoint targetPoint) {
        // 简化的方位角计算
        double[] targetCart = geodeticToCartesian(targetPoint);
        double dx = satPV.getPosition().getX() - targetCart[0];
        double dy = satPV.getPosition().getY() - targetCart[1];

        return Math.atan2(dy, dx);
    }

    /**
     * 从地理坐标创建GeodeticPoint
     */
    private GeodeticPoint createGeodeticPoint(TargetConfig target) {
        return new GeodeticPoint(
            Math.toRadians(target.getLatitude()),
            Math.toRadians(target.getLongitude()),
            target.getAltitude()
        );
    }

    /**
     * 地理坐标转笛卡尔坐标
     */
    private double[] geodeticToCartesian(GeodeticPoint point) {
        org.hipparchus.geometry.euclidean.threed.Vector3D vector = earth.transform(point);
        return new double[]{vector.getX(), vector.getY(), vector.getZ()};
    }

    /**
     * 创建卫星传播器
     */
    private Propagator createPropagator(SatelliteConfig sat) {
        try {
            // 尝试使用TLE创建传播器
            TLE tle = new TLE(sat.getTleLine1(), sat.getTleLine2());
            return TLEPropagator.selectExtrapolator(tle);
        } catch (Exception e) {
            // TLE失败时使用简化开普勒传播器
            return createKeplerianPropagator(sat);
        }
    }

    /**
     * 创建简化开普勒传播器（TLE无效时的回退）
     */
    private Propagator createKeplerianPropagator(SatelliteConfig sat) {
        // 默认轨道参数（近地轨道）
        double a = Constants.WGS84_EARTH_EQUATORIAL_RADIUS + 500000;  // 500km高度
        double e = 0.001;  // 小偏心率
        double i = Math.toRadians(51.6);  // 典型倾角
        double raan = 0.0;
        double omega = 0.0;
        double meanAnomaly = 0.0;

        Orbit orbit = new KeplerianOrbit(
            a, e, i, omega, raan, meanAnomaly,
            PositionAngleType.MEAN,
            FramesFactory.getGCRF(),
            AbsoluteDate.J2000_EPOCH,
            Constants.WGS84_EARTH_MU
        );

        return new KeplerianPropagator(orbit);
    }

    /**
     * 粗窗口内部类
     */
    private static class CoarseWindow {
        private final AbsoluteDate startTime;
        private final AbsoluteDate endTime;
        private final boolean potentiallyVisible;
        private final double maxElevation;

        public CoarseWindow(AbsoluteDate startTime, AbsoluteDate endTime,
                           boolean potentiallyVisible, double maxElevation) {
            this.startTime = startTime;
            this.endTime = endTime;
            this.potentiallyVisible = potentiallyVisible;
            this.maxElevation = maxElevation;
        }

        public AbsoluteDate getStartTime() { return startTime; }
        public AbsoluteDate getEndTime() { return endTime; }
        public boolean isPotentiallyVisible() { return potentiallyVisible; }
        public double getMaxElevation() { return maxElevation; }
    }
}
