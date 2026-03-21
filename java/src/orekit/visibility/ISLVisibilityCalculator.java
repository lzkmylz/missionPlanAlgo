package orekit.visibility;

import orekit.visibility.model.ISLSatellitePairConfig;
import orekit.visibility.model.VisibilityWindow;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.Constants;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ISL (Inter-Satellite Link) 可见性计算器
 *
 * 计算LEO-LEO卫星间链路可见窗口。使用与目标/地面站/中继窗口相同的
 * 高精度HPOP轨道数据和两阶段扫描策略（5s粗扫 + 1s精扫）。
 *
 * 支持激光链路（laser）和微波链路（microwave）两种类型：
 * - 激光链路：检查距离 <= maxRangeKm 且无地球遮挡
 * - 微波链路：检查距离 <= maxRangeKm、离轴角 <= scanAngleDeg 且无地球遮挡
 *
 * 使用双向对称性：若(A,B)对已存在，不重复计算(B,A)。
 */
public class ISLVisibilityCalculator {

    private final OrbitStateCache orbitCache;
    private static final double EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS;
    private static final Logger logger = LoggerFactory.getLogger(ISLVisibilityCalculator.class);

    public ISLVisibilityCalculator(OrbitStateCache orbitCache) {
        this.orbitCache = orbitCache;
    }

    /**
     * 计算所有配置对的ISL可见窗口。
     *
     * @param islPairs   ISL卫星对配置列表
     * @param startTime  场景开始时间
     * @param endTime    场景结束时间
     * @param coarseStep 粗扫描步长（秒，建议5.0）
     * @param fineStep   精扫描步长（秒，建议1.0）
     * @return Map: "satAId_ISL:satBId" -> List<VisibilityWindow>
     */
    public Map<String, List<VisibilityWindow>> computeISLVisibilityWindows(
            List<ISLSatellitePairConfig> islPairs,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double coarseStep,
            double fineStep) {

        if (islPairs == null || islPairs.isEmpty()) {
            return new ConcurrentHashMap<>();
        }

        // 验证轨道缓存已预计算（检查所有涉及卫星均在缓存中）
        for (ISLSatellitePairConfig pair : islPairs) {
            if (!orbitCache.hasSatellite(pair.getSatAId())) {
                throw new IllegalStateException(
                    "Orbit cache not precomputed for ISL calculation. " +
                    "Call orbitCache.precomputeAllOrbits() first. " +
                    "Satellite not found in cache: " + pair.getSatAId()
                );
            }
            if (!orbitCache.hasSatellite(pair.getSatBId())) {
                throw new IllegalStateException(
                    "Orbit cache not precomputed for ISL calculation. " +
                    "Call orbitCache.precomputeAllOrbits() first. " +
                    "Satellite not found in cache: " + pair.getSatBId()
                );
            }
        }

        Map<String, List<VisibilityWindow>> islWindows = new ConcurrentHashMap<>();

        // 并行计算所有对（与RelayVisibilityCalculator保持一致）
        // 每个key唯一，直接put，避免computeIfAbsent+ArrayList的并发风险
        int totalWindows = islPairs.parallelStream()
            .mapToInt(pair -> {
                List<VisibilityWindow> windows = computePairWindowsTwoPhase(
                    pair, startTime, endTime, coarseStep, fineStep
                );
                if (!windows.isEmpty()) {
                    String key = pair.getSatAId() + "_ISL:" + pair.getSatBId();
                    islWindows.put(key, Collections.unmodifiableList(windows));
                }
                return windows.size();
            })
            .sum();

        logger.info("Computed {} ISL visibility windows for {} pairs", totalWindows, islPairs.size());
        return islWindows;
    }

    /**
     * 两阶段扫描计算单个ISL对的可见窗口。
     * 策略与RelayVisibilityCalculator.computeRelayWindowsTwoPhase()保持一致。
     */
    private List<VisibilityWindow> computePairWindowsTwoPhase(
            ISLSatellitePairConfig pair,
            AbsoluteDate startTime,
            AbsoluteDate endTime,
            double coarseStep,
            double fineStep) {

        List<VisibilityWindow> windows = new ArrayList<>();

        double duration = endTime.durationFrom(startTime);
        double maxRangeM = pair.getMaxRangeKm() * 1000.0;
        boolean isLaser = "laser".equalsIgnoreCase(pair.getLinkType());
        double maxOffAxisRad = isLaser
            ? Math.PI / 2.0  // 激光：只检查距离，不限制离轴角
            : Math.toRadians(pair.getMicrowaveScanAngleDeg());

        // ===== 阶段1: 粗扫描，定位窗口大致位置 =====
        List<double[]> coarseWindows = new ArrayList<>();
        boolean wasVisible = false;
        double windowStartT = 0;

        for (double t = 0; t <= duration; t += coarseStep) {
            boolean visible = checkISLVisibility(pair, t, maxRangeM, maxOffAxisRad, isLaser);
            if (visible != wasVisible) {
                if (wasVisible) {
                    coarseWindows.add(new double[]{windowStartT, t});
                } else {
                    windowStartT = t;
                }
                wasVisible = visible;
            }
        }
        if (wasVisible) {
            coarseWindows.add(new double[]{windowStartT, duration});
        }

        // ===== 阶段2: 精扫描窗口边界 =====
        for (double[] cw : coarseWindows) {
            double fineStartT = Math.max(0, cw[0] - coarseStep);
            double fineEndT = Math.min(duration, cw[1] + coarseStep);

            boolean inWindow = false;
            AbsoluteDate windowStart = null;
            double minDistInWindow = Double.MAX_VALUE;
            double sumRelVel = 0.0;
            int velSamples = 0;

            for (double t = fineStartT; t <= fineEndT; t += fineStep) {
                boolean visible = checkISLVisibility(pair, t, maxRangeM, maxOffAxisRad, isLaser);

                if (visible != inWindow) {
                    if (inWindow && windowStart != null) {
                        // 窗口结束，构建窗口对象
                        AbsoluteDate windowEnd = startTime.shiftedBy(t);
                        VisibilityWindow window = buildISLWindow(
                            pair, windowStart, windowEnd,
                            isLaser, minDistInWindow, sumRelVel, velSamples
                        );
                        windows.add(window);
                        // 重置累计量
                        windowStart = null;
                        minDistInWindow = Double.MAX_VALUE;
                        sumRelVel = 0.0;
                        velSamples = 0;
                    } else {
                        windowStart = startTime.shiftedBy(t);
                    }
                    inWindow = visible;
                }

                // 在可见窗口内跟踪链路参数
                if (visible) {
                    OrbitStateCache.OrbitState stateA = orbitCache.getStateAtTime(pair.getSatAId(), t);
                    OrbitStateCache.OrbitState stateB = orbitCache.getStateAtTime(pair.getSatBId(), t);
                    if (stateA != null && stateB != null) {
                        Vector3D posA = new Vector3D(stateA.x, stateA.y, stateA.z);
                        Vector3D posB = new Vector3D(stateB.x, stateB.y, stateB.z);
                        double dist = posA.distance(posB);
                        if (dist < minDistInWindow) {
                            minDistInWindow = dist;
                        }
                        Vector3D velA = new Vector3D(stateA.vx, stateA.vy, stateA.vz);
                        Vector3D velB = new Vector3D(stateB.vx, stateB.vy, stateB.vz);
                        // 相对速度（m/s -> km/s）
                        sumRelVel += velA.subtract(velB).getNorm() / 1000.0;
                        velSamples++;
                    }
                }
            }

            // 处理精扫描结尾仍在窗口内的情况
            if (inWindow && windowStart != null) {
                AbsoluteDate windowEnd = startTime.shiftedBy(Math.min(fineEndT, duration));
                VisibilityWindow window = buildISLWindow(
                    pair, windowStart, windowEnd,
                    isLaser, minDistInWindow, sumRelVel, velSamples
                );
                windows.add(window);
            }
        }

        return windows;
    }

    /**
     * 构建ISL可见窗口对象，计算链路参数并设置ISL元数据。
     */
    private VisibilityWindow buildISLWindow(
            ISLSatellitePairConfig pair,
            AbsoluteDate windowStart,
            AbsoluteDate windowEnd,
            boolean isLaser,
            double minDistM,
            double sumRelVel,
            int velSamples) {

        double avgRelVelKmS = velSamples > 0 ? sumRelVel / velSamples : 0.0;
        double distKm = (minDistM < Double.MAX_VALUE) ? minDistM / 1000.0 : pair.getMaxRangeKm();

        double dataRateMbps;
        double linkMarginDb;
        double atpSetupTimeS;

        if (isLaser) {
            linkMarginDb = calculateLaserLinkMargin(distKm, pair.getLaserTrackingAccuracyUrad(),
                                                     pair.getLaserReceiverSensitivityDbm());
            dataRateMbps = calculateLaserDataRate(distKm, pair.getLaserTrackingAccuracyUrad(),
                                                   pair.getLaserReceiverSensitivityDbm());
            // ATP建立时间 = 捕获阶段（随相对速度缩放） + 粗跟踪+精跟踪固定时间
            atpSetupTimeS = pair.getLaserAcquisitionTimeS() * Math.max(1.0, avgRelVelKmS / 3.0)
                          + pair.getLaserTrackingSetupTimeS();
        } else {
            dataRateMbps = calculateMicrowaveDataRate(distKm);
            linkMarginDb = calculateMicrowaveLinkMargin(distKm);
            atpSetupTimeS = 0.0;
        }

        VisibilityWindow window = new VisibilityWindow(
            pair.getSatAId(),
            "ISL:" + pair.getSatBId(),
            windowStart,
            windowEnd,
            0.0  // elevation不适用于ISL
        );
        window.setISLMetadata(
            pair.getLinkType(), dataRateMbps, linkMarginDb,
            distKm, avgRelVelKmS, atpSetupTimeS
        );
        return window;
    }

    /**
     * 检查时刻t两颗卫星之间是否满足ISL可见条件。
     */
    private boolean checkISLVisibility(ISLSatellitePairConfig pair, double t,
                                        double maxRangeM, double maxOffAxisRad, boolean isLaser) {
        OrbitStateCache.OrbitState stateA = orbitCache.getStateAtTime(pair.getSatAId(), t);
        OrbitStateCache.OrbitState stateB = orbitCache.getStateAtTime(pair.getSatBId(), t);
        if (stateA == null || stateB == null) return false;

        Vector3D posA = new Vector3D(stateA.x, stateA.y, stateA.z);
        Vector3D posB = new Vector3D(stateB.x, stateB.y, stateB.z);

        // 距离检查
        double distance = posA.distance(posB);
        if (distance > maxRangeM) return false;

        // 地球遮挡检查（视线是否穿越地球）
        if (isEarthBlocking(posA, posB)) return false;

        // 微波链路额外检查离轴角
        if (!isLaser) {
            double offAxisAngle = calculateOffAxisAngle(posA, posB);
            if (offAxisAngle > maxOffAxisRad) return false;
        }

        return true;
    }

    /**
     * 检查视线是否被地球遮挡（与RelayVisibilityCalculator.isEarthBlocking()逻辑相同）。
     */
    private boolean isEarthBlocking(Vector3D from, Vector3D to) {
        Vector3D direction = to.subtract(from);
        double t = -from.dotProduct(direction) / direction.getNormSq();
        if (t < 0 || t > 1) return false;
        Vector3D closestPoint = from.add(direction.scalarMultiply(t));
        return closestPoint.getNorm() < EARTH_RADIUS * 0.99;
    }

    /**
     * 计算从观测方看向目标的离轴角（相对于nadir方向）。
     */
    private double calculateOffAxisAngle(Vector3D observerPos, Vector3D targetPos) {
        Vector3D toTarget = targetPos.subtract(observerPos).normalize();
        Vector3D nadir = observerPos.negate().normalize(); // 指向地心
        double cosAngle = nadir.dotProduct(toTarget);
        return Math.acos(Math.max(-1.0, Math.min(1.0, cosAngle)));
    }

    /**
     * 激光链路余量计算（直接光功率法，与Python端 isl_physics.py 保持一致）。
     *
     * 公式：
     *   光斑半径 = theta_div_rad * distance_m
     *   功率分数 = (D_rx/2)^2 / spot_radius^2
     *   Pr [W]  = Pt * eta_tx * eta_rx * power_fraction
     *   Pr_dBm  = 10*log10(Pr * 1000)
     *   L_point = 8.69 * (sigma/theta)^2
     *   margin  = Pr_dBm - L_point - receiver_sensitivity_dBm
     *
     * @param distanceKm            链路距离（km）
     * @param trackingAccuracyUrad  跟踪精度（μrad, 1-sigma）
     * @param receiverSensitivityDbm 接收机灵敏度（dBm），APD典型值-31.0
     * @return 链路余量（dB）
     */
    private double calculateLaserLinkMargin(double distanceKm, double trackingAccuracyUrad,
                                             double receiverSensitivityDbm) {
        double distanceM = distanceKm * 1000.0;
        double thetaDivRad = 5e-6;                // 发散角，rad（5 μrad）
        double etaTx = 0.85;                       // 发射光学效率
        double etaRx = 0.82;                       // 接收光学效率
        double txPowerW = 2.0;                     // 发射功率，W
        double rxApertureM = 0.1;                  // 接收孔径，m

        double spotRadiusM = thetaDivRad * distanceM;
        double powerFraction = Math.pow(rxApertureM / 2.0, 2) / (spotRadiusM * spotRadiusM);
        double PrW = txPowerW * etaTx * etaRx * powerFraction;
        double PrDbm = 10.0 * Math.log10(Math.max(PrW * 1000.0, 1e-30));

        double sigmaRad = trackingAccuracyUrad * 1e-6;
        double LpointDb = 8.69 * Math.pow(sigmaRad / thetaDivRad, 2);

        return PrDbm - LpointDb - receiverSensitivityDbm;
    }

    // Overload for backward compatibility (uses default receiver sensitivity -31.0 dBm)
    private double calculateLaserLinkMargin(double distanceKm, double trackingAccuracyUrad) {
        return calculateLaserLinkMargin(distanceKm, trackingAccuracyUrad, -31.0);
    }

    private double calculateLaserDataRate(double distanceKm, double trackingAccuracyUrad,
                                           double receiverSensitivityDbm) {
        double margin = calculateLaserLinkMargin(distanceKm, trackingAccuracyUrad, receiverSensitivityDbm);
        if (margin < 3.0) return 0.0;
        double baseRate = 10000.0;   // 10 Gbps @ 1000 km
        double rate = baseRate * Math.pow(1000.0 / distanceKm, 2);
        return Math.max(100.0, Math.min(rate, 100000.0));
    }

    private double calculateLaserDataRate(double distanceKm, double trackingAccuracyUrad) {
        return calculateLaserDataRate(distanceKm, trackingAccuracyUrad, -31.0);
    }

    /**
     * 微波链路余量计算（简化版，26 GHz，Pt=10W，G=30dBi，T=1000K，BW=1GHz）。
     */
    private double calculateMicrowaveLinkMargin(double distanceKm) {
        double lambda = 3e8 / 26e9;                         // 26 GHz波长，≈0.01154 m
        double Pt_dBm = 10 * Math.log10(10.0 * 1000);       // 40 dBm
        double G = 30.0;                                     // dBi
        double Lfs = 20 * Math.log10(4.0 * Math.PI * distanceKm * 1000.0 / lambda);
        double k = 1.38e-23;
        double T = 1000.0;
        double BW = 1e9;
        double noise_dBm = 10 * Math.log10(k * T * BW * 1000.0);
        double SNR = Pt_dBm + G + G - Lfs - noise_dBm;
        return SNR - 15.0;   // SNRreq = 15 dB
    }

    private double calculateMicrowaveDataRate(double distanceKm) {
        double margin = calculateMicrowaveLinkMargin(distanceKm);
        if (margin < 0) return 0.0;
        double snrLinear = Math.pow(10, margin / 10.0);
        double bwMhz = 1000.0;
        return bwMhz * (Math.log(1.0 + snrLinear) / Math.log(2.0));
    }
}
