package orekit.visibility.model;

/**
 * ISL (Inter-Satellite Link) configuration for a satellite pair.
 *
 * Each instance represents one directional ISL configuration:
 * satellite satAId can connect to satBId with the given link type.
 */
public class ISLSatellitePairConfig {
    private final String satAId;
    private final String satBId;
    private final String linkType;                   // "laser" or "microwave"
    private final double maxRangeKm;
    private final double laserAcquisitionTimeS;      // ATP acquisition time (laser only)
    private final double laserTrackingAccuracyUrad;
    private final double microwaveScanAngleDeg;      // max off-axis angle (microwave only)
    private final double laserReceiverSensitivityDbm; // APD receiver sensitivity (laser only), default -31.0 dBm
    private final double laserTrackingSetupTimeS;      // coarse+fine ATP tracking time (seconds), default 7.0

    public ISLSatellitePairConfig(
            String satAId, String satBId, String linkType,
            double maxRangeKm,
            double laserAcquisitionTimeS, double laserTrackingAccuracyUrad,
            double microwaveScanAngleDeg) {
        this(satAId, satBId, linkType, maxRangeKm,
             laserAcquisitionTimeS, laserTrackingAccuracyUrad,
             microwaveScanAngleDeg, -31.0, 7.0);
    }

    public ISLSatellitePairConfig(
            String satAId, String satBId, String linkType,
            double maxRangeKm,
            double laserAcquisitionTimeS, double laserTrackingAccuracyUrad,
            double microwaveScanAngleDeg, double laserReceiverSensitivityDbm,
            double laserTrackingSetupTimeS) {
        if (satAId == null || satAId.isEmpty()) throw new IllegalArgumentException("satAId must not be empty");
        if (satBId == null || satBId.isEmpty()) throw new IllegalArgumentException("satBId must not be empty");
        if (!linkType.equals("laser") && !linkType.equals("microwave"))
            throw new IllegalArgumentException("linkType must be 'laser' or 'microwave', got: " + linkType);
        if (maxRangeKm <= 0) throw new IllegalArgumentException("maxRangeKm must be positive");
        this.satAId = satAId;
        this.satBId = satBId;
        this.linkType = linkType;
        this.maxRangeKm = maxRangeKm;
        this.laserAcquisitionTimeS = laserAcquisitionTimeS;
        this.laserTrackingAccuracyUrad = laserTrackingAccuracyUrad;
        this.microwaveScanAngleDeg = microwaveScanAngleDeg;
        this.laserReceiverSensitivityDbm = laserReceiverSensitivityDbm;
        this.laserTrackingSetupTimeS = laserTrackingSetupTimeS;
    }

    // Getters
    public String getSatAId() { return satAId; }
    public String getSatBId() { return satBId; }
    public String getLinkType() { return linkType; }
    public double getMaxRangeKm() { return maxRangeKm; }
    public double getLaserAcquisitionTimeS() { return laserAcquisitionTimeS; }
    public double getLaserTrackingAccuracyUrad() { return laserTrackingAccuracyUrad; }
    public double getMicrowaveScanAngleDeg() { return microwaveScanAngleDeg; }
    public double getLaserReceiverSensitivityDbm() { return laserReceiverSensitivityDbm; }
    public double getLaserTrackingSetupTimeS() { return laserTrackingSetupTimeS; }
}
