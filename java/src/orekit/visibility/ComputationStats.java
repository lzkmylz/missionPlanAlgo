package orekit.visibility;

import java.io.Serializable;

/**
 * 计算统计信息
 *
 * 记录批量计算的性能指标
 */
public class ComputationStats implements Serializable {

    private static final long serialVersionUID = 1L;

    private long computationTimeMs;
    private int nSatellites;
    private int nTargets;
    private int nGroundStations;
    private int nWindowsFound;
    private long memoryUsageMb;

    public ComputationStats(long computationTimeMs, int nSatellites,
                           int nTargets, int nGroundStations,
                           int nWindowsFound, long memoryUsageMb) {
        this.computationTimeMs = computationTimeMs;
        this.nSatellites = nSatellites;
        this.nTargets = nTargets;
        this.nGroundStations = nGroundStations;
        this.nWindowsFound = nWindowsFound;
        this.memoryUsageMb = memoryUsageMb;
    }

    // Getters
    public long getComputationTimeMs() {
        return computationTimeMs;
    }

    public int getNSatellites() {
        return nSatellites;
    }

    public int getNTargets() {
        return nTargets;
    }

    public int getNGroundStations() {
        return nGroundStations;
    }

    public int getNWindowsFound() {
        return nWindowsFound;
    }

    public long getMemoryUsageMb() {
        return memoryUsageMb;
    }

    @Override
    public String toString() {
        return String.format(
            "ComputationStats{time=%dms, sats=%d, targets=%d, gs=%d, windows=%d, mem=%dMB}",
            computationTimeMs, nSatellites, nTargets, nGroundStations,
            nWindowsFound, memoryUsageMb
        );
    }
}
