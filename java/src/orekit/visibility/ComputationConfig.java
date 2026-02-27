package orekit.visibility;

import java.io.Serializable;

/**
 * 计算配置
 *
 * 控制批量计算的参数
 */
public class ComputationConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    private double coarseStep = 300.0;      // 粗扫描步长（秒）
    private double fineStep = 60.0;         // 精化步长（秒）
    private double minElevation = 0.0;      // 最小仰角（度）
    private boolean useParallel = true;     // 是否并行传播
    private int maxBatchSize = 100;         // 最大批次大小

    public ComputationConfig() {
    }

    // Getters and Setters
    public double getCoarseStep() {
        return coarseStep;
    }

    public void setCoarseStep(double coarseStep) {
        this.coarseStep = coarseStep;
    }

    public double getFineStep() {
        return fineStep;
    }

    public void setFineStep(double fineStep) {
        this.fineStep = fineStep;
    }

    public double getMinElevation() {
        return minElevation;
    }

    public void setMinElevation(double minElevation) {
        this.minElevation = minElevation;
    }

    public boolean isUseParallel() {
        return useParallel;
    }

    public void setUseParallel(boolean useParallel) {
        this.useParallel = useParallel;
    }

    public int getMaxBatchSize() {
        return maxBatchSize;
    }

    public void setMaxBatchSize(int maxBatchSize) {
        this.maxBatchSize = maxBatchSize;
    }

    @Override
    public String toString() {
        return "ComputationConfig{" +
                "coarseStep=" + coarseStep +
                ", fineStep=" + fineStep +
                ", minElevation=" + minElevation +
                ", useParallel=" + useParallel +
                ", maxBatchSize=" + maxBatchSize +
                '}';
    }
}
