package orekit.helper;

import org.orekit.propagation.sampling.OrekitFixedStepHandler;
import org.orekit.propagation.SpacecraftState;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.PVCoordinates;
import java.util.ArrayList;

/**
 * BatchStepHandler - 批量轨道状态收集器
 *
 * 实现OrekitFixedStepHandler接口，在Java端收集所有轨道状态数据，
 * 传播结束后一次性返回double[][]数组，通过JPype零拷贝映射为numpy数组。
 *
 * 性能优化：将86,400次JNI调用降为1次，大幅提升批量传播性能。
 *
 * 数据格式：每行7列 [seconds_since_j2000, px, py, pz, vx, vy, vz]
 * - seconds_since_j2000: 距J2000历元的秒数
 * - px, py, pz: 位置坐标 (m)
 * - vx, vy, vz: 速度矢量 (m/s)
 */
public class BatchStepHandler implements OrekitFixedStepHandler {
    private final ArrayList<double[]> data = new ArrayList<>();

    /**
     * 处理每一步的轨道状态
     *
     * @param currentState 当前轨道状态
     * @param isLast 是否为最后一步
     */
    @Override
    public void handleStep(SpacecraftState currentState) {
        AbsoluteDate date = currentState.getDate();
        PVCoordinates pv = currentState.getPVCoordinates();

        // [seconds_since_epoch, px, py, pz, vx, vy, vz]
        double[] row = new double[7];
        row[0] = date.durationFrom(AbsoluteDate.J2000_EPOCH);
        row[1] = pv.getPosition().getX();
        row[2] = pv.getPosition().getY();
        row[3] = pv.getPosition().getZ();
        row[4] = pv.getVelocity().getX();
        row[5] = pv.getVelocity().getY();
        row[6] = pv.getVelocity().getZ();

        data.add(row);
    }

    /**
     * 获取所有收集的轨道状态数据
     *
     * @return double[][] 二维数组，每行包含 [seconds_since_j2000, px, py, pz, vx, vy, vz]
     */
    public double[][] getResults() {
        return data.toArray(new double[0][]);
    }

    /**
     * 获取收集的数据点数量
     *
     * @return int 数据点数量
     */
    public int getCount() {
        return data.size();
    }

    /**
     * 清空已收集的数据
     */
    public void clear() {
        data.clear();
    }
}
