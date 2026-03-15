package orekit.visibility;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.utils.Constants;

/**
 * 姿态计算工具类
 *
 * 计算卫星成像所需的姿态角（滚转角、俯仰角、偏航角）。
 * 使用LVLH（Local Vertical Local Horizontal）坐标系。
 *
 * 算法与Python端保持一致：
 * 1. 构建LVLH坐标系（Z轴指向地心，X轴沿飞行方向，Y轴完成右手系）
 * 2. 计算目标视线向量
 * 3. 将视线向量转换到LVLH坐标系
 * 4. 计算滚转和俯仰角
 */
public class AttitudeCalculator {

    // 地球半径（米）
    private static final double EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS;

    // 默认容差系数（1.1表示10%容差）
    private static final double DEFAULT_TOLERANCE = 1.1;

    /**
     * 姿态角结果
     */
    public static class AttitudeAngles {
        public final double roll;   // 滚转角（度）
        public final double pitch;  // 俯仰角（度）
        public final double yaw;    // 偏航角（度，始终为0）

        public AttitudeAngles(double roll, double pitch, double yaw) {
            this.roll = roll;
            this.pitch = pitch;
            this.yaw = yaw;
        }

        @Override
        public String toString() {
            return String.format("AttitudeAngles[roll=%.2f°, pitch=%.2f°, yaw=%.2f°]", roll, pitch, yaw);
        }
    }

    /**
     * 计算姿态角
     *
     * @param satPosition 卫星位置（ECEF，米）
     * @param satVelocity 卫星速度（ECEF，m/s）
     * @param targetLat 目标纬度（度）
     * @param targetLon 目标经度（度）
     * @param targetAlt 目标高度（米）
     * @return 姿态角（roll, pitch, yaw，单位：度）
     */
    public static AttitudeAngles calculateAttitude(
            double[] satPosition,
            double[] satVelocity,
            double targetLat,
            double targetLon,
            double targetAlt) {

        // 输入验证
        if (satPosition == null || satPosition.length != 3) {
            throw new IllegalArgumentException("satPosition must be non-null array of length 3");
        }
        if (satVelocity == null || satVelocity.length != 3) {
            throw new IllegalArgumentException("satVelocity must be non-null array of length 3");
        }

        // 1. 构建LVLH坐标系
        LVLHFrame lvlh = constructLVLHFrame(
            new Vector3D(satPosition[0], satPosition[1], satPosition[2]),
            new Vector3D(satVelocity[0], satVelocity[1], satVelocity[2])
        );

        // 2. 计算目标在ECEF坐标系中的位置
        Vector3D targetECEF = geodeticToECEF(targetLat, targetLon, targetAlt);

        // 3. 计算视线向量（从卫星指向目标）
        Vector3D satPos = new Vector3D(satPosition[0], satPosition[1], satPosition[2]);
        Vector3D losVector = targetECEF.subtract(satPos);

        // 4. 将视线向量转换到LVLH坐标系
        Vector3D losInLVLH = transformToLVLH(losVector, lvlh);

        // 5. 计算姿态角
        return calculateRollPitchYaw(losInLVLH);
    }

    /**
     * 简化的姿态计算（默认高度为0）
     */
    public static AttitudeAngles calculateAttitude(
            double[] satPosition,
            double[] satVelocity,
            double targetLat,
            double targetLon) {
        return calculateAttitude(satPosition, satVelocity, targetLat, targetLon, 0.0);
    }

    /**
     * LVLH坐标系
     */
    private static class LVLHFrame {
        final Vector3D X;  // 沿飞行方向（单位向量）
        final Vector3D Y;  // 轨道面法向（单位向量）
        final Vector3D Z;  // 指向地心（单位向量）

        LVLHFrame(Vector3D X, Vector3D Y, Vector3D Z) {
            this.X = X;
            this.Y = Y;
            this.Z = Z;
        }
    }

    /**
     * 构建LVLH坐标系
     *
     * Z轴：指向地心（负位置方向）
     * X轴：沿飞行方向（速度在垂直于Z方向的分量）
     * Y轴：Z × X（完成右手系）
     */
    private static LVLHFrame constructLVLHFrame(Vector3D position, Vector3D velocity) {
        // Z轴：指向地心（负位置方向，单位化）
        Vector3D Z = position.normalize().negate();

        // X轴：沿飞行方向（速度在垂直于Z方向的分量）
        // v_parallel = v - (v·Z) * Z
        double vDotZ = velocity.dotProduct(Z);
        Vector3D vParallel = velocity.subtract(Z.scalarMultiply(vDotZ));
        Vector3D X = vParallel.normalize();

        // Y轴：Z × X（完成右手系）
        Vector3D Y = Z.crossProduct(X).normalize();

        return new LVLHFrame(X, Y, Z);
    }

    /**
     * 将地心坐标转换为ECEF坐标
     */
    private static Vector3D geodeticToECEF(double lat, double lon, double alt) {
        // 使用WGS84椭球参数
        double a = Constants.WGS84_EARTH_EQUATORIAL_RADIUS;  // 长半轴
        double f = Constants.WGS84_EARTH_FLATTENING;         // 扁率
        double e2 = 2 * f - f * f;                           // 第一偏心率的平方

        double latRad = Math.toRadians(lat);
        double lonRad = Math.toRadians(lon);

        double sinLat = Math.sin(latRad);
        double cosLat = Math.cos(latRad);
        double sinLon = Math.sin(lonRad);
        double cosLon = Math.cos(lonRad);

        // 卯酉圈曲率半径
        double N = a / Math.sqrt(1 - e2 * sinLat * sinLat);

        // ECEF坐标
        double x = (N + alt) * cosLat * cosLon;
        double y = (N + alt) * cosLat * sinLon;
        double z = (N * (1 - e2) + alt) * sinLat;

        return new Vector3D(x, y, z);
    }

    /**
     * 将向量从ECEF转换到LVLH坐标系
     */
    private static Vector3D transformToLVLH(Vector3D vector, LVLHFrame lvlh) {
        // 在LVLH坐标系中的分量 = 向量与各轴的点积
        double x = vector.dotProduct(lvlh.X);
        double y = vector.dotProduct(lvlh.Y);
        double z = vector.dotProduct(lvlh.Z);

        return new Vector3D(x, y, z);
    }

    /**
     * 计算滚转、俯仰和偏航角
     *
     * roll = atan2(y, z)   // 绕X轴旋转
     * pitch = atan2(x, z)  // 绕Y轴旋转
     * yaw = 0               // 零偏航模式
     *
     * 注意：使用+z因为视线向量指向地心（Z轴方向）
     */
    private static AttitudeAngles calculateRollPitchYaw(Vector3D losInLVLH) {
        double x = losInLVLH.getX();
        double y = losInLVLH.getY();
        double z = losInLVLH.getZ();

        // 滚转角：atan2(y, z)
        double roll = Math.toDegrees(Math.atan2(y, z));

        // 俯仰角：atan2(x, z)
        double pitch = Math.toDegrees(Math.atan2(x, z));

        // 偏航角：0（零偏航模式）
        double yaw = 0.0;

        return new AttitudeAngles(roll, pitch, yaw);
    }

    /**
     * 检查姿态约束
     *
     * @param roll 滚转角（度）
     * @param pitch 俯仰角（度）
     * @param maxRoll 最大滚转角（度）
     * @param maxPitch 最大俯仰角（度）
     * @param tolerance 容差系数（如1.1表示10%容差）
     * @return 是否满足约束
     */
    public static boolean checkAttitudeConstraints(
            double roll, double pitch,
            double maxRoll, double maxPitch,
            double tolerance) {
        return Math.abs(roll) <= maxRoll * tolerance &&
               Math.abs(pitch) <= maxPitch * tolerance;
    }

    /**
     * 检查姿态约束（默认10%容差）
     */
    public static boolean checkAttitudeConstraints(
            double roll, double pitch,
            double maxRoll, double maxPitch) {
        return checkAttitudeConstraints(roll, pitch, maxRoll, maxPitch, DEFAULT_TOLERANCE);
    }
}
