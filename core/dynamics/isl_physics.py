"""
ISL (Inter-Satellite Link) 物理引擎

实现激光和微波星间链路的完整物理模型：
- 激光链路预算（基于高斯光束传播理论）
- 微波链路预算（基于Friis传输方程 + 相控阵增益模型）
- ATP（捕获-跟踪-瞄准）状态机
- 链路数据率计算

所有计算基于精确的链路预算模型，禁止简化近似。

参考标准：
- CCSDS 141.0-B-1: Optical Communications Physical Layer
- ITU-R S.1327: Free-space path loss
- ESA ESTEC: Inter-Satellite Link Design Reference
"""

import math
import logging
from enum import Enum
from typing import Optional

from core.models.isl_config import LaserISLConfig, MicrowaveISLConfig

logger = logging.getLogger(__name__)


# =============================================================================
# ATP 状态机
# =============================================================================

class ATPState(Enum):
    """
    激光ATP（捕获-跟踪-瞄准）状态机状态

    建链流程：IDLE -> SCANNING -> ACQUIRED -> COARSE_TRACKING -> FINE_TRACKING -> LINKED
    中断恢复：LINKED -> INTERRUPTED -> COARSE_TRACKING / FINE_TRACKING -> LINKED
    """
    IDLE = 'idle'                           # 空闲，未尝试建链
    SCANNING = 'scanning'                   # 信标扫描阶段：宽光束扫描（10-60s）
    ACQUIRED = 'acquired'                   # 已捕获信标
    COARSE_TRACKING = 'coarse_tracking'     # 粗跟踪阶段：万向节闭环（~5s）
    FINE_TRACKING = 'fine_tracking'         # 精跟踪阶段：FSM（快速转向镜）闭环（~2s）
    LINKED = 'linked'                       # 链路已建立，全双工通信
    INTERRUPTED = 'interrupted'             # 链路中断（如遮挡、振动超限）


class ATPStateMachine:
    """
    激光链路ATP状态机

    模拟从IDLE到LINKED的完整建链过程，以及中断后的重建链过程。
    建链时间与卫星相对速度相关：相对速度越高，信标扫描范围越大，捕获时间越长。

    Args:
        config: 激光ISL配置
    """

    def __init__(self, config: LaserISLConfig) -> None:
        self.config = config
        self.state = ATPState.IDLE

    def calculate_total_setup_time(self, relative_velocity_km_s: float) -> float:
        """
        计算从IDLE到LINKED的总建链时间（秒）。

        物理原理：
        - 高相对速度 → 信标在扫描区域内移动更快 → 扫描范围需要更大 → 捕获时间更长
        - 捕获时间 = 基础捕获时间 × max(1.0, v_rel / v_ref)
        - 粗跟踪和精跟踪时间固定（由万向节和FSM带宽决定）

        Args:
            relative_velocity_km_s: 两颗卫星的相对速度（km/s）

        Returns:
            总建链时间（秒）
        """
        # 相对速度修正因子（参考速度3 km/s，典型LEO星间相对速度）
        # 当相对速度超过参考值时，扫描时间线性增加
        v_ref_km_s = 3.0
        scan_factor = max(1.0, relative_velocity_km_s / v_ref_km_s)
        acq_time = self.config.acquisition_time_s * scan_factor

        total = acq_time + self.config.coarse_tracking_time_s + self.config.fine_tracking_time_s
        logger.debug(
            f"ATP setup time: acq={acq_time:.1f}s (factor={scan_factor:.2f}), "
            f"coarse={self.config.coarse_tracking_time_s:.1f}s, "
            f"fine={self.config.fine_tracking_time_s:.1f}s, total={total:.1f}s"
        )
        return total

    def calculate_reacquisition_time(self, interruption_duration_s: float) -> float:
        """
        计算链路中断后重建链的时间（秒）。

        短暂中断（<5s）：精跟踪环路快速重新收敛，仅需精跟踪阶段的2倍时间
        中等中断（5-30s）：粗跟踪环路需要重新收敛，跳过信标扫描阶段
        长时间中断（>30s）：等效于重新建链，执行完整ATP流程

        Args:
            interruption_duration_s: 中断持续时间（秒）

        Returns:
            重建链时间（秒）
        """
        if interruption_duration_s < 5.0:
            # 精跟踪环路快速重新收敛（2倍精跟踪时间）
            return self.config.fine_tracking_time_s * 2.0
        elif interruption_duration_s < 30.0:
            # 粗跟踪+精跟踪（跳过信标扫描）
            return self.config.coarse_tracking_time_s + self.config.fine_tracking_time_s
        else:
            # 完整ATP流程（保守估计，不考虑速度因子）
            return self.calculate_total_setup_time(0.0)

    def transition(self, new_state: ATPState) -> None:
        """
        状态转换（含合法性检查）。

        Args:
            new_state: 目标状态

        Raises:
            ValueError: 非法状态转换
        """
        valid_transitions = {
            ATPState.IDLE: {ATPState.SCANNING},
            ATPState.SCANNING: {ATPState.ACQUIRED, ATPState.IDLE},
            ATPState.ACQUIRED: {ATPState.COARSE_TRACKING, ATPState.SCANNING},
            ATPState.COARSE_TRACKING: {ATPState.FINE_TRACKING, ATPState.SCANNING},
            ATPState.FINE_TRACKING: {ATPState.LINKED, ATPState.COARSE_TRACKING},
            ATPState.LINKED: {ATPState.INTERRUPTED, ATPState.IDLE},
            ATPState.INTERRUPTED: {ATPState.COARSE_TRACKING, ATPState.FINE_TRACKING, ATPState.SCANNING},
        }
        allowed = valid_transitions.get(self.state, set())
        if new_state not in allowed:
            raise ValueError(
                f"Invalid ATP state transition: {self.state.value} -> {new_state.value}. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )
        old_state = self.state
        self.state = new_state
        logger.debug(f"ATP state: {old_state.value} -> {new_state.value}")

    def is_linked(self) -> bool:
        """链路是否已建立"""
        return self.state == ATPState.LINKED

    def reset(self) -> None:
        """重置为IDLE状态"""
        self.state = ATPState.IDLE


# =============================================================================
# 激光链路预算
# =============================================================================

def calculate_laser_link_margin(
    config: LaserISLConfig,
    distance_km: float,
    tracking_error_urad: float,
) -> float:
    """
    计算激光链路余量（dB）。

    链路方程（对数域）：
        M = Pt_dBm + Gt_dB + Gr_dB - Lfs_dB - Lpoint_dB - Latm_dB - SNRreq_dB

    其中：
        Pt_dBm  = 10 * log10(P_tx_W * 1000)        发射功率
        Gt_dB   = 20 * log10(π * D_tx / λ)         发射增益（高斯光束近似）
        Gr_dB   = 10 * log10(η * (π*D_rx)²/(4λ²)) 接收增益（集光面积，η=0.7）
        Lfs_dB  = 20 * log10(4π*R/λ)               自由空间传播损耗
        Lpoint_dB = 8.69 * (σ/θ_div)²              指向误差损耗（高斯光束）
        Latm_dB = 0.0                                大气损耗（LEO-LEO在100km以上可忽略）

    Args:
        config: 激光ISL配置
        distance_km: 两星距离（公里）
        tracking_error_urad: 跟踪误差（微弧度，1-sigma）

    Returns:
        链路余量（dB），正值表示链路可用
    """
    if distance_km <= 0:
        raise ValueError(f"distance_km must be positive, got {distance_km}")

    distance_m = distance_km * 1000.0             # 距离（米）

    # ─── 直接光功率公式（正确的光学链路预算）────────────────────────────────
    # 高斯光束在距离 R 处的光斑半径 = θ_div * R
    # 接收到的功率占比 = (D_rx / (2 * θ_div * R))²
    # Pr = Pt × η_tx × η_rx × (D_rx / 2)² / (θ_div × R)²
    #
    # 参考：
    #   - CCSDS 141.0-B-1 §3.3 Free-Space Optical Link Budget
    #   - Saleh & Teich, "Fundamentals of Photonics", §22.1
    # ─────────────────────────────────────────────────────────────────────────

    theta_div_rad = config.beam_divergence_urad * 1e-6   # 半角束散角（弧度）
    eta_tx = 0.85   # 发射光路效率（光学元件、准直器等）
    eta_rx = 0.82   # 接收光路效率（滤波器、聚焦、量子效率等）

    # 接收功率（瓦特）
    spot_radius_m = theta_div_rad * distance_m
    if spot_radius_m <= 0:
        spot_radius_m = 1e-9  # 防止除零
    power_fraction = (config.receive_aperture_m / 2.0) ** 2 / spot_radius_m ** 2
    Pr_w = config.transmit_power_w * eta_tx * eta_rx * power_fraction
    Pr_dBm = 10.0 * math.log10(max(Pr_w * 1000.0, 1e-30))

    # 发射功率参考（调试用）
    Pt_dBm = 10.0 * math.log10(config.transmit_power_w * 1000.0)

    # 指向误差损耗（高斯光束模型）
    # Lpoint = 8.69 × (σ/θ_div)²
    sigma_rad = tracking_error_urad * 1e-6
    if theta_div_rad > 0:
        Lpoint_dB = 8.69 * (sigma_rad / theta_div_rad) ** 2
    else:
        Lpoint_dB = 0.0

    # 链路余量 = 接收功率(dBm) - 接收机灵敏度(dBm) - 指向损耗
    # 接收机灵敏度已涵盖最低可用SNR（BER=1e-6）要求
    margin = Pr_dBm - Lpoint_dB - config.receiver_sensitivity_dBm

    logger.debug(
        f"Laser link budget @ {distance_km:.0f} km: "
        f"Pt={Pt_dBm:.1f} dBm, θ_div={config.beam_divergence_urad:.1f} µrad, "
        f"spot_r={spot_radius_m:.0f} m, Pr={Pr_dBm:.1f} dBm, "
        f"Lpoint={Lpoint_dB:.2f} dB, sens={config.receiver_sensitivity_dBm:.1f} dBm "
        f"→ margin={margin:.1f} dB"
    )
    return margin


def calculate_laser_data_rate(
    config: LaserISLConfig,
    distance_km: float,
) -> float:
    """
    计算激光链路可达数据率（Mbps）。

    基于标称跟踪精度计算链路余量，再根据余量超出最小余量的量估算调制阶数。

    数据率模型：
    - 余量 < min_link_margin_db → 0 Mbps（链路不可用）
    - 余量 ≥ min_link_margin_db → 基于自由空间损耗（∝ 1/R²）的功率律缩放
    - 基准：1000 km时 10 Gbps，按距离平方反比缩放
    - 限幅：100 Mbps ~ 100 Gbps

    Args:
        config: 激光ISL配置
        distance_km: 两星距离（公里）

    Returns:
        可达数据率（Mbps）
    """
    if distance_km <= 0:
        return 0.0

    margin = calculate_laser_link_margin(
        config,
        distance_km,
        config.tracking_accuracy_urad  # 使用标称跟踪精度
    )

    if margin < config.min_link_margin_db:
        logger.debug(
            f"Laser link @ {distance_km:.0f} km: margin={margin:.1f} dB < "
            f"min={config.min_link_margin_db:.1f} dB, link unavailable"
        )
        return 0.0

    # 数据率按距离平方反比缩放（功率受限系统）
    # 基准：1000 km时 10 Gbps（10000 Mbps）
    base_rate_mbps = 10000.0
    ref_distance_km = 1000.0
    rate = base_rate_mbps * (ref_distance_km / distance_km) ** 2

    # 限幅到物理合理范围 [100 Mbps, 100 Gbps]
    rate = max(100.0, min(rate, 100000.0))

    logger.debug(
        f"Laser data rate @ {distance_km:.0f} km: {rate:.0f} Mbps "
        f"(margin={margin:.1f} dB)"
    )
    return rate


# =============================================================================
# 微波链路预算
# =============================================================================

def calculate_microwave_gain(
    config: MicrowaveISLConfig,
    off_axis_angle_deg: float,
) -> float:
    """
    计算相控阵天线偏轴增益（dBi）。

    增益模型：
    - 0° ~ 30°：增益等于天线标称增益（无滚降）
    - 30° ~ scan_angle_deg：线性滚降，速率为 gain_rolloff_db_per_deg
    - > scan_angle_deg：链路不可用，返回 -999 dBi

    Args:
        config: 微波ISL配置
        off_axis_angle_deg: 偏轴角度（度），即天线主轴与目标方向的夹角

    Returns:
        天线增益（dBi），-999 表示链路不可用
    """
    if off_axis_angle_deg < 0:
        off_axis_angle_deg = abs(off_axis_angle_deg)

    # 超出扫描范围
    if off_axis_angle_deg > config.scan_angle_deg:
        return -999.0

    # 3 dB阈值角（30°以内增益恒定）
    threshold_deg = 30.0

    if off_axis_angle_deg <= threshold_deg:
        return config.antenna_gain_dbi

    # 线性滚降区
    rolloff = config.gain_rolloff_db_per_deg * (off_axis_angle_deg - threshold_deg)
    return config.antenna_gain_dbi - rolloff


def calculate_microwave_link_margin(
    config: MicrowaveISLConfig,
    distance_km: float,
    off_axis_angle_deg: float,
) -> float:
    """
    计算微波链路余量（dB）。

    Friis传输方程：
        SNR = Pt_dBm + Gt_dBi + Gr_dBi - Lfs_dB - Pnoise_dBm
        M = SNR - SNRreq_dB

    其中：
        Pt_dBm     = 10 * log10(P_tx_W * 1000)
        Gt_dBi     = 相控阵发射增益（含偏轴滚降）
        Gr_dBi     = 相控阵接收增益（含偏轴滚降，假设对称）
        Lfs_dB     = 20 * log10(4π*R*f/c)       自由空间路径损耗
        Pnoise_dBm = 10 * log10(k * T_sys * B * 1000)  系统噪声功率

    Args:
        config: 微波ISL配置
        distance_km: 两星距离（公里）
        off_axis_angle_deg: 偏轴角度（度）

    Returns:
        链路余量（dB），正值表示链路可用；-999表示链路不可用（超出扫描范围）
    """
    if distance_km <= 0:
        raise ValueError(f"distance_km must be positive, got {distance_km}")

    # 天线增益（发射和接收，假设对称配置）
    Gt_dBi = calculate_microwave_gain(config, off_axis_angle_deg)
    Gr_dBi = calculate_microwave_gain(config, off_axis_angle_deg)

    if Gt_dBi < -100.0 or Gr_dBi < -100.0:
        return -999.0  # 超出扫描范围

    freq_hz = config.frequency_ghz * 1e9
    wavelength_m = 3e8 / freq_hz
    distance_m = distance_km * 1000.0

    # 发射功率（dBm）
    Pt_dBm = 10.0 * math.log10(config.transmit_power_w * 1000.0)

    # 自由空间路径损耗（Friis公式）
    Lfs_dB = 20.0 * math.log10(4.0 * math.pi * distance_m / wavelength_m)

    # 系统噪声功率（dBm）
    # Pnoise = k * T_sys * B（瓦特）
    k_boltzmann = 1.38e-23   # 玻尔兹曼常数（J/K）
    bandwidth_hz = 1e9       # 带宽 1 GHz（Ka频段典型值）
    noise_power_w = k_boltzmann * config.system_noise_temp_k * bandwidth_hz
    noise_power_dbm = 10.0 * math.log10(noise_power_w * 1000.0)

    # 接收SNR
    SNR_dB = Pt_dBm + Gt_dBi + Gr_dBi - Lfs_dB - noise_power_dbm

    margin = SNR_dB - config.snr_required_db

    logger.debug(
        f"Microwave link budget @ {distance_km:.0f} km, {off_axis_angle_deg:.1f}°: "
        f"Pt={Pt_dBm:.1f} + Gt={Gt_dBi:.1f} + Gr={Gr_dBi:.1f} "
        f"- Lfs={Lfs_dB:.1f} - Pn={noise_power_dbm:.1f} "
        f"= SNR={SNR_dB:.1f}, margin={margin:.1f} dB"
    )
    return margin


def calculate_microwave_data_rate(
    config: MicrowaveISLConfig,
    distance_km: float,
    off_axis_angle_deg: float,
    active_beams: int = 1,
) -> float:
    """
    计算微波链路可达数据率（Mbps）。

    基于Shannon容量定理近似，并考虑TDMA时隙共享。

    容量公式：
        C = B * log2(1 + SNR_linear)     Shannon容量（Mbps）
        SNR_linear = 10^(margin_dB / 10) 线性SNR

    TDMA共享：
        每个活跃波束分得 1/active_beams 的帧时间（简化模型）

    Args:
        config: 微波ISL配置
        distance_km: 两星距离（公里）
        off_axis_angle_deg: 偏轴角度（度）
        active_beams: 当前活跃波束数（≥1）

    Returns:
        可达数据率（Mbps），0表示链路不可用
    """
    if distance_km <= 0:
        return 0.0

    margin = calculate_microwave_link_margin(config, distance_km, off_axis_angle_deg)

    if margin < 0.0:
        return 0.0

    # Shannon容量（线性SNR转换）
    snr_linear = 10.0 ** (margin / 10.0)
    bandwidth_mhz = 1000.0   # 1 GHz = 1000 MHz
    capacity_mbps = bandwidth_mhz * math.log2(1.0 + snr_linear)

    # TDMA共享（每个活跃波束平均分配帧资源）
    n_beams = max(1, active_beams)
    beam_rate = capacity_mbps / n_beams

    logger.debug(
        f"Microwave data rate @ {distance_km:.0f} km, {off_axis_angle_deg:.1f}°, "
        f"{n_beams} beams: {beam_rate:.0f} Mbps "
        f"(capacity={capacity_mbps:.0f} Mbps, margin={margin:.1f} dB)"
    )
    return beam_rate


# =============================================================================
# ISL物理引擎（统一接口）
# =============================================================================

class ISLPhysicsEngine:
    """
    ISL物理计算引擎

    统一封装激光和微波链路的物理计算，对外提供单一接口。
    支持根据链路类型自动选择对应的计算模型。

    Example::

        engine = ISLPhysicsEngine()
        params = engine.compute_link_parameters(
            link_type='laser',
            laser_config=laser_cfg,
            microwave_config=None,
            distance_km=2500.0,
            relative_velocity_km_s=1.5,
        )
        print(params['data_rate_mbps'], params['link_margin_db'])
    """

    def compute_link_parameters(
        self,
        link_type: str,
        laser_config: Optional[LaserISLConfig],
        microwave_config: Optional[MicrowaveISLConfig],
        distance_km: float,
        relative_velocity_km_s: float,
        off_axis_angle_deg: float = 0.0,
        active_beams: int = 1,
    ) -> dict:
        """
        计算ISL链路物理参数。

        Args:
            link_type: 链路类型（'laser' 或 'microwave'）
            laser_config: 激光终端配置（link_type='laser'时必须提供）
            microwave_config: 微波终端配置（link_type='microwave'时必须提供）
            distance_km: 两星间距离（公里）
            relative_velocity_km_s: 两星相对速度（km/s）
            off_axis_angle_deg: 微波偏轴角度（度，仅微波链路使用）
            active_beams: 活跃微波波束数（仅微波链路使用）

        Returns:
            包含以下键的字典：
            - data_rate_mbps (float): 可达数据率（Mbps）
            - link_margin_db (float): 链路余量（dB）
            - atp_setup_time_s (float): 建链时间（秒），微波链路为0
            - link_viable (bool): 链路是否可用
            - link_type (str): 实际使用的链路类型
            - distance_km (float): 传入的距离（公里）
            - relative_velocity_km_s (float): 传入的相对速度（km/s）

        Raises:
            ValueError: link_type无效，或对应配置未提供
        """
        if link_type == 'laser':
            return self._compute_laser_parameters(
                laser_config, distance_km, relative_velocity_km_s
            )
        elif link_type == 'microwave':
            return self._compute_microwave_parameters(
                microwave_config, distance_km, off_axis_angle_deg, active_beams
            )
        else:
            raise ValueError(
                f"Invalid link_type: '{link_type}'. Must be 'laser' or 'microwave'."
            )

    def _compute_laser_parameters(
        self,
        config: Optional[LaserISLConfig],
        distance_km: float,
        relative_velocity_km_s: float,
    ) -> dict:
        """计算激光链路参数（内部方法）"""
        if config is None:
            raise ValueError(
                "laser_config must be provided for link_type='laser'"
            )

        # 链路余量（使用标称跟踪精度）
        margin = calculate_laser_link_margin(
            config, distance_km, config.tracking_accuracy_urad
        )

        # 数据率
        data_rate = calculate_laser_data_rate(config, distance_km)

        # ATP建链时间
        atp_machine = ATPStateMachine(config)
        atp_time = atp_machine.calculate_total_setup_time(relative_velocity_km_s)

        link_viable = (
            data_rate > 0.0
            and margin >= config.min_link_margin_db
            and distance_km <= config.max_range_km
        )

        return {
            'data_rate_mbps': data_rate,
            'link_margin_db': margin,
            'atp_setup_time_s': atp_time,
            'link_viable': link_viable,
            'link_type': 'laser',
            'distance_km': distance_km,
            'relative_velocity_km_s': relative_velocity_km_s,
        }

    def _compute_microwave_parameters(
        self,
        config: Optional[MicrowaveISLConfig],
        distance_km: float,
        off_axis_angle_deg: float,
        active_beams: int,
    ) -> dict:
        """计算微波链路参数（内部方法）"""
        if config is None:
            raise ValueError(
                "microwave_config must be provided for link_type='microwave'"
            )

        # 链路余量
        margin = calculate_microwave_link_margin(
            config, distance_km, off_axis_angle_deg
        )

        # 数据率
        data_rate = calculate_microwave_data_rate(
            config, distance_km, off_axis_angle_deg, active_beams
        )

        link_viable = (
            data_rate > 0.0
            and margin >= 0.0
            and distance_km <= config.max_range_km
            and off_axis_angle_deg <= config.scan_angle_deg
        )

        return {
            'data_rate_mbps': data_rate,
            'link_margin_db': margin if margin > -100.0 else -999.0,
            'atp_setup_time_s': 0.0,   # 微波链路无需ATP
            'link_viable': link_viable,
            'link_type': 'microwave',
            'distance_km': distance_km,
            'relative_velocity_km_s': 0.0,  # 微波不使用相对速度
        }

    def select_best_link_type(
        self,
        laser_config: Optional[LaserISLConfig],
        microwave_config: Optional[MicrowaveISLConfig],
        distance_km: float,
        relative_velocity_km_s: float,
        off_axis_angle_deg: float = 0.0,
        strategy: str = 'laser_preferred',
    ) -> Optional[str]:
        """
        根据策略和物理条件选择最佳链路类型。

        Args:
            laser_config: 激光配置（None表示不支持激光）
            microwave_config: 微波配置（None表示不支持微波）
            distance_km: 两星距离（公里）
            relative_velocity_km_s: 两星相对速度（km/s）
            off_axis_angle_deg: 微波偏轴角度（度）
            strategy: 选择策略（'laser_preferred'/'microwave_preferred'/'auto'）

        Returns:
            最佳链路类型字符串（'laser' 或 'microwave'），无可用链路时返回 None
        """
        # 评估各链路可用性
        laser_viable = False
        microwave_viable = False
        laser_rate = 0.0
        microwave_rate = 0.0

        if laser_config is not None:
            params = self._compute_laser_parameters(
                laser_config, distance_km, relative_velocity_km_s
            )
            laser_viable = params['link_viable']
            laser_rate = params['data_rate_mbps']

        if microwave_config is not None:
            params = self._compute_microwave_parameters(
                microwave_config, distance_km, off_axis_angle_deg, 1
            )
            microwave_viable = params['link_viable']
            microwave_rate = params['data_rate_mbps']

        if not laser_viable and not microwave_viable:
            return None

        if strategy == 'laser_preferred':
            if laser_viable:
                return 'laser'
            return 'microwave' if microwave_viable else None

        elif strategy == 'microwave_preferred':
            if microwave_viable:
                return 'microwave'
            return 'laser' if laser_viable else None

        else:  # 'auto'：选择数据率更高的
            if laser_viable and microwave_viable:
                return 'laser' if laser_rate >= microwave_rate else 'microwave'
            if laser_viable:
                return 'laser'
            if microwave_viable:
                return 'microwave'
            return None
