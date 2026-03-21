"""
ISL (Inter-Satellite Link) 配置模块

定义激光和微波星间链路的配置参数、物理模型参数及能力配置。

支持链路类型：
- 激光ISL: 1550nm波段，高速率（10 Gbps+），需要ATP（捕获、跟踪、瞄准）过程
- 微波ISL: 26 GHz Ka频段，相控阵天线，速率适中（1 Gbps），无需ATP

关键设计：
- ISL可见性窗口数据必须来自Java后端预计算（强制）
- 窗口键格式：{satA_id}_ISL:{satB_id}，目标字段tgt以"ISL:"开头
- 禁止Python几何回退计算
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ISLLinkType(Enum):
    """ISL链路类型"""
    LASER = 'laser'
    MICROWAVE = 'microwave'


class ISLLinkSelectionStrategy(Enum):
    """链路选择策略"""
    LASER_PREFERRED = 'laser_preferred'       # 优先选择激光链路
    MICROWAVE_PREFERRED = 'microwave_preferred'  # 优先选择微波链路
    AUTO = 'auto'                              # 根据距离和角度自动选择


@dataclass
class LaserISLConfig:
    """
    激光星间链路配置

    基于1550nm波段激光通信，需要精密的ATP（Acquisition-Tracking-Pointing）过程。

    Attributes:
        wavelength_nm: 激光波长（纳米），默认1550 nm（电信C波段）
        transmit_power_w: 发射功率（瓦特）
        transmit_aperture_m: 发射望远镜口径（米）
        receive_aperture_m: 接收望远镜口径（米）
        beam_divergence_urad: 半角束散角（微弧度）
        max_range_km: 最大通信距离（公里）
        acquisition_time_s: ATP捕获阶段时间（秒）—— 信标扫描阶段
        coarse_tracking_time_s: ATP粗跟踪时间（秒）—— 万向节闭环阶段
        fine_tracking_time_s: ATP精跟踪时间（秒）—— FSM（快速转向镜）闭环阶段
        tracking_accuracy_urad: 跟踪精度（微弧度，1-sigma）
        point_ahead_urad: 超前瞄准角补偿量（微弧度），补偿信号传播时延
        min_link_margin_db: 最小链路余量（dB），链路可用性判据
        snr_required_db: 所需信噪比（dB），对应BER=1e-6
    """
    wavelength_nm: float = 1550.0
    transmit_power_w: float = 2.0
    transmit_aperture_m: float = 0.1
    receive_aperture_m: float = 0.1
    beam_divergence_urad: float = 5.0
    max_range_km: float = 7000.0
    acquisition_time_s: float = 30.0      # ATP捕获阶段：信标扫描（宽束）
    coarse_tracking_time_s: float = 5.0   # ATP粗跟踪阶段：万向节闭环
    fine_tracking_time_s: float = 2.0     # ATP精跟踪阶段：FSM闭环
    tracking_accuracy_urad: float = 2.0
    point_ahead_urad: float = 30.0
    min_link_margin_db: float = 3.0
    snr_required_db: float = 20.0         # BER=1e-6对应的SNR要求
    receiver_sensitivity_dBm: float = -31.0  # 接收机灵敏度（dBm），APD探测器典型值

    def __post_init__(self) -> None:
        """参数有效性验证"""
        if self.wavelength_nm <= 0:
            raise ValueError(f"wavelength_nm must be positive, got {self.wavelength_nm}")
        if self.transmit_power_w <= 0:
            raise ValueError(f"transmit_power_w must be positive, got {self.transmit_power_w}")
        if self.transmit_aperture_m <= 0:
            raise ValueError(f"transmit_aperture_m must be positive, got {self.transmit_aperture_m}")
        if self.receive_aperture_m <= 0:
            raise ValueError(f"receive_aperture_m must be positive, got {self.receive_aperture_m}")
        if self.beam_divergence_urad <= 0:
            raise ValueError(f"beam_divergence_urad must be positive, got {self.beam_divergence_urad}")
        if self.max_range_km <= 0:
            raise ValueError(f"max_range_km must be positive, got {self.max_range_km}")
        if self.acquisition_time_s < 0:
            raise ValueError(f"acquisition_time_s must be non-negative, got {self.acquisition_time_s}")
        if self.coarse_tracking_time_s < 0:
            raise ValueError(f"coarse_tracking_time_s must be non-negative, got {self.coarse_tracking_time_s}")
        if self.fine_tracking_time_s < 0:
            raise ValueError(f"fine_tracking_time_s must be non-negative, got {self.fine_tracking_time_s}")
        if self.receiver_sensitivity_dBm > 0:
            raise ValueError(
                f"receiver_sensitivity_dBm must be negative (power below 1mW), got {self.receiver_sensitivity_dBm}"
            )

    @property
    def total_atp_time_s(self) -> float:
        """ATP总建链时间（秒）= 捕获 + 粗跟踪 + 精跟踪"""
        return self.acquisition_time_s + self.coarse_tracking_time_s + self.fine_tracking_time_s

    @classmethod
    def from_dict(cls, d: dict) -> 'LaserISLConfig':
        """
        从字典创建 LaserISLConfig，缺失字段使用默认值。

        Args:
            d: 配置字典

        Returns:
            LaserISLConfig 实例
        """
        return cls(
            wavelength_nm=float(d.get('wavelength_nm', 1550.0)),
            transmit_power_w=float(d.get('transmit_power_w', 2.0)),
            transmit_aperture_m=float(d.get('transmit_aperture_m', 0.1)),
            receive_aperture_m=float(d.get('receive_aperture_m', 0.1)),
            beam_divergence_urad=float(d.get('beam_divergence_urad', 5.0)),
            max_range_km=float(d.get('max_range_km', 7000.0)),
            acquisition_time_s=float(d.get('acquisition_time_s', 30.0)),
            coarse_tracking_time_s=float(d.get('coarse_tracking_time_s', 5.0)),
            fine_tracking_time_s=float(d.get('fine_tracking_time_s', 2.0)),
            tracking_accuracy_urad=float(d.get('tracking_accuracy_urad', 2.0)),
            point_ahead_urad=float(d.get('point_ahead_urad', 30.0)),
            min_link_margin_db=float(d.get('min_link_margin_db', 3.0)),
            snr_required_db=float(d.get('snr_required_db', 20.0)),
            receiver_sensitivity_dBm=float(d.get('receiver_sensitivity_dBm', -31.0)),
        )

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'wavelength_nm': self.wavelength_nm,
            'transmit_power_w': self.transmit_power_w,
            'transmit_aperture_m': self.transmit_aperture_m,
            'receive_aperture_m': self.receive_aperture_m,
            'beam_divergence_urad': self.beam_divergence_urad,
            'max_range_km': self.max_range_km,
            'acquisition_time_s': self.acquisition_time_s,
            'coarse_tracking_time_s': self.coarse_tracking_time_s,
            'fine_tracking_time_s': self.fine_tracking_time_s,
            'tracking_accuracy_urad': self.tracking_accuracy_urad,
            'point_ahead_urad': self.point_ahead_urad,
            'min_link_margin_db': self.min_link_margin_db,
            'snr_required_db': self.snr_required_db,
            'receiver_sensitivity_dBm': self.receiver_sensitivity_dBm,
            'total_atp_time_s': self.total_atp_time_s,
        }


@dataclass
class MicrowaveISLConfig:
    """
    微波星间链路配置

    基于Ka频段（26 GHz）相控阵天线，支持多波束，无需ATP建链过程。

    Attributes:
        frequency_ghz: 载波频率（GHz），默认26 GHz（Ka频段）
        transmit_power_w: 发射功率（瓦特）
        antenna_gain_dbi: 相控阵天线增益（dBi），星下方向
        max_beam_count: 最大同时波束数（MIMO/多波束相控阵）
        scan_angle_deg: 相控阵最大扫描角（度），超出则链路不可用
        max_range_km: 最大通信距离（公里）
        tdma_slots: TDMA时隙数量，决定多用户共享帧长
        gain_rolloff_db_per_deg: 偏轴增益滚降率（dB/度），从30°到60°区间
        system_noise_temp_k: 系统噪声温度（K），包含天线+接收机
        snr_required_db: 所需信噪比（dB），对应目标误码率
    """
    frequency_ghz: float = 26.0
    transmit_power_w: float = 10.0
    antenna_gain_dbi: float = 30.0
    max_beam_count: int = 4
    scan_angle_deg: float = 60.0
    max_range_km: float = 3500.0
    tdma_slots: int = 8
    gain_rolloff_db_per_deg: float = 0.067   # 从30°到60°约1 dB/15° ≈ 0.067 dB/deg
    system_noise_temp_k: float = 1000.0
    snr_required_db: float = 15.0

    def __post_init__(self) -> None:
        """参数有效性验证"""
        if self.frequency_ghz <= 0:
            raise ValueError(f"frequency_ghz must be positive, got {self.frequency_ghz}")
        if self.transmit_power_w <= 0:
            raise ValueError(f"transmit_power_w must be positive, got {self.transmit_power_w}")
        if self.antenna_gain_dbi < 0:
            raise ValueError(f"antenna_gain_dbi must be non-negative, got {self.antenna_gain_dbi}")
        if self.max_beam_count < 1:
            raise ValueError(f"max_beam_count must be >= 1, got {self.max_beam_count}")
        if not (0 < self.scan_angle_deg <= 90):
            raise ValueError(f"scan_angle_deg must be in (0, 90], got {self.scan_angle_deg}")
        if self.max_range_km <= 0:
            raise ValueError(f"max_range_km must be positive, got {self.max_range_km}")
        if self.tdma_slots < 1:
            raise ValueError(f"tdma_slots must be >= 1, got {self.tdma_slots}")
        if self.system_noise_temp_k <= 0:
            raise ValueError(f"system_noise_temp_k must be positive, got {self.system_noise_temp_k}")

    @classmethod
    def from_dict(cls, d: dict) -> 'MicrowaveISLConfig':
        """
        从字典创建 MicrowaveISLConfig，缺失字段使用默认值。

        Args:
            d: 配置字典

        Returns:
            MicrowaveISLConfig 实例
        """
        return cls(
            frequency_ghz=float(d.get('frequency_ghz', 26.0)),
            transmit_power_w=float(d.get('transmit_power_w', 10.0)),
            antenna_gain_dbi=float(d.get('antenna_gain_dbi', 30.0)),
            max_beam_count=int(d.get('max_beam_count', 4)),
            scan_angle_deg=float(d.get('scan_angle_deg', 60.0)),
            max_range_km=float(d.get('max_range_km', 3500.0)),
            tdma_slots=int(d.get('tdma_slots', 8)),
            gain_rolloff_db_per_deg=float(d.get('gain_rolloff_db_per_deg', 0.067)),
            system_noise_temp_k=float(d.get('system_noise_temp_k', 1000.0)),
            snr_required_db=float(d.get('snr_required_db', 15.0)),
        )

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'frequency_ghz': self.frequency_ghz,
            'transmit_power_w': self.transmit_power_w,
            'antenna_gain_dbi': self.antenna_gain_dbi,
            'max_beam_count': self.max_beam_count,
            'scan_angle_deg': self.scan_angle_deg,
            'max_range_km': self.max_range_km,
            'tdma_slots': self.tdma_slots,
            'gain_rolloff_db_per_deg': self.gain_rolloff_db_per_deg,
            'system_noise_temp_k': self.system_noise_temp_k,
            'snr_required_db': self.snr_required_db,
        }


@dataclass
class ISLPeerConfig:
    """
    ISL对等节点配置

    描述本星与某颗对等星之间的链路配置。

    Attributes:
        peer_satellite_id: 对等卫星ID
        link_type: 链路类型（'laser' 或 'microwave'）
        enabled: 是否启用该链路
        preferred: 是否为首选链路（当多条链路可用时优先使用）
    """
    peer_satellite_id: str
    link_type: str         # 'laser' 或 'microwave'
    enabled: bool = True
    preferred: bool = False

    def __post_init__(self) -> None:
        """参数有效性验证"""
        if not self.peer_satellite_id:
            raise ValueError("peer_satellite_id must not be empty")
        if self.link_type not in ('laser', 'microwave'):
            raise ValueError(
                f"link_type must be 'laser' or 'microwave', got '{self.link_type}'"
            )

    @classmethod
    def from_dict(cls, d: dict) -> 'ISLPeerConfig':
        """
        从字典创建 ISLPeerConfig。

        Args:
            d: 配置字典，必须包含 peer_satellite_id 和 link_type

        Returns:
            ISLPeerConfig 实例

        Raises:
            KeyError: 缺少必要字段
        """
        return cls(
            peer_satellite_id=str(d['peer_satellite_id']),
            link_type=str(d['link_type']),
            enabled=bool(d.get('enabled', True)),
            preferred=bool(d.get('preferred', False)),
        )

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            'peer_satellite_id': self.peer_satellite_id,
            'link_type': self.link_type,
            'enabled': self.enabled,
            'preferred': self.preferred,
        }


@dataclass
class ISLCapabilityConfig:
    """
    卫星ISL能力总配置

    聚合单颗卫星的所有ISL相关配置，包括激光终端、微波终端及对等链路列表。

    Attributes:
        enabled: 是否启用ISL功能
        laser: 激光终端配置，None表示不装备激光终端
        microwave: 微波终端配置，None表示不装备微波终端
        peer_links: 与其他卫星的链路配置列表
        max_simultaneous_laser: 最大同时激活的激光链路数（受硬件限制）
        link_selection: 链路选择策略字符串（'laser_preferred'/'microwave_preferred'/'auto'）
    """
    enabled: bool = False
    laser: Optional[LaserISLConfig] = None
    microwave: Optional[MicrowaveISLConfig] = None
    peer_links: List[ISLPeerConfig] = field(default_factory=list)
    max_simultaneous_laser: int = 2
    link_selection: str = 'laser_preferred'

    def __post_init__(self) -> None:
        """参数有效性验证"""
        if self.link_selection not in ('laser_preferred', 'microwave_preferred', 'auto'):
            raise ValueError(
                f"link_selection must be one of 'laser_preferred', 'microwave_preferred', 'auto', "
                f"got '{self.link_selection}'"
            )
        if self.max_simultaneous_laser < 0:
            raise ValueError(
                f"max_simultaneous_laser must be non-negative, got {self.max_simultaneous_laser}"
            )

    @classmethod
    def from_dict(cls, d: dict) -> 'ISLCapabilityConfig':
        """
        从字典创建 ISLCapabilityConfig。

        Args:
            d: 配置字典

        Returns:
            ISLCapabilityConfig 实例
        """
        # 解析激光配置
        laser_data = d.get('laser')
        laser = LaserISLConfig.from_dict(laser_data) if laser_data else None

        # 解析微波配置
        microwave_data = d.get('microwave')
        microwave = MicrowaveISLConfig.from_dict(microwave_data) if microwave_data else None

        # 解析对等链路列表
        peer_links: List[ISLPeerConfig] = []
        for peer_data in d.get('peer_links', []):
            try:
                peer_links.append(ISLPeerConfig.from_dict(peer_data))
            except (KeyError, ValueError) as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to parse ISL peer link config: {e}, skipping entry: {peer_data}"
                )

        return cls(
            enabled=bool(d.get('enabled', False)),
            laser=laser,
            microwave=microwave,
            peer_links=peer_links,
            max_simultaneous_laser=int(d.get('max_simultaneous_laser', 2)),
            link_selection=str(d.get('link_selection', 'laser_preferred')),
        )

    def to_dict(self) -> dict:
        """序列化为字典"""
        result: dict = {
            'enabled': self.enabled,
            'max_simultaneous_laser': self.max_simultaneous_laser,
            'link_selection': self.link_selection,
            'peer_links': [p.to_dict() for p in self.peer_links],
        }
        if self.laser is not None:
            result['laser'] = self.laser.to_dict()
        if self.microwave is not None:
            result['microwave'] = self.microwave.to_dict()
        return result

    def get_peer_link(self, peer_id: str) -> Optional[ISLPeerConfig]:
        """
        获取指定对等星的链路配置（仅返回已启用的链路）。

        Args:
            peer_id: 对等卫星ID

        Returns:
            ISLPeerConfig 或 None（未配置或已禁用）
        """
        for p in self.peer_links:
            if p.peer_satellite_id == peer_id and p.enabled:
                return p
        return None

    def get_laser_peers(self) -> List[str]:
        """
        获取所有已启用激光链路的对等卫星ID列表。

        Returns:
            卫星ID列表
        """
        return [
            p.peer_satellite_id
            for p in self.peer_links
            if p.link_type == 'laser' and p.enabled
        ]

    def get_microwave_peers(self) -> List[str]:
        """
        获取所有已启用微波链路的对等卫星ID列表。

        Returns:
            卫星ID列表
        """
        return [
            p.peer_satellite_id
            for p in self.peer_links
            if p.link_type == 'microwave' and p.enabled
        ]

    def has_laser_capability(self) -> bool:
        """是否装备激光终端"""
        return self.laser is not None and self.enabled

    def has_microwave_capability(self) -> bool:
        """是否装备微波终端"""
        return self.microwave is not None and self.enabled

    def get_link_selection_strategy(self) -> ISLLinkSelectionStrategy:
        """
        获取链路选择策略枚举值。

        Returns:
            ISLLinkSelectionStrategy
        """
        mapping = {
            'laser_preferred': ISLLinkSelectionStrategy.LASER_PREFERRED,
            'microwave_preferred': ISLLinkSelectionStrategy.MICROWAVE_PREFERRED,
            'auto': ISLLinkSelectionStrategy.AUTO,
        }
        return mapping.get(self.link_selection, ISLLinkSelectionStrategy.LASER_PREFERRED)
