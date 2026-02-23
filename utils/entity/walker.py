"""Walker constellation generator module."""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from copy import deepcopy


@dataclass
class WalkerConfig:
    """Walker constellation configuration.

    Standard notation: i:T/P/F
    - i: inclination (degrees)
    - T: total satellites = P × S
    - P: number of orbital planes
    - F: phasing factor (0, 1, ..., P-1)

    Geometry:
    - RAAN spacing: ΔΩ = raan_spread / P
    - In-plane spacing: ΔM = 360° / S
    - Inter-plane phase: ΔM_inter = F × 360° / T
    """
    inclination: float  # degrees
    total_sats: int     # T
    n_planes: int       # P
    f_factor: int       # F (0 to P-1)
    raan_start: float = 0.0
    raan_spread: float = 360.0

    def __post_init__(self):
        """Validate configuration."""
        if self.total_sats % self.n_planes != 0:
            raise ValueError(
                f"total_sats ({self.total_sats}) must be divisible by n_planes ({self.n_planes})"
            )
        if not (0 <= self.f_factor < self.n_planes):
            raise ValueError(
                f"f_factor ({self.f_factor}) must be in range [0, {self.n_planes-1}]"
            )


class WalkerGenerator:
    """Generate Walker Delta/Star constellations."""

    # Preset configurations
    PRESETS = {
        'delta_24_3_1': WalkerConfig(55.0, 24, 3, 1),   # GPS-like
        'delta_66_6_1': WalkerConfig(55.0, 66, 6, 1),   # Iridium-like
        'star_24_3_0': WalkerConfig(90.0, 24, 3, 0),    # Polar
        'star_48_6_0': WalkerConfig(90.0, 48, 6, 0),    # Polar dense
        'delta_18_3_1': WalkerConfig(60.0, 18, 3, 1),   # Small
    }

    def generate(
        self,
        config: WalkerConfig,
        template: Dict[str, Any],
        prefix: str = "WALKER"
    ) -> List[Dict[str, Any]]:
        """Generate Walker constellation satellites.

        Args:
            config: Walker constellation configuration
            template: Satellite template to use
            prefix: Satellite ID prefix

        Returns:
            List of satellite dictionaries
        """
        n_sats_per_plane = config.total_sats // config.n_planes
        delta_raan = config.raan_spread / config.n_planes
        delta_m_plane = 360.0 / n_sats_per_plane

        satellites = []

        for p in range(config.n_planes):
            raan = config.raan_start + p * delta_raan
            # Phase offset based on f_factor and plane index
            # Formula: ΔM_inter = p × F × 360° / T
            phase_offset = (p * config.f_factor * 360.0) / config.total_sats
            for s in range(n_sats_per_plane):
                mean_anomaly = (s * delta_m_plane + phase_offset) % 360.0

                sat_id = f"{prefix}-{p+1:02d}-{s+1:02d}"

                satellite = self._create_satellite_from_template(
                    template, sat_id, raan, mean_anomaly,
                    config.inclination, p + 1, s + 1
                )
                satellites.append(satellite)

        return satellites

    def _create_satellite_from_template(
        self,
        template: Dict[str, Any],
        sat_id: str,
        raan: float,
        mean_anomaly: float,
        inclination: float,
        plane: int,
        sat_in_plane: int
    ) -> Dict[str, Any]:
        """Create satellite from template with orbital parameters."""
        sat = deepcopy(template)
        sat['id'] = sat_id
        sat['name'] = f"Walker卫星-{sat_id}"
        sat['_template_source'] = template.get('template_id', 'unknown')
        sat['_template_version'] = template.get('version', '1.0')
        sat['plane'] = plane
        sat['sat_in_plane'] = sat_in_plane

        # Set orbital parameters (at top level for easy access)
        sat['raan'] = round(raan, 2)
        sat['mean_anomaly'] = round(mean_anomaly, 2)
        if 'orbit' not in sat:
            sat['orbit'] = {}
        sat['orbit']['raan'] = round(raan, 2)
        sat['orbit']['mean_anomaly'] = round(mean_anomaly, 2)
        sat['orbit']['inclination'] = inclination

        return sat

    @classmethod
    def get_preset(cls, preset_name: str) -> Optional[WalkerConfig]:
        """Get preset configuration by name.

        Args:
            preset_name: Preset name (e.g., 'delta_24_3_1')

        Returns:
            WalkerConfig or None if not found
        """
        return cls.PRESETS.get(preset_name)

    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """List available presets with descriptions.

        Returns:
            Dictionary mapping preset names to descriptions
        """
        return {
            'delta_24_3_1': 'GPS-like: 55° inclination, 24 satellites, 3 planes, F=1',
            'delta_66_6_1': 'Iridium-like: 55° inclination, 66 satellites, 6 planes, F=1',
            'star_24_3_0': 'Polar: 90° inclination, 24 satellites, 3 planes, F=0',
            'star_48_6_0': 'Polar dense: 90° inclination, 48 satellites, 6 planes, F=0',
            'delta_18_3_1': 'Small: 60° inclination, 18 satellites, 3 planes, F=1',
        }
