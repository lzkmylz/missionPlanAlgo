"""Entity library management module."""
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

try:
    import click
except ImportError:
    click = None

from .repository.base import EntityRepository
from .repository.json_repository import JSONEntityRepository


class EntityLibrary:
    """Entity library management.

    Provides high-level operations for managing satellite templates,
    targets, and ground stations in the entity library.
    """

    def __init__(self, repository: Optional[EntityRepository] = None):
        """Initialize entity library.

        Args:
            repository: Repository instance. If None, creates default JSON repository.
        """
        self.repo = repository or JSONEntityRepository()

    def list_satellites(self) -> List[Dict[str, Any]]:
        """List all satellite templates."""
        return self.repo.list_satellite_templates()

    def get_satellite(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get satellite template by ID."""
        return self.repo.get_satellite_template(template_id)

    def list_targets(self) -> List[Dict[str, Any]]:
        """List all targets."""
        return self.repo.list_targets()

    def list_ground_stations(self) -> List[Dict[str, Any]]:
        """List all ground stations."""
        return self.repo.list_ground_stations()

    def get_ground_station(self, gs_id: str) -> Optional[Dict[str, Any]]:
        """Get ground station by ID."""
        return self.repo.get_ground_station(gs_id)

    def add_satellite_interactive(self) -> Dict[str, Any]:
        """Interactive wizard for adding satellite template.

        Uses click prompts to collect information from user.
        """
        if click is None:
            raise RuntimeError("click is required for interactive mode")

        template_id = click.prompt("Template ID", type=str)
        name = click.prompt("Satellite name", type=str)
        sat_type = click.prompt(
            "Satellite type",
            type=click.Choice(['optical_1', 'optical_2', 'sar_1', 'sar_2'])
        )

        template = {
            "template_id": template_id,
            "name": name,
            "sat_type": sat_type,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

        # Add capabilities
        if click.confirm("Add capabilities?"):
            capabilities = {}

            imaging_modes_str = click.prompt(
                "Imaging modes (comma-separated)",
                default="push_broom"
            )
            capabilities["imaging_modes"] = [
                m.strip() for m in imaging_modes_str.split(",")
            ]

            capabilities["max_off_nadir"] = click.prompt(
                "Max off-nadir angle (degrees)",
                type=float,
                default=30.0
            )
            capabilities["storage_capacity"] = click.prompt(
                "Storage capacity (GB)",
                type=int,
                default=500
            )
            capabilities["power_capacity"] = click.prompt(
                "Power capacity (Wh)",
                type=int,
                default=2000
            )
            capabilities["data_rate"] = click.prompt(
                "Data rate (Mbps)",
                type=int,
                default=300
            )

            template["capabilities"] = capabilities

        # Add orbit
        if click.confirm("Add orbit parameters?"):
            orbit = {}
            orbit["orbit_type"] = click.prompt(
                "Orbit type",
                type=click.Choice(['SSO', 'LEO', 'MEO', 'GEO']),
                default="SSO"
            )
            orbit["altitude"] = click.prompt(
                "Altitude (m)",
                type=int,
                default=645000
            )
            orbit["inclination"] = click.prompt(
                "Inclination (degrees)",
                type=float,
                default=97.9
            )
            template["orbit"] = orbit

        self.repo.save_satellite_template(template)
        return template

    def add_target(
        self,
        target_id: str,
        name: str,
        lon: float,
        lat: float,
        priority: int = 5,
        target_type: str = "point",
        **kwargs
    ) -> Dict[str, Any]:
        """Add point target to library.

        Args:
            target_id: Unique target identifier
            name: Target name
            lon: Longitude
            lat: Latitude
            priority: Priority level (1-10)
            target_type: Type of target ('point' or 'area')
            **kwargs: Additional target properties

        Returns:
            Created target dictionary
        """
        target = {
            "id": target_id,
            "name": name,
            "target_type": target_type,
            "position": {
                "longitude": lon,
                "latitude": lat,
                "altitude": 0
            },
            "priority": priority,
            "required_observations": kwargs.get("required_observations", 1),
            "resolution_required": kwargs.get("resolution_required", 10.0),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

        # Add any additional properties
        for key, value in kwargs.items():
            if key not in target:
                target[key] = value

        self.repo.save_target(target, target_type=target_type)
        return target

    def init_defaults(self) -> None:
        """Initialize entity library with default templates."""
        # Default satellite templates
        default_satellites = [
            {
                "template_id": "optical_1",
                "name": "光学卫星1型",
                "sat_type": "optical_1",
                "orbit": {
                    "orbit_type": "SSO",
                    "altitude": 645000,
                    "inclination": 97.9
                },
                "capabilities": {
                    "imaging_modes": ["push_broom"],
                    "max_off_nadir": 30.0,
                    "storage_capacity": 500,
                    "power_capacity": 2000,
                    "data_rate": 300
                }
            },
            {
                "template_id": "optical_2",
                "name": "光学卫星2型",
                "sat_type": "optical_2",
                "orbit": {
                    "orbit_type": "SSO",
                    "altitude": 645000,
                    "inclination": 97.9
                },
                "capabilities": {
                    "imaging_modes": ["frame"],
                    "max_off_nadir": 35.0,
                    "storage_capacity": 800,
                    "power_capacity": 2500,
                    "data_rate": 400
                }
            },
            {
                "template_id": "sar_1",
                "name": "SAR卫星1型",
                "sat_type": "sar_1",
                "orbit": {
                    "orbit_type": "SSO",
                    "altitude": 631000,
                    "inclination": 98.0
                },
                "capabilities": {
                    "imaging_modes": ["stripmap", "scan"],
                    "max_off_nadir": 35.0,
                    "storage_capacity": 1000,
                    "power_capacity": 3000,
                    "data_rate": 300
                }
            },
            {
                "template_id": "sar_2",
                "name": "SAR卫星2型",
                "sat_type": "sar_2",
                "orbit": {
                    "orbit_type": "SSO",
                    "altitude": 631000,
                    "inclination": 98.0
                },
                "capabilities": {
                    "imaging_modes": ["spotlight", "sliding_spotlight", "stripmap"],
                    "max_off_nadir": 45.0,
                    "storage_capacity": 1500,
                    "power_capacity": 4000,
                    "data_rate": 500
                }
            }
        ]

        for sat in default_satellites:
            self.repo.save_satellite_template(sat)

        # Default ground stations
        default_ground_stations = [
            {
                "id": "gs_beijing",
                "name": "北京地面站",
                "longitude": 116.4074,
                "latitude": 39.9042,
                "altitude": 0,
                "antennas": [
                    {
                        "id": "BJ-ANT-01",
                        "elevation_min": 5.0,
                        "elevation_max": 90.0,
                        "data_rate": 450.0
                    }
                ]
            },
            {
                "id": "gs_kashi",
                "name": "喀什地面站",
                "longitude": 75.9896,
                "latitude": 39.4704,
                "altitude": 0,
                "antennas": [
                    {
                        "id": "KS-ANT-01",
                        "elevation_min": 5.0,
                        "elevation_max": 90.0,
                        "data_rate": 300.0
                    }
                ]
            }
        ]

        for gs in default_ground_stations:
            self.repo.save_ground_station(gs)
