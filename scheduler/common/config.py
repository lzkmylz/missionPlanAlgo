"""Centralized configuration for satellite schedulers.

This module provides unified configuration management, replacing
dispersed boolean flags and parameters across scheduler implementations.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Dict


@dataclass
class ConstraintConfig:
    """Configuration for constraint checking.

    Replaces multiple boolean flags like:
    - consider_power
    - consider_storage
    - use_simplified_slew
    - enable_saa_check
    - enable_attitude_calculation
    - enable_thermal_check
    - enable_sun_exclusion_check
    """
    consider_power: bool = True
    consider_storage: bool = True
    mode: str = 'standard'  # 'simplified', 'standard', 'full'
    enable_saa_check: bool = True
    enable_attitude_calculation: bool = False
    enable_thermal_check: bool = False  # New: thermal control constraint
    enable_sun_exclusion_check: bool = False  # New: sun exclusion angle
    enable_solar_elevation_check: bool = True  # Enabled by default: target solar elevation angle (daylight check)
    max_slew_angle: Optional[float] = None  # degrees, None for satellite default
    min_slew_time: float = 10.0  # seconds
    sun_exclusion_angle: float = 30.0  # degrees, for optical satellites
    min_solar_elevation: float = 15.0  # degrees, minimum sun elevation for optical imaging (default 15°)

    def __post_init__(self):
        """Validate configuration."""
        valid_modes = ('simplified', 'standard', 'full')
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode}")


@dataclass
class ResourceConfig:
    """Configuration for resource management."""
    # Power settings
    power_safety_margin: float = 0.05  # 5% safety margin
    min_power_threshold: float = 0.10  # 10% minimum power

    # Storage settings
    storage_safety_margin: float = 0.05  # 5% safety margin
    max_storage_threshold: float = 0.95  # 95% maximum storage

    # Default values for missing satellite capabilities
    default_power_capacity: float = 2800.0  # Wh
    default_storage_capacity: float = 128.0  # GB
    default_data_rate: float = 300.0  # Mbps


@dataclass
class AlgorithmConfig:
    """Base algorithm configuration."""
    # Common parameters
    max_scheduling_time: float = 300.0  # seconds
    priority_weight: float = 1.0
    deadline_weight: float = 0.5
    balance_weight: float = 0.3

    # Imaging settings
    min_imaging_duration: float = 5.0  # seconds
    max_imaging_duration: float = 60.0  # seconds
    default_imaging_duration: float = 10.0  # seconds


@dataclass
class GreedyConfig(AlgorithmConfig):
    """Configuration for greedy schedulers."""
    sort_by_priority: bool = True
    sort_by_deadline: bool = False
    sort_by_processing_time: bool = False  # For SPT
    enable_clustering: bool = False  # For clustering greedy


@dataclass
class MetaheuristicConfig(AlgorithmConfig):
    """Configuration for metaheuristic schedulers.

    Consolidates parameters from GA, SA, ACO, PSO, Tabu schedulers.
    """
    # Constraint and resource settings
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    # Population/iteration settings
    max_iterations: int = 1000
    population_size: int = 50

    # GA-specific settings
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_count: int = 2

    # SA-specific settings
    initial_temperature: float = 100.0
    cooling_rate: float = 0.95
    min_temperature: float = 0.01

    # ACO-specific settings
    num_ants: int = 20
    pheromone_evaporation: float = 0.1
    pheromone_initial: float = 1.0

    # PSO-specific settings
    num_particles: int = 30
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.5
    social_coeff: float = 1.5

    # Tabu-specific settings
    tabu_list_size: int = 50
    neighborhood_size: int = 10

    # Convergence settings
    convergence_threshold: float = 0.001
    max_no_improvement: int = 50

    def __post_init__(self):
        """Validate metaheuristic configuration."""
        if self.population_size <= 0:
            raise ValueError(f"population_size must be positive, got {self.population_size}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {self.max_iterations}")


@dataclass
class SchedulerConfig:
    """Unified scheduler configuration.

    This class consolidates all configuration options that were previously
dispersed across scheduler implementations.

    Example:
        config = SchedulerConfig(
            constraints=ConstraintConfig(
                consider_power=True,
                consider_storage=True,
                mode='standard',
            ),
            resources=ResourceConfig(
                default_power_capacity=2800.0,
            ),
            algorithm=MetaheuristicConfig(
                max_iterations=500,
                population_size=30,
            ),
        )
    """
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SchedulerConfig':
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            SchedulerConfig instance
        """
        constraints = ConstraintConfig(
            **config_dict.get('constraints', {})
        )
        resources = ResourceConfig(
            **config_dict.get('resources', {})
        )

        # Determine algorithm type
        algo_type = config_dict.get('algorithm_type', 'greedy')
        algo_config = config_dict.get('algorithm', {})

        if algo_type == 'metaheuristic':
            algorithm = MetaheuristicConfig(**algo_config)
        elif algo_type == 'greedy':
            algorithm = GreedyConfig(**algo_config)
        else:
            algorithm = AlgorithmConfig(**algo_config)

        return cls(
            constraints=constraints,
            resources=resources,
            algorithm=algorithm,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'constraints': {
                'consider_power': self.constraints.consider_power,
                'consider_storage': self.constraints.consider_storage,
                'mode': self.constraints.mode,
                'enable_saa_check': self.constraints.enable_saa_check,
                'enable_attitude_calculation': self.constraints.enable_attitude_calculation,
                'enable_thermal_check': self.constraints.enable_thermal_check,
                'enable_sun_exclusion_check': self.constraints.enable_sun_exclusion_check,
                'enable_solar_elevation_check': self.constraints.enable_solar_elevation_check,
                'min_solar_elevation': self.constraints.min_solar_elevation,
            },
            'resources': {
                'default_power_capacity': self.resources.default_power_capacity,
                'default_storage_capacity': self.resources.default_storage_capacity,
                'default_data_rate': self.resources.default_data_rate,
            },
            'algorithm': {
                'max_scheduling_time': self.algorithm.max_scheduling_time,
            },
        }


# Backward compatibility helpers
def convert_legacy_config(
    consider_power: bool = True,
    consider_storage: bool = True,
    use_simplified_slew: bool = False,
    enable_saa_check: bool = True,
    enable_attitude_calculation: bool = False,
    **kwargs
) -> SchedulerConfig:
    """Convert legacy boolean flag parameters to new config format.

    This function helps migrate existing code to the new configuration system.

    Args:
        consider_power: Legacy flag
        consider_storage: Legacy flag
        use_simplified_slew: Legacy flag
        enable_saa_check: Legacy flag
        enable_attitude_calculation: Legacy flag
        **kwargs: Additional configuration parameters

    Returns:
        SchedulerConfig with equivalent settings
    """
    mode = 'simplified' if use_simplified_slew else 'standard'

    constraints = ConstraintConfig(
        consider_power=consider_power,
        consider_storage=consider_storage,
        mode=mode,
        enable_saa_check=enable_saa_check,
        enable_attitude_calculation=enable_attitude_calculation,
    )

    # Extract resource config from kwargs
    resources = ResourceConfig(
        default_power_capacity=kwargs.get('default_power_capacity', 2800.0),
        default_storage_capacity=kwargs.get('default_storage_capacity', 128.0),
        default_data_rate=kwargs.get('default_data_rate', 300.0),
    )

    # Determine algorithm config type
    if 'population_size' in kwargs or 'max_iterations' in kwargs:
        algorithm = MetaheuristicConfig(
            max_iterations=kwargs.get('max_iterations', 1000),
            population_size=kwargs.get('population_size', 50),
        )
    else:
        algorithm = AlgorithmConfig(
            max_scheduling_time=kwargs.get('max_scheduling_time', 300.0),
        )

    return SchedulerConfig(
        constraints=constraints,
        resources=resources,
        algorithm=algorithm,
    )
