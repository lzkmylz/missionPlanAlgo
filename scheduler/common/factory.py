"""Scheduler factory for dependency injection.

.. deprecated::
    此模块中的 ConstraintChecker 创建方法已弃用。
    调度器现在内部使用 BatchSlewConstraintChecker 和 UnifiedBatchConstraintChecker。
    直接使用工厂创建调度器仍然有效。

This module provides factory methods for creating schedulers
with proper dependency injection, reducing coupling and improving testability.
"""

from typing import Optional, Dict, Any

from core.models import Mission
from core.orbit.visibility.window_cache import VisibilityWindowCache

from scheduler.common import SchedulerConfig
from scheduler.common.resource_manager import ResourceManager
from scheduler.common.constraint_checker import ConstraintChecker


class SchedulerFactory:
    """Factory for creating schedulers with dependency injection.

    This factory centralizes scheduler creation and ensures proper
    initialization of all dependencies.

    Example:
        factory = SchedulerFactory(mission, window_cache)
        scheduler = factory.create_ga_scheduler(population_size=100)
    """

    def __init__(
        self,
        mission: Mission,
        window_cache: VisibilityWindowCache,
        config: Optional[SchedulerConfig] = None,
    ):
        """Initialize factory with common dependencies.

        Args:
            mission: Mission object
            window_cache: Visibility window cache
            config: Base configuration (optional)
        """
        self.mission = mission
        self.window_cache = window_cache
        self.config = config or SchedulerConfig()

    def _create_resource_manager(self) -> ResourceManager:
        """Create resource manager."""
        return ResourceManager(
            satellites=list(self.mission.satellites),
            start_time=self.mission.start_time,
            consider_power=self.config.constraints.consider_power,
            consider_storage=self.config.constraints.consider_storage,
        )

    def _create_constraint_checker(self) -> ConstraintChecker:
        """Create constraint checker.

        .. deprecated::
            此方法已弃用。调度器现在内部使用 BatchSlewConstraintChecker
            和 UnifiedBatchConstraintChecker，不需要外部创建约束检查器。
        """
        import warnings
        warnings.warn(
            "_create_constraint_checker 已弃用。"
            "调度器现在内部使用 BatchSlewConstraintChecker",
            DeprecationWarning,
            stacklevel=2
        )
        checker = ConstraintChecker(
            mission=self.mission,
            config=self.config.constraints,
        )
        checker.initialize()
        return checker

    def create_greedy_scheduler(
        self,
        consider_power: bool = True,
        consider_storage: bool = True,
    ) -> 'GreedyScheduler':
        """Create greedy scheduler.

        Args:
            consider_power: Whether to consider power constraints
            consider_storage: Whether to consider storage constraints

        Returns:
            Configured GreedyScheduler
        """
        from scheduler.greedy.greedy_scheduler import GreedyScheduler

        config = {
            'consider_power': consider_power,
            'consider_storage': consider_storage,
        }
        scheduler = GreedyScheduler(config)
        scheduler.initialize(self.mission, self.window_cache)
        return scheduler

    def create_spt_scheduler(
        self,
        consider_power: bool = True,
        consider_storage: bool = True,
    ) -> 'SPTScheduler':
        """Create SPT (Shortest Processing Time) scheduler.

        Args:
            consider_power: Whether to consider power constraints
            consider_storage: Whether to consider storage constraints

        Returns:
            Configured SPTScheduler
        """
        from scheduler.greedy.spt_scheduler import SPTScheduler

        config = {
            'consider_power': consider_power,
            'consider_storage': consider_storage,
        }
        scheduler = SPTScheduler(config)
        scheduler.initialize(self.mission, self.window_cache)
        return scheduler

    def create_edd_scheduler(
        self,
        consider_power: bool = True,
        consider_storage: bool = True,
    ) -> 'EDDScheduler':
        """Create EDD (Earliest Due Date) scheduler.

        Args:
            consider_power: Whether to consider power constraints
            consider_storage: Whether to consider storage constraints

        Returns:
            Configured EDDScheduler
        """
        from scheduler.greedy.edd_scheduler import EDDScheduler

        config = {
            'consider_power': consider_power,
            'consider_storage': consider_storage,
        }
        scheduler = EDDScheduler(config)
        scheduler.initialize(self.mission, self.window_cache)
        return scheduler

    def create_ga_scheduler(
        self,
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elitism: int = 5,
        **kwargs
    ) -> 'GAScheduler':
        """Create Genetic Algorithm scheduler.

        Args:
            population_size: Population size
            generations: Number of generations
            crossover_rate: Crossover probability
            mutation_rate: Mutation probability
            elitism: Number of elite individuals
            **kwargs: Additional configuration

        Returns:
            Configured GAScheduler
        """
        from scheduler.metaheuristic.ga_scheduler import GAScheduler

        config = {
            'population_size': population_size,
            'generations': generations,
            'crossover_rate': crossover_rate,
            'mutation_rate': mutation_rate,
            'elitism': elitism,
            'consider_power': self.config.constraints.consider_power,
            'consider_storage': self.config.constraints.consider_storage,
            **kwargs,
        }
        scheduler = GAScheduler(config)
        scheduler.initialize(self.mission, self.window_cache)
        return scheduler

    def create_sa_scheduler(
        self,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        iterations_per_temp: int = 100,
        **kwargs
    ) -> 'SAScheduler':
        """Create Simulated Annealing scheduler.

        Args:
            initial_temperature: Starting temperature
            cooling_rate: Cooling rate
            min_temperature: Minimum temperature
            iterations_per_temp: Iterations per temperature
            **kwargs: Additional configuration

        Returns:
            Configured SAScheduler
        """
        from scheduler.metaheuristic.sa_scheduler import SAScheduler

        config = {
            'initial_temperature': initial_temperature,
            'cooling_rate': cooling_rate,
            'min_temperature': min_temperature,
            'iterations_per_temp': iterations_per_temp,
            'consider_power': self.config.constraints.consider_power,
            'consider_storage': self.config.constraints.consider_storage,
            **kwargs,
        }
        scheduler = SAScheduler(config)
        scheduler.initialize(self.mission, self.window_cache)
        return scheduler

    def create_aco_scheduler(
        self,
        num_ants: int = 30,
        alpha: float = 1.0,
        beta: float = 2.0,
        evaporation_rate: float = 0.1,
        **kwargs
    ) -> 'ACOScheduler':
        """Create Ant Colony Optimization scheduler.

        Args:
            num_ants: Number of ants
            alpha: Pheromone importance
            beta: Heuristic importance
            evaporation_rate: Pheromone evaporation rate
            **kwargs: Additional configuration

        Returns:
            Configured ACOScheduler
        """
        from scheduler.metaheuristic.aco_scheduler import ACOScheduler

        config = {
            'num_ants': num_ants,
            'alpha': alpha,
            'beta': beta,
            'evaporation_rate': evaporation_rate,
            'consider_power': self.config.constraints.consider_power,
            'consider_storage': self.config.constraints.consider_storage,
            **kwargs,
        }
        scheduler = ACOScheduler(config)
        scheduler.initialize(self.mission, self.window_cache)
        return scheduler

    def create_pso_scheduler(
        self,
        num_particles: int = 30,
        inertia_weight: float = 0.7,
        cognitive_coeff: float = 1.5,
        social_coeff: float = 1.5,
        **kwargs
    ) -> 'PSOScheduler':
        """Create Particle Swarm Optimization scheduler.

        Args:
            num_particles: Number of particles
            inertia_weight: Inertia weight
            cognitive_coeff: Cognitive coefficient
            social_coeff: Social coefficient
            **kwargs: Additional configuration

        Returns:
            Configured PSOScheduler
        """
        from scheduler.metaheuristic.pso_scheduler import PSOScheduler

        config = {
            'num_particles': num_particles,
            'inertia_weight': inertia_weight,
            'cognitive_coeff': cognitive_coeff,
            'social_coeff': social_coeff,
            'consider_power': self.config.constraints.consider_power,
            'consider_storage': self.config.constraints.consider_storage,
            **kwargs,
        }
        scheduler = PSOScheduler(config)
        scheduler.initialize(self.mission, self.window_cache)
        return scheduler

    def create_tabu_scheduler(
        self,
        tabu_tenure: int = 50,
        neighborhood_size: int = 20,
        aspiration_threshold: float = 0.1,
        **kwargs
    ) -> 'TabuScheduler':
        """Create Tabu Search scheduler.

        Args:
            tabu_tenure: Tabu list size
            neighborhood_size: Neighborhood size
            aspiration_threshold: Aspiration threshold
            **kwargs: Additional configuration

        Returns:
            Configured TabuScheduler
        """
        from scheduler.metaheuristic.tabu_scheduler import TabuScheduler

        config = {
            'tabu_tenure': tabu_tenure,
            'neighborhood_size': neighborhood_size,
            'aspiration_threshold': aspiration_threshold,
            'consider_power': self.config.constraints.consider_power,
            'consider_storage': self.config.constraints.consider_storage,
            **kwargs,
        }
        scheduler = TabuScheduler(config)
        scheduler.initialize(self.mission, self.window_cache)
        return scheduler


def create_scheduler(
    algorithm: str,
    mission: Mission,
    window_cache: VisibilityWindowCache,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Convenience function to create a scheduler by name.

    Args:
        algorithm: Algorithm name ('greedy', 'spt', 'edd', 'ga', 'sa', 'aco', 'pso', 'tabu')
        mission: Mission object
        window_cache: Visibility window cache
        config: Configuration dictionary

    Returns:
        Configured scheduler instance

    Raises:
        ValueError: If algorithm name is not recognized
    """
    config = config or {}
    factory = SchedulerFactory(mission, window_cache)

    creators = {
        'greedy': factory.create_greedy_scheduler,
        'spt': factory.create_spt_scheduler,
        'edd': factory.create_edd_scheduler,
        'ga': factory.create_ga_scheduler,
        'sa': factory.create_sa_scheduler,
        'aco': factory.create_aco_scheduler,
        'pso': factory.create_pso_scheduler,
        'tabu': factory.create_tabu_scheduler,
    }

    if algorithm.lower() not in creators:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Supported: {list(creators.keys())}"
        )

    return creators[algorithm.lower()](**config)
