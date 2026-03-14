"""Particle Swarm Optimization Scheduler (PSO Scheduler)

Satellite task scheduling using particle swarm optimization algorithm.
Uses swarm intelligence with velocity and position updates.

This refactored version uses MetaheuristicScheduler base class,
reducing code duplication by ~70%.
"""

import random
from typing import List, Dict, Any

from .base_metaheuristic import MetaheuristicScheduler, Solution


class PSOScheduler(MetaheuristicScheduler):
    """Particle Swarm Optimization Scheduler.

    Simulates bird flocking behavior with collaborative search.
    Each particle tracks personal best, swarm shares global best.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PSO scheduler.

        Args:
            config: Configuration parameters
                - num_particles: Number of particles (default 30)
                - inertia_weight: Inertia weight (default 0.7)
                - cognitive_coeff: Cognitive coefficient (default 1.5)
                - social_coeff: Social coefficient (default 1.5)
                - random_seed: Random seed (optional)
        """
        # Set PSO-specific defaults before calling parent init - 平衡模式
        config = config or {}
        if 'max_iterations' not in config and 'generations' not in config:
            config['max_iterations'] = 50  # 从100降低

        super().__init__("PSO", config)

        # PSO-specific parameters - 平衡模式优化
        self.num_particles = self._validate_positive_int(
            config.get('num_particles', 25), 'num_particles'  # 从30降低
        )
        self.inertia_weight = config.get('inertia_weight', 0.9)
        self.cognitive_coeff = config.get('cognitive_coeff', 2.0)
        self.social_coeff = config.get('social_coeff', 2.0)

        # Validate parameters
        if not (0 <= self.inertia_weight <= 1):
            raise ValueError(f"inertia_weight must be between 0 and 1, got {self.inertia_weight}")
        if self.cognitive_coeff < 0:
            raise ValueError(f"cognitive_coeff must be non-negative, got {self.cognitive_coeff}")
        if self.social_coeff < 0:
            raise ValueError(f"social_coeff must be non-negative, got {self.social_coeff}")

        # Set random seed
        self.random_seed = config.get('random_seed')
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # PSO state
        self._velocities: List[List[float]] = []
        self._personal_bests: List[Solution] = []
        self._global_best: Solution = Solution(encoding=[])

        # Backward compatibility: expose swarm attribute
        self.swarm: List[Solution] = []

    @property
    def global_best_fitness(self) -> float:
        """Backward compatibility: global best fitness."""
        return self._global_best.fitness if self._global_best else None

    @property
    def max_iterations(self) -> int:
        """Backward compatibility: max iterations."""
        return self._config.max_iterations

    def get_parameters(self) -> Dict[str, Any]:
        """Return algorithm parameters."""
        return {
            'num_particles': self.num_particles,
            'inertia_weight': self.inertia_weight,
            'cognitive_coeff': self.cognitive_coeff,
            'social_coeff': self.social_coeff,
            'max_iterations': self._config.max_iterations,
            'random_seed': self.random_seed,
        }

    def initialize_population(self) -> List[Solution]:
        """Initialize particle swarm."""
        population = []
        self._velocities = []
        self._personal_bests = []

        for _ in range(self.num_particles):
            # Random position (use integers for encoding)
            encoding = [
                random.randint(0, self.sat_count - 1)
                for _ in range(self.task_count)
            ]
            solution = Solution(encoding=encoding)
            population.append(solution)

            # Initialize velocity
            velocity = [random.uniform(-1, 1) for _ in range(self.task_count)]
            self._velocities.append(velocity)

            # Personal best
            self._personal_bests.append(Solution(
                encoding=encoding.copy(),
                fitness=float('-inf')
            ))

        # Backward compatibility: update swarm reference
        self.swarm = population

        return population

    def evolve(self, population: List[Solution]) -> List[Solution]:
        """Perform PSO iteration.

        Updates velocities and positions based on personal and global best.

        Args:
            population: Current particle positions

        Returns:
            Updated population
        """
        new_population = []

        for i, particle in enumerate(population):
            # Update velocity
            self._update_velocity(i, particle)

            # Update position
            new_solution = self._update_position(i, particle)
            new_population.append(new_solution)

            # Update personal best
            if new_solution.fitness > self._personal_bests[i].fitness:
                self._personal_bests[i] = Solution(
                    encoding=new_solution.encoding.copy(),
                    fitness=new_solution.fitness
                )

            # Update global best
            if new_solution.fitness > self._global_best.fitness:
                self._global_best = Solution(
                    encoding=new_solution.encoding.copy(),
                    fitness=new_solution.fitness
                )

        return new_population

    def _update_velocity(self, particle_idx: int, particle: Solution) -> None:
        """Update particle velocity.

        Args:
            particle_idx: Particle index
            particle: Current particle
        """
        velocity = self._velocities[particle_idx]
        personal_best = self._personal_bests[particle_idx]

        # Use particle as global best if not initialized
        global_best = self._global_best if self._global_best.encoding else particle

        for d in range(self.task_count):
            # Inertia component
            inertia = self.inertia_weight * velocity[d]

            # Cognitive component (towards personal best)
            cognitive = self.cognitive_coeff * random.random() * \
                       (personal_best.encoding[d] - particle.encoding[d])

            # Social component (towards global best)
            social = self.social_coeff * random.random() * \
                    (global_best.encoding[d] - particle.encoding[d])

            # Update velocity
            velocity[d] = inertia + cognitive + social

            # Clamp velocity
            velocity[d] = max(-self.sat_count, min(self.sat_count, velocity[d]))

    def _update_position(self, particle_idx: int, particle: Solution) -> Solution:
        """Update particle position.

        Args:
            particle_idx: Particle index
            particle: Current particle

        Returns:
            New solution with updated position
        """
        velocity = self._velocities[particle_idx]
        new_encoding = []

        for d in range(self.task_count):
            # Update position
            new_pos = particle.encoding[d] + velocity[d]

            # Discretize and wrap to valid satellite index
            sat_idx = int(round(new_pos)) % self.sat_count
            new_encoding.append(sat_idx)

        return Solution(encoding=new_encoding)
