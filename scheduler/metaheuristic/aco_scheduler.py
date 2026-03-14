"""Ant Colony Optimization Scheduler (ACO Scheduler)

Satellite task scheduling using ant colony optimization algorithm.
Uses pheromone trails to guide search towards optimal solutions.

This refactored version uses MetaheuristicScheduler base class,
reducing code duplication by ~70%.
"""

import random
from typing import List, Dict, Any

import numpy as np

from .base_metaheuristic import MetaheuristicScheduler, Solution


class ACOScheduler(MetaheuristicScheduler):
    """Ant Colony Optimization Scheduler.

    Simulates ant foraging behavior with pheromone-based guidance.
    Positive feedback: good paths accumulate more pheromone.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ACO scheduler.

        Args:
            config: Configuration parameters
                - num_ants: Number of ants (default 30)
                - alpha: Pheromone importance (default 1.0)
                - beta: Heuristic importance (default 2.0)
                - evaporation_rate: Pheromone evaporation (default 0.1)
                - random_seed: Random seed (optional)
        """
        config = config or {}
        # 平衡模式: 默认50代
        if 'max_iterations' not in config and 'generations' not in config:
            config['max_iterations'] = 50

        super().__init__("ACO", config)

        # ACO-specific parameters - 平衡模式优化
        self.num_ants = self._validate_positive_int(
            config.get('num_ants', 25), 'num_ants'  # 从30降低
        )
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 2.0)
        self.evaporation_rate = self._validate_probability(
            config.get('evaporation_rate', 0.1), 'evaporation_rate'
        )

        # Set random seed
        self.random_seed = config.get('random_seed')
        if self.random_seed is not None:
            random.seed(self.random_seed)

        # Pheromone matrix: pheromone[task_idx][sat_idx]
        self._pheromone: List[List[float]] = []
        self.initial_pheromone = config.get('initial_pheromone', 1.0) if config else 1.0  # Backward compatibility

    @property
    def max_iterations(self) -> int:
        """Backward compatibility: max iterations."""
        return self._config.max_iterations

    @property
    def pheromone_matrix(self) -> np.ndarray:
        """Backward compatibility: pheromone matrix as numpy array."""
        return np.array(self._pheromone)

    def get_parameters(self) -> Dict[str, Any]:
        """Return algorithm parameters."""
        return {
            'num_ants': self.num_ants,
            'alpha': self.alpha,
            'beta': self.beta,
            'evaporation_rate': self.evaporation_rate,
            'max_iterations': self._config.max_iterations,
            'random_seed': self.random_seed,
        }

    def _initialize_pheromone(self) -> None:
        """Initialize pheromone matrix with uniform values."""
        # Use the configured initial_pheromone value
        self._pheromone = [
            [self.initial_pheromone for _ in range(self.sat_count)]
            for _ in range(self.task_count)
        ]

    def initialize_population(self) -> List[Solution]:
        """Initialize ant population."""
        self._initialize_pheromone()
        return self._construct_solutions()

    def evolve(self, population: List[Solution]) -> List[Solution]:
        """Perform ACO iteration.

        Updates pheromone trails and constructs new solutions.

        Args:
            population: Current ant solutions

        Returns:
            New population after pheromone update
        """
        # Update pheromones based on solution quality
        self._update_pheromones(population)

        # Construct new solutions
        return self._construct_solutions()

    def _construct_solutions(self) -> List[Solution]:
        """Construct solutions for all ants using pheromone guidance.

        Returns:
            List of ant solutions
        """
        solutions = []
        for _ in range(self.num_ants):
            encoding = self._construct_ant_path()
            solutions.append(Solution(encoding=encoding))
        return solutions

    def _construct_ant_path(self) -> List[int]:
        """Construct a single ant path using pheromone and heuristic.

        Returns:
            Path encoding (task to satellite assignment)
        """
        path = []
        for task_idx in range(self.task_count):
            # Calculate selection probabilities
            probabilities = self._calculate_probabilities(task_idx)

            # Select satellite based on probabilities
            sat_idx = self._select_satellite(probabilities)
            path.append(sat_idx)

        return path

    def _calculate_probabilities(self, task_idx: int) -> List[float]:
        """Calculate selection probabilities for a task.

        Args:
            task_idx: Task index

        Returns:
            Probability distribution over satellites
        """
        probabilities = []
        total = 0.0

        for sat_idx in range(self.sat_count):
            pheromone = self._pheromone[task_idx][sat_idx] ** self.alpha
            heuristic = 1.0  # Uniform heuristic (can be improved)

            prob = pheromone * (heuristic ** self.beta)
            probabilities.append(prob)
            total += prob

        # Normalize
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / self.sat_count] * self.sat_count

        return probabilities

    def _select_satellite(self, probabilities: List[float]) -> int:
        """Select satellite based on probability distribution.

        Args:
            probabilities: Selection probabilities

        Returns:
            Selected satellite index
        """
        r = random.random()
        cumulative = 0.0
        for sat_idx, prob in enumerate(probabilities):
            cumulative += prob
            if r <= cumulative:
                return sat_idx
        return len(probabilities) - 1

    def _update_pheromones(self, solutions: List[Solution]) -> None:
        """Update pheromone trails based on solution quality.

        Args:
            solutions: Current ant solutions
        """
        # Evaporate pheromones
        for task_idx in range(self.task_count):
            for sat_idx in range(self.sat_count):
                self._pheromone[task_idx][sat_idx] *= (1 - self.evaporation_rate)

        # Deposit pheromones based on solution fitness
        for solution in solutions:
            if solution.fitness > 0:
                deposit = solution.fitness / 100.0  # Normalize
                for task_idx, sat_idx in enumerate(solution.encoding):
                    self._pheromone[task_idx][sat_idx] += deposit
