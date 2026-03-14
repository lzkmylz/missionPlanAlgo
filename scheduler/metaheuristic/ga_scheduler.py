"""Genetic Algorithm Scheduler (GA Scheduler)

Satellite task scheduling using genetic algorithm principles.
Uses integer encoding where each gene represents a satellite assignment.

This refactored version uses MetaheuristicScheduler base class,
reducing code duplication by ~70%.
"""

import random
from typing import List, Dict, Any

from .base_metaheuristic import MetaheuristicScheduler, Solution


class GAScheduler(MetaheuristicScheduler):
    """Genetic Algorithm Scheduler.

    Encoding scheme:
    - Chromosome length = number of tasks
    - Each gene = satellite index (0 to n_satellites-1)

    Fitness function:
    - Base: number of successfully scheduled tasks
    - Reward: resource balance, time efficiency
    - Penalty: constraint violations
    """

    # 平衡模式默认参数 - 经过收敛分析优化
    DEFAULT_POPULATION_SIZE = 80  # 从100降低，平衡搜索能力和计算时间
    DEFAULT_GENERATIONS = 50  # 从200降低，超过50代边际收益极低
    DEFAULT_ELITISM = 5  # 精英保留数量

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize GA scheduler.

        Args:
            config: Configuration parameters
                - population_size: Population size (default 80, 平衡模式)
                - generations: Number of generations (default 50, 平衡模式)
                - crossover_rate: Crossover probability (default 0.8)
                - mutation_rate: Mutation probability (default 0.2)
                - elitism: Number of elite individuals to preserve (default 5)
                - tournament_size: Tournament selection size (default 3)
                - random_seed: Random seed (optional)
        """
        super().__init__("GA", config)
        config = config or {}

        # GA-specific parameters - 使用平衡模式默认值
        self.population_size = self._validate_positive_int(
            config.get('population_size', self.DEFAULT_POPULATION_SIZE), 'population_size'
        )
        self.generations = self._validate_positive_int(
            config.get('generations', self.DEFAULT_GENERATIONS), 'generations'
        )
        self.crossover_rate = self._validate_probability(
            config.get('crossover_rate', 0.8), 'crossover_rate'
        )
        self.mutation_rate = self._validate_probability(
            config.get('mutation_rate', 0.2), 'mutation_rate'
        )
        self.elitism = config.get('elitism', self.DEFAULT_ELITISM)
        self.tournament_size = config.get('tournament_size', 3)

        # Set random seed
        self.random_seed = config.get('random_seed')
        if self.random_seed is not None:
            random.seed(self.random_seed)

    def get_parameters(self) -> Dict[str, Any]:
        """Return algorithm parameters."""
        return {
            'population_size': self.population_size,
            'generations': self.generations,
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate,
            'elitism': self.elitism,
            'tournament_size': self.tournament_size,
            'random_seed': self.random_seed,
        }

    def initialize_population(self) -> List[Solution]:
        """Initialize population with random solutions.

        Returns:
            List of Solution objects
        """
        population = []
        for _ in range(self.population_size):
            # Random assignment of each task to a satellite
            encoding = [
                random.randint(0, self.sat_count - 1)
                for _ in range(self.task_count)
            ]
            population.append(Solution(encoding=encoding))
        return population

    def evolve(self, population: List[Solution]) -> List[Solution]:
        """Evolve population for one generation.

        Performs selection, crossover, and mutation.

        Args:
            population: Current population

        Returns:
            New population after evolution
        """
        # Selection
        selected = self._selection(population)

        # Crossover
        offspring = self._crossover(selected)

        # Mutation
        offspring = self._mutation(offspring)

        # Elitism
        if self.elitism > 0:
            return self._elitism(population, offspring)
        return offspring

    def _selection(self, population: List[Solution]) -> List[Solution]:
        """Tournament selection.

        Args:
            population: Current population

        Returns:
            Selected individuals
        """
        selected = []
        for _ in range(len(population)):
            # Tournament
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            winner = max(tournament, key=lambda sol: sol.fitness)
            selected.append(Solution(
                encoding=winner.encoding.copy(),
                fitness=winner.fitness
            ))
        return selected

    def _crossover(self, population: List[Solution]) -> List[Solution]:
        """Single-point crossover.

        Args:
            population: Selected population

        Returns:
            Population after crossover
        """
        offspring = []
        for i in range(0, len(population), 2):
            parent1 = population[i]
            parent2 = population[i + 1] if i + 1 < len(population) else population[0]

            if random.random() < self.crossover_rate and self.task_count > 1:
                # Perform crossover
                point = random.randint(1, self.task_count - 1)
                child1_encoding = parent1.encoding[:point] + parent2.encoding[point:]
                child2_encoding = parent2.encoding[:point] + parent1.encoding[point:]

                offspring.append(Solution(encoding=child1_encoding))
                offspring.append(Solution(encoding=child2_encoding))
            else:
                # No crossover
                offspring.append(Solution(encoding=parent1.encoding.copy()))
                if i + 1 < len(population):
                    offspring.append(Solution(encoding=parent2.encoding.copy()))

        return offspring[:len(population)]

    def _mutation(self, population: List[Solution]) -> List[Solution]:
        """Random mutation.

        Args:
            population: Population after crossover

        Returns:
            Population after mutation
        """
        for solution in population:
            for i in range(len(solution.encoding)):
                if random.random() < self.mutation_rate:
                    # Mutate: assign to random satellite
                    solution.encoding[i] = random.randint(0, self.sat_count - 1)
        return population

    def _elitism(
        self,
        old_population: List[Solution],
        new_population: List[Solution]
    ) -> List[Solution]:
        """Preserve elite individuals.

        Args:
            old_population: Population before evolution
            new_population: Population after evolution

        Returns:
            Population with elites preserved
        """
        # Sort by fitness
        sorted_old = sorted(old_population, key=lambda sol: sol.fitness, reverse=True)

        # Preserve elites
        elites = [
            Solution(encoding=sol.encoding.copy(), fitness=sol.fitness)
            for sol in sorted_old[:self.elitism]
        ]

        # Replace worst individuals with elites
        result = new_population[:len(new_population) - self.elitism] + elites
        return result
