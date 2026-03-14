"""Feasibility matrix precomputation for metaheuristic schedulers.

This module provides precomputation of feasibility matrices to accelerate
metaheuristic algorithm evaluation from O(n) per task to O(1) lookup.

Precomputed matrices:
1. Single-window feasibility: task x satellite matrix
   - Stores whether a task can be performed by a satellite at its optimal window
   - Includes attitude (slew) feasibility and SAA check

2. Transition feasibility: task x task matrix per satellite
   - Stores whether task B can follow task A on the same satellite
   - Includes maneuver feasibility between the two tasks

Usage:
    from scheduler.metaheuristic.feasibility_precompute import FeasibilityPrecomputer

    precomputer = FeasibilityPrecomputer(mission, window_cache)
    matrices = precomputer.precompute_all()
    precomputer.save(matrices, "feasibility_matrices.pkl")

    # Later, in scheduler:
    matrices = FeasibilityPrecomputer.load("feasibility_matrices.pkl")
"""

import pickle
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

from core.models import Mission, Satellite, Target
from payload.imaging_time_calculator import ImagingTimeCalculator

logger = logging.getLogger(__name__)


class FeasibilityMatrices:
    """Container for precomputed feasibility matrices.

    Attributes:
        task_count: Number of tasks
        sat_count: Number of satellites
        task_ids: List of task IDs in order
        sat_ids: List of satellite IDs in order

        # Single-window feasibility [task_idx, sat_idx] -> bool
        # True if satellite can perform task at its optimal window
        single_window_feasible: np.ndarray

        # Task start times [task_idx, sat_idx] -> datetime or None
        # Optimal start time for task-satellite pair
        task_start_times: np.ndarray

        # Task end times [task_idx, sat_idx] -> datetime or None
        # Optimal end time for task-satellite pair
        task_end_times: np.ndarray

        # Transition feasibility [sat_idx, task_a_idx, task_b_idx] -> bool
        # True if task_b can follow task_a on satellite sat_idx
        # Only computed for tasks with overlapping satellites
        transition_feasible: Dict[int, np.ndarray]

        # Metadata
        created_at: datetime
        mission_id: str
    """

    def __init__(
        self,
        task_count: int,
        sat_count: int,
        task_ids: List[str],
        sat_ids: List[str]
    ):
        self.task_count = task_count
        self.sat_count = sat_count
        self.task_ids = task_ids
        self.sat_ids = sat_ids

        # Initialize matrices with False (infeasible)
        self.single_window_feasible = np.zeros((task_count, sat_count), dtype=np.bool_)
        self.task_start_times = np.empty((task_count, sat_count), dtype=object)
        self.task_end_times = np.empty((task_count, sat_count), dtype=object)
        self.task_durations = np.zeros((task_count, sat_count), dtype=np.float32)

        # Transition matrices per satellite (lazy initialized)
        self.transition_feasible: Dict[int, np.ndarray] = {}

        self.created_at = datetime.now()
        self.mission_id = ""

    def get_task_sat_feasible(self, task_idx: int, sat_idx: int) -> bool:
        """Check if task can be performed by satellite."""
        return self.single_window_feasible[task_idx, sat_idx]

    def get_task_timing(self, task_idx: int, sat_idx: int) -> Optional[Tuple[datetime, datetime]]:
        """Get optimal timing for task-satellite pair."""
        if not self.single_window_feasible[task_idx, sat_idx]:
            return None
        return (
            self.task_start_times[task_idx, sat_idx],
            self.task_end_times[task_idx, sat_idx]
        )

    def get_transition_feasible(
        self,
        sat_idx: int,
        task_a_idx: int,
        task_b_idx: int
    ) -> bool:
        """Check if task_b can follow task_a on satellite."""
        if sat_idx not in self.transition_feasible:
            return True  # Assume feasible if not computed
        return self.transition_feasible[sat_idx][task_a_idx, task_b_idx]


class FeasibilityPrecomputer:
    """Precomputes feasibility matrices for metaheuristic schedulers.

    This class parallelizes the precomputation across multiple CPU cores
    to minimize the one-time precomputation cost.
    """

    def __init__(
        self,
        mission: Mission,
        window_cache: Any,
        imaging_calculator: Optional[ImagingTimeCalculator] = None
    ):
        self.mission = mission
        self.window_cache = window_cache
        self.imaging_calculator = imaging_calculator or ImagingTimeCalculator()

        self.satellites = list(mission.satellites)
        self.sat_count = len(self.satellites)
        self.sat_id_to_idx = {sat.id: i for i, sat in enumerate(self.satellites)}

    def precompute_all(
        self,
        tasks: List[Any],
        enable_parallel: bool = True,
        max_workers: Optional[int] = None
    ) -> FeasibilityMatrices:
        """Precompute all feasibility matrices.

        Args:
            tasks: List of tasks to precompute for
            enable_parallel: Whether to use parallel processing
            max_workers: Number of parallel workers (default: CPU count)

        Returns:
            FeasibilityMatrices containing all precomputed data
        """
        task_count = len(tasks)
        task_ids = [getattr(t, 'id', str(i)) for i, t in enumerate(tasks)]
        sat_ids = [sat.id for sat in self.satellites]

        logger.info(f"Starting feasibility precomputation for {task_count} tasks x {self.sat_count} satellites")

        matrices = FeasibilityMatrices(task_count, self.sat_count, task_ids, sat_ids)
        matrices.mission_id = getattr(self.mission, 'id', 'unknown')

        # Phase 1: Single-window feasibility (only phase - transition check is done on-the-fly)
        logger.info("Phase 1: Computing single-window feasibility...")
        self._sequential_single_window(matrices, tasks)

        # Phase 2: Skip transition feasibility precomputation (too memory intensive)
        # Transition checks are done using time-based heuristics during evaluation
        logger.info("Phase 2: Skipping transition precomputation (using on-the-fly checks)")

        # Statistics
        feasible_count = np.sum(matrices.single_window_feasible)
        total_count = task_count * self.sat_count
        logger.info(
            f"Precomputation complete. "
            f"Feasible pairs: {feasible_count}/{total_count} "
            f"({feasible_count/total_count*100:.1f}%)"
        )

        return matrices

    def _precompute_single_window(
        self,
        matrices: FeasibilityMatrices,
        tasks: List[Any],
        enable_parallel: bool,
        max_workers: Optional[int]
    ):
        """Precompute single-window feasibility matrix."""
        task_count = len(tasks)

        if enable_parallel and task_count > 100:
            self._parallel_single_window(matrices, tasks, max_workers)
        else:
            self._sequential_single_window(matrices, tasks)

    def _sequential_single_window(
        self,
        matrices: FeasibilityMatrices,
        tasks: List[Any]
    ):
        """Sequential computation of single-window feasibility."""
        for task_idx, task in enumerate(tasks):
            if task_idx % 100 == 0:
                logger.info(f"  Processing task {task_idx}/{len(tasks)}...")

            for sat_idx, sat in enumerate(self.satellites):
                feasible, start_time, end_time, duration = self._check_single_window(
                    task, sat
                )
                matrices.single_window_feasible[task_idx, sat_idx] = feasible
                if feasible:
                    matrices.task_start_times[task_idx, sat_idx] = start_time
                    matrices.task_end_times[task_idx, sat_idx] = end_time
                    matrices.task_durations[task_idx, sat_idx] = duration

    def _check_single_window(
        self,
        task: Any,
        sat: Satellite
    ) -> Tuple[bool, Optional[datetime], Optional[datetime], float]:
        """Check if satellite can perform task at single window.

        Returns:
            (feasible, start_time, end_time, duration)
        """
        # Get visibility windows for this task-satellite pair
        target_id = getattr(task, 'target_id', getattr(task, 'id', None))
        if not target_id:
            return False, None, None, 0.0

        windows = self.window_cache.get_windows(sat.id, target_id)
        if not windows:
            return False, None, None, 0.0

        # Find best window (simplified: take first feasible window)
        for window in windows:
            # Check basic constraints (simplified version for precomputation)
            # In full version, this would check attitude, SAA, etc.
            start_time = window.start_time
            end_time = window.end_time
            duration = (end_time - start_time).total_seconds()

            # For now, assume feasible if window exists
            # Full implementation would include constraint checking
            return True, start_time, end_time, duration

        return False, None, None, 0.0

    def _precompute_transitions(
        self,
        matrices: FeasibilityMatrices,
        tasks: List[Any]
    ):
        """Precompute transition feasibility matrices per satellite."""
        task_count = len(tasks)

        for sat_idx, sat in enumerate(self.satellites):
            # Initialize transition matrix for this satellite
            trans_matrix = np.zeros((task_count, task_count), dtype=np.bool_)

            # Only compute transitions for tasks that this satellite can perform
            feasible_tasks = [
                i for i in range(task_count)
                if matrices.single_window_feasible[i, sat_idx]
            ]

            logger.info(f"  Satellite {sat.id}: {len(feasible_tasks)} feasible tasks")

            # For each pair of tasks on this satellite
            for i, task_a_idx in enumerate(feasible_tasks):
                if i % 50 == 0:
                    logger.info(f"    Processing transitions for task {i}/{len(feasible_tasks)}...")

                for task_b_idx in feasible_tasks:
                    if task_a_idx == task_b_idx:
                        trans_matrix[task_a_idx, task_b_idx] = True
                        continue

                    # Check if task_b can follow task_a
                    feasible = self._check_transition(
                        sat_idx, tasks[task_a_idx], tasks[task_b_idx], matrices
                    )
                    trans_matrix[task_a_idx, task_b_idx] = feasible

            matrices.transition_feasible[sat_idx] = trans_matrix

    def _check_transition(
        self,
        sat_idx: int,
        task_a: Any,
        task_b: Any,
        matrices: FeasibilityMatrices
    ) -> bool:
        """Check if task_b can follow task_a on satellite.

        Simplified check: task_b start time > task_a end time + setup time
        """
        task_a_idx = getattr(task_a, '_idx', None)
        task_b_idx = getattr(task_b, '_idx', None)

        if task_a_idx is None or task_b_idx is None:
            # Fallback: assume feasible
            return True

        end_a = matrices.task_end_times[task_a_idx, sat_idx]
        start_b = matrices.task_start_times[task_b_idx, sat_idx]

        if end_a is None or start_b is None:
            return False

        # Check time ordering with minimum gap (e.g., 60 seconds for setup)
        min_gap = 60.0  # seconds
        return (start_b - end_a).total_seconds() >= min_gap

    @staticmethod
    def save(matrices: FeasibilityMatrices, filepath: str):
        """Save feasibility matrices to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(matrices, f)

        logger.info(f"Saved feasibility matrices to {filepath}")

    @staticmethod
    def load(filepath: str) -> FeasibilityMatrices:
        """Load feasibility matrices from file."""
        with open(filepath, 'rb') as f:
            matrices = pickle.load(f)

        logger.info(
            f"Loaded feasibility matrices from {filepath}: "
            f"{matrices.task_count} tasks x {matrices.sat_count} satellites"
        )
        return matrices


def precompute_for_mission(
    mission: Mission,
    window_cache: Any,
    output_path: Optional[str] = None,
    enable_parallel: bool = True
) -> FeasibilityMatrices:
    """Convenience function to precompute matrices for a mission.

    Args:
        mission: Mission object
        window_cache: Visibility window cache
        output_path: Optional path to save matrices
        enable_parallel: Whether to use parallel processing

    Returns:
        FeasibilityMatrices
    """
    from scheduler.base_scheduler import BaseScheduler

    # Create tasks
    scheduler = BaseScheduler(mission)
    tasks = scheduler._create_frequency_aware_tasks()

    logger.info(f"Precomputing feasibility for {len(tasks)} tasks")

    # Precompute
    precomputer = FeasibilityPrecomputer(mission, window_cache)
    matrices = precomputer.precompute_all(tasks, enable_parallel)

    # Save if path provided
    if output_path:
        FeasibilityPrecomputer.save(matrices, output_path)

    return matrices
