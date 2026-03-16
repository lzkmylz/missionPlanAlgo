"""Scheduler common components.

This module provides shared components for all schedulers:
- ResourceManager: Unified resource management
- ConstraintChecker: Unified constraint checking
- SchedulerConfig: Centralized configuration
- SchedulerFactory: Dependency injection factory
"""

from .resource_manager import ResourceManager, ResourceSnapshot, ResourceAllocation, TaskTimeManager
from .config import SchedulerConfig, ConstraintConfig, ResourceConfig, MetaheuristicConfig
from .constraint_checker import ConstraintChecker, ConstraintContext, ConstraintResult, ConstraintType
from .factory import SchedulerFactory, create_scheduler
from .footprint_utils import (
    calculate_task_footprint,
    fill_footprint_to_task,
    calculate_haversine_distance,
    calculate_footprint_center_from_attitude,
    calculate_center_distance_score,
)

__all__ = [
    # Resource management
    'ResourceManager',
    'ResourceSnapshot',
    'ResourceAllocation',
    'TaskTimeManager',
    # Configuration
    'SchedulerConfig',
    'ConstraintConfig',
    'ResourceConfig',
    'MetaheuristicConfig',
    # Constraint checking
    'ConstraintChecker',
    'ConstraintContext',
    'ConstraintResult',
    'ConstraintType',
    # Factory
    'SchedulerFactory',
    'create_scheduler',
    # Footprint utilities
    'calculate_task_footprint',
    'fill_footprint_to_task',
    'calculate_haversine_distance',
    'calculate_footprint_center_from_attitude',
    'calculate_center_distance_score',
]
