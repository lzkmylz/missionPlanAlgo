"""Resource management for satellite schedulers.

This module provides unified resource tracking and management for all scheduler types.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from copy import deepcopy


@dataclass
class ResourceAllocation:
    """Result of a resource allocation attempt."""
    feasible: bool
    power_before: float = 0.0
    power_after: float = 0.0
    storage_before: float = 0.0
    storage_after: float = 0.0
    power_consumed: float = 0.0
    storage_used: float = 0.0


@dataclass
class ResourceSnapshot:
    """Snapshot of resource state for metaheuristic algorithms."""
    power: Dict[str, float] = field(default_factory=dict)
    storage: Dict[str, float] = field(default_factory=dict)
    last_task_end: Dict[str, datetime] = field(default_factory=dict)
    scheduled_tasks: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


class SatelliteResourceState:
    """Resource state for a single satellite."""

    def __init__(self, satellite: Any, start_time: datetime):
        self.satellite_id = satellite.id
        self.start_time = start_time

        # Initialize power with fallback for Mock objects
        current_power = getattr(satellite, 'current_power', None)
        try:
            if current_power is not None and isinstance(current_power, (int, float)) and current_power > 0:
                self.power = float(current_power)
            else:
                cap = getattr(satellite.capabilities, 'power_capacity', None)
                self.power = float(cap) if isinstance(cap, (int, float)) else 1000.0
        except (TypeError, AttributeError, ValueError):
            self.power = 1000.0  # Default fallback value

        # Initialize storage
        self.storage = 0.0

        # Track scheduled tasks for conflict detection
        self.last_task_end = start_time
        self.scheduled_tasks: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backward compatibility)."""
        return {
            'power': self.power,
            'storage': self.storage,
            'last_task_end': self.last_task_end,
            'scheduled_tasks': self.scheduled_tasks.copy(),
        }


class ResourceManager:
    """Unified resource manager for satellite schedulers.

    This class centralizes resource tracking and constraint checking,
    eliminating duplicate code across scheduler implementations.

    Attributes:
        consider_power: Whether to enforce power constraints
        consider_storage: Whether to enforce storage constraints
    """

    def __init__(
        self,
        satellites: List[Any],
        start_time: datetime,
        consider_power: bool = True,
        consider_storage: bool = True,
    ):
        """Initialize resource manager.

        Args:
            satellites: List of satellite objects
            start_time: Mission start time
            consider_power: Whether to track power usage
            consider_storage: Whether to track storage usage
        """
        self.consider_power = consider_power
        self.consider_storage = consider_storage
        self._states: Dict[str, SatelliteResourceState] = {
            sat.id: SatelliteResourceState(sat, start_time)
            for sat in satellites
        }
        self._satellites: Dict[str, Any] = {sat.id: sat for sat in satellites}
        self._start_time = start_time

    def get_state(self, sat_id: str) -> Optional[SatelliteResourceState]:
        """Get resource state for a satellite."""
        return self._states.get(sat_id)

    def get_power(self, sat_id: str) -> float:
        """Get current power level for a satellite."""
        state = self._states.get(sat_id)
        return state.power if state else 0.0

    def get_storage(self, sat_id: str) -> float:
        """Get current storage usage for a satellite."""
        state = self._states.get(sat_id)
        return state.storage if state else 0.0

    def check_resources(
        self,
        sat_id: str,
        power_needed: float = 0.0,
        storage_needed: float = 0.0,
    ) -> bool:
        """Check if satellite has sufficient resources.

        Args:
            sat_id: Satellite ID
            power_needed: Power required (if consider_power is True)
            storage_needed: Storage required (if consider_storage is True)

        Returns:
            True if resources are sufficient
        """
        state = self._states.get(sat_id)
        if not state:
            return False

        # Check power constraint
        if self.consider_power and power_needed > 0:
            if state.power < power_needed:
                return False

        # Check storage constraint
        if self.consider_storage and storage_needed > 0:
            satellite = self._satellites.get(sat_id)
            if satellite:
                capacity = satellite.capabilities.storage_capacity
                if state.storage + storage_needed > capacity:
                    return False

        return True

    def allocate(
        self,
        sat_id: str,
        power_consumed: float = 0.0,
        storage_used: float = 0.0,
        task_start: Optional[datetime] = None,
        task_end: Optional[datetime] = None,
        task_info: Optional[Dict[str, Any]] = None,
    ) -> ResourceAllocation:
        """Allocate resources for a task.

        Args:
            sat_id: Satellite ID
            power_consumed: Power to consume
            storage_used: Storage to use
            task_start: Task start time
            task_end: Task end time
            task_info: Additional task information for tracking

        Returns:
            ResourceAllocation with before/after states
        """
        state = self._states.get(sat_id)
        if not state:
            return ResourceAllocation(feasible=False)

        # Check feasibility
        power_needed = power_consumed if self.consider_power else 0.0
        storage_needed = storage_used if self.consider_storage else 0.0

        if not self.check_resources(sat_id, power_needed, storage_needed):
            return ResourceAllocation(feasible=False)

        # Record allocation
        result = ResourceAllocation(
            feasible=True,
            power_before=state.power,
            storage_before=state.storage,
        )

        # Update power
        if self.consider_power:
            state.power -= power_consumed
            result.power_after = state.power
            result.power_consumed = power_consumed

        # Update storage
        if self.consider_storage:
            state.storage += storage_used
            result.storage_after = state.storage
            result.storage_used = storage_used

        # Update task tracking
        if task_end:
            state.last_task_end = task_end

        if task_info and task_start and task_end:
            state.scheduled_tasks.append({
                'start': task_start,
                'end': task_end,
                **task_info,
            })

        return result

    def release(
        self,
        sat_id: str,
        power_released: float = 0.0,
        storage_released: float = 0.0,
    ) -> None:
        """Release resources (for metaheuristic rollback).

        Args:
            sat_id: Satellite ID
            power_released: Power to restore
            storage_released: Storage to free
        """
        state = self._states.get(sat_id)
        if not state:
            return

        if self.consider_power:
            state.power += power_released

        if self.consider_storage:
            state.storage = max(0.0, state.storage - storage_released)

    def has_time_conflict(
        self,
        sat_id: str,
        start: datetime,
        end: datetime,
    ) -> bool:
        """Check if time range conflicts with scheduled tasks.

        Args:
            sat_id: Satellite ID
            start: Proposed start time
            end: Proposed end time

        Returns:
            True if there is a conflict
        """
        state = self._states.get(sat_id)
        if not state:
            return False

        for task in state.scheduled_tasks:
            existing_start = task['start']
            existing_end = task['end']
            if not (end <= existing_start or start >= existing_end):
                return True

        return False

    def get_scheduled_tasks(self, sat_id: str) -> List[Dict[str, Any]]:
        """Get list of scheduled tasks for a satellite."""
        state = self._states.get(sat_id)
        return state.scheduled_tasks.copy() if state else []

    def get_last_task_end(self, sat_id: str) -> datetime:
        """Get the end time of the last scheduled task."""
        state = self._states.get(sat_id)
        return state.last_task_end if state else self._start_time

    def snapshot(self) -> ResourceSnapshot:
        """Create a snapshot of current resource states.

        Used by metaheuristic algorithms for state rollback.

        Returns:
            ResourceSnapshot containing all states
        """
        return ResourceSnapshot(
            power={sid: s.power for sid, s in self._states.items()},
            storage={sid: s.storage for sid, s in self._states.items()},
            last_task_end={sid: s.last_task_end for sid, s in self._states.items()},
            scheduled_tasks={sid: s.scheduled_tasks.copy() for sid, s in self._states.items()},
        )

    def restore(self, snapshot: ResourceSnapshot) -> None:
        """Restore resource states from a snapshot.

        Args:
            snapshot: ResourceSnapshot to restore from
        """
        for sat_id, state in self._states.items():
            if sat_id in snapshot.power:
                state.power = snapshot.power[sat_id]
            if sat_id in snapshot.storage:
                state.storage = snapshot.storage[sat_id]
            if sat_id in snapshot.last_task_end:
                state.last_task_end = snapshot.last_task_end[sat_id]
            if sat_id in snapshot.scheduled_tasks:
                state.scheduled_tasks = snapshot.scheduled_tasks[sat_id].copy()

    def reset(self) -> None:
        """Reset all resource states to initial values."""
        for state in self._states.values():
            # Note: This requires keeping initial values or reinitializing
            # For now, we'll just clear scheduled tasks
            state.scheduled_tasks = []

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert all states to dictionary format (backward compatibility)."""
        return {sid: state.to_dict() for sid, state in self._states.items()}


class TaskTimeManager:
    """Manages task time tracking for metaheuristic algorithms.

    This is a lighter-weight alternative to ResourceManager for algorithms
    that only need time conflict tracking without full resource management.
    """

    def __init__(self, sat_count: int):
        """Initialize task time manager.

        Args:
            sat_count: Number of satellites
        """
        self.sat_count = sat_count
        self._task_times: Dict[int, List[Tuple[datetime, datetime]]] = {
            i: [] for i in range(sat_count)
        }

    def is_time_feasible(
        self,
        sat_idx: int,
        start: datetime,
        end: datetime,
    ) -> bool:
        """Check if time range is feasible (no conflicts).

        Args:
            sat_idx: Satellite index
            start: Proposed start time
            end: Proposed end time

        Returns:
            True if no conflicts
        """
        for existing_start, existing_end in self._task_times.get(sat_idx, []):
            if not (end <= existing_start or start >= existing_end):
                return False
        return True

    def add_task_time(
        self,
        sat_idx: int,
        start: datetime,
        end: datetime,
    ) -> None:
        """Add a task time entry.

        Args:
            sat_idx: Satellite index
            start: Task start time
            end: Task end time
        """
        if sat_idx not in self._task_times:
            self._task_times[sat_idx] = []
        self._task_times[sat_idx].append((start, end))

    def remove_task_time(
        self,
        sat_idx: int,
        start: datetime,
        end: datetime,
    ) -> bool:
        """Remove a task time entry.

        Args:
            sat_idx: Satellite index
            start: Task start time
            end: Task end time

        Returns:
            True if removed successfully
        """
        if sat_idx not in self._task_times:
            return False

        try:
            self._task_times[sat_idx].remove((start, end))
            return True
        except ValueError:
            return False

    def get_task_times(self, sat_idx: int) -> List[Tuple[datetime, datetime]]:
        """Get all task times for a satellite."""
        return self._task_times.get(sat_idx, []).copy()

    def reset(self) -> None:
        """Reset all task times."""
        self._task_times = {i: [] for i in range(self.sat_count)}

    def snapshot(self) -> Dict[int, List[Tuple[datetime, datetime]]]:
        """Create a snapshot of task times."""
        return {k: v.copy() for k, v in self._task_times.items()}

    def restore(self, snapshot: Dict[int, List[Tuple[datetime, datetime]]]) -> None:
        """Restore task times from snapshot."""
        self._task_times = {k: v.copy() for k, v in snapshot.items()}
