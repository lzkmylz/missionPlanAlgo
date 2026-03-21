"""
scheduler.isl — ISL (Inter-Satellite Link) downlink scheduling.

This package implements the third downlink strategy tier:

    1. Direct GS downlink (primary)   — scheduler.ground_station
    2. GEO relay (secondary)          — scheduler.relay
    3. ISL multi-hop relay (tertiary) — scheduler.isl  ← this package

Public API
----------
ISLDownlinkTask
    Data model representing a single ISL-routed relay task.
ISLDownlinkScheduler
    Scheduler that assigns imaging tasks to ISL relay paths.
"""

from .isl_downlink_task import ISLDownlinkTask
from .isl_downlink_scheduler import ISLDownlinkScheduler

__all__ = [
    'ISLDownlinkTask',
    'ISLDownlinkScheduler',
]
