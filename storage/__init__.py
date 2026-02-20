"""
Storage module for satellite mission planning.

Provides unified storage interface with support for:
- Relational databases (MySQL, SQLite)
- Time-series storage (Parquet, HDF5)

Usage:
    from storage import StorageManager, StorageConfig

    config = StorageConfig(
        relational_backend=StorageBackend.SQLITE,
        sqlite_path="./data/experiments.db"
    )
    storage = StorageManager(config)
"""

from .storage_manager import StorageManager, StorageConfig, StorageBackend
from .schema import get_create_table_sql, TABLES

__all__ = [
    'StorageManager',
    'StorageConfig',
    'StorageBackend',
    'get_create_table_sql',
    'TABLES'
]
