"""
Unified storage manager for satellite mission planning data.

Implements hybrid storage architecture:
- Relational data (MySQL/SQLite) for configuration and metadata
- Time-series data (Parquet/HDF5) for high-frequency samples
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class StorageBackend(Enum):
    """Storage backend types"""
    MYSQL = "mysql"
    SQLITE = "sqlite"
    PARQUET = "parquet"
    HDF5 = "hdf5"


@dataclass
class StorageConfig:
    """Storage configuration

    Attributes:
        relational_backend: Backend for relational data (MySQL/SQLite)
        timeseries_backend: Backend for time-series data (Parquet/HDF5)
        mysql_host: MySQL server host
        mysql_port: MySQL server port
        mysql_database: MySQL database name
        mysql_user: MySQL username
        mysql_password: MySQL password
        sqlite_path: Path to SQLite database file
        timeseries_base_path: Base directory for time-series files
        parquet_compression: Compression codec for Parquet files
    """
    relational_backend: StorageBackend = StorageBackend.SQLITE
    timeseries_backend: StorageBackend = StorageBackend.PARQUET

    # MySQL configuration
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "satellite_mission"
    mysql_user: str = "root"
    mysql_password: str = ""

    # SQLite configuration
    sqlite_path: str = "./data/experiments.db"

    # Time-series configuration
    timeseries_base_path: str = "./data/timeseries"
    parquet_compression: str = "zstd"

    def __post_init__(self):
        """Validate configuration"""
        if isinstance(self.relational_backend, str):
            self.relational_backend = StorageBackend(self.relational_backend)
        if isinstance(self.timeseries_backend, str):
            self.timeseries_backend = StorageBackend(self.timeseries_backend)


class StorageManager:
    """Unified storage manager

    Coordinates relational and time-series storage backends.
    Provides high-level interface for experiment data persistence.

    Example:
        config = StorageConfig(
            relational_backend=StorageBackend.SQLITE,
            sqlite_path="./experiments.db"
        )
        storage = StorageManager(config)

        # Save experiment result
        experiment_id = storage.save_experiment_result({
            'name': 'benchmark_test',
            'algorithm': 'GA',
            'metrics': {...}
        })
    """

    def __init__(self, config: StorageConfig):
        """Initialize storage manager

        Args:
            config: Storage configuration
        """
        self.config = config
        self.relational: Optional[Any] = None
        self.timeseries: Optional[Any] = None

        self._initialize_storage()

    def _initialize_storage(self) -> None:
        """Initialize storage backends based on configuration"""
        # Initialize relational storage
        if self.config.relational_backend == StorageBackend.MYSQL:
            from .mysql_storage import MySQLStorage
            self.relational = MySQLStorage(self.config)
            logger.info("Initialized MySQL storage")
        elif self.config.relational_backend == StorageBackend.SQLITE:
            from .sqlite_storage import SQLiteStorage
            self.relational = SQLiteStorage(self.config)
            logger.info("Initialized SQLite storage")
        else:
            raise ValueError(f"Unsupported relational backend: {self.config.relational_backend}")

        # Initialize time-series storage
        if self.config.timeseries_backend == StorageBackend.PARQUET:
            from .parquet_storage import ParquetStorage
            self.timeseries = ParquetStorage(self.config)
            logger.info("Initialized Parquet storage")
        elif self.config.timeseries_backend == StorageBackend.HDF5:
            # TODO: Implement HDF5 storage
            raise NotImplementedError("HDF5 storage not yet implemented")

        # Create base directories
        Path(self.config.timeseries_base_path).mkdir(parents=True, exist_ok=True)

    def connect(self) -> None:
        """Connect to all storage backends"""
        if self.relational:
            self.relational.connect()
        logger.info("Storage manager connected")

    def close(self) -> None:
        """Close all storage connections"""
        if self.relational:
            self.relational.close()
        logger.info("Storage manager closed")

    def create_tables(self) -> None:
        """Create all database tables"""
        if self.relational:
            self.relational.create_tables()
        logger.info("Database tables created")

    def save_experiment_result(self, result: Dict[str, Any]) -> int:
        """Save complete experiment result

        Saves experiment metadata to relational database and
        time-series data to appropriate backend.

        Args:
            result: Experiment result dictionary containing:
                - name: Experiment name
                - scenario_id: Scenario ID
                - algorithm_name: Algorithm used
                - metrics: Performance metrics
                - scheduled_tasks: List of scheduled tasks
                - convergence_data: Optional convergence curve data

        Returns:
            experiment_id: ID of saved experiment
        """
        # Ensure connection and tables
        if not self.relational or not self.relational.is_connected():
            self.connect()
            self.create_tables()

        try:
            self.relational.begin_transaction()

            # 1. Insert experiment record
            experiment_data = {
                'name': result.get('name', 'Unnamed Experiment'),
                'scenario_id': result.get('scenario_id'),
                'description': result.get('description', ''),
                'algorithms': str([result.get('algorithm_name')]),
                'status': 'completed'
            }

            experiment_id = self.relational.insert('experiments', experiment_data)

            # 2. Insert experiment run
            metrics = result.get('metrics', {})
            run_data = {
                'experiment_id': experiment_id,
                'run_number': 1,
                'algorithm_name': result.get('algorithm_name', 'Unknown'),
                'status': 'success',
                'demand_satisfaction_rate': metrics.get('demand_satisfaction_rate', 0.0),
                'makespan': metrics.get('makespan', 0.0),
                'avg_revisit_time': metrics.get('avg_revisit_time'),
                'data_delivery_time': metrics.get('data_delivery_time'),
                'computation_time': metrics.get('computation_time', 0.0),
                'solution_quality': metrics.get('solution_quality')
            }

            run_id = self.relational.insert('experiment_runs', run_data)

            # 3. Insert scheduled tasks (skip incomplete tasks)
            from datetime import datetime
            for task in result.get('scheduled_tasks', []):
                # Skip tasks without required timing info
                if not task.get('imaging_start') or not task.get('imaging_end'):
                    continue
                task_data = {
                    'run_id': run_id,
                    'target_id': task.get('target_id'),
                    'satellite_id': task.get('satellite_id'),
                    'imaging_start': task.get('imaging_start'),
                    'imaging_end': task.get('imaging_end'),
                    'imaging_mode': task.get('imaging_mode')
                }
                self.relational.insert('schedule_tasks', task_data)

            # 4. Insert performance metrics
            metrics_data = {
                'run_id': run_id,
                'total_tasks': metrics.get('total_tasks', 0),
                'scheduled_tasks': metrics.get('scheduled_tasks', 0),
                'unscheduled_tasks': metrics.get('unscheduled_tasks', 0),
                'demand_satisfaction_rate': metrics.get('demand_satisfaction_rate', 0.0),
                'makespan': metrics.get('makespan', 0.0),
                'computation_time': metrics.get('computation_time', 0.0),
                'solution_quality': metrics.get('solution_quality')
            }
            self.relational.insert('performance_metrics', metrics_data)

            # 5. Save time-series data (convergence curve)
            if 'convergence_data' in result and self.timeseries:
                self.timeseries.write_convergence_data(
                    run_id,
                    result.get('algorithm_name', ''),
                    result['convergence_data']
                )

            self.relational.commit()
            logger.info(f"Saved experiment result with ID: {experiment_id}")
            return experiment_id

        except Exception as e:
            self.relational.rollback()
            logger.error(f"Failed to save experiment result: {e}")
            raise

    def get_experiment_result(self, experiment_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve experiment result by ID

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment result dictionary or None if not found
        """
        if not self.relational:
            return None

        # Fetch experiment
        experiment = self.relational.fetch_one(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,)
        )

        if not experiment:
            return None

        # Fetch associated runs
        runs = self.relational.fetch_all(
            "SELECT * FROM experiment_runs WHERE experiment_id = ?",
            (experiment_id,)
        )

        # Build result dictionary
        result = {
            'id': experiment['id'],
            'name': experiment['name'],
            'scenario_id': experiment['scenario_id'],
            'runs': runs
        }

        return result

    def list_experiments(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all experiments

        Args:
            limit: Maximum number of experiments to return

        Returns:
            List of experiment dictionaries
        """
        if not self.relational:
            return []

        return self.relational.fetch_all(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )

    def save_satellite_state_snapshots(self, run_id: int, satellite_id: str,
                                       snapshots: List[Dict[str, Any]]) -> str:
        """Save satellite state snapshots to time-series storage

        Args:
            run_id: Experiment run ID
            satellite_id: Satellite ID
            snapshots: List of state snapshot dictionaries

        Returns:
            Path to saved file
        """
        if not self.timeseries:
            raise RuntimeError("Time-series storage not configured")

        return self.timeseries.write_satellite_snapshots(
            run_id, satellite_id, snapshots
        )
