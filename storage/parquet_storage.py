"""
Parquet time-series storage implementation.

Optimized for high-frequency satellite state snapshots and convergence data.
Uses columnar storage with compression for efficient querying.
"""
from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

from .storage_manager import StorageConfig

logger = logging.getLogger(__name__)


class ParquetStorage:
    """Parquet time-series storage backend

    Optimized for:
    - High-frequency state snapshots (50 satellites × 1 sample/minute)
    - Convergence curve data from metaheuristic algorithms
    - Compression ratios of 5-10x compared to CSV

    File structure:
        timeseries/
        ├── run_{run_id}/
        │   ├── snapshots_{satellite_id}.parquet
        │   └── convergence.parquet
    """

    def __init__(self, config: StorageConfig):
        """Initialize Parquet storage

        Args:
            config: Storage configuration
        """
        self.config = config
        self.base_path = Path(config.timeseries_base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        if not HAS_PYARROW:
            logger.warning(
                "PyArrow not installed. Parquet storage will not function. "
                "Install with: pip install pyarrow pandas"
            )

    def _get_run_path(self, run_id: int) -> Path:
        """Get path for specific run"""
        run_path = self.base_path / f"run_{run_id}"
        run_path.mkdir(exist_ok=True)
        return run_path

    def write_satellite_snapshots(self, run_id: int, satellite_id: str,
                                   snapshots: List[Dict[str, Any]]) -> str:
        """Write satellite state snapshots to Parquet

        Args:
            run_id: Experiment run ID
            satellite_id: Satellite ID
            snapshots: List of state snapshot dictionaries

        Returns:
            Path to saved Parquet file
        """
        if not HAS_PYARROW:
            raise RuntimeError("PyArrow not installed")

        if not snapshots:
            return ""

        run_path = self._get_run_path(run_id)
        file_path = run_path / f"snapshots_{satellite_id}.parquet"

        # Convert to DataFrame
        df = pd.DataFrame(snapshots)

        # Optimize data types
        df = self._optimize_dataframe(df)

        # Write Parquet
        df.to_parquet(
            file_path,
            engine='pyarrow',
            compression=self.config.parquet_compression,
            index=False
        )

        logger.info(f"Wrote {len(snapshots)} snapshots to {file_path}")
        return str(file_path)

    def read_satellite_snapshots(self, run_id: int, satellite_id: Optional[str] = None,
                                  start_time: Optional[Any] = None,
                                  end_time: Optional[Any] = None) -> Any:
        """Read satellite state snapshots from Parquet

        Args:
            run_id: Experiment run ID
            satellite_id: Optional satellite ID filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            Pandas DataFrame with snapshots
        """
        if not HAS_PYARROW:
            raise RuntimeError("PyArrow not installed")

        run_path = self._get_run_path(run_id)

        if satellite_id:
            file_path = run_path / f"snapshots_{satellite_id}.parquet"
            if not file_path.exists():
                return pd.DataFrame()

            # Build filters for predicate pushdown
            filters = None
            if start_time and end_time:
                filters = [
                    ('timestamp', '>=', pd.Timestamp(start_time)),
                    ('timestamp', '<=', pd.Timestamp(end_time))
                ]

            return pd.read_parquet(file_path, filters=filters)
        else:
            # Read all satellites
            all_dfs = []
            for file_path in run_path.glob("snapshots_*.parquet"):
                all_dfs.append(pd.read_parquet(file_path))

            if not all_dfs:
                return pd.DataFrame()

            return pd.concat(all_dfs, ignore_index=True)

    def write_convergence_data(self, run_id: int, algorithm_name: str,
                               convergence_data: List[float]) -> str:
        """Write convergence curve data

        Args:
            run_id: Experiment run ID
            algorithm_name: Algorithm name
            convergence_data: List of fitness values

        Returns:
            Path to saved Parquet file
        """
        if not HAS_PYARROW:
            raise RuntimeError("PyArrow not installed")

        run_path = self._get_run_path(run_id)
        file_path = run_path / "convergence.parquet"

        # Create DataFrame
        df = pd.DataFrame({
            'iteration': range(len(convergence_data)),
            'fitness': convergence_data,
            'algorithm': algorithm_name
        })

        df.to_parquet(
            file_path,
            engine='pyarrow',
            compression=self.config.parquet_compression,
            index=False
        )

        logger.info(f"Wrote {len(convergence_data)} convergence points to {file_path}")
        return str(file_path)

    def _optimize_dataframe(self, df: Any) -> Any:
        """Optimize DataFrame for storage efficiency

        - Compress float64 to float32
        - Use appropriate integer types
        - Convert low-cardinality strings to categorical
        """
        df = df.copy()

        for col in df.columns:
            col_type = df[col].dtype

            # Optimize floats
            if col_type == 'float64':
                df[col] = df[col].astype('float32')

            # Optimize integers
            elif col_type == 'int64':
                if df[col].min() >= 0 and df[col].max() < 65535:
                    df[col] = df[col].astype('uint16')
                else:
                    df[col] = df[col].astype('int32')

        return df
