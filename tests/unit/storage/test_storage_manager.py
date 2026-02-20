"""
Test suite for storage management layer.
Tests StorageManager, SQLiteStorage, and MySQLStorage backends.
"""
import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

# Skip if storage module not yet implemented
pytest.importorskip("storage", reason="Storage module not yet implemented")

from storage.storage_manager import StorageManager, StorageConfig, StorageBackend
from storage.sqlite_storage import SQLiteStorage
from storage.schema import get_create_table_sql


class TestStorageConfig:
    """Test storage configuration"""

    def test_default_config(self):
        """Test default storage configuration"""
        config = StorageConfig()
        assert config.relational_backend == StorageBackend.SQLITE
        assert config.timeseries_backend == StorageBackend.PARQUET
        assert config.sqlite_path == "./data/experiments.db"

    def test_mysql_config(self):
        """Test MySQL configuration"""
        config = StorageConfig(
            relational_backend=StorageBackend.MYSQL,
            mysql_host="localhost",
            mysql_port=3306,
            mysql_database="test_db"
        )
        assert config.relational_backend == StorageBackend.MYSQL
        assert config.mysql_host == "localhost"


class TestSQLiteStorage:
    """Test SQLite storage backend"""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def storage(self, temp_db):
        """Create SQLite storage instance"""
        config = StorageConfig(
            relational_backend=StorageBackend.SQLITE,
            sqlite_path=temp_db
        )
        storage = SQLiteStorage(config)
        storage.connect()
        # Create tables
        storage.create_tables()
        yield storage
        storage.close()

    def test_connect(self, temp_db):
        """Test database connection"""
        config = StorageConfig(sqlite_path=temp_db)
        storage = SQLiteStorage(config)
        storage.connect()
        assert storage.conn is not None
        storage.close()

    def test_create_tables(self, storage):
        """Test table creation"""
        # Verify tables exist
        tables = storage.get_tables()
        expected_tables = [
            'satellites', 'ground_stations', 'antennas', 'targets', 'scenarios',
            'scenario_satellites', 'scenario_targets', 'scenario_ground_stations',
            'experiments', 'experiment_runs', 'algorithm_params',
            'schedule_tasks', 'performance_metrics', 'constraint_violations'
        ]
        for table in expected_tables:
            assert table in tables, f"Table {table} should exist"

    def test_insert_and_fetch_satellite(self, storage):
        """Test satellite CRUD operations"""
        satellite_data = {
            'id': 'OPT-01',
            'name': '光学卫星01',
            'sat_type': 'optical_1',
            'orbit_type': 'SSO',
            'altitude': 500000,
            'inclination': 97.4,
            'max_off_nadir': 30.0,
            'storage_capacity': 500,
            'power_capacity': 2000,
            'data_rate': 300
        }

        # Insert
        storage.insert('satellites', satellite_data)

        # Fetch
        result = storage.fetch_one(
            "SELECT * FROM satellites WHERE id = ?",
            ('OPT-01',)
        )

        assert result is not None
        assert result['id'] == 'OPT-01'
        assert result['name'] == '光学卫星01'
        assert result['sat_type'] == 'optical_1'

    def test_insert_experiment_run(self, storage):
        """Test experiment run insertion"""
        # First insert parent records
        storage.insert('scenarios', {
            'id': 1,
            'name': '测试场景',
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(hours=24)
        })

        storage.insert('experiments', {
            'id': 1,
            'name': '测试实验',
            'scenario_id': 1,
            'algorithms': '["GA", "SA"]'
        })

        # Insert experiment run
        run_data = {
            'id': 1,
            'experiment_id': 1,
            'run_number': 1,
            'algorithm_name': 'GA',
            'status': 'success',
            'demand_satisfaction_rate': 0.85,
            'makespan': 3600.0,
            'computation_time': 45.5
        }
        storage.insert('experiment_runs', run_data)

        result = storage.fetch_one(
            "SELECT * FROM experiment_runs WHERE id = ?",
            (1,)
        )
        assert result is not None
        assert result['algorithm_name'] == 'GA'
        assert result['demand_satisfaction_rate'] == 0.85

    def test_transaction_rollback(self, storage):
        """Test transaction rollback on error"""
        storage.begin_transaction()

        try:
            storage.insert('satellites', {
                'id': 'TEST-01',
                'name': 'Test Satellite',
                'sat_type': 'optical_1'
            })
            # Simulate error
            raise Exception("Simulated error")
        except Exception:
            storage.rollback()

        # Verify record was not inserted
        result = storage.fetch_one(
            "SELECT * FROM satellites WHERE id = ?",
            ('TEST-01',)
        )
        assert result is None


class TestStorageManager:
    """Test unified storage manager"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_initialization(self, temp_dir):
        """Test storage manager initialization"""
        config = StorageConfig(
            relational_backend=StorageBackend.SQLITE,
            sqlite_path=os.path.join(temp_dir, "test.db")
        )
        manager = StorageManager(config)
        assert manager.relational is not None
        assert manager.config == config

    def test_save_experiment_result(self, temp_dir):
        """Test saving complete experiment result"""
        config = StorageConfig(
            relational_backend=StorageBackend.SQLITE,
            sqlite_path=os.path.join(temp_dir, "test.db")
        )
        manager = StorageManager(config)

        # First create a scenario (required by foreign key)
        manager.connect()
        manager.create_tables()
        from datetime import datetime, timedelta
        scenario_id = manager.relational.insert('scenarios', {
            'name': 'Test Scenario',
            'start_time': datetime.now(),
            'end_time': datetime.now() + timedelta(hours=24)
        })

        # Create mock result
        result = {
            'name': '基准测试',
            'scenario_id': scenario_id,
            'algorithm_name': 'GA',
            'metrics': {
                'demand_satisfaction_rate': 0.92,
                'makespan': 7200.0,
                'computation_time': 120.5
            },
            'scheduled_tasks': [
                {'task_id': 'T001', 'satellite_id': 'OPT-01'}
            ]
        }

        # Should not raise exception
        experiment_id = manager.save_experiment_result(result)
        assert experiment_id is not None
        assert isinstance(experiment_id, int)


class TestSchema:
    """Test database schema definitions"""

    def test_satellites_table_schema(self):
        """Test satellites table schema"""
        sql = get_create_table_sql('satellites')
        assert 'CREATE TABLE satellites' in sql
        assert 'id VARCHAR(32) PRIMARY KEY' in sql
        assert 'sat_type' in sql
        assert 'storage_capacity' in sql

    def test_experiment_runs_table_schema(self):
        """Test experiment_runs table schema"""
        sql = get_create_table_sql('experiment_runs')
        assert 'CREATE TABLE experiment_runs' in sql
        assert 'experiment_id' in sql
        assert 'demand_satisfaction_rate' in sql
        assert 'makespan' in sql

    def test_all_tables_defined(self):
        """Test that all 21 tables have schema definitions"""
        from storage.schema import TABLES
        expected_tables = 21
        assert len(TABLES) == expected_tables, f"Expected {expected_tables} tables, got {len(TABLES)}"
