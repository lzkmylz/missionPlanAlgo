"""
Database schema definitions for satellite mission planning.

Defines all 21 tables across 7 layers as specified in design document Chapter 10:
1. Configuration Layer (5 tables)
2. Scenario Instance Layer (3 tables)
3. Target Decomposition Layer (2 tables)
4. Experiment Layer (3 tables)
5. Results Layer (3 tables)
6. Status Layer (1 table)
7. Network Layer (4 tables)
"""

from typing import Dict

# All table names
TABLES = [
    # Configuration Layer
    'satellites', 'ground_stations', 'antennas', 'targets', 'scenarios',
    # Scenario Instance Layer
    'scenario_satellites', 'scenario_targets', 'scenario_ground_stations',
    # Target Decomposition Layer
    'decomposition_configs', 'sub_targets',
    # Experiment Layer
    'experiments', 'experiment_runs', 'algorithm_params',
    # Results Layer
    'schedule_tasks', 'performance_metrics', 'task_sequences',
    # Status Layer
    'constraint_violations',
    # Network Layer
    'isl_windows', 'data_routing_paths', 'uplink_commands', 'relay_satellites'
]

# Table schema definitions (MySQL syntax)
SCHEMAS: Dict[str, str] = {
    # ============== Configuration Layer ==============
    'satellites': """
        CREATE TABLE satellites (
            id VARCHAR(32) PRIMARY KEY,
            name VARCHAR(64) NOT NULL,
            sat_type VARCHAR(16) NOT NULL,
            orbit_type VARCHAR(16) DEFAULT 'SSO',
            altitude INT,
            inclination DECIMAL(6,2),
            orbit_params TEXT,
            max_off_nadir DECIMAL(5,2),
            agility TEXT,
            storage_capacity INT,
            power_capacity INT,
            data_rate INT,
            supported_modes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    'ground_stations': """
        CREATE TABLE ground_stations (
            id VARCHAR(32) PRIMARY KEY,
            name VARCHAR(64) NOT NULL,
            longitude DECIMAL(9,6) NOT NULL,
            latitude DECIMAL(8,6) NOT NULL,
            altitude DECIMAL(8,2) DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    'antennas': """
        CREATE TABLE antennas (
            id VARCHAR(32) PRIMARY KEY,
            ground_station_id VARCHAR(32) NOT NULL,
            name VARCHAR(64) NOT NULL,
            elevation_min DECIMAL(5,2) DEFAULT 5.0,
            elevation_max DECIMAL(5,2) DEFAULT 90.0,
            data_rate INT,
            slew_rate DECIMAL(5,2),
            FOREIGN KEY (ground_station_id) REFERENCES ground_stations(id) ON DELETE CASCADE
        )
    """,

    'targets': """
        CREATE TABLE targets (
            id VARCHAR(32) PRIMARY KEY,
            name VARCHAR(128),
            target_type VARCHAR(16) NOT NULL,
            longitude DECIMAL(9,6),
            latitude DECIMAL(8,6),
            area_vertices TEXT,
            priority INT DEFAULT 1,
            required_observations INT DEFAULT 1,
            time_window_start TIMESTAMP NULL,
            time_window_end TIMESTAMP NULL,
            immediate_downlink BOOLEAN DEFAULT FALSE,
            resolution_required DECIMAL(6,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,

    'scenarios': """
        CREATE TABLE scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(128) NOT NULL,
            version INT DEFAULT 1,
            description TEXT,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP NOT NULL,
            satellite_count INT,
            target_count INT,
            config_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            UNIQUE(name, version)
        )
    """,

    # ============== Scenario Instance Layer ==============
    'scenario_satellites': """
        CREATE TABLE scenario_satellites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id INT NOT NULL,
            satellite_id VARCHAR(32) NOT NULL,
            initial_storage_used DECIMAL(5,2) DEFAULT 0,
            initial_power_used DECIMAL(5,2) DEFAULT 0,
            FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE,
            FOREIGN KEY (satellite_id) REFERENCES satellites(id),
            UNIQUE(scenario_id, satellite_id)
        )
    """,

    'scenario_targets': """
        CREATE TABLE scenario_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id INT NOT NULL,
            target_id VARCHAR(32) NOT NULL,
            adjusted_priority INT,
            adjusted_observations INT,
            FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES targets(id),
            UNIQUE(scenario_id, target_id)
        )
    """,

    'scenario_ground_stations': """
        CREATE TABLE scenario_ground_stations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id INT NOT NULL,
            ground_station_id VARCHAR(32) NOT NULL,
            FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE,
            FOREIGN KEY (ground_station_id) REFERENCES ground_stations(id),
            UNIQUE(scenario_id, ground_station_id)
        )
    """,

    # ============== Target Decomposition Layer ==============
    'decomposition_configs': """
        CREATE TABLE decomposition_configs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target_id VARCHAR(32) NOT NULL,
            scenario_id INT NOT NULL,
            strategy VARCHAR(16) NOT NULL,
            resolution DECIMAL(6,2),
            sat_type VARCHAR(16),
            strip_direction DECIMAL(6,2),
            strip_overlap DECIMAL(4,2) DEFAULT 0.1,
            grid_size DECIMAL(8,2),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (target_id) REFERENCES targets(id),
            FOREIGN KEY (scenario_id) REFERENCES scenarios(id),
            UNIQUE(target_id, scenario_id, sat_type)
        )
    """,

    'sub_targets': """
        CREATE TABLE sub_targets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_target_id VARCHAR(32) NOT NULL,
            scenario_id INT NOT NULL,
            sub_target_code VARCHAR(64) NOT NULL,
            sub_type VARCHAR(16) NOT NULL,
            center_lon DECIMAL(9,6),
            center_lat DECIMAL(8,6),
            vertices TEXT,
            area_sqkm DECIMAL(10,2),
            strip_length DECIMAL(10,2),
            required_mode VARCHAR(32),
            estimated_duration INT,
            status VARCHAR(16) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_target_id) REFERENCES targets(id),
            FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
        )
    """,

    # ============== Experiment Layer ==============
    'experiments': """
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(128) NOT NULL,
            scenario_id INT NOT NULL,
            description TEXT,
            algorithms TEXT NOT NULL,
            repetitions INT DEFAULT 10,
            status VARCHAR(16) DEFAULT 'pending',
            started_at TIMESTAMP NULL,
            completed_at TIMESTAMP NULL,
            best_algorithm VARCHAR(64),
            best_makespan DECIMAL(12,2),
            FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
        )
    """,

    'experiment_runs': """
        CREATE TABLE experiment_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INT NOT NULL,
            run_number INT NOT NULL,
            algorithm_name VARCHAR(64) NOT NULL,
            algorithm_version VARCHAR(32),
            status VARCHAR(16) DEFAULT 'running',
            started_at TIMESTAMP NULL,
            completed_at TIMESTAMP NULL,
            demand_satisfaction_rate DECIMAL(5,4),
            makespan DECIMAL(12,2),
            avg_revisit_time DECIMAL(12,2),
            data_delivery_time DECIMAL(12,2),
            computation_time DECIMAL(10,3),
            solution_quality DECIMAL(5,4),
            FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
            UNIQUE(experiment_id, run_number, algorithm_name)
        )
    """,

    'algorithm_params': """
        CREATE TABLE algorithm_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INT NOT NULL,
            param_name VARCHAR(64) NOT NULL,
            param_value VARCHAR(256),
            FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE,
            UNIQUE(run_id, param_name)
        )
    """,

    # ============== Results Layer ==============
    'schedule_tasks': """
        CREATE TABLE schedule_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INT NOT NULL,
            sub_target_id INT,
            target_id VARCHAR(32),
            satellite_id VARCHAR(32) NOT NULL,
            antenna_id VARCHAR(32),
            imaging_start TIMESTAMP NOT NULL,
            imaging_end TIMESTAMP NOT NULL,
            imaging_mode VARCHAR(32),
            slew_start TIMESTAMP,
            slew_end TIMESTAMP,
            slew_angle DECIMAL(6,2),
            downlink_start TIMESTAMP,
            downlink_end TIMESTAMP,
            downlink_station_id VARCHAR(32),
            storage_before DECIMAL(5,2),
            storage_after DECIMAL(5,2),
            sequence_number INT,
            FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
        )
    """,

    'performance_metrics': """
        CREATE TABLE performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INT NOT NULL,
            total_tasks INT,
            scheduled_tasks INT,
            unscheduled_tasks INT,
            demand_satisfaction_rate DECIMAL(5,4),
            makespan DECIMAL(12,2),
            avg_revisit_time DECIMAL(12,2),
            max_revisit_time DECIMAL(12,2),
            data_delivery_time DECIMAL(12,2),
            computation_time DECIMAL(10,3),
            solution_quality DECIMAL(5,4),
            avg_satellite_utilization DECIMAL(5,4),
            max_satellite_load INT,
            ground_station_utilization DECIMAL(5,4),
            load_balance_std DECIMAL(10,4),
            FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE,
            UNIQUE(run_id)
        )
    """,

    'task_sequences': """
        CREATE TABLE task_sequences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INT NOT NULL,
            satellite_id VARCHAR(32) NOT NULL,
            task_sequence TEXT NOT NULL,
            total_idle_time INT,
            idle_segments INT,
            FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE,
            UNIQUE(run_id, satellite_id)
        )
    """,

    # ============== Status Layer ==============
    'constraint_violations': """
        CREATE TABLE constraint_violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INT NOT NULL,
            violation_type VARCHAR(16) NOT NULL,
            satellite_id VARCHAR(32),
            violation_time TIMESTAMP,
            expected_value DECIMAL(10,2),
            limit_value DECIMAL(10,2),
            description TEXT,
            FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
        )
    """,

    # ============== Network Layer ==============
    'isl_windows': """
        CREATE TABLE isl_windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id INT NOT NULL,
            satellite_a_id VARCHAR(32) NOT NULL,
            satellite_b_id VARCHAR(32) NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP NOT NULL,
            link_quality DECIMAL(3,2),
            max_data_rate DECIMAL(10,2),
            distance DECIMAL(10,2),
            FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE
        )
    """,

    'data_routing_paths': """
        CREATE TABLE data_routing_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INT NOT NULL,
            task_id VARCHAR(32) NOT NULL,
            source_satellite VARCHAR(32),
            destination_gs VARCHAR(32),
            route_hops TEXT,
            total_latency DECIMAL(10,3),
            used_relay BOOLEAN DEFAULT FALSE,
            relay_satellite_id VARCHAR(32),
            FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
        )
    """,

    'uplink_commands': """
        CREATE TABLE uplink_commands (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INT NOT NULL,
            satellite_id VARCHAR(32) NOT NULL,
            command_type VARCHAR(64),
            uplink_time TIMESTAMP,
            scheduled_task_id VARCHAR(32),
            link_type VARCHAR(16),
            ground_station_id VARCHAR(32),
            relay_satellite_id VARCHAR(32),
            FOREIGN KEY (run_id) REFERENCES experiment_runs(id) ON DELETE CASCADE
        )
    """,

    'relay_satellites': """
        CREATE TABLE relay_satellites (
            id VARCHAR(32) PRIMARY KEY,
            name VARCHAR(64),
            orbit_type VARCHAR(32),
            longitude DECIMAL(9,6),
            uplink_capacity DECIMAL(10,2),
            downlink_capacity DECIMAL(10,2),
            coverage_zones TEXT
        )
    """
}


def get_create_table_sql(table_name: str) -> str:
    """Get CREATE TABLE SQL for specified table

    Args:
        table_name: Name of the table

    Returns:
        SQL CREATE TABLE statement

    Raises:
        KeyError: If table name is not recognized
    """
    if table_name not in SCHEMAS:
        raise KeyError(f"Unknown table: {table_name}. Available tables: {list(SCHEMAS.keys())}")
    return SCHEMAS[table_name]


def get_all_schemas() -> Dict[str, str]:
    """Get all table schemas

    Returns:
        Dictionary mapping table names to SQL statements
    """
    return SCHEMAS.copy()
