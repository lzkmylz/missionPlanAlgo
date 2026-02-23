"""Tests for scenario validator."""
import pytest
import json
from unittest.mock import Mock, patch, mock_open
from datetime import datetime

from utils.entity.validator import ScenarioValidator


class TestScenarioValidatorBasic:
    """Test basic validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create validator with default schema."""
        return ScenarioValidator()

    def test_validator_loads_default_schema(self, validator):
        """Should load default JSON schema."""
        assert validator.schema is not None
        assert "type" in validator.schema

    def test_validate_valid_scenario(self, validator):
        """Should pass validation for valid scenario."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is True
        assert errors == []

    def test_validate_invalid_time_range(self, validator):
        """Should fail if end_time <= start_time."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-02T00:00:00Z",
            "end_time": "2024-01-01T00:00:00Z",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is False
        assert any("end_time" in e for e in errors)


class TestScenarioValidatorSatelliteRules:
    """Test satellite validation rules."""

    @pytest.fixture
    def validator(self):
        return ScenarioValidator()

    def test_validate_satellite_altitude_out_of_range(self, validator):
        """Should fail if satellite altitude out of range."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [
                {
                    "id": "SAT-01",
                    "orbit": {"altitude": 200000}  # Too low
                }
            ],
            "targets": [],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is False
        assert any("altitude" in e.lower() for e in errors)

    def test_validate_satellite_inclination_out_of_range(self, validator):
        """Should fail if satellite inclination out of range."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [
                {
                    "id": "SAT-01",
                    "orbit": {"altitude": 645000, "inclination": 200}  # Too high
                }
            ],
            "targets": [],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is False
        assert any("inclination" in e.lower() for e in errors)

    def test_validate_satellite_eccentricity_out_of_range(self, validator):
        """Should fail if satellite eccentricity out of range."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [
                {
                    "id": "SAT-01",
                    "orbit": {"altitude": 645000, "eccentricity": 0.5}  # Too high
                }
            ],
            "targets": [],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is False
        assert any("eccentricity" in e.lower() for e in errors)

    def test_validate_multiple_satellites(self, validator):
        """Should validate all satellites in scenario."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [
                {"id": "SAT-01", "orbit": {"altitude": 200000}},  # Bad
                {"id": "SAT-02", "orbit": {"altitude": 645000}}   # Good
            ],
            "targets": [],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is False
        assert any("SAT-01" in e for e in errors)


class TestScenarioValidatorTargetRules:
    """Test target validation rules."""

    @pytest.fixture
    def validator(self):
        return ScenarioValidator()

    def test_validate_target_longitude_out_of_range(self, validator):
        """Should fail if target longitude out of range."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [
                {
                    "id": "TARGET-01",
                    "position": {"longitude": 200, "latitude": 39.9}
                }
            ],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is False
        assert any("longitude" in e.lower() for e in errors)

    def test_validate_target_latitude_out_of_range(self, validator):
        """Should fail if target latitude out of range."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [
                {
                    "id": "TARGET-01",
                    "position": {"longitude": 116.4, "latitude": 100}
                }
            ],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is False
        assert any("latitude" in e.lower() for e in errors)

    def test_validate_target_priority_out_of_range(self, validator):
        """Should fail if target priority out of range."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [
                {
                    "id": "TARGET-01",
                    "priority": 15  # Too high
                }
            ],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is False
        assert any("priority" in e.lower() for e in errors)

    def test_validate_target_with_position_dict(self, validator):
        """Should handle target with position dict."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [
                {
                    "id": "TARGET-01",
                    "longitude": 116.4,  # Direct field
                    "latitude": 39.9
                }
            ],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is True


class TestScenarioValidatorGroundStationRules:
    """Test ground station validation rules."""

    @pytest.fixture
    def validator(self):
        return ScenarioValidator()

    def test_validate_ground_station_elevation_out_of_range(self, validator):
        """Should fail if antenna elevation out of range."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [],
            "targets": [],
            "ground_stations": [
                {
                    "id": "GS-01",
                    "antennas": [
                        {"elevation_min": -10, "elevation_max": 90}  # Bad
                    ]
                }
            ]
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert is_valid is False
        assert any("elevation" in e.lower() for e in errors)


class TestScenarioValidatorTemplateVersionCheck:
    """Test template version warnings."""

    @pytest.fixture
    def validator(self):
        return ScenarioValidator()

    @patch('utils.entity.validator.JSONEntityRepository')
    def test_warns_on_outdated_template_version(self, mock_repo_class, validator):
        """Should warn when scenario uses old template version."""
        # Mock repository to return newer template
        mock_repo = Mock()
        mock_repo.get_satellite_template.return_value = {
            "template_id": "optical_1",
            "version": "2.0"
        }
        mock_repo_class.return_value = mock_repo

        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [
                {
                    "id": "SAT-01",
                    "_template_source": "optical_1",
                    "_template_version": "1.0",
                    "orbit": {"altitude": 645000}
                }
            ],
            "targets": [],
            "ground_stations": []
        }

        is_valid, errors, warnings = validator.validate(scenario)

        assert any("version" in w.lower() for w in warnings)


class TestScenarioValidatorVerboseOutput:
    """Test verbose validation output."""

    @pytest.fixture
    def validator(self):
        return ScenarioValidator()

    @patch('utils.entity.validator.RICH_AVAILABLE', True)
    @patch('rich.console.Console')
    def test_verbose_output_prints_report(self, mock_console_class, validator):
        """Should print formatted report in verbose mode."""
        scenario = {
            "name": "test",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-02T00:00:00Z",
            "satellites": [
                {"id": "SAT-01", "orbit": {"altitude": 645000}}
            ],
            "targets": [],
            "ground_stations": []
        }

        # Create a mock console instance
        mock_console = Mock()
        mock_console_class.return_value = mock_console

        # Recreate validator to use mocked console
        validator_with_mock = ScenarioValidator()
        validator_with_mock.console = mock_console

        is_valid, errors, warnings = validator_with_mock.validate(scenario, verbose=True)

        # Console print should be called
        mock_console.print.assert_called()
