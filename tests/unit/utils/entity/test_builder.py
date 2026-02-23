"""Tests for scenario builder."""
import pytest
import json
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, mock_open
from copy import deepcopy

from utils.entity.builder import ScenarioBuilder


class TestScenarioBuilderInit:
    """Test scenario initialization."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository."""
        return Mock()

    @pytest.fixture
    def builder(self, mock_repo):
        """Create ScenarioBuilder with mock repository."""
        return ScenarioBuilder(repository=mock_repo)

    def test_init_scenario_basic(self, builder, tmp_path):
        """Should initialize empty scenario with basic structure."""
        output_path = str(tmp_path / "test_scenario.json")

        scenario = builder.init_scenario(output_path, with_metadata=False)

        assert scenario["name"] == "test_scenario"
        assert scenario["description"] == ""
        assert scenario["start_time"] == "2024-01-01T00:00:00Z"
        assert scenario["end_time"] == "2024-01-02T00:00:00Z"
        assert scenario["satellites"] == []
        assert scenario["targets"] == []
        assert scenario["ground_stations"] == []
        assert "metadata" not in scenario

    def test_init_scenario_with_metadata(self, builder, tmp_path):
        """Should initialize scenario with metadata."""
        output_path = str(tmp_path / "test_scenario.json")

        scenario = builder.init_scenario(output_path, with_metadata=True)

        assert "metadata" in scenario
        assert scenario["metadata"]["coordinate_system"] == "WGS84"
        assert scenario["metadata"]["time_system"] == "UTC"
        assert scenario["metadata"]["time_format"] == "ISO8601"
        assert "created_at" in scenario["metadata"]
        assert scenario["metadata"]["version"] == "1.0"

    def test_init_scenario_extracts_name_from_path(self, builder, tmp_path):
        """Should extract scenario name from output path."""
        output_path = str(tmp_path / "my_custom_scenario.json")

        scenario = builder.init_scenario(output_path)

        assert scenario["name"] == "my_custom_scenario"


class TestScenarioBuilderAddSatellite:
    """Test adding satellites to scenario."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository."""
        mock = Mock()
        mock.get_satellite_template.return_value = {
            "template_id": "optical_1",
            "name": "光学卫星1型",
            "sat_type": "optical_1",
            "orbit": {"altitude": 645000, "inclination": 97.9},
            "capabilities": {"storage_capacity": 500}
        }
        return mock

    @pytest.fixture
    def builder(self, mock_repo):
        """Create ScenarioBuilder with mock repository."""
        return ScenarioBuilder(repository=mock_repo)

    @pytest.fixture
    def empty_scenario(self):
        """Create empty scenario."""
        return {
            "name": "test",
            "satellites": [],
            "targets": [],
            "ground_stations": []
        }

    def test_add_satellite_from_template(self, builder, mock_repo, empty_scenario):
        """Should add satellite from template."""
        result = builder.add_satellite_to_scenario(
            empty_scenario,
            template_id="optical_1",
            sat_id="SAT-01"
        )

        assert len(result["satellites"]) == 1
        sat = result["satellites"][0]
        assert sat["id"] == "SAT-01"
        assert sat["_template_source"] == "optical_1"
        assert sat["_template_version"] == "1.0"
        assert sat["orbit"]["altitude"] == 645000

        # Should be a deep copy, not reference
        mock_repo.get_satellite_template.assert_called_once_with("optical_1")

    def test_add_satellite_with_overrides(self, builder, mock_repo, empty_scenario):
        """Should add satellite with override parameters."""
        overrides = {"orbit": {"raan": 180.0}}

        result = builder.add_satellite_to_scenario(
            empty_scenario,
            template_id="optical_1",
            sat_id="SAT-01",
            overrides=overrides
        )

        sat = result["satellites"][0]
        assert sat["orbit"]["raan"] == 180.0
        # Original fields preserved
        assert sat["orbit"]["altitude"] == 645000

    def test_add_multiple_satellites(self, builder, mock_repo, empty_scenario):
        """Should add multiple satellites to scenario."""
        builder.add_satellite_to_scenario(
            empty_scenario, template_id="optical_1", sat_id="SAT-01"
        )
        builder.add_satellite_to_scenario(
            empty_scenario, template_id="optical_1", sat_id="SAT-02"
        )

        assert len(empty_scenario["satellites"]) == 2
        assert empty_scenario["satellites"][0]["id"] == "SAT-01"
        assert empty_scenario["satellites"][1]["id"] == "SAT-02"

    def test_add_satellite_template_not_found(self, builder, mock_repo, empty_scenario):
        """Should raise error when template not found."""
        mock_repo.get_satellite_template.return_value = None

        with pytest.raises(ValueError, match="Template not found"):
            builder.add_satellite_to_scenario(
                empty_scenario, template_id="nonexistent", sat_id="SAT-01"
            )

    def test_add_satellite_is_deep_copy(self, builder, mock_repo, empty_scenario):
        """Should create deep copy of template, not reference."""
        # Modify the scenario satellite
        builder.add_satellite_to_scenario(
            empty_scenario, template_id="optical_1", sat_id="SAT-01"
        )
        empty_scenario["satellites"][0]["orbit"]["altitude"] = 999999

        # Original template should not be modified
        template = mock_repo.get_satellite_template.return_value
        assert template["orbit"]["altitude"] == 645000


class TestScenarioBuilderClone:
    """Test cloning scenarios."""

    @pytest.fixture
    def mock_repo(self):
        return Mock()

    @pytest.fixture
    def builder(self, mock_repo):
        return ScenarioBuilder(repository=mock_repo)

    def test_clone_scenario(self, builder, tmp_path):
        """Should clone existing scenario."""
        source_path = tmp_path / "source.json"
        source_scenario = {
            "name": "source",
            "description": "Original scenario",
            "satellites": [{"id": "SAT-01"}],
            "metadata": {"version": "1.0", "created_at": "2024-01-01T00:00:00Z"}
        }
        source_path.write_text(json.dumps(source_scenario))

        output_path = str(tmp_path / "cloned.json")
        result = builder.clone_scenario(str(source_path), output_path)

        assert result["name"] == "cloned"
        assert result["description"] == "Original scenario"
        assert len(result["satellites"]) == 1
        assert result["satellites"][0]["id"] == "SAT-01"
        # Should update created_at
        assert "created_at" in result["metadata"]

    def test_clone_preserves_all_fields(self, builder, tmp_path):
        """Should preserve all fields when cloning."""
        source_path = tmp_path / "source.json"
        source_scenario = {
            "name": "source",
            "description": "Test",
            "start_time": "2024-06-01T00:00:00Z",
            "end_time": "2024-06-02T00:00:00Z",
            "satellites": [],
            "targets": [{"id": "T1"}],
            "ground_stations": [{"id": "GS1"}]
        }
        source_path.write_text(json.dumps(source_scenario))

        output_path = str(tmp_path / "cloned.json")
        result = builder.clone_scenario(str(source_path), output_path)

        assert result["start_time"] == "2024-06-01T00:00:00Z"
        assert result["end_time"] == "2024-06-02T00:00:00Z"
        assert len(result["targets"]) == 1
        assert len(result["ground_stations"]) == 1


class TestScenarioBuilderSave:
    """Test saving scenarios."""

    @pytest.fixture
    def mock_repo(self):
        return Mock()

    @pytest.fixture
    def builder(self, mock_repo):
        return ScenarioBuilder(repository=mock_repo)

    def test_save_scenario(self, builder, tmp_path):
        """Should save scenario to file."""
        scenario = {
            "name": "test",
            "satellites": [{"id": "SAT-01"}]
        }
        output_path = str(tmp_path / "output.json")

        builder.save_scenario(scenario, output_path)

        assert Path(output_path).exists()
        with open(output_path, 'r') as f:
            saved = json.load(f)
        assert saved["name"] == "test"
        assert saved["satellites"][0]["id"] == "SAT-01"

    def test_save_scenario_creates_directory(self, builder, tmp_path):
        """Should create output directory if not exists."""
        scenario = {"name": "test"}
        output_path = str(tmp_path / "subdir" / "nested" / "output.json")

        builder.save_scenario(scenario, output_path)

        assert Path(output_path).exists()

    def test_save_scenario_preserves_unicode(self, builder, tmp_path):
        """Should preserve unicode characters."""
        scenario = {
            "name": "测试场景",
            "satellites": [{"id": "卫星01", "name": "光学卫星"}]
        }
        output_path = str(tmp_path / "output.json")

        builder.save_scenario(scenario, output_path)

        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "测试场景" in content
        assert "光学卫星" in content
