"""Tests for entity repository pattern."""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.entity.repository.base import EntityRepository
from utils.entity.repository.json_repository import JSONEntityRepository


class TestEntityRepositoryInterface:
    """Test that EntityRepository defines the correct interface."""

    def test_repository_is_abstract(self):
        """EntityRepository should be abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            EntityRepository()

    def test_repository_defines_satellite_methods(self):
        """Repository should define satellite template methods."""
        # Check abstract methods exist
        assert hasattr(EntityRepository, 'get_satellite_template')
        assert hasattr(EntityRepository, 'list_satellite_templates')
        assert hasattr(EntityRepository, 'save_satellite_template')
        assert hasattr(EntityRepository, 'delete_satellite_template')

    def test_repository_defines_target_methods(self):
        """Repository should define target methods."""
        assert hasattr(EntityRepository, 'get_target')
        assert hasattr(EntityRepository, 'list_targets')
        assert hasattr(EntityRepository, 'save_target')
        assert hasattr(EntityRepository, 'delete_target')
        assert hasattr(EntityRepository, 'query_targets_by_region')

    def test_repository_defines_ground_station_methods(self):
        """Repository should define ground station methods."""
        assert hasattr(EntityRepository, 'get_ground_station')
        assert hasattr(EntityRepository, 'list_ground_stations')
        assert hasattr(EntityRepository, 'save_ground_station')
        assert hasattr(EntityRepository, 'delete_ground_station')


class TestJSONEntityRepository:
    """Test JSON file-based repository implementation."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository for testing."""
        repo_path = tmp_path / "entity_lib"
        return JSONEntityRepository(str(repo_path))

    def test_repository_creates_directory_structure(self, temp_repo):
        """Repository should create necessary directories on init."""
        base_path = Path(temp_repo.base_path)
        assert (base_path / "satellites").exists()
        assert (base_path / "targets" / "point").exists()
        assert (base_path / "targets" / "area").exists()
        assert (base_path / "ground_stations").exists()

    def test_save_and_get_satellite_template(self, temp_repo):
        """Should save and retrieve satellite template."""
        template = {
            "template_id": "optical_1",
            "name": "光学卫星1型",
            "sat_type": "optical_1",
            "orbit": {"altitude": 645000, "inclination": 97.9}
        }

        temp_repo.save_satellite_template(template)
        retrieved = temp_repo.get_satellite_template("optical_1")

        assert retrieved == template

    def test_get_nonexistent_satellite_returns_none(self, temp_repo):
        """Getting non-existent template should return None."""
        result = temp_repo.get_satellite_template("nonexistent")
        assert result is None

    def test_list_satellite_templates(self, temp_repo):
        """Should list all satellite templates."""
        templates = [
            {"template_id": "optical_1", "name": "光学1型"},
            {"template_id": "optical_2", "name": "光学2型"},
            {"template_id": "sar_1", "name": "SAR1型"}
        ]

        for template in templates:
            temp_repo.save_satellite_template(template)

        result = temp_repo.list_satellite_templates()

        assert len(result) == 3
        template_ids = [t["template_id"] for t in result]
        assert "optical_1" in template_ids
        assert "optical_2" in template_ids
        assert "sar_1" in template_ids

    def test_delete_satellite_template(self, temp_repo):
        """Should delete satellite template."""
        template = {"template_id": "test_sat", "name": "测试卫星"}
        temp_repo.save_satellite_template(template)

        assert temp_repo.get_satellite_template("test_sat") is not None

        result = temp_repo.delete_satellite_template("test_sat")

        assert result is True
        assert temp_repo.get_satellite_template("test_sat") is None

    def test_delete_nonexistent_satellite_returns_false(self, temp_repo):
        """Deleting non-existent template should return False."""
        result = temp_repo.delete_satellite_template("nonexistent")
        assert result is False

    def test_save_and_get_target(self, temp_repo):
        """Should save and retrieve target."""
        target = {
            "id": "beijing",
            "name": "北京",
            "target_type": "point",
            "position": {"longitude": 116.4, "latitude": 39.9, "altitude": 0}
        }

        temp_repo.save_target(target, target_type="point")
        retrieved = temp_repo.get_target("beijing", target_type="point")

        assert retrieved == target

    def test_save_and_get_ground_station(self, temp_repo):
        """Should save and retrieve ground station."""
        gs = {
            "id": "gs_beijing",
            "name": "北京地面站",
            "longitude": 116.4,
            "latitude": 39.9
        }

        temp_repo.save_ground_station(gs)
        retrieved = temp_repo.get_ground_station("gs_beijing")

        assert retrieved == gs

    def test_query_targets_by_region(self, temp_repo):
        """Should query targets within geographic region."""
        targets = [
            {"id": "beijing", "name": "北京", "position": {"longitude": 116.4, "latitude": 39.9}},
            {"id": "shanghai", "name": "上海", "position": {"longitude": 121.5, "latitude": 31.2}},
            {"id": "guangzhou", "name": "广州", "position": {"longitude": 113.3, "latitude": 23.1}},
        ]

        for target in targets:
            temp_repo.save_target(target, target_type="point")

        # Query around Beijing (within 1 degree)
        result = temp_repo.query_targets_by_region(115, 117, 39, 40)

        assert len(result) == 1
        assert result[0]["id"] == "beijing"

    def test_query_targets_by_region_no_results(self, temp_repo):
        """Should return empty list when no targets in region."""
        target = {"id": "beijing", "name": "北京", "position": {"longitude": 116.4, "latitude": 39.9}}
        temp_repo.save_target(target, target_type="point")

        result = temp_repo.query_targets_by_region(0, 10, 0, 10)

        assert result == []


class TestJSONEntityRepositoryEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def temp_repo(self, tmp_path):
        """Create a temporary repository for testing."""
        repo_path = tmp_path / "entity_lib"
        return JSONEntityRepository(str(repo_path))

    def test_handles_invalid_json_in_file(self, temp_repo):
        """Should handle invalid JSON in existing file gracefully."""
        # Create an invalid JSON file
        sat_path = Path(temp_repo.base_path) / "satellites" / "broken.json"
        sat_path.parent.mkdir(parents=True, exist_ok=True)
        sat_path.write_text("invalid json {{{")

        result = temp_repo.get_satellite_template("broken")
        assert result is None

    def test_handles_missing_directory(self, tmp_path):
        """Should handle missing base directory."""
        nonexistent_path = tmp_path / "nonexistent" / "path"
        repo = JSONEntityRepository(str(nonexistent_path))

        # Should create directories automatically
        assert Path(repo.base_path).exists()

    def test_overwrites_existing_template(self, temp_repo):
        """Saving should overwrite existing template with same ID."""
        template_v1 = {"template_id": "test", "name": "Test v1", "version": "1.0"}
        template_v2 = {"template_id": "test", "name": "Test v2", "version": "2.0"}

        temp_repo.save_satellite_template(template_v1)
        temp_repo.save_satellite_template(template_v2)

        retrieved = temp_repo.get_satellite_template("test")
        assert retrieved["version"] == "2.0"
        assert retrieved["name"] == "Test v2"

    def test_list_returns_empty_list_when_no_templates(self, temp_repo):
        """Should return empty list when no templates exist."""
        result = temp_repo.list_satellite_templates()
        assert result == []

    def test_list_returns_empty_list_when_directory_missing(self, tmp_path):
        """Should return empty list when satellites directory doesn't exist."""
        repo_path = tmp_path / "empty_repo"
        repo = JSONEntityRepository(str(repo_path))

        # Delete the satellites directory that was created
        import shutil
        shutil.rmtree(repo.base_path / "satellites")

        result = repo.list_satellite_templates()
        assert result == []

    def test_save_target_without_id_raises_error(self, temp_repo):
        """Saving target without id should raise ValueError."""
        target = {"name": "No ID Target"}
        with pytest.raises(ValueError, match="id"):
            temp_repo.save_target(target)

    def test_save_satellite_without_template_id_raises_error(self, temp_repo):
        """Saving satellite without template_id should raise ValueError."""
        template = {"name": "No ID Satellite"}
        with pytest.raises(ValueError, match="template_id"):
            temp_repo.save_satellite_template(template)

    def test_save_ground_station_without_id_raises_error(self, temp_repo):
        """Saving ground station without id should raise ValueError."""
        gs = {"name": "No ID Ground Station"}
        with pytest.raises(ValueError, match="id"):
            temp_repo.save_ground_station(gs)

    def test_delete_nonexistent_target_returns_false(self, temp_repo):
        """Deleting non-existent target should return False."""
        result = temp_repo.delete_target("nonexistent")
        assert result is False

    def test_delete_nonexistent_ground_station_returns_false(self, temp_repo):
        """Deleting non-existent ground station should return False."""
        result = temp_repo.delete_ground_station("nonexistent")
        assert result is False

    def test_list_targets_by_type(self, temp_repo):
        """Should list targets filtered by type."""
        point_target = {"id": "point1", "name": "Point Target"}
        area_target = {"id": "area1", "name": "Area Target"}

        temp_repo.save_target(point_target, target_type="point")
        temp_repo.save_target(area_target, target_type="area")

        point_results = temp_repo.list_targets(target_type="point")
        area_results = temp_repo.list_targets(target_type="area")

        assert len(point_results) == 1
        assert point_results[0]["id"] == "point1"
        assert len(area_results) == 1
        assert area_results[0]["id"] == "area1"

    def test_list_targets_target_type_none_lists_all(self, temp_repo):
        """list_targets with target_type=None should list all targets."""
        point_target = {"id": "point1", "name": "Point Target"}
        area_target = {"id": "area1", "name": "Area Target"}

        temp_repo.save_target(point_target, target_type="point")
        temp_repo.save_target(area_target, target_type="area")

        all_results = temp_repo.list_targets(target_type=None)

        assert len(all_results) == 2
