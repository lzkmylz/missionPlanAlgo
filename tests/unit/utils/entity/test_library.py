"""Tests for entity library management."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from utils.entity.library import EntityLibrary


class TestEntityLibrary:
    """Test EntityLibrary class."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository."""
        return Mock()

    @pytest.fixture
    def library(self, mock_repo):
        """Create EntityLibrary with mock repository."""
        return EntityLibrary(repository=mock_repo)

    def test_library_initializes_with_repository(self, mock_repo):
        """Should initialize with provided repository."""
        lib = EntityLibrary(repository=mock_repo)
        assert lib.repo == mock_repo

    def test_library_creates_default_repository(self):
        """Should create default JSON repository if none provided."""
        with patch('utils.entity.library.JSONEntityRepository') as mock_json_repo:
            mock_instance = Mock()
            mock_json_repo.return_value = mock_instance

            lib = EntityLibrary()

            mock_json_repo.assert_called_once()
            assert lib.repo == mock_instance

    def test_list_satellites_delegates_to_repository(self, library, mock_repo):
        """list_satellites should delegate to repository."""
        expected = [{"template_id": "optical_1"}, {"template_id": "sar_1"}]
        mock_repo.list_satellite_templates.return_value = expected

        result = library.list_satellites()

        mock_repo.list_satellite_templates.assert_called_once()
        assert result == expected

    def test_get_satellite_delegates_to_repository(self, library, mock_repo):
        """get_satellite should delegate to repository."""
        expected = {"template_id": "optical_1", "name": "光学1型"}
        mock_repo.get_satellite_template.return_value = expected

        result = library.get_satellite("optical_1")

        mock_repo.get_satellite_template.assert_called_once_with("optical_1")
        assert result == expected

    def test_list_targets_delegates_to_repository(self, library, mock_repo):
        """list_targets should delegate to repository."""
        expected = [{"id": "target_1"}, {"id": "target_2"}]
        mock_repo.list_targets.return_value = expected

        result = library.list_targets()

        mock_repo.list_targets.assert_called_once()
        assert result == expected

    def test_list_ground_stations_delegates_to_repository(self, library, mock_repo):
        """list_ground_stations should delegate to repository."""
        expected = [
            {"id": "gs_beijing", "name": "北京地面站"},
            {"id": "gs_kashi", "name": "喀什地面站"}
        ]
        mock_repo.list_ground_stations.return_value = expected

        result = library.list_ground_stations()

        mock_repo.list_ground_stations.assert_called_once()
        assert result == expected

    def test_get_ground_station_delegates_to_repository(self, library, mock_repo):
        """get_ground_station should delegate to repository."""
        expected = {"id": "gs_beijing", "name": "北京地面站"}
        mock_repo.get_ground_station.return_value = expected

        result = library.get_ground_station("gs_beijing")

        mock_repo.get_ground_station.assert_called_once_with("gs_beijing")
        assert result == expected

    def test_add_target_creates_target_dict(self, library, mock_repo):
        """add_target should create proper target dictionary."""
        library.add_target(
            target_id="beijing",
            name="北京",
            lon=116.4,
            lat=39.9,
            priority=8
        )

        mock_repo.save_target.assert_called_once()
        saved_target = mock_repo.save_target.call_args[0][0]

        assert saved_target["id"] == "beijing"
        assert saved_target["name"] == "北京"
        assert saved_target["target_type"] == "point"
        assert saved_target["position"]["longitude"] == 116.4
        assert saved_target["position"]["latitude"] == 39.9
        assert saved_target["position"]["altitude"] == 0
        assert saved_target["priority"] == 8

    def test_add_target_with_custom_priority(self, library, mock_repo):
        """add_target should use custom priority."""
        library.add_target(
            target_id="test",
            name="Test",
            lon=0.0,
            lat=0.0,
            priority=5
        )

        saved_target = mock_repo.save_target.call_args[0][0]
        assert saved_target["priority"] == 5

    def test_add_target_with_additional_kwargs(self, library, mock_repo):
        """add_target should accept additional keyword arguments."""
        library.add_target(
            target_id="test",
            name="Test",
            lon=0.0,
            lat=0.0,
            priority=5,
            resolution_required=2.0,
            required_observations=3
        )

        saved_target = mock_repo.save_target.call_args[0][0]
        assert saved_target["resolution_required"] == 2.0
        assert saved_target["required_observations"] == 3


class TestEntityLibraryInitDefaults:
    """Test initializing entity library with default templates."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository."""
        return Mock()

    @pytest.fixture
    def library(self, mock_repo):
        """Create EntityLibrary with mock repository."""
        return EntityLibrary(repository=mock_repo)

    def test_init_defaults_saves_satellite_templates(self, library, mock_repo):
        """init_defaults should save default satellite templates."""
        library.init_defaults()

        # Should save at least 4 default satellite templates
        assert mock_repo.save_satellite_template.call_count >= 4

    def test_init_defaults_saves_ground_stations(self, library, mock_repo):
        """init_defaults should save default ground stations."""
        library.init_defaults()

        # Should save at least 2 default ground stations
        assert mock_repo.save_ground_station.call_count >= 2


class TestEntityLibraryInteractiveWizard:
    """Test interactive wizard for adding satellites."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository."""
        return Mock()

    @pytest.fixture
    def library(self, mock_repo):
        """Create EntityLibrary with mock repository."""
        return EntityLibrary(repository=mock_repo)

    @patch('utils.entity.library.click')
    def test_add_satellite_interactive_prompts_for_fields(self, mock_click, library, mock_repo):
        """add_satellite_interactive should prompt for required fields."""
        mock_click.prompt.side_effect = [
            "custom_sat",      # template_id
            "Custom Satellite", # name
            "optical_1"        # sat_type
        ]
        mock_click.confirm.return_value = False  # Don't add capabilities

        result = library.add_satellite_interactive()

        # Should call prompt for each field
        assert mock_click.prompt.call_count >= 3

        # Should save the template
        mock_repo.save_satellite_template.assert_called_once()
        saved = mock_repo.save_satellite_template.call_args[0][0]
        assert saved["template_id"] == "custom_sat"
        assert saved["name"] == "Custom Satellite"
        assert saved["sat_type"] == "optical_1"

    @patch('utils.entity.library.click')
    def test_add_satellite_interactive_with_capabilities(self, mock_click, library, mock_repo):
        """add_satellite_interactive should handle capabilities."""
        mock_click.prompt.side_effect = [
            "custom_sat",
            "Custom Satellite",
            "optical_1",
            "push_broom,frame",  # imaging_modes
            "30.0",              # max_off_nadir
            "500",               # storage_capacity
            "2000",              # power_capacity
            "300",               # data_rate
            "SSO",               # orbit_type
            "645000",            # altitude
            "97.9"               # inclination
        ]
        mock_click.confirm.side_effect = [True, True]  # Add capabilities, add orbit

        result = library.add_satellite_interactive()

        saved = mock_repo.save_satellite_template.call_args[0][0]
        assert "capabilities" in saved
        # click.prompt returns strings which are converted to int by type=int
        assert int(saved["capabilities"]["storage_capacity"]) == 500
