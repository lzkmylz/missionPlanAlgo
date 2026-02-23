"""Tests for datetime timezone consistency across entity module.

This module ensures that all datetime operations use timezone-aware UTC
instead of deprecated utcnow() method.
"""
import pytest
import re
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from utils.entity.library import EntityLibrary
from utils.entity.builder import ScenarioBuilder


class TestTimezoneConsistency:
    """Test that all datetime operations use timezone-aware UTC."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository."""
        return Mock()

    @pytest.fixture
    def library(self, mock_repo):
        """Create EntityLibrary with mock repository."""
        return EntityLibrary(repository=mock_repo)

    def test_add_target_created_at_is_timezone_aware(self, library, mock_repo):
        """add_target should use timezone-aware UTC datetime."""
        library.add_target(
            target_id="test_target",
            name="Test Target",
            lon=116.4,
            lat=39.9,
            priority=5
        )

        mock_repo.save_target.assert_called_once()
        saved_target = mock_repo.save_target.call_args[0][0]

        # Verify created_at exists and is in ISO8601 format with Z suffix
        assert "created_at" in saved_target
        created_at = saved_target["created_at"]

        # Should end with Z (UTC indicator)
        assert created_at.endswith("Z"), f"Expected created_at to end with Z, got: {created_at}"

        # Should be valid ISO8601 format
        iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$'
        assert re.match(iso8601_pattern, created_at), f"Invalid ISO8601 format: {created_at}"

    @patch('utils.entity.library.click')
    def test_add_satellite_interactive_created_at_is_timezone_aware(self, mock_click, library, mock_repo):
        """add_satellite_interactive should use timezone-aware UTC datetime."""
        mock_click.prompt.side_effect = [
            "custom_sat",
            "Custom Satellite",
            "optical_1"
        ]
        mock_click.confirm.return_value = False

        library.add_satellite_interactive()

        mock_repo.save_satellite_template.assert_called_once()
        saved = mock_repo.save_satellite_template.call_args[0][0]

        # Verify created_at exists and is in ISO8601 format with Z suffix
        assert "created_at" in saved
        created_at = saved["created_at"]

        # Should end with Z (UTC indicator)
        assert created_at.endswith("Z"), f"Expected created_at to end with Z, got: {created_at}"

        # Should be valid ISO8601 format
        iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$'
        assert re.match(iso8601_pattern, created_at), f"Invalid ISO8601 format: {created_at}"


class TestScenarioBuilderTimezoneConsistency:
    """Test that ScenarioBuilder uses timezone-aware UTC datetime."""

    @pytest.fixture
    def mock_repo(self):
        """Create a mock repository."""
        return Mock()

    @pytest.fixture
    def builder(self, mock_repo):
        """Create ScenarioBuilder with mock repository."""
        return ScenarioBuilder(repository=mock_repo)

    def test_init_scenario_metadata_created_at_is_timezone_aware(self, builder, tmp_path):
        """init_scenario should use timezone-aware UTC datetime in metadata."""
        output_path = str(tmp_path / "test_scenario.json")

        scenario = builder.init_scenario(output_path, with_metadata=True)

        assert "metadata" in scenario
        created_at = scenario["metadata"]["created_at"]

        # Should end with Z (UTC indicator)
        assert created_at.endswith("Z"), f"Expected created_at to end with Z, got: {created_at}"

        # Should be valid ISO8601 format
        iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$'
        assert re.match(iso8601_pattern, created_at), f"Invalid ISO8601 format: {created_at}"

    def test_clone_scenario_updates_created_at_with_timezone_aware(self, builder, tmp_path):
        """clone_scenario should use timezone-aware UTC datetime."""
        import json

        source_path = tmp_path / "source.json"
        source_scenario = {
            "name": "source",
            "description": "Original scenario",
            "satellites": [],
            "metadata": {
                "version": "1.0",
                "created_at": "2024-01-01T00:00:00Z"
            }
        }
        source_path.write_text(json.dumps(source_scenario))

        output_path = str(tmp_path / "cloned.json")
        result = builder.clone_scenario(str(source_path), output_path)

        # Should update created_at with new timezone-aware timestamp
        assert "created_at" in result["metadata"]
        created_at = result["metadata"]["created_at"]

        # Should end with Z (UTC indicator)
        assert created_at.endswith("Z"), f"Expected created_at to end with Z, got: {created_at}"

        # Should be valid ISO8601 format
        iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$'
        assert re.match(iso8601_pattern, created_at), f"Invalid ISO8601 format: {created_at}"


class TestDatetimeFormatConsistency:
    """Test that datetime format is consistent across the codebase."""

    def test_datetime_now_timezone_utc_format(self):
        """Verify datetime.now(timezone.utc) produces expected format."""
        now = datetime.now(timezone.utc)
        formatted = now.isoformat().replace("+00:00", "Z")

        # Should end with Z
        assert formatted.endswith("Z")

        # Should be valid ISO8601 with microseconds
        iso8601_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$'
        assert re.match(iso8601_pattern, formatted), f"Unexpected format: {formatted}"
