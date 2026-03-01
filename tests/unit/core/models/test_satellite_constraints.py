"""
Tests for satellite-specific and mode-specific imaging duration constraints.

TDD approach: Write failing tests first, then implement to make them pass.
"""

import pytest
from dataclasses import dataclass, field
from typing import Dict, Any

from core.models.satellite import (
    SatelliteCapabilities,
    ImagingMode,
    Satellite,
    SatelliteType,
    Orbit,
)


class TestSatelliteCapabilitiesConstraints:
    """Test SatelliteCapabilities imaging mode constraints."""

    def test_default_constraints_empty_dict(self):
        """Test that default imaging_mode_constraints is an empty dict."""
        capabilities = SatelliteCapabilities()
        assert capabilities.imaging_mode_constraints == {}

    def test_custom_per_mode_constraints(self):
        """Test setting custom per-mode constraints."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
            ImagingMode.FRAME: {'min_duration': 10.0, 'max_duration': 300.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )
        assert capabilities.imaging_mode_constraints == constraints

    def test_get_constraints_for_mode_exists(self):
        """Test getting constraints for a mode that has configuration."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )
        result = capabilities.get_imaging_constraints(ImagingMode.PUSH_BROOM)
        assert result == {'min_duration': 30.0, 'max_duration': 600.0}

    def test_get_constraints_for_mode_not_exists(self):
        """Test getting constraints for a mode without configuration returns None."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )
        result = capabilities.get_imaging_constraints(ImagingMode.FRAME)
        assert result is None

    def test_get_constraints_with_defaults_fallback(self):
        """Test getting constraints with fallback to defaults."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )
        # For mode without constraints, should return None
        result = capabilities.get_imaging_constraints(ImagingMode.SPOTLIGHT)
        assert result is None

    def test_validate_constraints_valid(self):
        """Test validation of valid constraints (min < max)."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )
        # Should not raise
        assert capabilities.validate_constraints() is True

    def test_validate_constraints_invalid_min_greater_than_max(self):
        """Test validation fails when min > max."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 600.0, 'max_duration': 30.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )
        with pytest.raises(ValueError, match="min_duration.*greater than.*max_duration"):
            capabilities.validate_constraints()

    def test_validate_constraints_invalid_min_equals_max(self):
        """Test validation fails when min == max."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 300.0, 'max_duration': 300.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )
        with pytest.raises(ValueError, match="min_duration.*equal to.*max_duration"):
            capabilities.validate_constraints()

    def test_validate_constraints_missing_min(self):
        """Test validation with missing min_duration."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'max_duration': 600.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )
        with pytest.raises(ValueError, match="min_duration"):
            capabilities.validate_constraints()

    def test_validate_constraints_missing_max(self):
        """Test validation with missing max_duration."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )
        with pytest.raises(ValueError, match="max_duration"):
            capabilities.validate_constraints()

    def test_validate_constraints_empty(self):
        """Test validation with empty constraints passes."""
        capabilities = SatelliteCapabilities(imaging_mode_constraints={})
        assert capabilities.validate_constraints() is True

    def test_multiple_modes_constraints(self):
        """Test constraints for multiple imaging modes."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
            ImagingMode.FRAME: {'min_duration': 10.0, 'max_duration': 300.0},
            ImagingMode.SPOTLIGHT: {'min_duration': 60.0, 'max_duration': 1800.0},
            ImagingMode.STRIPMAP: {'min_duration': 45.0, 'max_duration': 1200.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_mode_constraints=constraints
        )

        for mode, expected in constraints.items():
            result = capabilities.get_imaging_constraints(mode)
            assert result == expected


class TestSatelliteWithConstraints:
    """Test Satellite with imaging mode constraints integration."""

    def test_satellite_with_mode_constraints(self):
        """Test creating satellite with mode-specific constraints."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
            ImagingMode.FRAME: {'min_duration': 10.0, 'max_duration': 300.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
            imaging_mode_constraints=constraints
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_2,
            orbit=Orbit(),
            capabilities=capabilities
        )

        assert satellite.capabilities.imaging_mode_constraints == constraints
        assert satellite.capabilities.get_imaging_constraints(ImagingMode.PUSH_BROOM) == {
            'min_duration': 30.0, 'max_duration': 600.0
        }

    def test_satellite_default_capabilities_no_constraints(self):
        """Test satellite with default capabilities has expected constraints."""
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
        )

        # OPTICAL_1 now has default constraints: min=6s, max=12s
        assert satellite.capabilities.imaging_mode_constraints == {
            ImagingMode.PUSH_BROOM: {'min_duration': 6.0, 'max_duration': 12.0}
        }


class TestSatelliteCapabilitiesSerialization:
    """Test serialization/deserialization of imaging mode constraints."""

    def test_to_dict_includes_constraints(self):
        """Test that to_dict includes imaging_mode_constraints."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
        }
        # Also include imaging_modes to prevent _set_default_capabilities from overriding
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM],
            imaging_mode_constraints=constraints
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_1,
            orbit=Orbit(),
            capabilities=capabilities
        )

        data = satellite.to_dict()
        assert 'imaging_mode_constraints' in data['capabilities']
        assert data['capabilities']['imaging_mode_constraints'] == {
            'push_broom': {'min_duration': 30.0, 'max_duration': 600.0}
        }

    def test_from_dict_with_constraints(self):
        """Test that from_dict correctly parses imaging_mode_constraints."""
        data = {
            'id': 'test_sat',
            'name': 'Test Satellite',
            'sat_type': 'optical_1',
            'orbit': {
                'altitude': 500000,
                'inclination': 97.4,
            },
            'capabilities': {
                'imaging_modes': ['push_broom'],
                'imaging_mode_constraints': {
                    'push_broom': {'min_duration': 30.0, 'max_duration': 600.0}
                }
            }
        }

        satellite = Satellite.from_dict(data)
        constraints = satellite.capabilities.get_imaging_constraints(ImagingMode.PUSH_BROOM)
        assert constraints == {'min_duration': 30.0, 'max_duration': 600.0}

    def test_from_dict_without_constraints(self):
        """Test that from_dict works without imaging_mode_constraints."""
        data = {
            'id': 'test_sat',
            'name': 'Test Satellite',
            'sat_type': 'optical_1',
            'orbit': {
                'altitude': 500000,
                'inclination': 97.4,
            },
            'capabilities': {
                'imaging_modes': ['push_broom'],
            }
        }

        satellite = Satellite.from_dict(data)
        assert satellite.capabilities.imaging_mode_constraints == {}

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization preserves constraints."""
        constraints = {
            ImagingMode.PUSH_BROOM: {'min_duration': 30.0, 'max_duration': 600.0},
            ImagingMode.FRAME: {'min_duration': 10.0, 'max_duration': 300.0},
        }
        capabilities = SatelliteCapabilities(
            imaging_modes=[ImagingMode.PUSH_BROOM, ImagingMode.FRAME],
            imaging_mode_constraints=constraints
        )
        satellite = Satellite(
            id="test_sat",
            name="Test Satellite",
            sat_type=SatelliteType.OPTICAL_2,
            orbit=Orbit(),
            capabilities=capabilities
        )

        # Serialize
        data = satellite.to_dict()

        # Deserialize
        restored = Satellite.from_dict(data)

        # Verify
        assert restored.capabilities.imaging_mode_constraints == constraints
