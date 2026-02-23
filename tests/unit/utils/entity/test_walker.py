"""Tests for Walker constellation generator."""
import pytest
import math
from unittest.mock import Mock

from utils.entity.walker import WalkerConfig, WalkerGenerator


class TestWalkerConfig:
    """Test Walker configuration dataclass."""

    def test_valid_walker_config(self):
        """Should create valid Walker configuration."""
        config = WalkerConfig(
            inclination=55.0,
            total_sats=24,
            n_planes=3,
            f_factor=1
        )

        assert config.inclination == 55.0
        assert config.total_sats == 24
        assert config.n_planes == 3
        assert config.f_factor == 1
        assert config.raan_start == 0.0
        assert config.raan_spread == 360.0

    def test_walker_config_with_custom_raan(self):
        """Should create config with custom RAAN parameters."""
        config = WalkerConfig(
            inclination=97.9,
            total_sats=6,
            n_planes=2,
            f_factor=1,
            raan_start=180.0,
            raan_spread=180.0
        )

        assert config.raan_start == 180.0
        assert config.raan_spread == 180.0

    def test_invalid_total_sats_not_divisible(self):
        """Should raise error if total_sats not divisible by n_planes."""
        with pytest.raises(ValueError, match="divisible"):
            WalkerConfig(
                inclination=55.0,
                total_sats=25,  # Not divisible by 3
                n_planes=3,
                f_factor=1
            )

    def test_invalid_f_factor_negative(self):
        """Should raise error if f_factor negative."""
        with pytest.raises(ValueError, match="f_factor"):
            WalkerConfig(
                inclination=55.0,
                total_sats=24,
                n_planes=3,
                f_factor=-1
            )

    def test_invalid_f_factor_too_large(self):
        """Should raise error if f_factor >= n_planes."""
        with pytest.raises(ValueError, match="f_factor"):
            WalkerConfig(
                inclination=55.0,
                total_sats=24,
                n_planes=3,
                f_factor=3  # Must be < n_planes
            )


class TestWalkerGenerator:
    """Test Walker constellation generator."""

    @pytest.fixture
    def template(self):
        """Create a sample satellite template."""
        return {
            "template_id": "optical_1",
            "name": "光学卫星1型",
            "sat_type": "optical_1",
            "capabilities": {"storage_capacity": 500}
        }

    @pytest.fixture
    def generator(self):
        """Create Walker generator."""
        return WalkerGenerator()

    def test_generate_walker_delta(self, generator, template):
        """Should generate Walker Delta constellation."""
        config = WalkerConfig(
            inclination=55.0,
            total_sats=6,
            n_planes=2,
            f_factor=1
        )

        satellites = generator.generate(config, template, prefix="WALKER")

        assert len(satellites) == 6

        # Check all satellites have correct structure
        for sat in satellites:
            assert "id" in sat
            assert "raan" in sat
            assert "mean_anomaly" in sat
            assert sat["orbit"]["inclination"] == 55.0
            assert sat["_template_source"] == "optical_1"

    def test_walker_satellite_ids(self, generator, template):
        """Should generate correct satellite IDs."""
        config = WalkerConfig(
            inclination=55.0,
            total_sats=6,
            n_planes=2,
            f_factor=1
        )

        satellites = generator.generate(config, template, prefix="WALKER")

        expected_ids = ["WALKER-01-01", "WALKER-01-02", "WALKER-01-03",
                       "WALKER-02-01", "WALKER-02-02", "WALKER-02-03"]
        actual_ids = [sat["id"] for sat in satellites]

        assert actual_ids == expected_ids

    def test_walker_raan_distribution(self, generator, template):
        """Should distribute RAAN evenly across planes."""
        config = WalkerConfig(
            inclination=55.0,
            total_sats=6,
            n_planes=2,
            f_factor=1,
            raan_start=0.0,
            raan_spread=180.0
        )

        satellites = generator.generate(config, template)

        # Plane 1 should have RAAN around 0
        plane1_sats = [s for s in satellites if s["plane"] == 1]
        assert all(abs(s["raan"] - 0.0) < 1.0 for s in plane1_sats)

        # Plane 2 should have RAAN around 90 (180/2)
        plane2_sats = [s for s in satellites if s["plane"] == 2]
        assert all(abs(s["raan"] - 90.0) < 1.0 for s in plane2_sats)

    def test_walker_mean_anomaly_distribution(self, generator, template):
        """Should distribute mean anomaly evenly within planes."""
        config = WalkerConfig(
            inclination=55.0,
            total_sats=6,
            n_planes=2,
            f_factor=1
        )

        satellites = generator.generate(config, template)

        # Plane 1 satellites should be at 0, 120, 240 degrees
        plane1_sats = sorted([s for s in satellites if s["plane"] == 1],
                            key=lambda x: x["sat_in_plane"])
        assert abs(plane1_sats[0]["mean_anomaly"] - 0.0) < 1.0
        assert abs(plane1_sats[1]["mean_anomaly"] - 120.0) < 1.0
        assert abs(plane1_sats[2]["mean_anomaly"] - 240.0) < 1.0

    def test_walker_phase_factor_effect(self, generator, template):
        """Phase factor should affect inter-plane spacing."""
        # F=0: satellites aligned across planes
        config_f0 = WalkerConfig(
            inclination=55.0,
            total_sats=4,
            n_planes=2,
            f_factor=0
        )
        sats_f0 = generator.generate(config_f0, template)

        # F=1: satellites offset between planes
        config_f1 = WalkerConfig(
            inclination=55.0,
            total_sats=4,
            n_planes=2,
            f_factor=1
        )
        sats_f1 = generator.generate(config_f1, template)

        # First satellite in each plane should have different mean anomaly
        plane1_f0 = [s for s in sats_f0 if s["plane"] == 1][0]
        plane2_f0 = [s for s in sats_f0 if s["plane"] == 2][0]

        plane1_f1 = [s for s in sats_f1 if s["plane"] == 1][0]
        plane2_f1 = [s for s in sats_f1 if s["plane"] == 2][0]

        # With F=0, first satellites should be aligned (same mean anomaly)
        assert abs(plane1_f0["mean_anomaly"] - plane2_f0["mean_anomaly"]) < 1.0

        # With F=1, there should be a phase offset
        assert abs(plane1_f1["mean_anomaly"] - plane2_f1["mean_anomaly"]) > 1.0

    def test_generate_creates_deep_copy(self, generator, template):
        """Should create deep copy of template."""
        config = WalkerConfig(
            inclination=55.0,
            total_sats=2,
            n_planes=1,
            f_factor=0
        )

        satellites = generator.generate(config, template)

        # Modify generated satellite
        satellites[0]["capabilities"]["storage_capacity"] = 9999

        # Original template should not be modified
        assert template["capabilities"]["storage_capacity"] == 500

    def test_plane_and_sat_in_plane_fields(self, generator, template):
        """Should include plane and sat_in_plane fields."""
        config = WalkerConfig(
            inclination=55.0,
            total_sats=6,
            n_planes=2,
            f_factor=1
        )

        satellites = generator.generate(config, template)

        for sat in satellites:
            assert "plane" in sat
            assert "sat_in_plane" in sat
            assert 1 <= sat["plane"] <= 2
            assert 1 <= sat["sat_in_plane"] <= 3


class TestWalkerPresets:
    """Test Walker preset configurations."""

    def test_get_preset_exists(self):
        """Should return preset config if exists."""
        config = WalkerGenerator.get_preset("delta_24_3_1")

        assert config is not None
        assert config.inclination == 55.0
        assert config.total_sats == 24
        assert config.n_planes == 3
        assert config.f_factor == 1

    def test_get_preset_not_exists(self):
        """Should return None for non-existent preset."""
        config = WalkerGenerator.get_preset("nonexistent")

        assert config is None

    def test_list_presets(self):
        """Should list available presets."""
        presets = WalkerGenerator.list_presets()

        assert "delta_24_3_1" in presets
        assert "star_24_3_0" in presets
        assert "GPS-like" in presets["delta_24_3_1"]

    def test_preset_delta_24_3_1(self):
        """delta_24_3_1 preset should be GPS-like."""
        config = WalkerGenerator.get_preset("delta_24_3_1")

        assert config.inclination == 55.0
        assert config.total_sats == 24
        assert config.n_planes == 3
        assert config.f_factor == 1

    def test_preset_star_24_3_0(self):
        """star_24_3_0 preset should be polar orbit."""
        config = WalkerGenerator.get_preset("star_24_3_0")

        assert config.inclination == 90.0
        assert config.total_sats == 24
        assert config.n_planes == 3
        assert config.f_factor == 0
