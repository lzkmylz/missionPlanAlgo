"""
Tests for FootprintCalculator - Coverage Geometry Calculator

TDD Phase: RED - Write failing tests first
"""

import pytest
import math
from typing import List, Tuple
from dataclasses import dataclass

from core.models import Target, TargetType, ImagingMode


@dataclass
class Footprint:
    """地面成像条带足迹"""
    center: Tuple[float, float]  # (lon, lat)
    polygon: List[Tuple[float, float]]  # 多边形顶点
    width_km: float
    length_km: float


class TestFootprintDataclass:
    """Test Footprint data class"""

    def test_footprint_creation(self):
        """Test basic footprint creation"""
        polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
        footprint = Footprint(
            center=(0.5, 0.5),
            polygon=polygon,
            width_km=10.0,
            length_km=50.0
        )
        assert footprint.center == (0.5, 0.5)
        assert footprint.polygon == polygon
        assert footprint.width_km == 10.0
        assert footprint.length_km == 50.0


class TestFootprintCalculatorInit:
    """Test FootprintCalculator initialization"""

    def test_default_altitude(self):
        """Test default satellite altitude"""
        from core.coverage.footprint_calculator import FootprintCalculator
        calc = FootprintCalculator()
        assert calc.satellite_altitude_km == 500.0

    def test_custom_altitude(self):
        """Test custom satellite altitude"""
        from core.coverage.footprint_calculator import FootprintCalculator
        calc = FootprintCalculator(satellite_altitude_km=700.0)
        assert calc.satellite_altitude_km == 700.0


class TestCalculateFootprint:
    """Test footprint calculation at nadir and off-nadir"""

    def test_footprint_at_nadir(self):
        """Test footprint calculation at nadir (0° look angle)"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        # Satellite at 500km altitude directly above equator
        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)  # ECEF (m)
        nadir_position = (0.0, 0.0)  # (lon, lat) at equator

        footprint = calc.calculate_footprint(
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            look_angle=0.0,  # Nadir
            swath_width_km=10.0,
            imaging_mode=ImagingMode.PUSH_BROOM
        )

        # Center should be at nadir
        assert footprint.center == nadir_position
        # Width should match swath width
        assert footprint.width_km == 10.0
        # Should have a polygon with 4+ vertices
        assert len(footprint.polygon) >= 4

    def test_footprint_with_off_nadir_angle(self):
        """Test footprint calculation with off-nadir look angle"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        # Satellite at 500km altitude
        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)  # ECEF (m)
        nadir_position = (0.0, 0.0)  # (lon, lat)

        look_angle = 15.0  # degrees
        footprint = calc.calculate_footprint(
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            look_angle=look_angle,
            swath_width_km=10.0,
            imaging_mode=ImagingMode.PUSH_BROOM
        )

        # Center should be displaced from nadir
        assert footprint.center != nadir_position
        # Displacement should be roughly altitude * tan(look_angle)
        expected_displacement_km = 500.0 * math.tan(math.radians(look_angle))
        actual_displacement_km = self._haversine_distance(
            nadir_position[1], nadir_position[0],
            footprint.center[1], footprint.center[0]
        )
        # Allow 10% tolerance for Earth curvature
        assert abs(actual_displacement_km - expected_displacement_km) < expected_displacement_km * 0.1

    def test_footprint_different_imaging_modes(self):
        """Test footprint for different imaging modes"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)
        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        # Test optical push broom
        footprint_optical = calc.calculate_footprint(
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            look_angle=0.0,
            swath_width_km=10.0,
            imaging_mode=ImagingMode.PUSH_BROOM
        )
        assert footprint_optical.width_km == 10.0

        # Test SAR stripmap
        footprint_sar = calc.calculate_footprint(
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            look_angle=0.0,
            swath_width_km=20.0,
            imaging_mode=ImagingMode.STRIPMAP
        )
        assert footprint_sar.width_km == 20.0

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371.0  # Earth radius in km
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c


class TestCalculateOffNadirAngle:
    """Test off-nadir angle calculation"""

    def test_off_nadir_at_nadir(self):
        """Test off-nadir angle for target directly below satellite"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        # Satellite at 500km altitude above equator
        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)  # ECEF (m)
        target_position = (0.0, 0.0)  # (lon, lat) at nadir

        angle = calc.calculate_off_nadir_angle(satellite_position, target_position)

        # Should be approximately 0 degrees
        assert abs(angle) < 0.1  # Allow small numerical error

    def test_off_nadir_with_displacement(self):
        """Test off-nadir angle for displaced target"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        # Satellite at 500km altitude
        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)

        # Target 50km away from nadir (approximately 0.45 degrees)
        # At equator, 1 degree longitude ~ 111km
        target_position = (0.45, 0.0)  # (lon, lat)

        angle = calc.calculate_off_nadir_angle(satellite_position, target_position)

        # Expected angle: atan2(50, 500) ~ 5.7 degrees
        expected_angle = math.degrees(math.atan2(50, 500))
        assert abs(angle - expected_angle) < 1.0  # Allow 1 degree tolerance

    def test_off_nadir_symmetry(self):
        """Test off-nadir angle is symmetric for left/right displacement"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)

        # Targets at same distance but opposite sides
        target_left = (-0.45, 0.0)
        target_right = (0.45, 0.0)

        angle_left = calc.calculate_off_nadir_angle(satellite_position, target_left)
        angle_right = calc.calculate_off_nadir_angle(satellite_position, target_right)

        # Absolute angles should be equal
        assert abs(abs(angle_left) - abs(angle_right)) < 0.1


class TestIsTargetInFootprint:
    """Test target-in-footprint check"""

    @pytest.fixture
    def sample_footprint(self):
        """Create a sample footprint for testing"""
        return Footprint(
            center=(0.0, 0.0),
            polygon=[(-0.05, -0.05), (0.05, -0.05), (0.05, 0.05), (-0.05, 0.05)],
            width_km=10.0,
            length_km=10.0
        )

    def test_target_in_footprint_center(self, sample_footprint):
        """Test target at footprint center"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator()
        target = Target(
            id="target_1",
            target_type=TargetType.POINT,
            longitude=0.0,
            latitude=0.0
        )

        assert calc.is_target_in_footprint(target, sample_footprint) is True

    def test_target_in_footprint_edge(self, sample_footprint):
        """Test target at footprint edge"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator()
        target = Target(
            id="target_2",
            target_type=TargetType.POINT,
            longitude=0.04,  # Near edge
            latitude=0.0
        )

        assert calc.is_target_in_footprint(target, sample_footprint) is True

    def test_target_outside_footprint(self, sample_footprint):
        """Test target outside footprint"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator()
        target = Target(
            id="target_3",
            target_type=TargetType.POINT,
            longitude=0.1,  # Outside
            latitude=0.1
        )

        assert calc.is_target_in_footprint(target, sample_footprint) is False

    def test_target_on_polygon_vertex(self, sample_footprint):
        """Test target exactly on polygon vertex"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator()
        target = Target(
            id="target_4",
            target_type=TargetType.POINT,
            longitude=-0.05,
            latitude=-0.05
        )

        # Point on vertex should be considered inside
        assert calc.is_target_in_footprint(target, sample_footprint) is True


class TestCanCoverTargets:
    """Test multi-target coverage capability"""

    def test_can_cover_single_target(self):
        """Test coverage of single target"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        target = Target(
            id="target_1",
            target_type=TargetType.POINT,
            longitude=0.0,
            latitude=0.0
        )

        can_cover, angle = calc.can_cover_targets(
            targets=[target],
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            max_off_nadir=30.0,
            swath_width_km=10.0
        )

        assert can_cover is True
        assert abs(angle) < 0.1  # Near nadir

    def test_can_cover_multiple_targets_within_swath(self):
        """Test coverage of multiple targets within same swath"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        # Two targets close together (within 5km of each other)
        targets = [
            Target(id="t1", target_type=TargetType.POINT, longitude=0.0, latitude=0.0),
            Target(id="t2", target_type=TargetType.POINT, longitude=0.02, latitude=0.0),  # ~2km away
        ]

        can_cover, angle = calc.can_cover_targets(
            targets=targets,
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            max_off_nadir=30.0,
            swath_width_km=10.0  # 10km swath
        )

        assert can_cover is True

    def test_cannot_cover_targets_too_far_apart(self):
        """Test that targets too far apart cannot be covered together"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        # Two targets far apart (15km apart, but swath is 10km)
        targets = [
            Target(id="t1", target_type=TargetType.POINT, longitude=0.0, latitude=0.0),
            Target(id="t2", target_type=TargetType.POINT, longitude=0.15, latitude=0.0),  # ~15km away
        ]

        can_cover, angle = calc.can_cover_targets(
            targets=targets,
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            max_off_nadir=30.0,
            swath_width_km=10.0
        )

        assert can_cover is False

    def test_max_off_nadir_constraint(self):
        """Test that max off-nadir angle constraint is enforced"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        # Target at 45 degrees off-nadir would require aggressive maneuvering
        # At 500km altitude, 45 degrees = 500km ground displacement
        # 1 degree longitude ~ 111km at equator, so ~4.5 degrees
        target = Target(
            id="target_far",
            target_type=TargetType.POINT,
            longitude=4.5,
            latitude=0.0
        )

        # With max_off_nadir=30, should not be able to cover
        can_cover, angle = calc.can_cover_targets(
            targets=[target],
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            max_off_nadir=30.0,  # Conservative limit
            swath_width_km=100.0  # Wide swath
        )

        assert can_cover is False

        # With max_off_nadir=50, should be able to cover
        can_cover, angle = calc.can_cover_targets(
            targets=[target],
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            max_off_nadir=50.0,  # Aggressive limit
            swath_width_km=100.0
        )

        assert can_cover is True

    def test_empty_targets_list(self):
        """Test with empty targets list"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        can_cover, angle = calc.can_cover_targets(
            targets=[],
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            max_off_nadir=30.0,
            swath_width_km=10.0
        )

        # Empty list should return True (vacuously true)
        assert can_cover is True
        assert angle == 0.0


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_null_satellite_position(self):
        """Test handling of null/None satellite position"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator()

        with pytest.raises((ValueError, TypeError)):
            calc.calculate_off_nadir_angle(None, (0.0, 0.0))

    def test_invalid_target_type(self):
        """Test handling of area target (not point)"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator()

        # Create an area target
        area_target = Target(
            id="area_1",
            target_type=TargetType.AREA,
            area_vertices=[(0, 0), (1, 0), (1, 1), (0, 1)]
        )

        footprint = Footprint(
            center=(0.5, 0.5),
            polygon=[(0, 0), (1, 0), (1, 1), (0, 1)],
            width_km=100.0,
            length_km=100.0
        )

        # Should handle area target (check center)
        result = calc.is_target_in_footprint(area_target, footprint)
        assert isinstance(result, bool)

    def test_footprint_at_high_latitude(self):
        """Test footprint calculation at high latitude"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        # Satellite above 60N
        satellite_position = (3185500.0, 0.0, 5516888.0)  # Approx ECEF for 60N
        nadir_position = (0.0, 60.0)

        footprint = calc.calculate_footprint(
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            look_angle=10.0,
            swath_width_km=10.0,
            imaging_mode=ImagingMode.PUSH_BROOM
        )

        # Should produce valid footprint
        assert len(footprint.polygon) >= 4
        assert footprint.width_km == 10.0

    def test_very_small_swath_width(self):
        """Test with very small swath width"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        footprint = calc.calculate_footprint(
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            look_angle=0.0,
            swath_width_km=0.1,  # Very small
            imaging_mode=ImagingMode.SPOTLIGHT
        )

        assert footprint.width_km == 0.1

    def test_zero_swath_width(self):
        """Test with zero swath width"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6371000.0 + 500000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        # Zero swath should still produce valid (degenerate) footprint
        footprint = calc.calculate_footprint(
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            look_angle=0.0,
            swath_width_km=0.0,
            imaging_mode=ImagingMode.PUSH_BROOM
        )

        assert footprint.width_km == 0.0


class TestRealisticScenarios:
    """Test realistic satellite scenarios"""

    def test_typical_leo_optical_satellite(self):
        """Test typical LEO optical satellite at 500km"""
        from core.coverage.footprint_calculator import FootprintCalculator

        # Typical optical satellite
        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6871000.0, 0.0, 0.0)  # 500km above equator
        nadir_position = (0.0, 0.0)

        # At 30° off-nadir (typical max for optical)
        footprint = calc.calculate_footprint(
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            look_angle=30.0,
            swath_width_km=10.0,
            imaging_mode=ImagingMode.PUSH_BROOM
        )

        # Displacement should be ~500 * tan(30°) = ~289km
        expected_displacement = 500.0 * math.tan(math.radians(30.0))
        actual_displacement = self._haversine_distance(
            nadir_position[1], nadir_position[0],
            footprint.center[1], footprint.center[0]
        )

        # Allow 15% tolerance for Earth curvature effects at 30°
        assert abs(actual_displacement - expected_displacement) < expected_displacement * 0.15

    def test_typical_leo_sar_satellite(self):
        """Test typical LEO SAR satellite with wider swath"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6871000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        # SAR with 20km swath at 45° off-nadir
        footprint = calc.calculate_footprint(
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            look_angle=45.0,
            swath_width_km=20.0,
            imaging_mode=ImagingMode.STRIPMAP
        )

        assert footprint.width_km == 20.0
        # Should be displaced from nadir
        assert footprint.center != nadir_position

    def test_multiple_targets_cluster(self):
        """Test coverage of a cluster of targets"""
        from core.coverage.footprint_calculator import FootprintCalculator

        calc = FootprintCalculator(satellite_altitude_km=500.0)

        satellite_position = (6871000.0, 0.0, 0.0)
        nadir_position = (0.0, 0.0)

        # Cluster of 5 targets within 5km radius
        targets = [
            Target(id=f"cluster_{i}", target_type=TargetType.POINT,
                   longitude=0.0 + i*0.01, latitude=0.0 + i*0.005)
            for i in range(5)
        ]

        can_cover, angle = calc.can_cover_targets(
            targets=targets,
            satellite_position=satellite_position,
            nadir_position=nadir_position,
            max_off_nadir=45.0,
            swath_width_km=20.0
        )

        assert can_cover is True

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371.0
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
