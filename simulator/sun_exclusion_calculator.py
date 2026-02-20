"""
Sun exclusion angle calculator for optical satellites.

Calculates sun position and ensures imaging LOS (Line of Sight)
maintains minimum angular separation from the sun to protect
delicate optical sensors.
"""
import math
from typing import Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SunExclusionCalculator:
    """Sun exclusion angle calculator

    Ensures optical satellite imaging LOS does not point too close
to the sun, protecting focal plane detectors from damage.

    The calculator uses astronomical algorithms to compute sun position
    in ECI (Earth-Centered Inertial) coordinates.

    Example:
        calculator = SunExclusionCalculator(exclusion_angle=30.0)

        # Check if imaging is safe
        is_valid, separation_angle = calculator.check_sun_exclusion(
            satellite_pos=(xs, ys, zs),
            target_pos=(xt, yt, zt),
            t=imaging_time
        )

        if not is_valid:
            logger.warning(f"Sun too close: {separation_angle:.1f}°")

    Attributes:
        exclusion_angle: Minimum allowed sun-LOS separation angle (radians)
    """

    def __init__(self, exclusion_angle: float = 30.0):
        """Initialize calculator

        Args:
            exclusion_angle: Exclusion angle in degrees (default 30°)
        """
        self.exclusion_angle = math.radians(exclusion_angle)
        self._sun_cache: dict = {}  # Cache sun positions

    def _datetime_to_julian_day(self, t: datetime) -> float:
        """Convert datetime to Julian Day

        Args:
            t: Date/time

        Returns:
            Julian day number
        """
        # Julian day calculation
        year = t.year
        month = t.month
        day = t.day
        hour = t.hour
        minute = t.minute
        second = t.second

        if month <= 2:
            year -= 1
            month += 12

        a = int(year / 100)
        b = 2 - a + int(a / 4)

        jd = (int(365.25 * (year + 4716)) +
              int(30.6001 * (month + 1)) +
              day + hour / 24.0 + b - 1524.5)

        return jd

    def calculate_sun_position(self, t: datetime) -> Tuple[float, float, float]:
        """Calculate sun position in ECI coordinates

        Uses simplified astronomical algorithms based on:
        - Astronomical Algorithms by Jean Meeus

        Args:
            t: Date/time

        Returns:
            Sun position (x, y, z) in meters
        """
        # Check cache
        cache_key = (t.year, t.month, t.day, t.hour)
        if cache_key in self._sun_cache:
            return self._sun_cache[cache_key]

        jd = self._datetime_to_julian_day(t)

        # Julian centuries from J2000.0
        n = (jd - 2451545.0) / 36525.0

        # Mean longitude of sun (degrees)
        L0 = (280.460 + 36000.770 * n) % 360.0
        if L0 < 0:
            L0 += 360.0

        # Mean anomaly of sun (degrees)
        M = (357.529 + 35999.050 * n) % 360.0
        if M < 0:
            M += 360.0

        # Sun's equation of center (degrees)
        M_rad = math.radians(M)
        C = (1.9146 - 0.004817 * n) * math.sin(M_rad) + \
            0.019993 * math.sin(2 * M_rad) + \
            0.000289 * math.sin(3 * M_rad)

        # Sun's true longitude (degrees)
        sun_lon = (L0 + C) % 360.0

        # Obliquity of ecliptic (simplified)
        obliquity = math.radians(23.44)

        # Sun latitude (degrees)
        sun_lat = math.degrees(math.asin(
            math.sin(obliquity) * math.sin(math.radians(sun_lon))
        ))

        # Convert to Cartesian coordinates
        AU = 149597870700  # 1 AU in meters
        sun_lon_rad = math.radians(sun_lon)
        sun_lat_rad = math.radians(sun_lat)

        x = AU * math.cos(sun_lat_rad) * math.cos(sun_lon_rad)
        y = AU * math.cos(sun_lat_rad) * math.sin(sun_lon_rad)
        z = AU * math.sin(sun_lat_rad)

        sun_pos = (x, y, z)

        # Cache result
        self._sun_cache[cache_key] = sun_pos

        return sun_pos

    def check_sun_exclusion(self,
                           satellite_pos: Tuple[float, float, float],
                           target_pos: Tuple[float, float, float],
                           t: datetime) -> Tuple[bool, float]:
        """Check if imaging LOS satisfies sun exclusion constraint

        Args:
            satellite_pos: Satellite position (x, y, z) in meters
            target_pos: Target position (x, y, z) in meters
            t: Date/time for sun position calculation

        Returns:
            Tuple of (is_valid, separation_angle_degrees)
        """
        # Calculate sun position
        sun_pos = self.calculate_sun_position(t)

        # Calculate LOS vector (satellite to target)
        los_vector = (
            target_pos[0] - satellite_pos[0],
            target_pos[1] - satellite_pos[1],
            target_pos[2] - satellite_pos[2]
        )

        # Normalize LOS vector
        los_norm = math.sqrt(sum(x**2 for x in los_vector))
        if los_norm < 1e-10:  # Avoid division by zero
            return False, 0.0

        los_unit = tuple(x / los_norm for x in los_vector)

        # Calculate sun direction vector (satellite to sun)
        sun_vector = (
            sun_pos[0] - satellite_pos[0],
            sun_pos[1] - satellite_pos[1],
            sun_pos[2] - satellite_pos[2]
        )

        # Normalize sun vector
        sun_norm = math.sqrt(sum(x**2 for x in sun_vector))
        if sun_norm < 1e-10:
            return False, 0.0

        sun_unit = tuple(x / sun_norm for x in sun_vector)

        # Calculate angle between LOS and sun direction
        dot_product = sum(los_unit[i] * sun_unit[i] for i in range(3))

        # Clamp to valid range for acos
        dot_product = max(-1.0, min(1.0, dot_product))

        angle = math.acos(dot_product)

        # Check against exclusion angle
        is_valid = angle >= self.exclusion_angle

        return is_valid, math.degrees(angle)

    def get_safe_imaging_windows(self,
                                 satellite_pos_func,
                                 target_pos,
                                 start_time: datetime,
                                 end_time: datetime,
                                 time_step: int = 60) -> list:
        """Find safe imaging windows satisfying sun exclusion

        Args:
            satellite_pos_func: Function(t) -> (x, y, z) for satellite position
            target_pos: Target position (x, y, z)
            start_time: Search start time
            end_time: Search end time
            time_step: Time step in seconds

        Returns:
            List of safe (start, end) datetime tuples
        """
        safe_windows = []
        current_start = None

        t = start_time
        while t <= end_time:
            sat_pos = satellite_pos_func(t)
            is_valid, angle = self.check_sun_exclusion(sat_pos, target_pos, t)

            if is_valid:
                if current_start is None:
                    current_start = t
            else:
                if current_start is not None:
                    safe_windows.append((current_start, t))
                    current_start = None

            t = t + datetime.timedelta(seconds=time_step)

        # Close final window if open
        if current_start is not None:
            safe_windows.append((current_start, end_time))

        return safe_windows
