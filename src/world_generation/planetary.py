"""Planetary systems module for EmergenWorld.

This module simulates planetary characteristics such as axial tilt, rotation,
and orbital properties to generate realistic day/night cycles, seasons,
and solar radiation patterns across the world.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
import ephem  # For astronomical calculations
from datetime import datetime, timedelta


class PlanetarySystem:
    """Simulates planetary systems for the EmergenWorld simulation.

    Models planetary characteristics including orbital properties,
    day/night cycles, seasonal variations, and
    solar radiation patterns.
    """

    def __init__(
            self,
            world_size: int = 1024,
            axial_tilt_degrees: float = 23.5,
            day_length_hours: float = 24.0,
            year_length_days: float = 365.25,
            eccentricity: float = 0.0167,
            perihelion_day: float = 14.0,
            start_day: int = 0,
            seasonal_factor: float = 1.0,
            earth_scale: float = 0.0083
    ):
        """Initialize the PlanetarySystem with configurable parameters.

        Args:
            world_size: Size of the world grid
            axial_tilt_degrees: Axial tilt in degrees (Earth is 23.5)
            day_length_hours: Length of a day in hours (Earth is 24)
            year_length_days: Length of a year in days (Earth is 365.25)
            eccentricity: Orbital eccentricity (Earth is 0.0167)
            perihelion_day: Day of year when the planet is closest to its sun
            start_day: Starting day of the simulation (0 to year_length_days-1)
            seasonal_factor: Multiplier to increase or decrease seasonal effects
            earth_scale: Scale relative to Earth (default 0.83% of Earth)
        """
        self.world_size = world_size
        self.axial_tilt = np.radians(axial_tilt_degrees)
        self.day_length_hours = day_length_hours
        self.year_length_days = year_length_days
        self.eccentricity = eccentricity
        self.perihelion_day = perihelion_day
        self.current_day = start_day % year_length_days
        self.current_hour = 0.0
        self.seasonal_factor = seasonal_factor
        self.earth_scale = earth_scale

        # Derived planetary properties
        # Earth radius × scale
        self.planet_radius_km = 6371.0 * np.sqrt(earth_scale)
        self.planet_circumference_km = 2 * np.pi * self.planet_radius_km
        self.km_per_cell = self.planet_circumference_km / world_size

        # Initialize coordinate system
        self._initialize_coordinates()

        # Set up the sun and observer for ephem calculations
        self._setup_celestial_objects()

        # Create day/night mask and solar radiation map
        self.day_night_mask = np.zeros((world_size, world_size), dtype=bool)
        self.solar_radiation = np.zeros((world_size, world_size))

        # Update the initial state
        self.update_sun_position()

        print(
            f"Initialized planetary system with "
            f"{axial_tilt_degrees}° axial tilt"
        )
        print(
            f"Year length: {year_length_days} days, "
            f"Day length: {day_length_hours} hours"
        )
        msg = f"Planet radius: {self.planet_radius_km:.1f} km, "
        msg += f"Cell size: {self.km_per_cell:.1f} km"
        print(msg)
        print(f"Starting on day {start_day} (day {self.current_day} of year)")

    def _initialize_coordinates(self) -> None:
        """Initialize the latitude and longitude grids for the world."""
        # Create arrays to hold the latitude and longitude of each cell
        self.latitudes = np.zeros((self.world_size, self.world_size))
        self.longitudes = np.zeros((self.world_size, self.world_size))

        # Calculate latitudes (from +90° at the top to -90° at the bottom)
        for y in range(self.world_size):
            latitude = 90 - (y / (self.world_size - 1) * 180)
            self.latitudes[y, :] = latitude

        # Calculate longitudes (from -180° at the left to +180° at the right)
        for x in range(self.world_size):
            longitude = (x / (self.world_size - 1) * 360) - 180
            self.longitudes[:, x] = longitude

    def _setup_celestial_objects(self) -> None:
        """Set up the sun and observer for astronomical calculations."""
        self.sun = ephem.Sun()

        # Create an observer - will be updated for each grid cell
        self.observer = ephem.Observer()
        self.observer.pressure = 0.0  # Ignore atmospheric refraction

        # Set the epoch for calculations (doesn't matter much for our purposes)
        self.epoch = datetime(2000, 1, 1)

    def _update_sun_position(self) -> None:
        """Update the sun's position based on the current day and hour."""
        # Calculate the hour of day as a fraction
        hour_fraction = self.current_hour / self.day_length_hours

        # Calculate the julian date for ephem
        # This is simplified - we're just using a base date
        # and adding our simulation time
        days_since_epoch = (
            self.current_day
            + (hour_fraction * self.day_length_hours / 24.0)
        )
        self.observer.date = self.epoch + timedelta(days=days_since_epoch)

        # Update the sun's position
        self.sun.compute(self.observer)

    def update_sun_position(self) -> None:
        """Update the sun's position based on current date and time."""
        self._update_sun_position()

    def update_day_night_cycle(self) -> None:
        """Update the day/night mask for the entire world."""
        self._update_day_night_cycle()

    def update_solar_radiation(self) -> None:
        """Update the solar radiation map for the entire world."""
        self._update_solar_radiation()

    def update_all(self) -> None:
        """Update sun position, day/night cycle and solar radiation."""
        self.update_sun_position()
        self.update_day_night_cycle()
        self.update_solar_radiation()

    def advance_time(self, hours: float = 1.0) -> None:
        """Advance the simulation time by a specified number of hours.

        Args:
            hours: Number of hours to advance the simulation
        """
        self.current_hour += hours

        # Handle day rollover
        while self.current_hour >= self.day_length_hours:
            self.current_hour -= self.day_length_hours
            self.current_day += 1

            # Handle year rollover
            if self.current_day >= self.year_length_days:
                self.current_day = 0

        # Update the sun's position
        self.update_sun_position()

        # Update day/night and solar radiation maps
        self._update_day_night_cycle()
        self._update_solar_radiation()

    def _update_day_night_cycle(self) -> None:
        """Update the day/night mask for the entire world."""
        # For each position, calculate if it's day or night
        for y in range(self.world_size):
            for x in range(self.world_size):
                latitude = self.latitudes[y, x]
                longitude = self.longitudes[y, x]

                # Update observer position
                self.observer.lat = str(latitude)
                self.observer.lon = str(longitude)

                # Compute the sun's position for this location
                self.sun.compute(self.observer)

                # It's day if the sun's altitude is above 0
                self.day_night_mask[y, x] = float(self.sun.alt) > 0

    def _update_solar_radiation(self) -> None:
        """Update the solar radiation map for the entire world."""
        # For each position, calculate solar radiation intensity
        for y in range(self.world_size):
            for x in range(self.world_size):
                latitude = self.latitudes[y, x]
                longitude = self.longitudes[y, x]

                # Update observer position
                self.observer.lat = str(latitude)
                self.observer.lon = str(longitude)

                # Compute the sun's position for this location
                self.sun.compute(self.observer)

                # Solar radiation is 0 at night,
                # otherwise proportional to sin(altitude)
                if float(self.sun.alt) <= 0:
                    self.solar_radiation[y, x] = 0.0
                else:
                    # Calculate radiation factor based on sun's altitude
                    radiation = np.sin(float(self.sun.alt))

                    # Apply orbital distance factor
                    year_angle = (
                        2 * np.pi * (self.current_day / self.year_length_days)
                    )
                    perihelion_angle = (
                        2 * np.pi
                        * (self.perihelion_day / self.year_length_days)
                    )
                    angle_diff = (year_angle - perihelion_angle) % (2 * np.pi)

                    # Distance factor due to eccentricity (1 - e*cos(angle))
                    distance_factor = 1 - self.eccentricity * np.cos(angle_diff)

                    # Solar radiation is inversely proportional
                    # to square of distance of the planet from its sun
                    orbital_factor = 1 / (distance_factor ** 2)

                    self.solar_radiation[y, x] = radiation * orbital_factor

    def get_current_date(self) -> Tuple[int, float]:
        """Get the current simulation date.

        Returns:
            Tuple of (day_of_year, hour_of_day)
        """
        return self.current_day, self.current_hour

    def get_formatted_date(self) -> str:
        """Get a formatted string of the current simulation date.

        Returns:
            String representation of the current date and time
        """
        day = int(self.current_day)
        hour = int(self.current_hour)
        minute = int((self.current_hour - hour) * 60)

        # Use 1-based calendar days for display
        calendar_day = day + 1

        # Calculate month and day assuming 12 equal months
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        days_per_month = self.year_length_days / 12
        month_idx = int(day / days_per_month)
        day_of_month = int(day % days_per_month) + 1

        if month_idx >= 12:  # Just in case of numerical errors
            month_idx = 11

        month_name = months[month_idx]

        return (
            f"Day {calendar_day} ({month_name} {day_of_month}), "
            f"{hour:02d}:{minute:02d}"
        )

    def get_season(self) -> str:
        """Get the current season based on day of year.

        Returns:
            String name of the current season
        """
        # Convert day of year to position in the year (0 to 1)
        year_position = self.current_day / self.year_length_days

        # Northern hemisphere seasons
        if 0.0 <= year_position < 0.25:
            return "Winter"
        elif 0.25 <= year_position < 0.5:
            return "Spring"
        elif 0.5 <= year_position < 0.75:
            return "Summer"
        else:
            return "Fall"

    def get_seasonal_factor(self, latitude: float) -> float:
        """Calculate seasonal factor for a given latitude.

        This represents how much the season affects the given latitude,
        with positive values for summer-like conditions and negative for winter.

        Args:
            latitude: Latitude in degrees (-90 to 90)

        Returns:
            Seasonal factor (-1.0 to 1.0)
        """
        # Convert to observer at this latitude
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.pressure = 0

        # Set date based on current day
        observer.date = self.epoch + timedelta(days=self.current_day)

        # Calculate sun's position at noon
        sun = ephem.Sun()
        # Set to local noon
        observer.date = observer.date.datetime() + timedelta(hours=12)
        sun.compute(observer)

        # Get the sun's altitude at noon
        max_altitude = float(sun.alt)

        # Normalize to -1 to 1 range and apply seasonal factor
        # At equinox, sun at equator is at 90° at noon, and at poles is at 0°
        equinox_altitude = np.radians(90 - abs(latitude))
        seasonal_effect = (max_altitude - equinox_altitude) / (np.pi/2)
        seasonal_effect = np.clip(
            seasonal_effect * self.seasonal_factor, -1.0, 1.0
        )

        return seasonal_effect

    def get_day_length(self, latitude: float) -> float:
        """Calculate day length in hours for a given
           latitude on the current day.

        Args:
            latitude: Latitude in degrees (-90 to 90)

        Returns:
            Day length in hours
        """
        # Set up observer at this latitude
        observer = ephem.Observer()
        observer.lat = str(latitude)
        observer.pressure = 0

        # Set date based on current day
        observer.date = self.epoch + timedelta(days=self.current_day)

        # Get sunrise and sunset times
        try:
            sunrise = observer.next_rising(ephem.Sun())
            sunset = observer.next_setting(ephem.Sun())

            # Calculate time difference in hours
            time_diff = (
                (sunset.datetime()
                 - sunrise.datetime()).total_seconds()
                 / 3600.0
            )

            # Scale to match our custom day length
            scaled_time = time_diff * (self.day_length_hours / 24.0)

            return scaled_time
        except (ephem.AlwaysUpError, ephem.NeverUpError):
            # Handle polar day/night
            sun = ephem.Sun()
            sun.compute(observer)

            if float(sun.alt) > 0:
                return self.day_length_hours  # Polar day
            else:
                return 0.0  # Polar night

    def visualize_day_night(self, ax=None, title: str = "Day/Night Cycle"):
        """Visualize the current day/night cycle.

        Args:
            ax: Optional matplotlib axis for plotting
            title: Title for the plot

        Returns:
            Matplotlib image object
        """
        if ax is None:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()

        # Create a world map projection
        img = ax.imshow(
            self.day_night_mask, cmap="Blues",
            extent=[-180, 180, -90, 90],
            interpolation="nearest",
            origin="upper"
        )

        # Add grid lines
        ax.grid(linestyle=":", color="gray", alpha=0.5)

        # Add date/time information
        date_str = self.get_formatted_date()
        ax.text(
            0.02, 0.02, date_str, transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
        )

        # Add season information
        season = self.get_season()
        ax.text(
            0.98, 0.02, f"Season: {season}", transform=ax.transAxes,
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
        )

        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        return img

    def visualize_solar_radiation(self, ax=None,
                                  title: str = "Solar Radiation"):
        """Visualize the current solar radiation pattern.

        Args:
            ax: Optional matplotlib axis for plotting
            title: Title for the plot

        Returns:
            Matplotlib image object
        """
        if ax is None:
            plt.figure(figsize=(12, 6))
            ax = plt.gca()

        # Create a world map projection
        img = ax.imshow(
            self.solar_radiation, cmap="YlOrRd",
            extent=[-180, 180, -90, 90],
            interpolation="nearest",
            origin="upper"
        )

        # Add grid lines
        ax.grid(linestyle=":", color="gray", alpha=0.5)

        # Add date/time information
        date_str = self.get_formatted_date()
        ax.text(
            0.02, 0.02, date_str, transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
        )

        # Add a colorbar
        plt.colorbar(img, ax=ax, label="Radiation Intensity")

        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        return img

    def visualize_day_length(self, latitudes=None):
        """Visualize day length variation across
           the year for selected latitudes.

        Args:
            latitudes: List of latitudes to plot,
                       default is [0, 23.5, 45, 66.5, 90]
        """
        if latitudes is None:
            latitudes = [0, 23.5, 45, 66.5, 90]

        # Save current day
        original_day = self.current_day

        plt.figure(figsize=(12, 6))

        # For each latitude, calculate day length through the year
        for current_lat in latitudes:
            days = np.arange(0, self.year_length_days, self.year_length_days/50)
            day_lengths = []

            for day in days:
                self.current_day = int(day)
                day_lengths.append(self.get_day_length(current_lat))

            plt.plot(days, day_lengths, label=f"{current_lat}° latitude")

        plt.axhline(
            y=self.day_length_hours/2,
            color="black",
            linestyle="--",
            alpha=0.3,
            label="12 hours"
        )

        # Mark seasons (for northern hemisphere)
        season_days = [
            0,
            self.year_length_days/4,
            self.year_length_days/2,
            3*self.year_length_days/4
        ]
        season_names = ["Winter", "Spring", "Summer", "Fall"]

        for day, name in zip(season_days, season_names):
            plt.axvline(x=day, color="gray", linestyle=":", alpha=0.5)
            plt.text(
                day,
                self.day_length_hours * 0.1,
                name,
                rotation=90,
                va="bottom",
                ha="center",
                alpha=0.7
            )

        plt.title("Day Length Throughout the Year")
        plt.xlabel("Day of Year")
        plt.ylabel("Hours of Daylight")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.ylim(0, self.day_length_hours)
        plt.xlim(0, self.year_length_days - 1)
        plt.tight_layout()
        plt.show()

        # Restore original day
        self.current_day = original_day

    def get_seasonal_temperature_factor(self, latitude: float) -> float:
        """Calculate a temperature modifier based on season and latitude.

        Args:
            latitude: Latitude in degrees (-90 to 90)

        Returns:
            Temperature modifier in degrees Celsius
        """
        seasonal_factor = self.get_seasonal_factor(latitude)
        return seasonal_factor * 20  # Scale to ±20°C variation
