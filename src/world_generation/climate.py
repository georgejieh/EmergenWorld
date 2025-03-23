"""Climate system module for EmergenWorld.

This module simulates climate patterns including temperature, precipitation,
humidity, and wind patterns based on terrain features, planetary conditions,
and geographical factors, with support for fantasy elements.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
import xclim.indices as xcind
import verde as vd
from opensimplex import OpenSimplex
from typing import Dict, List, Optional, Tuple, Union


class ClimateSystem:
    """Simulates climate patterns for the EmergenWorld simulation.

    Models climate characteristics including temperature, precipitation,
    wind patterns, and humidity across the world based on terrain elevation,
    water bodies, latitude, and planetary conditions.
    """

    def __init__(
            self,
            terrain_heightmap: np.ndarray,
            water_mask: np.ndarray,
            planetary_system: object,
            world_size: int = 1024,
            base_temperature: float = 14.0,  # Global average temperature (°C)
            seasonal_variation_strength: float = 1.0,
            random_seed: Optional[int] = None,
            fantasy_climate_features: Optional[Dict[str, float]] = None,
            elevation_map: Optional[np.ndarray] = None
    ):
        """Initialize the ClimateSystem with configurable parameters.

        Args:
            terrain_heightmap: 2D array representing terrain elevation (0.0-1.0)
            water_mask: 2D array indicating water bodies (1 for water, 0 for land)
            planetary_system: PlanetarySystem object with day/night and seasonal data
            world_size: Size of the world grid (should match terrain and planetary)
            base_temperature: Global average temperature in degrees Celsius
            seasonal_variation_strength: Multiplier for seasonal effects (1.0 = Earth-like)
            random_seed: Seed for random number generation
            fantasy_climate_features: Optional dictionary of fantasy climate elements
                Supported keys:
                - 'magical_hotspots': Strength of magical thermal anomalies (0.0-1.0)
                - 'elemental_zones': Strength of elemental climate influence (0.0-1.0)
                - 'aether_currents': Strength of supernatural wind patterns (0.0-1.0)
                - 'reality_flux': Degree of climate unpredictability (0.0-1.0)
            elevation_map: Optional precalculated elevation in meters (if not provided, will calculate from heightmap)
        """
        self.heightmap = terrain_heightmap
        self.water_mask = water_mask
        self.planetary = planetary_system
        self.world_size = world_size
        self.base_temperature = base_temperature
        self.seasonal_variation_strength = seasonal_variation_strength
    
        # Initialize random number generator
        self.rng = np.random.RandomState(random_seed)
        self.seed = random_seed if random_seed is not None else np.random.randint(0, 1000000)
    
        # Initialize noise generator for coherent noise patterns
        self.noise_gen = OpenSimplex(seed=self.seed)
    
        # Default fantasy features if none provided
        if fantasy_climate_features is None:
            self.fantasy_features = {
                'magical_hotspots': 0.0,
                'elemental_zones': 0.0,
                'aether_currents': 0.0,
                'reality_flux': 0.0
            }
        else:
            self.fantasy_features = fantasy_climate_features
    
        # Extract unique latitudes and longitudes
        self.lats = self.planetary.latitudes[:, 0]  # Extract unique latitudes
        self.lons = self.planetary.longitudes[0, :]  # Extract unique longitudes
    
        # Create a proper coordinate grid for xarray
        self.lat_grid, self.lon_grid = np.meshgrid(self.lats, self.lons, indexing='ij')
    
        # Calculate Coriolis parameter (f = 2Ω*sin(φ))
        # Earth's angular velocity Ω = 7.292 × 10^-5 rad/s
        omega = 7.292e-5
        self.coriolis = 2 * omega * np.sin(np.radians(self.lat_grid))
    
        # Use provided elevation map or calculate from heightmap
        if elevation_map is not None:
            # Use custom elevation map (in meters)
            actual_elevation = elevation_map
        else:
            # Apply custom elevation scaling with more moderate heights
            actual_elevation = np.zeros_like(self.heightmap)
            for y in range(world_size):
                for x in range(world_size):
                    h = self.heightmap[y, x]
                    if h < 0.3:  # Ocean and shores
                        actual_elevation[y, x] = h * 2000  # Lower elevation
                    elif h < 0.6:  # Normal land
                        # Map 0.3-0.6 to 600-3000m
                        normalized_h = (h - 0.3) / 0.3
                        actual_elevation[y, x] = 600 + normalized_h * 2400
                    else:  # Mountains
                        # Map 0.6-1.0 to 3000-8000m
                        normalized_h = (h - 0.6) / 0.4
                        actual_elevation[y, x] = 3000 + normalized_h * 5000
    
        # Set up climate data using xarray for better scientific data handling
        self.climate_data = xr.Dataset(
            data_vars={
                "temperature": (["y", "x"], np.zeros((world_size, world_size))),
                "precipitation": (["y", "x"], np.zeros((world_size, world_size))),
                "humidity": (["y", "x"], np.zeros((world_size, world_size))),
                "pressure": (["y", "x"], np.ones((world_size, world_size)) * 1013.25),
                "wind_u": (["y", "x"], np.zeros((world_size, world_size))),
                "wind_v": (["y", "x"], np.zeros((world_size, world_size))),
                "elevation": (["y", "x"], actual_elevation),  # Use calculated elevation
                "water": (["y", "x"], self.water_mask),
                "coriolis": (["y", "x"], self.coriolis),
            },
            coords={
                "lat": (["y", "x"], self.lat_grid),
                "lon": (["y", "x"], self.lon_grid),
                "y": np.arange(world_size),
                "x": np.arange(world_size),
            }
        )
    
        # Set up additional metadata for better scientific analysis
        self.climate_data.temperature.attrs["units"] = "degC"
        self.climate_data.precipitation.attrs["units"] = "mm/day"
        self.climate_data.humidity.attrs["units"] = "fraction"
        self.climate_data.pressure.attrs["units"] = "hPa"
        self.climate_data.wind_u.attrs["units"] = "m/s"
        self.climate_data.wind_v.attrs["units"] = "m/s"
        self.climate_data.elevation.attrs["units"] = "m"
        self.climate_data.coriolis.attrs["units"] = "rad/s"
    
        # Cache for biome lookup
        self.biome_cache = {}
    
        # Generate base climate patterns
        self._initialize_base_climate()
    
        print(f"Initialized climate system with {world_size}x{world_size} resolution")
        print(f"Base temperature: {base_temperature}°C")
        print(f"Fantasy climate features enabled: {[k for k, v in self.fantasy_features.items() if v > 0]}")
        print(f"Elevation range: {actual_elevation.min():.1f}m to {actual_elevation.max():.1f}m")

    def _get_simplex_noise(self, x, y, z=0.0, scale=1.0, octaves=1, persistence=0.5, lacunarity=2.0):
        """Generate coherent noise using OpenSimplex.
        
        Args:
            x: X coordinate
            y: Y coordinate
            z: Z coordinate (default 0.0)
            scale: Noise scale factor
            octaves: Number of octaves for noise generation
            persistence: Persistence value for octaves
            lacunarity: Lacunarity value for octaves
            
        Returns:
            Noise value in range [-1, 1]
        """
        # Apply scaling
        nx = x * scale
        ny = y * scale
        nz = z * scale
        
        # Generate noise with multiple octaves
        value = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            if hasattr(self.noise_gen, 'noise3d'):  # Newer versions of opensimplex
                n = self.noise_gen.noise3d(nx * frequency, ny * frequency, nz * frequency)
            else:  # Older versions of opensimplex
                n = self.noise_gen.noise3(nx * frequency, ny * frequency, nz * frequency)
                
            value += n * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        
        # Normalize to [-1, 1]
        return value / max_value

    def _initialize_base_climate(self) -> None:
        """Initialize the base climate patterns before any simulation."""
        # Calculate base pressure based on elevation and latitude
        self._generate_base_pressure()
        
        # Calculate base temperature based on latitude and elevation
        self._generate_base_temperature()
        
        # Generate initial wind patterns based on pressure gradients and Coriolis force
        self._generate_wind_patterns()
        
        # Calculate initial humidity based on temperature, pressure, and water proximity
        self._generate_base_humidity()
        
        # Generate initial precipitation based on humidity, wind, and orographic effects
        self._generate_precipitation()
        
        # Apply magical/fantasy elements if enabled
        self._apply_fantasy_climate_features()
        
        # Calculate derived climate variables and indices
        self._calculate_climate_indices()
    
    def _generate_base_pressure(self) -> None:
        """Generate the base atmospheric pressure map using barometric formula and global circulation patterns."""
        # Get elevation in meters
        elevation = self.climate_data.elevation.values
    
        # Calculate pressure using barometric formula with standard parameters
        # P = P_0 * exp(-elevation / scale_height)
        scale_height = 8500  # Scale height of Earth's atmosphere (m)
        sea_level_pressure = 1013.25  # Standard sea-level pressure (hPa)
    
        # Apply the barometric formula directly
        pressure = sea_level_pressure * np.exp(-elevation / scale_height)
    
        # Apply global circulation patterns to create realistic pressure systems
        for y in range(self.world_size):
            for x in range(self.world_size):
                latitude = self.lat_grid[y, x]
            
                # Create global pressure belts similar to Earth's
                # - High pressure at subtropical highs (30°N/S) and polar regions
                # - Low pressure at equator (ITCZ) and subpolar lows (60°N/S)
                if abs(latitude) < 15:  # Equatorial low (ITCZ)
                    pressure_mod = -5.0
                elif 15 <= abs(latitude) < 35:  # Subtropical high
                    pressure_mod = 5.0
                elif 35 <= abs(latitude) < 65:  # Subpolar low
                    pressure_mod = -5.0
                else:  # Polar high
                    pressure_mod = 5.0
                
                # Add the modification
                pressure[y, x] += pressure_mod
    
        # Use Verde's BlockReduce to apply regional smoothing for more realistic patterns
        # This creates larger-scale pressure systems with proper spatial correlation
        block_mean = vd.BlockReduce(np.mean, spacing=3)
    
        # Create a meshgrid of coordinates
        y_coords, x_coords = np.meshgrid(
            np.arange(self.world_size),
            np.arange(self.world_size),
            indexing='ij'
        )
    
        # Flatten the arrays to match what Verde expects
        coordinates = (y_coords.flatten(), x_coords.flatten())
        pressure_flat = pressure.flatten()
    
        # Apply the filtering
        reduced_coords, reduced_pressure = block_mean.filter(coordinates, pressure_flat)
    
        # Now we can use ScipyGridder to interpolate back to our original grid
        pressure_grid = vd.ScipyGridder(method="cubic").fit(reduced_coords, reduced_pressure)
    
        # Grid back to the original resolution
        pressure_interp = pressure_grid.grid(
            region=(0, self.world_size, 0, self.world_size),
            shape=(self.world_size, self.world_size),
            data_names=["pressure"]
        )
    
        # Add random perturbations to create weather systems like highs and lows
        # Use OpenSimplex noise for spatially correlated pressure variations
        noise = np.zeros((self.world_size, self.world_size))
        scale = 0.05  # Controls the spatial scale of the noise
        for y in range(self.world_size):
            for x in range(self.world_size):
                noise[y, x] = self._get_simplex_noise(
                    x, y, 0.0, 
                    scale=scale,
                    octaves=4, 
                    persistence=0.5, 
                    lacunarity=2.0
                )
    
        # Scale noise to appropriate pressure variations (±5 hPa)
        noise = noise * 5.0
    
        # Add noise to pressure field
        pressure = pressure_interp.pressure.values + noise
    
        # Ensure pressure stays within realistic bounds (870-1090 hPa: extremes on Earth)
        pressure = np.clip(pressure, 870, 1090)
    
        # Update the dataset
        self.climate_data["pressure"].values = pressure

    def _generate_base_temperature(self) -> None:
        """Generate the base temperature map based on latitude, elevation and complex climate factors."""
        # Create initial temperatures based on latitudinal gradients
        temperatures = np.zeros((self.world_size, self.world_size))
        
        # Implement a more realistic temperature model using real climate science
        for y in range(self.world_size):
            for x in range(self.world_size):
                latitude = self.lat_grid[y, x]
                
                # More accurate Earth-like latitudinal temperature model
                # Based on empirical data of Earth's latitudinal temperature distribution
                if abs(latitude) < 10:  # Equatorial zone
                    # Warm, but not the warmest due to cloud cover and precipitation
                    base_temp = 26.0
                elif abs(latitude) < 30:  # Tropical/subtropical
                    # Typically the warmest zones (deserts, etc.)
                    base_temp = 28.0 - (abs(latitude) - 10) * 0.4
                elif abs(latitude) < 60:  # Temperate
                    # Rapid decline in temperature
                    base_temp = 16.0 - (abs(latitude) - 30) * 0.5
                else:  # Polar
                    # Very cold, approaching -30°C at poles
                    base_temp = -14.0 - (abs(latitude) - 60) * 0.4
                
                # Apply base temperature relative to global average
                temperatures[y, x] = self.base_temperature + (base_temp - 14.0)
        
        # Apply elevation using proper environmental lapse rate
        # Standard adiabatic lapse rate is 6.5°C/km but varies by climate zone
        elevation = self.climate_data.elevation.values
        
        for y in range(self.world_size):
            for x in range(self.world_size):
                # Different lapse rates for different climate zones
                latitude = abs(self.lat_grid[y, x])
                if latitude < 30:  # Tropical
                    lapse_rate = 5.5 / 1000.0  # °C/m (less than standard due to humidity)
                elif latitude < 60:  # Temperate
                    lapse_rate = 6.5 / 1000.0  # Standard lapse rate
                else:  # Polar
                    lapse_rate = 7.5 / 1000.0  # More than standard due to dry air
                
                # Apply elevation correction
                temperatures[y, x] -= elevation[y, x] * lapse_rate
        
        # Apply ocean influence using a proper ocean heat capacity model
        # Oceans have much higher heat capacity and moderate nearby land
        water_mask = self.climate_data.water.values
        land_mask = 1 - water_mask
        
        # Calculate ocean temperature with reduced seasonal variation
        # Ocean temperatures typically lag seasonal changes by about 2 months
        # and have reduced amplitude in yearly variation
        ocean_temps = temperatures.copy() * water_mask
        land_temps = temperatures.copy() * land_mask
        
        # Smooth ocean temperatures to reflect higher heat capacity and mixing
        ocean_temps = ndimage.gaussian_filter(ocean_temps, sigma=5.0)
        
        # Model ocean currents influence by shifting temperatures poleward
        # Use a simplified model of ocean currents based on latitude
        for y in range(self.world_size):
            for x in range(self.world_size):
                if water_mask[y, x] > 0:
                    latitude = self.lat_grid[y, x]
                    
                    # Western boundary currents (Gulf Stream, Kuroshio) warm high latitudes
                    # Eastern boundary currents cool lower latitudes
                    if 30 < abs(latitude) < 60:  # Mid-latitudes - warming effect
                        ocean_temps[y, x] += 5.0 * np.exp(-(abs(latitude) - 45)**2 / 200)
                    elif abs(latitude) < 30:  # Subtropical - cooling effect
                        ocean_temps[y, x] -= 2.0 * np.exp(-(abs(latitude) - 15)**2 / 200)
        
        # Land-sea temperature contrast - coastal moderation
        # Distance from coast affects temperature moderation
        distance_to_water = ndimage.distance_transform_edt(land_mask)
        coastal_influence = np.exp(-distance_to_water / 20)  # Exponential decay of influence
        
        # Apply coastal moderation - land temps become more like ocean temps near coasts
        moderation_strength = 0.7  # How strongly oceans moderate nearby land
        for y in range(self.world_size):
            for x in range(self.world_size):
                if land_mask[y, x] > 0:  # Only for land
                    # Find average ocean temperature in vicinity
                    y_min, y_max = max(0, y-10), min(self.world_size, y+10)
                    x_min, x_max = max(0, x-10), min(self.world_size, x+10)
                    
                    nearby_ocean = ocean_temps[y_min:y_max, x_min:x_max] * water_mask[y_min:y_max, x_min:x_max]
                    if np.sum(water_mask[y_min:y_max, x_min:x_max]) > 0:
                        mean_ocean_temp = np.sum(nearby_ocean) / np.sum(water_mask[y_min:y_max, x_min:x_max])
                        
                        # Moderate land temperature based on distance to water
                        moderation = coastal_influence[y, x] * moderation_strength
                        land_temps[y, x] = land_temps[y, x] * (1 - moderation) + mean_ocean_temp * moderation
        
        # Combine water and land temperatures
        temperatures = ocean_temps + land_temps
        
        # Apply continental climate effects (greater temperature extremes away from oceans)
        # Continental interiors have greater temperature variations
        continental_effect = distance_to_water / np.max(distance_to_water) * 10.0  # Up to 10°C effect
        
        # Apply the effect differently based on latitude
        # Mid-latitudes experience stronger continental effects
        for y in range(self.world_size):
            for x in range(self.world_size):
                if land_mask[y, x] > 0:
                    latitude = abs(self.lat_grid[y, x])
                    if 30 < latitude < 60:  # Strongest in mid-latitudes
                        latitude_factor = 1.0
                    else:
                        latitude_factor = max(0, 1.0 - abs(latitude - 45) / 45)
                    
                    # Continental interiors are cooler in this base map (will vary with seasons)
                    temperatures[y, x] -= continental_effect[y, x] * latitude_factor * 0.5
        
        # Update the dataset
        self.climate_data["temperature"].values = temperatures

    def _generate_wind_patterns(self) -> None:
        """Generate wind patterns based on pressure gradients, Coriolis force, and thermal effects."""
        # Get pressure field and calculate gradients
        pressure = self.climate_data.pressure.values
        
        # Calculate gradients (hPa per grid cell)
        dy, dx = np.gradient(pressure)
        
        # Convert gradients to proper units (Pa/m)
        # 1 hPa = 100 Pa
        # Need to divide by distance between grid cells
        cell_size_m = self.planetary.km_per_cell * 1000  # Convert km to m
        dx_pam = dx * 100 / cell_size_m  # Pa/m
        dy_pam = dy * 100 / cell_size_m  # Pa/m
        
        # Get Coriolis parameter
        f = self.climate_data.coriolis.values
        
        # Initialize wind components
        u_geo = np.zeros((self.world_size, self.world_size))
        v_geo = np.zeros((self.world_size, self.world_size))
        
        # Air density (kg/m³) - decreases with height
        rho = 1.225 * np.exp(-self.climate_data.elevation.values / 8500)
        
        # Calculate geostrophic wind, handling the equatorial case
        for y in range(self.world_size):
            for x in range(self.world_size):
                if abs(f[y, x]) > 1e-10:  # Away from equator
                    # Geostrophic wind equation
                    u_geo[y, x] = -1 / (rho[y, x] * f[y, x]) * dy_pam[y, x]
                    v_geo[y, x] = 1 / (rho[y, x] * f[y, x]) * dx_pam[y, x]
                else:  # Near equator
                    # Use thermal wind approximation
                    # Winds flow from high to low pressure directly
                    magnitude = np.sqrt(dx_pam[y, x]**2 + dy_pam[y, x]**2)
                    if magnitude > 0:
                        u_geo[y, x] = -dx_pam[y, x] / magnitude * 10
                        v_geo[y, x] = -dy_pam[y, x] / magnitude * 10
        
        # Apply thermal wind effects (vertical wind shear due to temperature gradients)
        temperature = self.climate_data.temperature.values
        dt_dy, dt_dx = np.gradient(temperature)
        
        # Apply thermal wind correction (simplified)
        for y in range(self.world_size):
            for x in range(self.world_size):
                if abs(f[y, x]) > 1e-10:  # Away from equator
                    # Stronger westerlies with stronger north-south temperature gradient
                    u_geo[y, x] -= dt_dy[y, x] * 0.5
                    v_geo[y, x] += dt_dx[y, x] * 0.5
        
        # Scale wind to realistic values (typically 0-30 m/s)
        max_wind = np.max(np.sqrt(u_geo**2 + v_geo**2))
        if max_wind > 30:
            scale_factor = 30 / max_wind
            u_geo *= scale_factor
            v_geo *= scale_factor
        
        # Apply boundary layer effects
        # Near surface, friction reduces wind speed and causes cross-isobaric flow
        terrain_roughness = np.zeros_like(self.heightmap)
        water_mask = self.climate_data.water.values
        land_mask = 1 - water_mask
        
        # Ocean has low roughness
        terrain_roughness[water_mask > 0] = 0.1
        
        # Land has variable roughness based on elevation variability
        for y in range(1, self.world_size-1):
            for x in range(1, self.world_size-1):
                if land_mask[y, x] > 0:
                    # Get neighboring elevations
                    neighbors = [
                        self.heightmap[y-1, x], self.heightmap[y+1, x],
                        self.heightmap[y, x-1], self.heightmap[y, x+1]
                    ]
                    # Variance as measure of roughness
                    terrain_roughness[y, x] = 0.3 + np.var(neighbors) * 5.0
        
        # Apply friction - reduce speed and turn wind toward low pressure
        u_sfc = np.zeros_like(u_geo)
        v_sfc = np.zeros_like(v_geo)
        
        for y in range(self.world_size):
            for x in range(self.world_size):
                # Reduction factor based on roughness
                reduction = 1.0 - 0.5 * terrain_roughness[y, x]
                reduction = max(0.2, min(0.9, reduction))  # Limit range
                
                # Turning angle based on roughness (10-30°)
                angle_rad = np.radians(10 + 20 * terrain_roughness[y, x])
                
                # Apply reduction and turning
                speed = np.sqrt(u_geo[y, x]**2 + v_geo[y, x]**2)
                if speed > 0:
                    dir_x, dir_y = u_geo[y, x] / speed, v_geo[y, x] / speed
                    
                    # Rotate wind vector (simple 2D rotation)
                    new_dir_x = dir_x * np.cos(angle_rad) - dir_y * np.sin(angle_rad)
                    new_dir_y = dir_x * np.sin(angle_rad) + dir_y * np.cos(angle_rad)
                    
                    # Apply new direction and reduced speed
                    u_sfc[y, x] = new_dir_x * speed * reduction
                    v_sfc[y, x] = new_dir_y * speed * reduction
        
        # Apply local circulation effects (sea/land breezes, mountain/valley winds)
        # These are driven by differential heating
        
        # Sea-land breeze effect near coastlines
        distance_to_coast = np.minimum(
            ndimage.distance_transform_edt(water_mask),
            ndimage.distance_transform_edt(land_mask)
        )
        coastal_zone = distance_to_coast < 10
        
        # Direction depends on time of day (simplified)
        is_day = np.mean(self.planetary.day_night_mask) > 0.5
        coastal_wind_strength = 2.0  # m/s
        
        for y in range(self.world_size):
            for x in range(self.world_size):
                if coastal_zone[y, x]:
                    # Find direction to nearest water
                    y_min, y_max = max(0, y-5), min(self.world_size, y+5)
                    x_min, x_max = max(0, x-5), min(self.world_size, x+5)
                    
                    local_water = water_mask[y_min:y_max, x_min:x_max]
                    local_land = land_mask[y_min:y_max, x_min:x_max]
                    
                    if np.sum(local_water) > 0 and np.sum(local_land) > 0:
                        # Calculate average water and land positions
                        y_indices, x_indices = np.mgrid[y_min:y_max, x_min:x_max]
                        
                        water_y = np.sum(y_indices * local_water) / np.sum(local_water) - y
                        water_x = np.sum(x_indices * local_water) / np.sum(local_water) - x
                        
                        # Normalize direction vector
                        dist = np.sqrt(water_y**2 + water_x**2)
                        if dist > 0:
                            dir_y, dir_x = water_y / dist, water_x / dist
                            
                            # During day: sea breeze (from sea to land)
                            # During night: land breeze (from land to sea)
                            if is_day:
                                dir_y, dir_x = -dir_y, -dir_x
                            
                            # Add to existing wind
                            strength = coastal_wind_strength * (1.0 - distance_to_coast[y, x] / 10)
                            u_sfc[y, x] += dir_x * strength
                            v_sfc[y, x] += dir_y * strength
        
        # Mountain/valley winds near significant slopes
        dy_terrain, dx_terrain = np.gradient(self.heightmap)
        slope_magnitude = np.sqrt(dx_terrain**2 + dy_terrain**2)
        mountain_areas = slope_magnitude > 0.05
        
        for y in range(self.world_size):
            for x in range(self.world_size):
                if mountain_areas[y, x] and land_mask[y, x] > 0:
                    # Direction of the slope
                    if slope_magnitude[y, x] > 0:
                        slope_y, slope_x = dy_terrain[y, x] / slope_magnitude[y, x], dx_terrain[y, x] / slope_magnitude[y, x]
                        
                        # During day: anabatic (upslope) winds
                        # During night: katabatic (downslope) winds
                        if is_day:
                            dir_y, dir_x = slope_y, slope_x
                        else:
                            dir_y, dir_x = -slope_y, -slope_x
                        
                        # Add to existing wind
                        mountain_strength = 3.0 * slope_magnitude[y, x]
                        u_sfc[y, x] += dir_x * mountain_strength
                        v_sfc[y, x] += dir_y * mountain_strength
        
        # Final smoothing for realistic wind fields
        u_final = ndimage.gaussian_filter(u_sfc, sigma=1.0)
        v_final = ndimage.gaussian_filter(v_sfc, sigma=1.0)
        
        # Update the dataset
        self.climate_data["wind_u"].values = u_final
        self.climate_data["wind_v"].values = v_final

    def _generate_base_humidity(self) -> None:
        """Generate humidity based on temperature, pressure and water proximity."""
        # Initialize with evaporation from water bodies
        humidity = np.zeros((self.world_size, self.world_size))
        
        # Water bodies have maximum humidity at surface
        water_mask = self.climate_data.water.values
        humidity[water_mask > 0] = 1.0
        
        # Land humidity depends on distance from water sources
        land_mask = 1 - water_mask
        
        # Calculate distance from water using distance transform
        distance_to_water = ndimage.distance_transform_edt(land_mask)
        
        # Calculate prevailing wind direction (coarse scale)
        u_wind = self.climate_data.wind_u.values
        v_wind = self.climate_data.wind_v.values
        
        # Smooth wind for large-scale transport calculation
        u_smooth = ndimage.gaussian_filter(u_wind, sigma=5.0)
        v_smooth = ndimage.gaussian_filter(v_wind, sigma=5.0)
        
        # Calculate humidity based on wind patterns and distance from water
        for y in range(self.world_size):
            for x in range(self.world_size):
                if land_mask[y, x] > 0:
                    # Basic exponential decay with distance
                    dist_effect = np.exp(-distance_to_water[y, x] / 50.0)
                    
                    # Calculate upwind direction
                    local_u = u_smooth[y, x]
                    local_v = v_smooth[y, x]
                    wind_speed = np.sqrt(local_u**2 + local_v**2)
                    
                    # Only consider wind transport if there's significant wind
                    if wind_speed > 0.5:
                        # Normalize the wind vector
                        wind_u = local_u / wind_speed
                        wind_v = local_v / wind_speed
                        
                        # Check humidity upwind (approximate)
                        steps = 10
                        upwind_humidity = 0.0
                        count = 0
                        
                        for i in range(1, steps+1):
                            # Calculate upwind position
                            upwind_y = int(y - wind_v * i * 2)
                            upwind_x = int(x - wind_u * i * 2)
                            
                            # Check if position is within bounds
                            if (0 <= upwind_y < self.world_size and 
                                0 <= upwind_x < self.world_size):
                                
                                # If upwind has water, increase humidity contribution
                                if water_mask[upwind_y, upwind_x] > 0:
                                    upwind_humidity += 1.0 / (i**0.5)  # Higher weight for closer water
                                    count += 1
                        
                        # Calculate wind-based humidity factor
                        if count > 0:
                            wind_humidity = upwind_humidity / count
                            wind_humidity = min(1.0, wind_humidity * 1.5)  # Scale up for effect
                        else:
                            wind_humidity = 0.0
                        
                        # Combine distance and wind effects (wind effect stronger)
                        humidity[y, x] = 0.3 * dist_effect + 0.7 * wind_humidity
                    else:
                        # Just use distance effect if wind is weak
                        humidity[y, x] = dist_effect
        
        # Apply temperature effects on humidity (warmer air can hold more moisture)
        temperature = self.climate_data.temperature.values
        
        for y in range(self.world_size):
            for x in range(self.world_size):
                if land_mask[y, x] > 0:  # Only for land
                    temp = temperature[y, x]
                    
                    # Maximum humidity decreases in cold regions
                    # Use a simplified version of Clausius-Clapeyron relation
                    if temp < 0:
                        # Very cold air is drier
                        max_humidity = 0.5 * (1 + temp / 30)  # 0.5 at 0°C, 0.0 at -30°C
                    else:
                        # Warmer air can hold more moisture
                        max_humidity = 0.5 + 0.5 * min(1.0, temp / 40)  # 0.5 at 0°C to 1.0 at 40°C+
                    
                    # Limit humidity by temperature constraint
                    humidity[y, x] = min(humidity[y, x], max_humidity)
        
        # Apply orographic effects (increasing humidity on windward slopes)
        self._apply_orographic_humidity(humidity)
        
        # Apply latitudinal patterns (ITCZ, subtropical highs, etc.)
        for y in range(self.world_size):
            for x in range(self.world_size):
                if land_mask[y, x] > 0:
                    latitude = abs(self.lat_grid[y, x])
                    
                    # ITCZ - very humid near equator
                    if latitude < 10:
                        humidity[y, x] = min(1.0, humidity[y, x] * 1.3)
                    
                    # Subtropical desert belts - drier
                    elif 15 < latitude < 35:
                        humidity[y, x] *= 0.7
        
        # Final smoothing for realistic transitions
        humidity = ndimage.gaussian_filter(humidity, sigma=1.0)
        
        # Ensure humidity stays in valid range [0,1]
        humidity = np.clip(humidity, 0.0, 1.0)
        
        # Update the dataset
        self.climate_data["humidity"].values = humidity

    def _apply_orographic_humidity(self, humidity: np.ndarray) -> None:
        """Apply orographic effects to humidity based on wind patterns and terrain."""
        # Get wind components and terrain gradient
        u_wind = self.climate_data.wind_u.values
        v_wind = self.climate_data.wind_v.values
        
        # Calculate terrain gradient
        dy_terrain, dx_terrain = np.gradient(self.heightmap)
        
        # Calculate slope direction and magnitude
        slope_magnitude = np.sqrt(dx_terrain**2 + dy_terrain**2)
        
        # Normalized wind vectors
        wind_magnitude = np.sqrt(u_wind**2 + v_wind**2)
        u_norm = np.zeros_like(u_wind)
        v_norm = np.zeros_like(v_wind)
        
        # Avoid division by zero
        mask = wind_magnitude > 0.5
        u_norm[mask] = u_wind[mask] / wind_magnitude[mask]
        v_norm[mask] = v_wind[mask] / wind_magnitude[mask]
        
        # Wind-slope dot product: positive when wind blows upslope
        orographic_effect = u_norm * dx_terrain + v_norm * dy_terrain
        
        # Apply to humidity
        for y in range(self.world_size):
            for x in range(self.world_size):
                if self.water_mask[y, x] == 0 and slope_magnitude[y, x] > 0.02 and mask[y, x]:
                    # Windward slopes: increased humidity
                    if orographic_effect[y, x] > 0:
                        effect = orographic_effect[y, x] * slope_magnitude[y, x] * 3.0
                        humidity[y, x] = min(1.0, humidity[y, x] + effect)
                    
                    # Leeward slopes: rain shadow effect
                    else:
                        effect = -orographic_effect[y, x] * slope_magnitude[y, x] * 5.0
                        humidity[y, x] = max(0.1, humidity[y, x] - effect)

    def _generate_precipitation(self) -> None:
        """Generate precipitation patterns based on humidity, temperature, and atmospheric dynamics."""
        # Get required variables
        humidity = self.climate_data.humidity.values
        temperature = self.climate_data.temperature.values
        u_wind = self.climate_data.wind_u.values
        v_wind = self.climate_data.wind_v.values
        
        # Initialize precipitation array
        precipitation = np.zeros((self.world_size, self.world_size))
        
        # Basic precipitation model - precipitation proportional to humidity
        for y in range(self.world_size):
            for x in range(self.world_size):
                # Use a non-linear relationship (higher humidity = much higher precipitation)
                # This reflects atmospheric physics where precipitation increases more than
                # linearly with humidity (supersaturation effects)
                humidity_factor = humidity[y, x] ** 2
                
                # Base precipitation rate (mm/day)
                base_precip = humidity_factor * 15.0
                
                # Temperature affects precipitation type and efficiency
                temp = temperature[y, x]
                
                # Highest precipitation in moderate temperatures
                # Lower in very cold (less water vapor) or very hot (less condensation) regions
                if temp < 0:
                    temp_factor = 0.7 + 0.3 * (temp / -30)  # 0.7 at 0°C, declining to 0.4 at -30°C
                elif temp < 25:
                    temp_factor = 0.7 + 0.3 * (temp / 25)  # 0.7 at 0°C, rising to 1.0 at 25°C
                else:
                    temp_factor = 1.0 - 0.3 * min(1.0, (temp - 25) / 15)  # 1.0 at 25°C, falling to 0.7 at 40°C+
                
                precipitation[y, x] = base_precip * temp_factor
        
        # Apply convergence zones - areas where winds converge have more precipitation
        # Calculate wind convergence (negative divergence)
        du_dx = np.zeros_like(u_wind)
        dv_dy = np.zeros_like(v_wind)
        
        # Calculate derivatives
        for y in range(1, self.world_size-1):
            for x in range(1, self.world_size-1):
                du_dx[y, x] = (u_wind[y, x+1] - u_wind[y, x-1]) / 2
                dv_dy[y, x] = (v_wind[y+1, x] - v_wind[y-1, x]) / 2
        
        # Convergence = -(du/dx + dv/dy)
        convergence = -(du_dx + dv_dy)
        
        # Normalize convergence for effect scaling
        if np.max(np.abs(convergence)) > 0:
            convergence = convergence / np.max(np.abs(convergence)) * 2.0
        
        # Apply convergence effect to precipitation
        for y in range(self.world_size):
            for x in range(self.world_size):
                if convergence[y, x] > 0:  # Convergence increases precipitation
                    precipitation[y, x] *= (1.0 + convergence[y, x])
        
        # Apply orographic precipitation effects
        self._apply_orographic_precipitation(precipitation)
        
        # Apply regional precipitation patterns based on global circulation
        self._apply_zonal_precipitation_patterns(precipitation)
        
        # Apply local effects like lake-effect precipitation
        water_mask = self.climate_data.water.values
        land_mask = 1 - water_mask
        
        # Lake effect snow/rain - enhanced precipitation downwind of lakes
        for y in range(self.world_size):
            for x in range(self.world_size):
                if land_mask[y, x] > 0:
                    # Consider only significant wind
                    wind_speed = np.sqrt(u_wind[y, x]**2 + v_wind[y, x]**2)
                    if wind_speed > 1.0:
                        # Find upwind position
                        upwind_factor = 5
                        upwind_y = int(y - v_wind[y, x] / wind_speed * upwind_factor)
                        upwind_x = int(x - u_wind[y, x] / wind_speed * upwind_factor)
                        
                        # Check if upwind position is within bounds and is water
                        if (0 <= upwind_y < self.world_size and 
                            0 <= upwind_x < self.world_size and 
                            water_mask[upwind_y, upwind_x] > 0):
                            
                            # Temperature conditions for lake effect
                            temp_diff = temperature[upwind_y, upwind_x] - temperature[y, x]
                            
                            # Lake effect strongest when lake warmer than land
                            if temp_diff > 5 and temperature[y, x] < 10:
                                # Increase precipitation (stronger effect with colder air)
                                lake_effect = 1.0 + 0.5 * min(1.0, -temperature[y, x] / 10)
                                precipitation[y, x] *= lake_effect
        
        # Apply final smoothing for realistic patterns
        precipitation = ndimage.gaussian_filter(precipitation, sigma=1.0)
        
        # Ensure precipitation is non-negative
        precipitation = np.maximum(0.0, precipitation)
        
        # Update the dataset
        self.climate_data["precipitation"].values = precipitation

    def _apply_orographic_precipitation(self, precipitation: np.ndarray) -> None:
        """Apply orographic effects to precipitation with realistic physics."""
        # Get wind components and terrain gradient
        u_wind = self.climate_data.wind_u.values
        v_wind = self.climate_data.wind_v.values
        
        # Calculate terrain gradient
        dy_terrain, dx_terrain = np.gradient(self.heightmap)
        
        # Calculate slope magnitude
        slope_magnitude = np.sqrt(dx_terrain**2 + dy_terrain**2)
        
        # Calculate wind direction and speed
        wind_magnitude = np.sqrt(u_wind**2 + v_wind**2)
        
        # Initialize normalized wind vectors
        u_norm = np.zeros_like(u_wind)
        v_norm = np.zeros_like(v_wind)
        
        # Avoid division by zero
        mask = wind_magnitude > 0.5  # Only consider significant wind
        u_norm[mask] = u_wind[mask] / wind_magnitude[mask]
        v_norm[mask] = v_wind[mask] / wind_magnitude[mask]
        
        # Calculate wind-slope dot product (upslope/downslope wind component)
        upslope_wind = u_norm * dx_terrain + v_norm * dy_terrain
        
        # Apply orographic effect to precipitation
        for y in range(1, self.world_size-1):
            for x in range(1, self.world_size-1):
                if self.water_mask[y, x] == 0 and mask[y, x]:  # Land areas with significant wind
                    # Slope must be significant
                    if slope_magnitude[y, x] > 0.01:
                        # Windward slopes: enhanced precipitation
                        if upslope_wind[y, x] > 0:
                            # Effect stronger with steeper slope and stronger upslope wind
                            effect = upslope_wind[y, x] * slope_magnitude[y, x] * 15.0
                            precipitation[y, x] += effect
                        
                        # Leeward slopes: rain shadow effect
                        else:
                            # Effect stronger with steeper slope
                            effect = -upslope_wind[y, x] * slope_magnitude[y, x] * 8.0
                            precipitation[y, x] = max(0.2, precipitation[y, x] - effect)
                            
                            # Track precipitation depletion downwind
                            # This creates extended rain shadows beyond the immediate lee slope
                            for i in range(1, 6):  # Check up to 5 cells downwind
                                shadow_y = int(y + v_norm[y, x] * i)
                                shadow_x = int(x + u_norm[y, x] * i)
                                
                                if (0 <= shadow_y < self.world_size and 
                                    0 <= shadow_x < self.world_size and 
                                    self.water_mask[shadow_y, shadow_x] == 0):
                                    
                                    # Rain shadow weakens with distance
                                    shadow_factor = 0.8 - 0.1 * i  # 0.7 to 0.3
                                    precipitation[shadow_y, shadow_x] *= shadow_factor

    def _apply_zonal_precipitation_patterns(self, precipitation: np.ndarray) -> None:
        """Apply global circulation related precipitation patterns by latitude zone."""
        # Apply latitude-based precipitation patterns from global circulation
        for y in range(self.world_size):
            latitude = self.lats[y]
            abs_lat = abs(latitude)
            
            # Equatorial convergence zone (ITCZ) - high rainfall
            if abs_lat < 10:
                for x in range(self.world_size):
                    if self.water_mask[y, x] == 0:  # Only modify land
                        precipitation[y, x] *= 1.4
            
            # Subtropical high pressure (desert belts) - low rainfall
            elif 15 < abs_lat < 35:
                for x in range(self.world_size):
                    if self.water_mask[y, x] == 0:  # Only modify land
                        precipitation[y, x] *= 0.6
            
            # Mid-latitude storm tracks - moderate to high rainfall
            elif 40 < abs_lat < 65:
                for x in range(self.world_size):
                    if self.water_mask[y, x] == 0:  # Only modify land
                        precipitation[y, x] *= 1.2
            
            # Polar regions - low precipitation (cold air holds less moisture)
            elif abs_lat > 70:
                for x in range(self.world_size):
                    if self.water_mask[y, x] == 0:  # Only modify land
                        precipitation[y, x] *= 0.5

    def _calculate_climate_indices(self) -> None:
        """Calculate various climate indices for analysis and biome determination."""
        # Create climate indices using xclim
        temperature = self.climate_data.temperature.values
        precipitation = self.climate_data.precipitation.values
        
        # Convert to proper dimensions and units for xclim
        # Monthly values would be ideal, but we'll use annual averages here
        tas_dataarray = xr.DataArray(
            temperature, 
            dims=["lat", "lon"], 
            coords={"lat": self.lats, "lon": self.lons}
        )
        tas_dataarray.attrs["units"] = "degC"
        
        pr_dataarray = xr.DataArray(
            precipitation * 365,  # Annual precipitation in mm/year
            dims=["lat", "lon"], 
            coords={"lat": self.lats, "lon": self.lons}
        )
        pr_dataarray.attrs["units"] = "mm/year"
        
        # Calculate various climate indices
        try:
            # Growing degree days (base 5°C)
            gdd = xcind.growing_degree_days(tas_dataarray, thresh="5.0 degC")
            
            # Aridity index (P/PET) - simplified calculation
            # Potential evapotranspiration estimated from temperature
            temp_k = temperature + 273.15  # Convert to Kelvin
            solar_energy = np.cos(np.radians(abs(self.lat_grid))) * 1000  # Simple solar input estimate
            pet = 0.0023 * solar_energy * (temp_k - 273.16) * np.sqrt(self.climate_data.humidity.values)
            pet = np.clip(pet, 0.1, 3000)  # Ensure reasonable range
            
            aridity = precipitation * 365 / pet  # Annual precipitation / PET
            aridity = np.clip(aridity, 0.01, 5.0)  # Limit range for numerical stability
            
            # Store calculated indices in the climate dataset
            self.climate_data["growing_degree_days"] = (["y", "x"], gdd.values)
            self.climate_data["aridity_index"] = (["y", "x"], aridity)
            
            # Add bioclimatic temperature indices
            self.climate_data["annual_mean_temp"] = (["y", "x"], temperature)
            
        except Exception as e:
            print(f"Warning: Could not calculate some climate indices: {e}")
            # Proceed without indices

    def _apply_fantasy_climate_features(self) -> None:
        """Apply fantasy-specific climate elements based on configured parameters."""
        # Skip if no fantasy features are enabled
        if all(value == 0 for value in self.fantasy_features.values()):
            return
        
        # Apply magical hotspots (thermal anomalies)
        if self.fantasy_features.get('magical_hotspots', 0) > 0:
            self._apply_magical_hotspots()
        
        # Apply elemental climate zones (extreme climate regions)
        if self.fantasy_features.get('elemental_zones', 0) > 0:
            self._apply_elemental_climate_zones()
        
        # Apply aether currents (supernatural wind patterns)
        if self.fantasy_features.get('aether_currents', 0) > 0:
            self._apply_aether_currents()
        
        # Apply reality flux (unpredictable climate pockets)
        if self.fantasy_features.get('reality_flux', 0) > 0:
            self._apply_reality_flux()

    def _apply_magical_hotspots(self) -> None:
        """Apply magical thermal anomalies that create localized climate variations."""
        # Determine strength of effect
        strength = self.fantasy_features['magical_hotspots']
        
        # Generate number of hotspots based on world size
        num_hotspots = int(np.sqrt(self.world_size) * strength * 0.5)
        
        # Hotspot types with distinct climate signatures
        hotspot_types = [
            {'name': 'arcane_nexus', 'temp': 25, 'humidity': 0.4, 'precip': 1.5, 'radius': 10},
            {'name': 'frost_well', 'temp': -25, 'humidity': -0.2, 'precip': -0.5, 'radius': 15},
            {'name': 'sunburst_vent', 'temp': 40, 'humidity': -0.8, 'precip': -0.8, 'radius': 12},
            {'name': 'ethereal_fountain', 'temp': 5, 'humidity': 0.9, 'precip': 2.0, 'radius': 14},
            {'name': 'void_tear', 'temp': -15, 'humidity': -0.5, 'precip': -0.7, 'radius': 8}
        ]
        
        # Create hotspots
        for _ in range(num_hotspots):
            # Select random hotspot type
            hotspot = self.rng.choice(hotspot_types)
            
            # Random location (prefer land)
            attempts = 0
            while attempts < 10:
                x = self.rng.randint(0, self.world_size)
                y = self.rng.randint(0, self.world_size)
                
                # Prefer land locations
                if self.water_mask[y, x] == 0 or attempts > 5:
                    break
                attempts += 1
            
            # Scale radius by strength
            radius = int(hotspot['radius'] * strength * (0.8 + 0.4 * self.rng.random()))
            
            # Apply the hotspot effects with a radial falloff
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    
                    if 0 <= ny < self.world_size and 0 <= nx < self.world_size:
                        # Calculate distance from center
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        if distance <= radius:
                            # Falloff effect based on distance
                            falloff = (1 - distance/radius)**1.5
                            
                            # Apply temperature effect
                            if hotspot['temp'] != 0:
                                temp_effect = hotspot['temp'] * falloff * strength
                                self.climate_data["temperature"].values[ny, nx] += temp_effect
                            
                            # Apply humidity effect
                            if hotspot['humidity'] != 0:
                                humid = self.climate_data["humidity"].values[ny, nx]
                                if hotspot['humidity'] > 0:
                                    # Increase humidity
                                    humid_effect = hotspot['humidity'] * falloff * strength
                                    self.climate_data["humidity"].values[ny, nx] = min(1.0, humid + humid_effect)
                                else:
                                    # Decrease humidity
                                    humid_effect = abs(hotspot['humidity']) * falloff * strength
                                    self.climate_data["humidity"].values[ny, nx] = max(0.0, humid * (1.0 - humid_effect))
                            
                            # Apply precipitation effect
                            if hotspot['precip'] != 0:
                                precip = self.climate_data["precipitation"].values[ny, nx]
                                if hotspot['precip'] > 0:
                                    # Increase precipitation
                                    precip_effect = hotspot['precip'] * falloff * strength
                                    self.climate_data["precipitation"].values[ny, nx] = precip * (1.0 + precip_effect)
                                else:
                                    # Decrease precipitation
                                    precip_effect = abs(hotspot['precip']) * falloff * strength
                                    self.climate_data["precipitation"].values[ny, nx] = precip * (1.0 - precip_effect)

    def _apply_elemental_climate_zones(self) -> None:
        """Apply elemental climate zones with distinct weather patterns."""
        # Determine strength of effect
        strength = self.fantasy_features['elemental_zones']
        
        # Number of elemental zones to create
        num_zones = int(np.sqrt(self.world_size) * strength * 0.3)
        
        # Define elemental zone types with comprehensive climate signatures
        elemental_zones = [
            {
                'name': 'fire', 
                'temp': 40, 
                'humidity': -0.8, 
                'precip': -0.9,
                'wind_strength': 1.5,
                'wind_pattern': 'spiral', 
                'radius': 30
            },
            {
                'name': 'ice', 
                'temp': -30, 
                'humidity': -0.5, 
                'precip': -0.2,
                'wind_strength': 0.7,
                'wind_pattern': 'outward', 
                'radius': 35
            },
            {
                'name': 'storm', 
                'temp': 0, 
                'humidity': 0.9, 
                'precip': 2.0,
                'wind_strength': 2.5,
                'wind_pattern': 'spiral', 
                'radius': 40
            },
            {
                'name': 'desert', 
                'temp': 15, 
                'humidity': -0.9, 
                'precip': -0.9,
                'wind_strength': 1.2,
                'wind_pattern': 'random', 
                'radius': 45
            },
            {
                'name': 'verdant', 
                'temp': 10, 
                'humidity': 0.8, 
                'precip': 1.5,
                'wind_strength': 0.8,
                'wind_pattern': 'gentle', 
                'radius': 38
            },
            {
                'name': 'mist', 
                'temp': -5, 
                'humidity': 1.0, 
                'precip': 0.5,
                'wind_strength': 0.4,
                'wind_pattern': 'stagnant', 
                'radius': 33
            }
        ]
        
        # Create elemental zones
        for _ in range(num_zones):
            # Select a random zone type
            zone = self.rng.choice(elemental_zones)
            
            # Random location (prefer land for most types)
            attempts = 0
            land_preference = zone['name'] != 'mist'  # Mist can be over water
            
            while attempts < 10:
                x = self.rng.randint(0, self.world_size)
                y = self.rng.randint(0, self.world_size)
                
                # Check land/water preference
                if (not land_preference or self.water_mask[y, x] == 0 or attempts > 5):
                    break
                attempts += 1
            
            # Scale radius by strength
            radius = int(zone['radius'] * strength * (0.7 + 0.6 * self.rng.random()))
            
            # Apply the elemental zone effects
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    
                    if 0 <= ny < self.world_size and 0 <= nx < self.world_size:
                        # Calculate distance from center
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        if distance <= radius:
                            # Falloff effect based on distance
                            falloff = (1 - distance/radius)**1.8  # Sharper falloff for distinct zones
                            
                            # Apply temperature effect
                            if zone['temp'] != 0:
                                temp_offset = zone['temp'] * falloff * strength
                                self.climate_data["temperature"].values[ny, nx] += temp_offset
                            
                            # Apply humidity effect
                            if zone['humidity'] != 0:
                                humid = self.climate_data["humidity"].values[ny, nx]
                                if zone['humidity'] > 0:
                                    humid_effect = zone['humidity'] * falloff * strength
                                    self.climate_data["humidity"].values[ny, nx] = min(1.0, humid + humid_effect)
                                else:
                                    humid_effect = abs(zone['humidity']) * falloff * strength
                                    self.climate_data["humidity"].values[ny, nx] = max(0.0, humid * (1.0 - humid_effect))
                            
                            # Apply precipitation effect
                            if zone['precip'] != 0:
                                precip = self.climate_data["precipitation"].values[ny, nx]
                                if zone['precip'] > 0:
                                    precip_effect = zone['precip'] * falloff * strength
                                    self.climate_data["precipitation"].values[ny, nx] = precip * (1.0 + precip_effect)
                                else:
                                    precip_effect = abs(zone['precip']) * falloff * strength
                                    self.climate_data["precipitation"].values[ny, nx] = max(0.1, precip * (1.0 - precip_effect))
                            
                            # Apply wind pattern effect
                            if zone['wind_pattern'] != 'none':
                                wind_u = self.climate_data["wind_u"].values[ny, nx]
                                wind_v = self.climate_data["wind_v"].values[ny, nx]
                                
                                # Calculate wind modification based on pattern
                                if zone['wind_pattern'] == 'spiral':
                                    # Create spiral wind pattern
                                    angle = np.arctan2(dy, dx) + distance / radius * np.pi
                                    new_u = np.cos(angle) * zone['wind_strength'] * 5
                                    new_v = np.sin(angle) * zone['wind_strength'] * 5
                                
                                elif zone['wind_pattern'] == 'outward':
                                    # Outward from center
                                    if distance > 0:
                                        new_u = dx / distance * zone['wind_strength'] * 5
                                        new_v = dy / distance * zone['wind_strength'] * 5
                                    else:
                                        new_u = new_v = 0
                                
                                elif zone['wind_pattern'] == 'random':
                                    # Random chaotic winds
                                    noise_val = self._get_simplex_noise(nx, ny, 0.0, scale=0.1)
                                    angle = noise_val * np.pi * 2
                                    new_u = np.cos(angle) * zone['wind_strength'] * 5
                                    new_v = np.sin(angle) * zone['wind_strength'] * 5
                                
                                elif zone['wind_pattern'] == 'gentle':
                                    # Gentle consistent breeze
                                    angle = np.pi / 4  # NE wind
                                    new_u = np.cos(angle) * zone['wind_strength'] * 3
                                    new_v = np.sin(angle) * zone['wind_strength'] * 3
                                
                                elif zone['wind_pattern'] == 'stagnant':
                                    # Very low wind speeds
                                    new_u = wind_u * 0.2
                                    new_v = wind_v * 0.2
                                
                                else:  # Default
                                    new_u = wind_u
                                    new_v = wind_v
                                
                                # Apply wind modification with falloff
                                self.climate_data["wind_u"].values[ny, nx] = wind_u * (1 - falloff) + new_u * falloff
                                self.climate_data["wind_v"].values[ny, nx] = wind_v * (1 - falloff) + new_v * falloff

    def _apply_aether_currents(self) -> None:
        """Apply supernatural wind patterns (aether currents) that affect climate and weather."""
        # Determine strength of effect
        strength = self.fantasy_features['aether_currents']
    
        # Number of aether currents
        num_currents = int(np.sqrt(self.world_size) * strength * 0.2)
    
        # Types of aether currents
        current_types = [
            {
                'name': 'life_stream',
                'temp_mod': 5,
                'humidity_mod': 0.4,
                'fertility_boost': True,
                'width': 8,
                'flow_pattern': 'meander'
            },
            {
                'name': 'astral_tide',
                'temp_mod': -10,
                'humidity_mod': -0.2,
                'psychic_resonance': True,
                'width': 12,
                'flow_pattern': 'sine'
            },
            {
                'name': 'phoenix_wind',
                'temp_mod': 15,
                'humidity_mod': -0.3,
                'rejuvenation': True,
                'width': 6,
                'flow_pattern': 'spiral'
            },
            {
                'name': 'void_current',
                'temp_mod': -20,
                'humidity_mod': -0.5,
                'arcane_dampening': True,
                'width': 10,
                'flow_pattern': 'linear'
            },
            {
                'name': 'fey_breeze',
                'temp_mod': 0,
                'humidity_mod': 0.7,
                'enchantment': True,
                'width': 7,
                'flow_pattern': 'meander'
            }
        ]
    
        # Generate each aether current
        for _ in range(num_currents):
            # Select a random current type
            current = self.rng.choice(current_types)
        
            # Random starting point
            x = self.rng.randint(0, self.world_size)
            y = self.rng.randint(0, self.world_size)
        
            # Current width adjusted by strength
            width = int(current['width'] * strength)
        
            # Random length
            length = int(self.rng.uniform(50, 200) * strength)
        
            # Random starting direction
            angle = self.rng.uniform(0, 2 * np.pi)
        
            # Generate path based on flow pattern
            points = []
            current_x, current_y = x, y
        
            if current['flow_pattern'] == 'meander':
                # Meandering river-like path
                curvature = 0
                for i in range(length):
                    if 0 <= current_x < self.world_size and 0 <= current_y < self.world_size:
                        points.append((int(current_x), int(current_y)))  # Convert to int to ensure exact matches later
                
                    # Random change in curvature
                    curvature += self.rng.uniform(-0.1, 0.1)
                    curvature = np.clip(curvature, -0.2, 0.2)
                
                    # Update angle with curvature
                    angle += curvature
                
                    # Move in the new direction
                    current_x += np.cos(angle) * 2
                    current_y += np.sin(angle) * 2
        
            elif current['flow_pattern'] == 'sine':
                # Sinusoidal wave pattern
                base_angle = angle
                amplitude = self.rng.uniform(10, 30)
                wavelength = self.rng.uniform(20, 60)
                for i in range(length):
                    if 0 <= current_x < self.world_size and 0 <= current_y < self.world_size:
                        points.append((int(current_x), int(current_y)))  # Convert to int
                
                    # Move along the base direction
                    current_x += np.cos(base_angle) * 2
                
                    # Add sine wave perpendicular to base direction
                    perp_angle = base_angle + np.pi/2
                    sine_offset = amplitude * np.sin(2 * np.pi * i / wavelength)
                    current_y += np.sin(base_angle) * 2 + np.sin(perp_angle) * sine_offset * 0.1
        
            elif current['flow_pattern'] == 'spiral':
                # Spiral pattern
                spiral_radius = 5
                spiral_growth = 0.2
                for i in range(length):
                    if 0 <= current_x < self.world_size and 0 <= current_y < self.world_size:
                        points.append((int(current_x), int(current_y)))  # Convert to int
                
                    # Update angle to create spiral
                    angle += 0.1
                
                    # Increase radius for outward spiral
                    spiral_radius += spiral_growth
                
                    # Move in spiral
                    current_x += np.cos(angle) * spiral_radius * 0.1
                    current_y += np.sin(angle) * spiral_radius * 0.1
        
            else:  # 'linear'
                # Straight or slightly curved path
                for i in range(length):
                    if 0 <= current_x < self.world_size and 0 <= current_y < self.world_size:
                        points.append((int(current_x), int(current_y)))  # Convert to int
                
                    # Small random angle adjustments
                    angle += self.rng.uniform(-0.05, 0.05)
                
                    # Move forward
                    current_x += np.cos(angle) * 2
                    current_y += np.sin(angle) * 2
        
            # Make sure we have points
            if not points:
                continue
                
            # Apply effects along the aether current path
            for i, (px, py) in enumerate(points):
                # Apply in a radius around the path
                for dy in range(-width, width + 1):
                    for dx in range(-width, width + 1):
                        nx, ny = px + dx, py + dy
                    
                        if 0 <= ny < self.world_size and 0 <= nx < self.world_size:
                            # Calculate distance from current's center line
                            distance = np.sqrt(dx**2 + dy**2)
                        
                            if distance <= width:
                                # Falloff effect based on distance
                                falloff = (1 - distance/width)**2
                            
                                # Apply temperature effect
                                if current['temp_mod'] != 0:
                                    temp_effect = current['temp_mod'] * falloff * strength
                                    self.climate_data["temperature"].values[ny, nx] += temp_effect
                            
                                # Apply humidity effect
                                if current['humidity_mod'] != 0:
                                    humid = self.climate_data["humidity"].values[ny, nx]
                                    humid_effect = current['humidity_mod'] * falloff * strength
                                    new_humid = np.clip(humid + humid_effect, 0, 1)
                                    self.climate_data["humidity"].values[ny, nx] = new_humid
                            
                                # Apply wind effect - create flow along the current
                                # Find the next point on the path to determine direction
                                if i < len(points) - 1:  # If not the last point
                                    next_point = points[i + 1]
                                    next_px, next_py = next_point
                                
                                    # Direction from current to next point
                                    dx_path = next_px - px
                                    dy_path = next_py - py
                                
                                    # Normalize direction
                                    path_length = np.sqrt(dx_path**2 + dy_path**2)
                                    if path_length > 0:
                                        dx_norm = dx_path / path_length
                                        dy_norm = dy_path / path_length
                                    
                                        # Create strong wind along the current
                                        flow_strength = 10 * strength * falloff
                                        self.climate_data["wind_u"].values[ny, nx] += dx_norm * flow_strength
                                        self.climate_data["wind_v"].values[ny, nx] += dy_norm * flow_strength

    def _apply_reality_flux(self) -> None:
        """Apply reality flux zones - areas of unstable and unpredictable weather patterns."""
        # Determine strength of effect
        strength = self.fantasy_features['reality_flux']
        
        # Number of flux zones
        num_zones = int(np.sqrt(self.world_size) * strength * 0.15)
        
        # Different types of reality flux
        flux_types = [
            {'name': 'chaotic_flux', 'temp_range': 40, 'humidity_range': 1.0, 'wind_chaos': 2.0, 'radius': 25},
            {'name': 'temporal_anomaly', 'temp_range': 30, 'humidity_range': 0.8, 'seasonal_shift': True, 'radius': 20},
            {'name': 'planar_bleed', 'temp_range': 50, 'humidity_range': 1.0, 'alien_weather': True, 'radius': 15},
            {'name': 'arcane_storm', 'temp_range': 25, 'humidity_range': 0.7, 'mana_surge': True, 'radius': 30}
        ]
        
        # Generate each flux zone
        for _ in range(num_zones):
            # Select flux type
            flux = self.rng.choice(flux_types)
            
            # Random location
            x = self.rng.randint(0, self.world_size)
            y = self.rng.randint(0, self.world_size)
            
            # Radius adjusted by strength
            radius = int(flux['radius'] * strength)
            
            # Apply flux effects
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    
                    if 0 <= ny < self.world_size and 0 <= nx < self.world_size:
                        # Calculate distance from center
                        distance = np.sqrt(dx**2 + dy**2)
                        
                        if distance <= radius:
                            # Falloff effect based on distance
                            falloff = (1 - distance/radius)**1.5
                            
                            # Generate noise pattern for this flux zone
                            # Use OpenSimplex noise for spatially coherent randomness
                            noise_val = self._get_simplex_noise(
                                nx, ny, self.seed * 0.1,
                                scale=0.1,
                                octaves=4,
                                persistence=0.5,
                                lacunarity=2.0
                            )
                            
                            # Apply temperature chaos
                            temp_chaos = noise_val * flux['temp_range'] * falloff * strength
                            self.climate_data["temperature"].values[ny, nx] += temp_chaos
                            
                            # Apply humidity chaos
                            humid = self.climate_data["humidity"].values[ny, nx]
                            humid_chaos = noise_val * flux['humidity_range'] * falloff * strength
                            self.climate_data["humidity"].values[ny, nx] = np.clip(humid + humid_chaos, 0, 1)
                            
                            # Apply wind chaos if specified
                            if 'wind_chaos' in flux:
                                angle = self.rng.uniform(0, 2 * np.pi)
                                wind_strength = flux['wind_chaos'] * falloff * strength * 5
                                
                                # Random wind vector
                                wind_u = np.cos(angle) * wind_strength
                                wind_v = np.sin(angle) * wind_strength
                                
                                # Add to existing winds
                                self.climate_data["wind_u"].values[ny, nx] += wind_u
                                self.climate_data["wind_v"].values[ny, nx] += wind_v
                            
                            # Apply special effects based on flux type
                            if 'seasonal_shift' in flux and flux['seasonal_shift']:
                                # Create pockets of different seasons
                                if noise_val > 0.3:  # Summer-like
                                    self.climate_data["temperature"].values[ny, nx] += 10 * falloff * strength
                                elif noise_val < -0.3:  # Winter-like
                                    self.climate_data["temperature"].values[ny, nx] -= 10 * falloff * strength
                            
                            if 'alien_weather' in flux and flux['alien_weather']:
                                # Completely bizarre weather patterns
                                if noise_val > 0.7:
                                    # Extremely hot and dry
                                    self.climate_data["temperature"].values[ny, nx] = 50 * falloff + (1 - falloff) * self.climate_data["temperature"].values[ny, nx]
                                    self.climate_data["humidity"].values[ny, nx] = 0.1 * falloff + (1 - falloff) * self.climate_data["humidity"].values[ny, nx]
                                elif noise_val < -0.7:
                                    # Freezing with strange precipitation
                                    self.climate_data["temperature"].values[ny, nx] = -40 * falloff + (1 - falloff) * self.climate_data["temperature"].values[ny, nx]
                                    self.climate_data["precipitation"].values[ny, nx] *= (3 * falloff + (1 - falloff))

    def update_climate(self, day_of_year: int, hour_of_day: float) -> None:
        """Update climate based on planetary conditions, seasonal changes, and time of day.
        
        Args:
            day_of_year: Current day of the year
            hour_of_day: Current hour of the day
        """
        # Apply seasonal temperature variations
        self._update_seasonal_temperature(day_of_year)
        
        # Apply diurnal (day/night) temperature cycle
        self._update_diurnal_temperature(hour_of_day)
        
        # Update precipitation based on seasonal patterns
        self._update_seasonal_precipitation(day_of_year)
        
        # Update wind patterns based on seasonal changes
        self._update_seasonal_winds(day_of_year)
        
        print(f"Updated climate for day {day_of_year}, hour {hour_of_day:.1f}")

    def _update_seasonal_temperature(self, day_of_year: int) -> None:
        """Update temperature based on seasonal changes.
        
        Args:
            day_of_year: Current day of the year
        """
        # Get the base temperature map without seasonal variations
        base_temps = self.climate_data["temperature"].values.copy()
        
        for y in range(self.world_size):
            for x in range(self.world_size):
                latitude = self.lat_grid[y, x]
                
                # Calculate seasonal factor for this latitude (-1 to 1)
                seasonal_factor = self.planetary.get_seasonal_factor(latitude)
                
                # Scale by seasonal variation strength
                seasonal_factor *= self.seasonal_variation_strength
                
                # Different seasonal temperature ranges based on climate zone
                if abs(latitude) < 15:  # Tropical
                    seasonal_temp_range = 5.0  # Small seasonal variation
                elif abs(latitude) < 35:  # Subtropical
                    seasonal_temp_range = 15.0  # Moderate variation
                elif abs(latitude) < 65:  # Temperate
                    seasonal_temp_range = 25.0  # Large variation
                else:  # Polar
                    seasonal_temp_range = 30.0  # Very large variation
                
                # Continental areas have greater seasonal variation
                land_mask = 1 - self.water_mask
                if land_mask[y, x] > 0:
                    # Calculate distance from coast
                    distance_to_water = ndimage.distance_transform_edt(land_mask)
                    continental_factor = min(1.0, distance_to_water[y, x] / 50.0)
                    seasonal_temp_range *= (1.0 + continental_factor)
                else:
                    # Oceans have reduced seasonal variation
                    seasonal_temp_range *= 0.5
                
                # Apply the seasonal temperature offset
                temp_offset = seasonal_factor * seasonal_temp_range
                self.climate_data["temperature"].values[y, x] = base_temps[y, x] + temp_offset

    def _update_diurnal_temperature(self, hour_of_day: float) -> None:
        """Update temperature based on time of day.
        
        Args:
            hour_of_day: Current hour of the day
        """
        # Use the day/night mask to determine solar heating
        day_mask = self.planetary.day_night_mask
        
        # Calculate time of day factor (0 at night, 1 at solar noon)
        day_length = self.planetary.day_length_hours
        
        # For simple diurnal variation, use a sine curve peaking at noon
        time_factor = np.sin(np.pi * (hour_of_day / day_length))
        time_factor = max(0, time_factor)  # Only positive during daytime
        
        # Apply diurnal temperature cycle
        for y in range(self.world_size):
            for x in range(self.world_size):
                # Skip water (water has much smaller diurnal variation)
                if self.water_mask[y, x] > 0:
                    continue
                
                # Get current temperature
                current_temp = self.climate_data["temperature"].values[y, x]
                
                # Diurnal range depends on:
                # 1. Humidity (drier = larger swing)
                # 2. Latitude (higher = smaller swing due to lower sun angle)
                # 3. Terrain (higher elevation = larger swing due to thinner air)
                
                humidity = self.climate_data["humidity"].values[y, x]
                latitude = abs(self.lat_grid[y, x])
                elevation = self.climate_data["elevation"].values[y, x] / 8000.0  # Normalize to 0-1
                
                # Base diurnal range
                diurnal_range = 15.0
                
                # Adjust for humidity (drier = larger swing)
                diurnal_range *= (1.0 - 0.6 * humidity)
                
                # Adjust for latitude (higher = smaller swing)
                diurnal_range *= (1.0 - 0.5 * (latitude / 90.0))
                
                # Adjust for elevation (higher = larger swing)
                diurnal_range *= (1.0 + 0.5 * elevation)
                
                # Calculate temperature offset based on time of day
                if day_mask[y, x]:  # Daytime
                    temp_offset = time_factor * diurnal_range
                else:  # Nighttime
                    # Night temperatures fall gradually
                    night_progress = hour_of_day / day_length
                    if night_progress > 0.5:  # After midnight, approaching dawn
                        night_factor = 1.0 - (night_progress - 0.5) * 2.0
                    else:  # Evening to midnight
                        night_factor = night_progress * 2.0
                    
                    temp_offset = -diurnal_range * (0.5 + 0.5 * night_factor)
                
                # Apply the offset
                self.climate_data["temperature"].values[y, x] = current_temp + temp_offset

    def _update_seasonal_precipitation(self, day_of_year: int) -> None:
        """Update precipitation patterns based on seasonal changes.
        
        Args:
            day_of_year: Current day of the year
        """
        # Calculate position in the year (0 to 1)
        year_position = day_of_year / self.planetary.year_length_days
        
        # Update precipitation based on latitude and season
        for y in range(self.world_size):
            for x in range(self.world_size):
                if self.water_mask[y, x] > 0:
                    continue  # Skip water bodies
                
                latitude = self.lat_grid[y, x]
                
                # Calculate seasonal factor (-1 to 1)
                seasonal_factor = self.planetary.get_seasonal_factor(latitude)
                
                # Different seasonal precipitation patterns by latitude zone
                if abs(latitude) < 10:  # Equatorial
                    # ITCZ shifts north and south with seasons, creating two wet seasons
                    # at most equatorial locations
                    season_mod = np.sin(year_position * 4 * np.pi) * 0.3
                    
                elif 10 <= abs(latitude) < 30:  # Tropical/Subtropical (monsoon regions)
                    # Single strong wet season during summer
                    season_mod = max(0, seasonal_factor) * 0.8
                    
                elif 30 <= abs(latitude) < 60:  # Temperate
                    # More complex patterns, generally wetter in winter in coastal regions
                    # but can vary significantly by location
                    if self._is_coastal(x, y):
                        # Coastal areas often have wetter winters
                        season_mod = -seasonal_factor * 0.4
                    else:
                        # Continental areas often have summer thunderstorms
                        season_mod = seasonal_factor * 0.3
                    
                else:  # Polar
                    # Generally drier in winter when it's colder
                    season_mod = seasonal_factor * 0.5
                
                # Apply the seasonal modifier
                base_precip = self.climate_data["precipitation"].values[y, x]
                season_effect = 1.0 + season_mod * self.seasonal_variation_strength
                self.climate_data["precipitation"].values[y, x] = base_precip * season_effect

    def _is_coastal(self, x: int, y: int, distance: int = 10) -> bool:
        """Determine if a location is coastal (near water).
        
        Args:
            x: X coordinate
            y: Y coordinate
            distance: Maximum distance to consider coastal
            
        Returns:
            True if the location is near water, False otherwise
        """
        # Get region around point
        y_min = max(0, y - distance)
        y_max = min(self.world_size, y + distance)
        x_min = max(0, x - distance)
        x_max = min(self.world_size, x + distance)
        
        # Check if water exists in this region
        region = self.water_mask[y_min:y_max, x_min:x_max]
        
        # It's coastal if there's both land and water in the region
        has_water = np.any(region > 0)
        has_land = np.any(region == 0)
        
        return has_water and has_land

    def _update_seasonal_winds(self, day_of_year: int) -> None:
        """Update wind patterns based on seasonal changes.
        
        Args:
            day_of_year: Current day of the year
        """
        # Seasonal shifts in global wind patterns
        year_position = day_of_year / self.planetary.year_length_days
        
        # ITCZ (Intertropical Convergence Zone) shifts with the thermal equator
        # Calculate ITCZ position (shifts north in northern summer, south in southern summer)
        itcz_shift = np.sin(2 * np.pi * year_position) * 15  # ±15° from equator
        
        # Apply shifts to the wind patterns
        for y in range(self.world_size):
            for x in range(self.world_size):
                # Current wind components
                u_wind = self.climate_data["wind_u"].values[y, x]
                v_wind = self.climate_data["wind_v"].values[y, x]
                
                # Get latitude and adjusted "thermal latitude"
                latitude = self.lat_grid[y, x]
                thermal_latitude = latitude - itcz_shift
                
                # Calculate effect based on difference between actual and thermal latitude
                lat_diff = latitude - thermal_latitude
                
                # Wind adjustments depend on latitude zone
                if abs(latitude) < 30:  # Tropical - stronger seasonal effect
                    # Trade winds and monsoon circulation shift with ITCZ
                    shift_factor = 0.05 * lat_diff * self.seasonal_variation_strength
                    
                    # Primarily affects meridional (north-south) wind component
                    self.climate_data["wind_v"].values[y, x] = v_wind + shift_factor * 3.0
                    
                elif 30 <= abs(latitude) < 60:  # Mid-latitudes
                    # Jet stream and storm tracks shift with seasons
                    shift_factor = 0.03 * lat_diff * self.seasonal_variation_strength
                    
                    # Affects both components but mainly zonal (east-west)
                    self.climate_data["wind_u"].values[y, x] = u_wind + shift_factor * 2.0
                    self.climate_data["wind_v"].values[y, x] = v_wind + shift_factor * 1.0
                    
                else:  # Polar regions
                    # Polar vortex strengthens in winter, weakens in summer
                    seasonal_factor = self.planetary.get_seasonal_factor(latitude)
                    
                    # Negative factor = winter = stronger polar winds
                    season_strength = -seasonal_factor * self.seasonal_variation_strength
                    
                    # Strengthen or weaken existing winds
                    strength_mod = 1.0 + 0.3 * season_strength
                    self.climate_data["wind_u"].values[y, x] = u_wind * strength_mod
                    self.climate_data["wind_v"].values[y, x] = v_wind * strength_mod

    def get_climate_at_position(self, x: int, y: int) -> Dict[str, float]:
        """Get climate data at a specific position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Dictionary of climate variables at the position
        """
        if not (0 <= x < self.world_size and 0 <= y < self.world_size):
            raise ValueError(f"Position ({x}, {y}) is out of bounds")
        
        return {
            "temperature": float(self.climate_data["temperature"].values[y, x]),
            "precipitation": float(self.climate_data["precipitation"].values[y, x]),
            "humidity": float(self.climate_data["humidity"].values[y, x]),
            "wind_speed": float(np.sqrt(self.climate_data["wind_u"].values[y, x]**2 + 
                                      self.climate_data["wind_v"].values[y, x]**2)),
            "wind_direction": float(np.arctan2(self.climate_data["wind_v"].values[y, x],
                                             self.climate_data["wind_u"].values[y, x]) * 180 / np.pi),
            "pressure": float(self.climate_data["pressure"].values[y, x]),
            "elevation": float(self.climate_data["elevation"].values[y, x]),
            "is_water": bool(self.water_mask[y, x])
        }

    def determine_climate_classification(self, x: int, y: int) -> str:
        """Determine the Köppen climate classification at a position.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            Köppen climate classification code
        """
        # Check cache first
        cache_key = (x, y)
        if cache_key in self.biome_cache:
            return self.biome_cache[cache_key]
        
        # Get climate data
        climate = self.get_climate_at_position(x, y)
        
        # If water, return ocean
        if climate["is_water"]:
            self.biome_cache[cache_key] = "Ocean"
            return "Ocean"
        
        # Extract key variables
        temp = climate["temperature"]
        precip = climate["precipitation"]
        
        # Calculate annual precipitation
        annual_precip = precip * 365  # mm/year
        
        # Calculate seasonal temperature variation
        lat = self.lat_grid[y, x]
        seasonal_variation = abs(self.planetary.get_seasonal_factor(lat)) * 20  # Approximate temperature range
        
        # Calculate warmest and coldest monthly temperatures (estimated)
        warmest_month = temp + seasonal_variation / 2
        coldest_month = temp - seasonal_variation / 2
        
        # Simplified Köppen climate classification
        # A - Tropical
        # B - Arid
        # C - Temperate
        # D - Continental
        # E - Polar
        
        # Define temperature thresholds
        if coldest_month >= 18:  # Tropical (A)
            if annual_precip >= 60 * 25:  # Af (rainforest) - at least 60mm in driest month
                climate_type = "Af"
            elif annual_precip >= 25 * (100 - min(60, annual_precip / 25)):  # Am (monsoon)
                climate_type = "Am"
            else:  # Aw (savanna)
                climate_type = "Aw"
                
        elif annual_precip < 10 * (temp + seasonal_variation/2):  # Arid (B)
            # Threshold depends on temperature and seasonality
            if annual_precip > 5 * (temp + seasonal_variation/2):  # Semi-arid
                if temp > 0:
                    climate_type = "BSh"  # Hot steppe
                else:
                    climate_type = "BSk"  # Cold steppe
            else:  # Desert
                if temp > 0:
                    climate_type = "BWh"  # Hot desert
                else:
                    climate_type = "BWk"  # Cold desert
                    
        elif coldest_month > 0 and warmest_month > 10:  # Temperate (C)
            if annual_precip > 40 * 12:  # Sufficient precipitation
                # Check seasonal distribution
                if precip * 6 > annual_precip * 0.7:  # Summer-concentrated
                    climate_type = "Cwa" if warmest_month > 22 else "Cwb"
                elif precip * 6 < annual_precip * 0.3:  # Winter-concentrated
                    climate_type = "Csa" if warmest_month > 22 else "Csb"
                else:  # Evenly distributed
                    climate_type = "Cfa" if warmest_month > 22 else "Cfb"
            else:
                climate_type = "Cs"  # Mediterranean (dry summer)
                
        elif warmest_month > 10:  # Continental (D)
            if annual_precip > 40 * 12:  # Sufficient precipitation
                # Check seasonal distribution
                if precip * 6 > annual_precip * 0.7:  # Summer-concentrated
                    climate_type = "Dwa" if warmest_month > 22 else "Dwb"
                elif precip * 6 < annual_precip * 0.3:  # Winter-concentrated
                    climate_type = "Dsa" if warmest_month > 22 else "Dsb"
                else:  # Evenly distributed
                    climate_type = "Dfa" if warmest_month > 22 else "Dfb"
            else:
                climate_type = "Df"  # Humid continental
                
        else:  # Polar (E)
            if warmest_month > 0:
                climate_type = "ET"  # Tundra
            else:
                climate_type = "EF"  # Ice cap
        
        # Cache and return
        self.biome_cache[cache_key] = climate_type
        return climate_type

    def visualize_temperature(self, ax=None, title="Temperature Map"):
        """Visualize the temperature map.
        
        Args:
            ax: Optional matplotlib axis
            title: Plot title
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
        
        # Create temperature colormap
        # Blue for cold, red for hot
        cmap = plt.cm.RdBu_r
        
        # Plot temperature map
        im = ax.imshow(
            self.climate_data["temperature"].values,
            cmap=cmap,
            extent=[-180, 180, -90, 90],
            vmin=-30,
            vmax=40,
            origin="upper"
        )
        
        # Add coastlines (water boundary)
        contour = ax.contour(
            self.water_mask,
            colors="black",
            levels=[0.5],
            extent=[-180, 180, -90, 90],
            linewidths=0.5,
            alpha=0.7,
            origin="upper"
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Temperature (°C)")
        
        # Add grid
        ax.grid(linestyle=":", color="gray", alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        
        return ax

    def visualize_precipitation(self, ax=None, title="Precipitation Map"):
        """Visualize the precipitation map.
        
        Args:
            ax: Optional matplotlib axis
            title: Plot title
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
        
        # Create precipitation colormap
        # Light to dark blue for precipitation
        cmap = plt.cm.YlGnBu
        
        # Plot precipitation map
        im = ax.imshow(
            self.climate_data["precipitation"].values,
            cmap=cmap,
            extent=[-180, 180, -90, 90],
            vmin=0,
            vmax=10,
            origin="upper"
        )
        
        # Add coastlines (water boundary)
        contour = ax.contour(
            self.water_mask,
            colors="black",
            levels=[0.5],
            extent=[-180, 180, -90, 90],
            linewidths=0.5,
            alpha=0.7,
            origin="upper"
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Precipitation (mm/day)")
        
        # Add grid
        ax.grid(linestyle=":", color="gray", alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        
        return ax

    def visualize_humidity(self, ax=None, title="Humidity Map"):
        """Visualize the humidity map.
        
        Args:
            ax: Optional matplotlib axis
            title: Plot title
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
        
        # Create humidity colormap
        cmap = plt.cm.YlGnBu
        
        # Plot humidity map
        im = ax.imshow(
            self.climate_data["humidity"].values,
            cmap=cmap,
            extent=[-180, 180, -90, 90],
            vmin=0,
            vmax=1,
            origin="upper"
        )
        
        # Add coastlines (water boundary)
        contour = ax.contour(
            self.water_mask,
            colors="black",
            levels=[0.5],
            extent=[-180, 180, -90, 90],
            linewidths=0.5,
            alpha=0.7,
            origin="upper"
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Relative Humidity")
        
        # Add grid
        ax.grid(linestyle=":", color="gray", alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        
        return ax
    
    def visualize_wind(self, ax=None, title="Wind Patterns", density=20):
        """Visualize wind patterns using quiver plot.
        
        Args:
            ax: Optional matplotlib axis
            title: Plot title
            density: Density of wind arrows
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
        
        # Subsample wind field for clearer visualization
        step = max(1, self.world_size // density)
        
        # Create meshgrid for quiver plot
        y, x = np.mgrid[0:self.world_size:step, 0:self.world_size:step]
        u = self.climate_data["wind_u"].values[::step, ::step]
        v = self.climate_data["wind_v"].values[::step, ::step]
        
        # Calculate wind speed for coloring
        speed = np.sqrt(u**2 + v**2)
        
        # Convert grid coordinates to lat/lon for plotting
        lons = np.linspace(-180, 180, len(x[0]))
        lats = np.linspace(-90, 90, len(y[:,0]))
        
        # Plot wind vectors
        quiv = ax.quiver(
            lons, lats, u, v,
            speed,
            cmap=plt.cm.viridis,
            scale=50,
            width=0.002,
            headwidth=4,
            headlength=5,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(quiv, ax=ax)
        cbar.set_label("Wind Speed (m/s)")
        
        # Add coastlines (water boundary)
        contour = ax.contour(
            self.water_mask,
            colors="black",
            levels=[0.5],
            extent=[-180, 180, -90, 90],
            linewidths=0.5,
            alpha=0.7,
            origin="upper"
        )
        
        # Add grid
        ax.grid(linestyle=":", color="gray", alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        
        return ax

    def visualize_pressure(self, ax=None, title="Atmospheric Pressure"):
        """Visualize the atmospheric pressure map.
        
        Args:
            ax: Optional matplotlib axis
            title: Plot title
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
        
        # Create pressure colormap
        # Higher values (high pressure) are shown in warmer colors
        cmap = plt.cm.RdYlBu_r
        
        # Plot pressure map
        im = ax.imshow(
            self.climate_data["pressure"].values,
            cmap=cmap,
            extent=[-180, 180, -90, 90],
            vmin=980,
            vmax=1040,
            origin="upper"
        )
        
        # Add coastlines (water boundary)
        contour = ax.contour(
            self.water_mask,
            colors="black",
            levels=[0.5],
            extent=[-180, 180, -90, 90],
            linewidths=0.5,
            alpha=0.7,
            origin="upper"
        )
        
        # Add pressure contours
        levels = np.arange(980, 1041, 4)
        pressure_contour = ax.contour(
            self.climate_data["pressure"].values,
            levels=levels,
            colors='black',
            alpha=0.5,
            extent=[-180, 180, -90, 90],
            origin="upper"
        )
        plt.clabel(pressure_contour, inline=1, fontsize=8, fmt='%1.0f')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Pressure (hPa)")
        
        # Add grid
        ax.grid(linestyle=":", color="gray", alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
        
        return ax

    def visualize_fantasy_features(self, ax=None, title="Fantasy Climate Features"):
        """Visualize fantasy climate features.
        
        Args:
            ax: Optional matplotlib axis
            title: Plot title
            
        Returns:
            Matplotlib axis
        """
        if not any(v > 0 for v in self.fantasy_features.values()):
            print("No fantasy features enabled")
            return None
        
        if ax is None:
            plt.figure(figsize=(10, 8))
            ax = plt.gca()
        
        # Calculate baseline temperature from latitude and elevation only
        baseline_temp = np.zeros((self.world_size, self.world_size))
        for y in range(self.world_size):
            for x in range(self.world_size):
                latitude = self.lat_grid[y, x]
                elevation = self.climate_data["elevation"].values[y, x]
                
                # Simple baseline model
                equator_temp = 27.0
                pole_temp = -44.0
                temp_factor = np.cos(np.abs(np.radians(latitude)))
                temp = self.base_temperature + equator_temp * temp_factor + pole_temp * (1 - temp_factor)
                
                # Apply elevation
                lapse_rate = 6.5 / 1000.0  # °C per meter
                temp -= elevation * lapse_rate
                
                baseline_temp[y, x] = temp
        
        # Calculate temperature anomaly (deviation from baseline)
        temp_anomaly = self.climate_data["temperature"].values - baseline_temp
        
        # Create a diverging colormap for anomalies
        cmap = plt.cm.RdBu_r  # Blue for cold anomalies, red for warm
        
        # Plot temperature anomalies
        im = ax.imshow(
            temp_anomaly,
            cmap=cmap,
            extent=[-180, 180, -90, 90],
            vmin=-20,
            vmax=20,
            origin="upper"
        )
        
        # Add coastlines
        contour = ax.contour(
            self.water_mask,
            colors="black",
            levels=[0.5],
            extent=[-180, 180, -90, 90],
            linewidths=0.5,
            alpha=0.7,
            origin="upper"
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Temperature Anomaly (°C)")
        
        # Add grid
        ax.grid(linestyle=":", color="gray", alpha=0.5)
        
        # Add labels and title
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        
        # Add fantasy feature info to title
        features_str = ", ".join([f"{k}: {v:.2f}" for k, v in self.fantasy_features.items() if v > 0])
        ax.set_title(f"{title}\n({features_str})")
        
        return ax

    def visualize_climate_classification(self, ax=None, title="Köppen Climate Classification", resolution=4):
        """Visualize Köppen climate classification.
        
        Args:
            ax: Optional matplotlib axis
            title: Plot title
            resolution: Sampling resolution (higher = faster but less detailed)
            
        Returns:
            Matplotlib axis
        """
        if ax is None:
            plt.figure(figsize=(12, 10))
            ax = plt.gca()
        
        # Define colors for major climate types
        climate_colors = {
            'A': [0.0, 0.8, 0.2],   # Tropical (green)
            'B': [0.9, 0.7, 0.2],   # Arid (yellow/brown)
            'C': [0.0, 0.6, 0.8],   # Temperate (blue-green)
            'D': [0.2, 0.4, 0.8],   # Continental (blue)
            'E': [0.8, 0.8, 0.9],   # Polar (light blue/white)
            'Ocean': [0.1, 0.1, 0.6]  # Ocean (dark blue)
        }
        
        # Create an image for the climate map
        climate_map = np.zeros((self.world_size, self.world_size, 3))
        
        # Sample the climate types at the specified resolution
        step = max(1, resolution)
        
        print(f"Generating climate classification map (this may take a moment)...")
        for y in range(0, self.world_size, step):
            for x in range(0, self.world_size, step):
                climate_type = self.determine_climate_classification(x, y)
                
                # Extract the main climate letter
                if climate_type == 'Ocean':
                    color = climate_colors['Ocean']
                else:
                    color = climate_colors.get(climate_type[0], [0.5, 0.5, 0.5])
                
                # Fill in a block
                y_end = min(y + step, self.world_size)
                x_end = min(x + step, self.world_size)
                climate_map[y:y_end, x:x_end] = color
        
        # Display the climate classification map
        ax.imshow(
            climate_map, 
            extent=[-180, 180, -90, 90], 
            origin='upper'
        )
        
        # Add coastlines
        ax.contour(
            self.water_mask,
            colors="black",
            levels=[0.5],
            extent=[-180, 180, -90, 90],
            linewidths=0.5,
            origin="upper"
        )
        
        # Add a legend
        for key, color in climate_colors.items():
            if key == 'A':
                label = 'Tropical'
            elif key == 'B':
                label = 'Arid'
            elif key == 'C':
                label = 'Temperate'
            elif key == 'D':
                label = 'Continental'
            elif key == 'E':
                label = 'Polar'
            else:
                label = key
                
            ax.plot([], [], color=color, label=label, linewidth=10)
        
        ax.legend(loc='lower right')
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(alpha=0.3)
        
        return ax

    def visualize_all(self, figsize=(15, 20)):
        """Visualize all climate variables in a single figure.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # Temperature plot
        self.visualize_temperature(ax=axes[0, 0])
        
        # Precipitation plot
        self.visualize_precipitation(ax=axes[0, 1])
        
        # Humidity plot
        self.visualize_humidity(ax=axes[1, 0])
        
        # Wind plot
        self.visualize_wind(ax=axes[1, 1])
        
        # Pressure plot
        self.visualize_pressure(ax=axes[2, 0])
        
        # Fantasy features or climate classification
        if any(v > 0 for v in self.fantasy_features.values()):
            self.visualize_fantasy_features(ax=axes[2, 1])
        else:
            self.visualize_climate_classification(ax=axes[2, 1], resolution=8)
        
        plt.tight_layout()
        return fig

    def save_climate_data(self, filename: str = "climate_data.nc") -> None:
        """Save the climate data to a file.
        
        Args:
            filename: Output filename (will use NetCDF format)
        """
        # Save to NetCDF format using xarray
        self.climate_data.to_netcdf(filename)
        print(f"Climate data saved to {filename}")

    def load_climate_data(self, filename: str) -> None:
        """Load climate data from a file.
        
        Args:
            filename: Input filename (NetCDF format)
        """
        # Load from NetCDF format using xarray
        self.climate_data = xr.open_dataset(filename)
        print(f"Climate data loaded from {filename}")