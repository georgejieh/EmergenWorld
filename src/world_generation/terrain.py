"""Terrain generation module for EmergenWorld.

This module handles the creation of realistic terrain with deep valleys,
better land distribution patterns, and enhanced geographic features for
a believable fantasy world.
"""

import numpy as np
from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from scipy.ndimage import distance_transform_edt, gaussian_filter


class TerrainGenerator:
    """Generates terrain with diverse features for the EmergenWorld simulation.
    
    Features deep valleys, realistic mountain ranges, improved land distribution
    across latitudes, and better continental connectivity.
    """

    def __init__(self, size: int = 1024, octaves: int = 6,
                persistence: float = 0.5, lacunarity: float = 2.0,
                seed: Optional[int] = None, earth_scale: float = 0.0083):
        """Initialize the TerrainGenerator with configurable parameters.

        Args:
            size: Grid size for the heightmap (size x size)
            octaves: Number of octaves for noise generation
            persistence: Persistence value for noise generation
            lacunarity: Lacunarity value for noise generation
            seed: Random seed for reproducible terrain generation
            earth_scale: Scale factor for display purposes
        """
        self.size = size
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.heightmap = None
        self.continental_mask = None
        self.valley_mask = None
        self.river_paths = []

        # Earth properties for scaling
        self.earth_scale = earth_scale

        # For display purposes - this creates a scaled-down map representation
        self.earth_radius_km = 6371.0  # Earth's radius in km
        self.scaled_radius_km = self.earth_radius_km * np.sqrt(earth_scale)
        self.scaled_circumference_km = 2 * np.pi * self.scaled_radius_km

        # For physical calculations - we use full Earth properties
        self.physics_radius_km = 6371.0  # Always use full Earth radius for physics
        self.physics_circumference_km = 2 * np.pi * self.physics_radius_km

        # Calculate km per grid cell for display
        self.km_per_cell = self.scaled_circumference_km / size
        self.area_per_cell_sqkm = self.km_per_cell ** 2
        self.area_per_cell_sqmiles = (self.area_per_cell_sqkm * 0.386102)  # Convert to sq miles

        # Create latitude bands for reference (y-coordinates to latitude in degrees)
        self.latitudes = np.zeros(self.size)
        for y in range(self.size):
            self.latitudes[y] = 90 - (y / (self.size - 1) * 180)  # +90° at top, -90° at bottom

        print(f"Initialized terrain generator "
              f"for world at {earth_scale:.4%} of Earth's size")
        print(f"World radius: {self.scaled_radius_km:.1f} km")
        print(f"Each grid cell represents {self.km_per_cell:.2f} km "
              f"({self.area_per_cell_sqmiles:.2f} sq miles)")

    def create_continental_mask(self, continent_size: float = 0.3) -> np.ndarray:
        """Generate continent masks with improved latitudinal distribution.
        
        Creates more land in temperate zones (25°-60° N/S) and less in equatorial regions.
        Ensures smooth continental connectivity.
        
        Args:
            continent_size: Target land percentage (0.0-1.0)
            
        Returns:
            2D numpy array with continental influence
        """
        print("Generating continental influence map with improved distribution...")
    
        # Base continental noise with variable frequencies
        continental_mask = np.zeros((self.size, self.size))
        noise_gen = OpenSimplex(seed=self.seed + 42)  # Different seed for continents

        # Use multiple noise octaves for more realistic continent shapes
        for y in range(self.size):
            latitude = self.latitudes[y]  # Current latitude in degrees
            
            # Create latitudinal bias - multiply land probability by this factor
            # Peaks at around 40° North and South, lowest at equator
            lat_bias = self.calculate_latitudinal_land_bias(latitude)
            
            for x in range(self.size):
                # Multiple noise scales for complex continents
                value1 = noise_gen.noise2(x / (self.size/2), y / (self.size/2))
                value2 = noise_gen.noise2(x / (self.size/4), y / (self.size/4)) * 0.5
                value3 = noise_gen.noise2(x / (self.size/8), y / (self.size/8)) * 0.25
                
                # Combine noise at different scales
                base_value = value1 + value2 + value3
                
                # Apply latitudinal bias
                continental_mask[y, x] = base_value * lat_bias

        # Normalize to 0-1 range
        min_val = np.min(continental_mask)
        max_val = np.max(continental_mask)
        continental_mask = (continental_mask - min_val) / (max_val - min_val)
        
        # Improve east-west continuity using a horizontal continuity pass
        continental_mask = self.enhance_horizontal_continuity(continental_mask)
        
        # Use percentile for land threshold to ensure target land percentage
        land_threshold = np.percentile(continental_mask, (1.0 - continent_size) * 100)
        binary_continents = (continental_mask > land_threshold).astype(float)
        
        # Apply distance field for coastal gradients
        distance = distance_transform_edt(1 - binary_continents)
        max_distance = np.max(distance)
        if max_distance > 0:
            coastal_gradient = 1.0 - (distance / max_distance)
            continental_mask = np.maximum(binary_continents, coastal_gradient * 0.7)
            
        # Store the continental mask for later use
        self.continental_mask = continental_mask
        return continental_mask

    def calculate_latitudinal_land_bias(self, latitude: float) -> float:
        """Calculate the land probability bias for a given latitude.
        
        Higher values in temperate zones (25°-60° N/S), lower values near equator.
        
        Args:
            latitude: Latitude in degrees (-90 to 90)
            
        Returns:
            Multiplier for land probability
        """
        # Convert to absolute latitude for symmetrical N/S distribution
        abs_lat = abs(latitude)
        
        # Base multiplier
        if abs_lat < 10:
            # Equatorial region (0°-10°) - reduce land probability
            return 0.7
        elif abs_lat < 25:
            # Transitional region (10°-25°) - slight increase
            return 0.8 + (abs_lat - 10) * 0.02
        elif abs_lat < 60:
            # Temperate region (25°-60°) - increase land probability
            return 1.3
        elif abs_lat < 80:
            # Sub-polar region (60°-80°) - gradual decrease
            return 1.3 - (abs_lat - 60) * 0.025
        else:
            # Polar region (80°-90°) - less land
            return 0.8
    
    def enhance_horizontal_continuity(self, mask: np.ndarray) -> np.ndarray:
        """Enhance the horizontal (east-west) continuity of land masses.
        
        Ensures land masses don't abruptly end at edges and wrap properly around the world.
        
        Args:
            mask: Continental mask to enhance
            
        Returns:
            Enhanced continental mask
        """
        enhanced_mask = mask.copy()
        
        # Apply a horizontal smoothing pass to ensure east-west continuity
        for y in range(self.size):
            # Extract row and create wrapped version for edge handling
            row = mask[y, :]
            wrapped_row = np.concatenate([row[-10:], row, row[:10]])
            
            # Apply horizontal smoothing with distance weighting
            for x in range(self.size):
                # Consider a wider neighborhood with distance weighting
                neighborhood = wrapped_row[x:x+20]  # 10 cells left, current, 9 cells right
                
                # Weighted average based on distance
                weights = np.linspace(0.5, 1.0, 10)  # Weights increase toward center
                weights = np.concatenate([weights, weights[::-1]])
                
                weighted_avg = np.sum(neighborhood * weights) / np.sum(weights)
                
                # Blend with original value (80% original, 20% smoothed)
                enhanced_mask[y, x] = mask[y, x] * 0.8 + weighted_avg * 0.2
        
        # Improve edge wrapping by explicitly copying edge data
        # Copy right edge to left buffer zone and vice versa
        for y in range(self.size):
            for x in range(1, 21):  # Wider buffer zone (20 cells)
                enhanced_mask[y, -x] = 0.3 * enhanced_mask[y, -x] + 0.7 * enhanced_mask[y, x]
                enhanced_mask[y, x-1] = 0.3 * enhanced_mask[y, x-1] + 0.7 * enhanced_mask[y, -x]
        
        # Additional smoothing pass focused on edges
        edge_weights = np.linspace(0.5, 1.0, 20)  # Stronger weights near edges
        for y in range(self.size):
            # Process left edge (with data from right)
            for x in range(20):
                weight = edge_weights[x]
                right_val = enhanced_mask[y, -20+x]
                enhanced_mask[y, x] = enhanced_mask[y, x] * (1-weight) + right_val * weight
                
            # Process right edge (with data from left)
            for x in range(1, 21):
                weight = edge_weights[20-x]
                left_val = enhanced_mask[y, x-1]
                enhanced_mask[y, -x] = enhanced_mask[y, -x] * (1-weight) + left_val * weight
        
        return enhanced_mask

    def generate_valley_mask(self) -> np.ndarray:
        """Generate a mask for deep valleys in the terrain.
        
        Valleys are more likely to appear near mountain ranges.
        
        Returns:
            2D numpy array representing valley locations
        """
        print("Generating deep valley patterns...")
        
        valley_mask = np.zeros((self.size, self.size))
        valley_noise = OpenSimplex(seed=self.seed + 75)
        
        # Different frequency for valleys
        for y in range(self.size):
            for x in range(self.size):
                # Valleys have a different pattern than the base heightmap
                nx = x / (self.size / 7)  # Higher frequency for narrower valleys
                ny = y / (self.size / 5)
                
                # Multiple octaves for valley detail
                value = 0.0
                amplitude = 1.0
                frequency = 1.0
                
                for _ in range(4):  # Fewer octaves for valley mask
                    value += valley_noise.noise2(nx * frequency, ny * frequency) * amplitude
                    amplitude *= 0.65
                    frequency *= 2.0
                
                valley_mask[y, x] = value
        
        # Normalize to 0-1
        valley_mask = (valley_mask - np.min(valley_mask)) / (np.max(valley_mask) - np.min(valley_mask))
        
        # Apply threshold to get only the deepest valleys
        valley_mask = (valley_mask > 0.65) * valley_mask
        
        # Smooth the valley transitions
        valley_mask = gaussian_filter(valley_mask, sigma=1.0)
        
        self.valley_mask = valley_mask
        return valley_mask

    def generate_heightmap(self, scale: float = 100.0) -> np.ndarray:
        """Generate a heightmap with improved features including deep valleys.
        
        Args:
            scale: Scale factor for noise generation
            
        Returns:
            2D numpy array representing the heightmap
        """
        print(f"Generating heightmap of size {self.size}x{self.size}...")

        heightmap = np.zeros((self.size, self.size))
        noise_gen = OpenSimplex(seed=self.seed)

        # Constants for cylindrical mapping
        two_pi = 2 * np.pi

        # Generate base noise with cylindrical mapping for proper wrapping
        for y in range(self.size):
            for x in range(self.size):
                value = 0.0
                amplitude = 1.0
                frequency = 1.0

                # Convert grid coordinates to cylindrical coordinates
                # Map x to longitude (0 to 2π) for east-west wrapping
                lon = (x / self.size) * two_pi

                # Map y to latitude (-π/2 to π/2) for proper pole distortion
                lat = ((y / self.size) * np.pi) - (np.pi / 2)

                for _ in range(self.octaves):
                    # Calculate 3D noise coordinates on a cylinder
                    nx = np.cos(lon) * scale / frequency
                    nz = np.sin(lon) * scale / frequency
                    ny = lat * scale / frequency
                
                    # Use 3D noise for better continuity
                    value += noise_gen.noise3(nx, ny, nz) * amplitude
                
                    amplitude *= self.persistence
                    frequency *= self.lacunarity

                heightmap[y][x] = value

        # Normalize to 0-1 range
        min_val = np.min(heightmap)
        max_val = np.max(heightmap)
        heightmap = (heightmap - min_val) / (max_val - min_val)

        # Create continental influence if not already done
        if self.continental_mask is None:
            continental_mask = self.create_continental_mask()
        else:
            continental_mask = self.continental_mask

        # Apply the continental mask to influence heightmap
        for y in range(self.size):
            for x in range(self.size):
                # Blend heightmap with continental influence
                heightmap[y, x] = heightmap[y, x] * 0.4 + continental_mask[y, x] * 0.6

        # Create valley mask if not already created
        if self.valley_mask is None:
            valley_mask = self.generate_valley_mask()
        else:
            valley_mask = self.valley_mask

        # Apply mountain and valley transformations
        heightmap = self.apply_mountain_valley_transformation(heightmap, valley_mask)

        # Enhance coastal areas to ensure proper elevation transitions
        heightmap = self.enhance_coastal_areas(heightmap)
        
        # Apply terrain variation to avoid uniform elevations
        heightmap = self.add_terrain_variation(heightmap)

        # Normalize again
        heightmap = (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))
        self.heightmap = heightmap
        
        return heightmap

    def apply_mountain_valley_transformation(self, heightmap: np.ndarray, 
                                             valley_mask: np.ndarray) -> np.ndarray:
        """Apply mountain and valley transformations to create dramatic terrain.
        
        Creates realistic terrain with tallest peaks ~10% higher than Everest (~9,700m)
        and deepest land depressions ~10% lower than Dead Sea (~-440m).
        
        Args:
            heightmap: Original heightmap
            valley_mask: Mask indicating where valleys should form
            
        Returns:
            Transformed heightmap with mountains and valleys
        """
        # Create a copy of the heightmap to modify
        transformed = heightmap.copy()
        
        # Reference elevations (in normalized height values 0-1)
        # These map to real-world elevations approximately:
        sea_level = 0.3           # Sea level = 0m
        coastal_plain = 0.32      # Coastal plains = 0-100m
        lowland = 0.4             # Lowlands/plains = 100-500m
        hills = 0.55              # Hills = 500-1500m
        highlands = 0.65          # Highlands = 1500-2500m
        mountain_start = 0.75     # Mountains begin = 2500m+
        high_peaks = 0.9          # High peaks = 5000m+
        
        # Apply height transformation for varied terrain features
        for y in range(self.size):
            for x in range(self.size):
                h = heightmap[y, x]
                valley_factor = valley_mask[y, x]

                if h < sea_level:
                    # Ocean depths - get deeper gradually
                    normalized_depth = h / sea_level
                    transformed[y, x] = -0.3 * (1.0 - normalized_depth**2)
                
                elif h < coastal_plain:
                    # Coastal plains (0-100m) - very gradual rise from shoreline
                    normalized_h = (h - sea_level) / (coastal_plain - sea_level)
                    transformed[y, x] = 0.01 + (normalized_h * 0.03)
                    
                elif h < lowland:
                    # Lowlands and plains (100-500m)
                    normalized_h = (h - coastal_plain) / (lowland - coastal_plain)
                    base_height = 0.04 + (normalized_h * 0.1)
                    
                    # Apply valley effects if needed - can create below sea level depressions
                    if valley_factor > 0.3:
                        # Deep valleys can go below sea level (-440m × 1.1 = ~ -480m)
                        valley_depth = valley_factor * 0.15  
                        transformed[y, x] = base_height - valley_depth
                    elif valley_factor > 0.1:
                        # Shallow valleys
                        valley_depth = valley_factor * 0.05
                        transformed[y, x] = base_height - valley_depth
                    else:
                        # Normal terrain with subtle variations
                        variation = np.sin(normalized_h * 5.0 * np.pi) * 0.02
                        transformed[y, x] = base_height + variation
                
                elif h < hills:
                    # Hills (500-1500m)
                    normalized_h = (h - lowland) / (hills - lowland)
                    base_height = 0.14 + (normalized_h * 0.12)
                    
                    if valley_factor > 0.2:
                        valley_depth = valley_factor * 0.12
                        transformed[y, x] = base_height - valley_depth
                    else:
                        variation = np.sin(normalized_h * 4.0 * np.pi) * 0.03
                        transformed[y, x] = base_height + variation
                
                elif h < highlands:
                    # Highlands (1500-2500m)
                    normalized_h = (h - hills) / (highlands - hills)
                    base_height = 0.26 + (normalized_h * 0.14)
                    
                    if valley_factor > 0.15:
                        valley_depth = valley_factor * 0.15
                        transformed[y, x] = base_height - valley_depth
                    else:
                        transformed[y, x] = base_height + np.random.random() * 0.02
                
                elif h < mountain_start:
                    # Mountain foothills (2500-4000m)
                    normalized_h = (h - highlands) / (mountain_start - highlands)
                    base_height = 0.4 + (normalized_h * 0.15)
                    
                    if valley_factor > 0.1:
                        # Mountain valleys
                        valley_depth = valley_factor * 0.2
                        transformed[y, x] = base_height - valley_depth
                    else:
                        transformed[y, x] = base_height
                
                elif h < high_peaks:
                    # Mountains (4000-7000m)
                    normalized_h = (h - mountain_start) / (high_peaks - mountain_start)
                    transformed[y, x] = 0.55 + normalized_h * 0.25
                
                else:
                    # Highest peaks (7000-9700m) - ~10% higher than Everest
                    normalized_h = (h - high_peaks) / (1.0 - high_peaks)
                    transformed[y, x] = 0.8 + normalized_h * 0.2
                    
                    # Add extra height to the tallest 2% for exceptional peaks
                    if h > 0.98:
                        bonus_height = (h - 0.98) / 0.02  # Normalize 0.98-1.0 to 0-1
                        transformed[y, x] += bonus_height * 0.05  # Additional 5% height

        return transformed

    def enhance_coastal_areas(self, heightmap: np.ndarray) -> np.ndarray:
        """Create realistic coastal plains extending from shorelines.
        
        Args:
            heightmap: Heightmap to enhance
            
        Returns:
            Heightmap with improved coastal transitions
        """
        enhanced = heightmap.copy()
        ocean_mask = heightmap < 0
        
        # For each land cell, check distance to nearest ocean
        for y in range(self.size):
            for x in range(self.size):
                if heightmap[y, x] <= 0:  # Skip ocean
                    continue
                    
                # Look for nearby ocean
                min_dist = float('inf')
                for r in range(1, 20):  # Check up to 20 cells away
                    found_ocean = False
                    
                    # Check cells at distance r
                    for angle in range(0, 360, 10):
                        rads = np.radians(angle)
                        dy = int(r * np.sin(rads))
                        dx = int(r * np.cos(rads))
                        
                        ny = max(0, min(self.size-1, y+dy))
                        nx = (x+dx) % self.size
                        
                        if heightmap[ny, nx] < 0:  # Found ocean
                            found_ocean = True
                            min_dist = r
                            break
                    
                    if found_ocean:
                        break
                
                # If within 15 cells of coast, apply coastal plain effect
                if min_dist <= 15:
                    # Calculate elevation based on distance from coast
                    # Closer to coast = lower elevation
                    coastal_factor = 1.0 - (min_dist / 15.0)
                    
                    # Maximum height reduction (when right next to coast)
                    max_reduction = enhanced[y, x] * 0.8  # Reduce height by up to 80%
                    
                    # Apply height reduction, stronger near coast
                    reduction = max_reduction * (coastal_factor ** 2)  # Squared for faster falloff
                    enhanced[y, x] -= reduction
                    
                    # Ensure minimum elevation for coastal plains (very subtle rise from shore)
                    if min_dist <= 3:
                        # First 3 cells from shore have very low elevation
                        enhanced[y, x] = min(enhanced[y, x], 0.02 + (min_dist * 0.01))
        
        return enhanced

    def add_terrain_variation(self, heightmap: np.ndarray) -> np.ndarray:
        """Add natural elevation variation across continents.
        
        Prevents uniform elevations in large areas and creates more interesting terrain.
        
        Args:
            heightmap: Heightmap to enhance
            
        Returns:
            Heightmap with improved elevation variation
        """
        varied = heightmap.copy()
        
        # Use large-scale noise for broad elevation patterns
        large_scale_noise = OpenSimplex(seed=self.seed + 200)
        
        for y in range(self.size):
            for x in range(self.size):
                # Skip ocean
                if heightmap[y, x] < 0:
                    continue
                    
                # Get large-scale noise value for this location
                nx = x / (self.size / 2)  # Very large scale
                ny = y / (self.size / 2)
                noise_val = large_scale_noise.noise2(nx, ny)
                
                # Add medium-scale variation
                nx2 = x / (self.size / 8)
                ny2 = y / (self.size / 8)
                medium_noise = large_scale_noise.noise2(nx2, ny2) * 0.3
                
                # Combine noise values
                combined_noise = noise_val * 0.7 + medium_noise * 0.3
                
                # Use noise to create broad elevation variations
                # Keep coastal areas low regardless
                if heightmap[y, x] < 0.2:
                    # Minimal adjustment near coasts
                    varied[y, x] += combined_noise * 0.03
                else:
                    # Larger adjustments inland
                    varied[y, x] += combined_noise * 0.1
        
        # Re-normalize the land areas to maintain proper elevation range
        land_mask = varied >= 0
        if np.any(land_mask):
            land = varied[land_mask]
            land_min = np.min(land)
            land_max = np.max(land)
            if land_max > land_min:
                # Scale land to 0-1 range again
                varied[land_mask] = (varied[land_mask] - land_min) / (land_max - land_min)
        
        return varied

    def smooth_mountain_slopes(self, heightmap: np.ndarray) -> np.ndarray:
        """Apply smoothing to mountain slopes while preserving peaks.
        
        Creates more gradual elevation transitions on mountainsides.
        
        Args:
            heightmap: Heightmap to smooth
            
        Returns:
            Heightmap with smoother mountain slopes
        """
        smoothed = heightmap.copy()
        
        # Only smooth areas above certain elevation (mountains)
        mountain_mask = heightmap > 0.6
        
        # For each mountain cell, apply a directional smoothing
        for y in range(1, self.size-1):
            for x in range(1, self.size-1):
                if not mountain_mask[y, x]:
                    continue
                    
                # Find downslope neighbors
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                            
                        ny = max(0, min(self.size-1, y+dy))
                        nx = (x+dx) % self.size
                        
                        if heightmap[ny, nx] < heightmap[y, x]:
                            neighbors.append((ny, nx, heightmap[ny, nx]))
                
                # If no downslope neighbors, continue
                if not neighbors:
                    continue
                    
                # Sort by height (highest first)
                neighbors.sort(key=lambda n: n[2], reverse=True)
                
                # Smooth transition to highest downslope neighbor
                ny, nx, n_height = neighbors[0]
                
                # Calculate how much to smooth
                height_diff = heightmap[y, x] - n_height
                if height_diff > 0.15:  # Only smooth significant drops
                    # Adjust height to create smoother slope
                    smoothed[y, x] -= height_diff * 0.3  # Reduce height by 30% of difference
                    smoothed[ny, nx] += height_diff * 0.1  # Add a small amount to neighbor
        
        return smoothed

    def add_mountains(self, mountain_scale: float = 1.2,
                     peak_threshold: float = 0.6,
                     epic_factor: float = 1.8) -> np.ndarray:
        """Add realistic mountain ranges with dramatic relief.
        
        Args:
            mountain_scale: Height scale for mountains
            peak_threshold: Elevation threshold for mountain features
            epic_factor: Multiplier for dramatic mountain features
            
        Returns:
            Heightmap with enhanced mountain features
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before adding mountains")

        print(f"Adding mountain ranges (epic factor: {epic_factor:.1f})...")

        # Generate a mountain mask using different noise parameters
        mountain_mask = np.zeros((self.size, self.size))
        mountain_seed = self.seed + 1000
        mountain_noise = OpenSimplex(seed=mountain_seed)

        # Generate ridge-like mountain patterns
        for y in range(self.size):
            for x in range(self.size):
                # Different scales for more dramatic mountains
                nx = x / 200
                ny = y / 200
                
                # Ridge noise algorithm - creates elongated ranges
                ridged_value = 0.0
                amplitude = 1.0
                frequency = 1.0
                
                for _ in range(3):
                    # Generate basic noise
                    n = mountain_noise.noise2(nx * frequency, ny * frequency)
                    # Convert to ridged noise (1 - abs(n))
                    n = 1.0 - abs(n)
                    # Square to sharpen ridges
                    n *= n
                    ridged_value += n * amplitude
                    
                    amplitude *= 0.55
                    frequency *= 2.5
                    
                mountain_mask[y][x] = ridged_value

        # Normalize mountain mask
        mountain_mask = (mountain_mask - np.min(mountain_mask)) / (
            np.max(mountain_mask) - np.min(mountain_mask))

        # Use the valley mask to ensure mountains and valleys are correctly positioned
        valley_influence = np.zeros_like(mountain_mask)
        if self.valley_mask is not None:
            # Enhance mountains adjacent to valleys
            for y in range(self.size):
                for x in range(self.size):
                    # Look for valley edges
                    is_valley_edge = False
                    if self.valley_mask[y, x] > 0.1:
                        continue  # Skip valley centers
                        
                    # Check if near a valley
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            ny = max(0, min(self.size - 1, y + dy))
                            nx = (x + dx) % self.size
                            
                            if self.valley_mask[ny, nx] > 0.3:
                                is_valley_edge = True
                                break
                        if is_valley_edge:
                            break
                            
                    if is_valley_edge:
                        # Boost mountains near valleys for dramatic relief
                        valley_influence[y, x] = 0.3

        # Apply mountains where the mask exceeds the threshold
        mountain_terrain = self.heightmap.copy()
        mountain_areas = mountain_mask > peak_threshold

        # First apply basic mountain height increase
        mountain_terrain[mountain_areas] += (
            mountain_scale * (mountain_mask[mountain_areas] - peak_threshold))

        # Add enhanced peaks more selectively
        high_peaks = ((mountain_mask > peak_threshold + 0.25) &  # Selective peaks
                      (mountain_terrain > 0.7))

        # Apply valley-influenced enhancement
        for y in range(self.size):
            for x in range(self.size):
                if mountain_areas[y, x]:
                    # Extra height near valleys for dramatic relief
                    mountain_terrain[y, x] += valley_influence[y, x]

        # Create epic peaks
        if np.any(high_peaks):
            # Get count of high peaks before modification
            peak_count = np.sum(high_peaks)

            # Keep about 20% of the highest peaks for truly epic mountains
            if peak_count > 10:
                # Find the highest peaks
                peak_values = mountain_terrain[high_peaks]
                peak_threshold_value = np.percentile(peak_values, 80)  # Keep top 20%

                # Create mask that only includes the highest peaks
                epic_peaks = high_peaks.copy()
                epic_peaks[high_peaks] = mountain_terrain[high_peaks] >= peak_threshold_value

                # Apply the epic height increase only to these selected peaks
                peak_height = mountain_terrain[epic_peaks]
                mountain_terrain[epic_peaks] = peak_height + (epic_factor * (peak_height - 0.7) ** 2)
            
                print(f"Created {np.sum(epic_peaks)} major mountain peaks out of {peak_count} candidates")
            else:
                # If we have very few peaks, keep them all
                peak_height = mountain_terrain[high_peaks]
                mountain_terrain[high_peaks] = peak_height + (epic_factor * (peak_height - 0.7) ** 2)

        # Add crags and ridges (small high-frequency variations)
        crag_noise = OpenSimplex(seed=mountain_seed + 500)
        for y in range(self.size):
            for x in range(self.size):
                if mountain_terrain[y, x] > 0.6:  # Only add detail to mountains
                    nx = x / 20  # Higher frequency for small details
                    ny = y / 20
                    # Small, sharp variations
                    crag_value = crag_noise.noise2(nx, ny) * 0.05 * epic_factor
                    mountain_terrain[y, x] += crag_value

        # Apply slope smoothing to create more gradual transitions
        mountain_terrain = self.smooth_mountain_slopes(mountain_terrain)

        # Renormalize if needed while maintaining ocean depths as negative
        ocean_mask = mountain_terrain < 0
        
        # Get max value of land to normalize
        land = mountain_terrain.copy()
        land[ocean_mask] = 0
        max_land = np.max(land)
        
        # Get min value of ocean to normalize
        ocean = mountain_terrain.copy()
        ocean[~ocean_mask] = 0
        min_ocean = np.min(ocean) if np.any(ocean_mask) else 0
        
        # Normalize separately to maintain structure
        if max_land > 1.0:
            land = land / max_land
            
        if min_ocean < 0:
            ocean = ocean / (2 * abs(min_ocean)) * (-0.5)
            
        # Combine normalized land and ocean
        mountain_terrain = np.zeros_like(mountain_terrain)
        mountain_terrain[~ocean_mask] = land[~ocean_mask]
        mountain_terrain[ocean_mask] = ocean[ocean_mask]
        
        self.heightmap = mountain_terrain
        return mountain_terrain
        
    def add_ridges_and_canyons(self, ridge_count: int = 15, canyon_count: int = 10) -> np.ndarray:
        """Add dramatic ridge lines and canyon features to the terrain.
        
        Creates linear features that enhance the dramatic relief of the terrain.
        
        Args:
            ridge_count: Number of major ridge features
            canyon_count: Number of major canyon features
            
        Returns:
            Updated heightmap with ridge and canyon features
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before adding features")
            
        print(f"Adding {ridge_count} major ridges and {canyon_count} canyons...")
        
        # Create a copy of the heightmap to modify
        enhanced_map = self.heightmap.copy()
        
        # Function to generate a random curved line across the terrain
        def generate_curve_points(length, curviness=0.3):
            """Generate a curved line with given length and curviness."""
            # Random starting point (away from edges)
            border = self.size // 10
            y = np.random.randint(border, self.size - border)
            x = np.random.randint(border, self.size - border)
            
            # Random direction (in radians)
            direction = np.random.random() * 2 * np.pi
            
            points = [(y, x)]
            
            # Generate each segment
            for _ in range(length):
                # Add some random variation to direction
                direction += (np.random.random() - 0.5) * curviness
                
                # Calculate new position
                dy = np.sin(direction) * 1.5  # Step size
                dx = np.cos(direction) * 1.5
                
                y = int(max(0, min(self.size - 1, y + dy)))
                x = int((x + int(dx)) % self.size)  # Wrap around for longitude
                
                points.append((y, x))
            
            return points
        
        # Add ridge lines (elevated features)
        for _ in range(ridge_count):
            # Random length between 50 and 200 cells
            length = np.random.randint(50, 200)
            ridge_points = generate_curve_points(length, curviness=0.4)
            
            # Ridges only appear on land
            valid_ridge = False
            for y, x in ridge_points:
                if enhanced_map[y, x] > 0:  # On land
                    valid_ridge = True
                    break
                    
            if not valid_ridge:
                continue
                
            # Random ridge height (higher for more dramatic effect)
            ridge_height = np.random.uniform(0.1, 0.25)
            
            # Apply ridge with falloff
            for y, x in ridge_points:
                if enhanced_map[y, x] <= 0:  # Skip ocean
                    continue
                    
                # Central ridge elevation
                enhanced_map[y, x] += ridge_height
                
                # Wider ridges with falloff
                for r in range(1, 5):
                    falloff = ridge_height * (1 - r/5)
                    
                    for dy in range(-r, r+1):
                        for dx in range(-r, r+1):
                            dist = np.sqrt(dy*dy + dx*dx)
                            if dist > r:
                                continue
                                
                            ny = max(0, min(self.size - 1, y + dy))
                            nx = (x + dx) % self.size
                            
                            if enhanced_map[ny, nx] <= 0:  # Skip ocean
                                continue
                                
                            enhanced_map[ny, nx] += falloff * (1 - dist/r)
        
        # Add canyon features (depressed lines)
        for _ in range(canyon_count):
            # Random length between 30 and 150 cells
            length = np.random.randint(30, 150)
            canyon_points = generate_curve_points(length, curviness=0.3)
            
            # Canyons only appear on land of certain elevation
            valid_canyon = False
            for y, x in canyon_points:
                if 0.1 < enhanced_map[y, x] < 0.7:  # On suitable land
                    valid_canyon = True
                    break
                    
            if not valid_canyon:
                continue
                
            # Random canyon depth
            canyon_depth = np.random.uniform(0.1, 0.3)
            
            # Apply canyon with falloff
            for y, x in canyon_points:
                if enhanced_map[y, x] <= 0:  # Skip ocean
                    continue
                    
                # Don't cut canyons into very high mountains
                if enhanced_map[y, x] > 0.7:
                    continue
                    
                # Central canyon depression
                enhanced_map[y, x] -= canyon_depth
                
                # Wider canyons with falloff
                for r in range(1, 4):
                    falloff = canyon_depth * (1 - r/4)
                    
                    for dy in range(-r, r+1):
                        for dx in range(-r, r+1):
                            dist = np.sqrt(dy*dy + dx*dx)
                            if dist > r:
                                continue
                                
                            ny = max(0, min(self.size - 1, y + dy))
                            nx = (x + dx) % self.size
                            
                            if enhanced_map[ny, nx] <= 0:  # Skip ocean
                                continue
                                
                            enhanced_map[ny, nx] -= falloff * (1 - dist/r)
        
        # Ensure we don't have negative elevation on land due to canyon depth
        enhanced_map[enhanced_map < 0] = 0
        
        # Normalize the heightmap (keeping oceans negative)
        ocean_mask = self.heightmap < 0
        enhanced_map[ocean_mask] = self.heightmap[ocean_mask]
        
        self.heightmap = enhanced_map
        return enhanced_map
        
    def add_plateaus(self, count: int = 8) -> np.ndarray:
        """Add plateau features to the terrain for more diverse landscapes.
        
        Creates flat-topped elevated areas typical of mesas and tablelands.
        
        Args:
            count: Number of plateau features to add
            
        Returns:
            Updated heightmap with plateau features
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before adding plateaus")
            
        print(f"Adding {count} plateau features...")
        
        # Create a copy of the heightmap to modify
        plateau_map = self.heightmap.copy()
        
        # Generate potential plateau centers
        plateau_centers = []
        border = self.size // 10
        
        # Look for suitable flat-ish regions
        for _ in range(count * 5):  # Try more spots than needed
            y = np.random.randint(border, self.size - border)
            x = np.random.randint(border, self.size - border)
            
            # Check if suitable for plateau (on land, somewhat elevated)
            if plateau_map[y, x] > 0.3 and plateau_map[y, x] < 0.6:
                # Check local variation - plateaus need relatively flat base
                local_var = 0
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        ny = max(0, min(self.size - 1, y + dy))
                        nx = (x + dx) % self.size
                        local_var += abs(plateau_map[y, x] - plateau_map[ny, nx])
                
                # Lower local variation is better for plateaus
                if local_var < 1.5:
                    plateau_centers.append((y, x, local_var))
        
        # Sort by local variation (flattest first)
        plateau_centers.sort(key=lambda p: p[2])
        
        plateaus_added = 0
        for y, x, _ in plateau_centers:
            if plateaus_added >= count:
                break
                
            # Random plateau size and height
            size = np.random.randint(15, 50)
            height_boost = np.random.uniform(0.1, 0.25)
            
            # Create base height for the plateau (current height + boost)
            plateau_height = plateau_map[y, x] + height_boost
            
            # Check if area is suitable for plateau
            too_close = False
            for py, px, _ in plateau_centers[:plateaus_added]:
                # Calculate distance with longitude wrap-around
                dx = min(abs(x - px), self.size - abs(x - px))
                dy = abs(y - py)
                if np.sqrt(dx*dx + dy*dy) < size * 2:
                    too_close = True
                    break
            
            if too_close:
                continue
                
            # Add plateau with different shapes (circle, oval, irregular)
            shape_type = np.random.choice(['circle', 'oval', 'irregular'])
            
            if shape_type == 'circle':
                # Simple circular plateau
                for dy in range(-size, size+1):
                    for dx in range(-size, size+1):
                        dist = np.sqrt(dy*dy + dx*dx)
                        if dist > size:
                            continue
                            
                        ny = max(0, min(self.size - 1, y + dy))
                        nx = (x + dx) % self.size
                        
                        if plateau_map[ny, nx] <= 0:  # Skip ocean
                            continue
                            
                        # Smooth falloff at edges
                        edge_falloff = 1.0
                        if dist > size * 0.8:
                            edge_falloff = 1.0 - ((dist - size * 0.8) / (size * 0.2))
                        
                        # Set to plateau height with falloff
                        if edge_falloff < 1.0:
                            # Edge - blend with original height
                            original = plateau_map[ny, nx]
                            plateau_map[ny, nx] = original * (1.0 - edge_falloff) + plateau_height * edge_falloff
                        else:
                            # Flat top with minor variations for realism
                            variation = (np.random.random() - 0.5) * 0.02
                            plateau_map[ny, nx] = plateau_height + variation
                            
            elif shape_type == 'oval':
                # Oval/elliptical plateau
                # Random orientation angle
                angle = np.random.random() * np.pi
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                
                # Random aspect ratio
                a = size
                b = size * np.random.uniform(0.5, 0.8)
                
                for dy in range(-size, size+1):
                    for dx in range(-size, size+1):
                        # Rotate and scale coordinates
                        rx = dx * cos_angle - dy * sin_angle
                        ry = dx * sin_angle + dy * cos_angle
                        
                        # Ellipse equation
                        dist = np.sqrt((rx/a)**2 + (ry/b)**2)
                        if dist > 1.0:
                            continue
                            
                        ny = max(0, min(self.size - 1, y + dy))
                        nx = (x + dx) % self.size
                        
                        if plateau_map[ny, nx] <= 0:  # Skip ocean
                            continue
                            
                        # Smooth falloff at edges
                        edge_falloff = 1.0
                        if dist > 0.8:
                            edge_falloff = 1.0 - ((dist - 0.8) / 0.2)
                        
                        # Set to plateau height with falloff
                        if edge_falloff < 1.0:
                            # Edge - blend with original height
                            original = plateau_map[ny, nx]
                            plateau_map[ny, nx] = original * (1.0 - edge_falloff) + plateau_height * edge_falloff
                        else:
                            # Flat top with minor variations
                            variation = (np.random.random() - 0.5) * 0.02
                            plateau_map[ny, nx] = plateau_height + variation
                            
            else:  # irregular
                # Create an irregular plateau with Perlin noise boundary
                noise_gen = OpenSimplex(seed=self.seed + plateaus_added)
                
                for dy in range(-size, size+1):
                    for dx in range(-size, size+1):
                        # Base distance from center
                        base_dist = np.sqrt(dy*dy + dx*dx) / size
                        if base_dist > 1.0:
                            continue
                            
                        # Add noise to the boundary
                        angle = np.arctan2(dy, dx)
                        noise_val = noise_gen.noise2(np.cos(angle), np.sin(angle)) * 0.2
                        
                        # Check if inside the irregular boundary
                        if base_dist > 0.9 + noise_val:
                            continue
                            
                        ny = max(0, min(self.size - 1, y + dy))
                        nx = (x + dx) % self.size
                        
                        if plateau_map[ny, nx] <= 0:  # Skip ocean
                            continue
                            
                        # Smooth falloff at edges
                        edge_falloff = 1.0
                        if base_dist > 0.7:
                            edge_falloff = 1.0 - ((base_dist - 0.7) / 0.3)
                        
                        # Set to plateau height with falloff
                        if edge_falloff < 1.0:
                            # Edge - blend with original height
                            original = plateau_map[ny, nx]
                            plateau_map[ny, nx] = original * (1.0 - edge_falloff) + plateau_height * edge_falloff
                        else:
                            # Flat top with minor variations
                            variation = (np.random.random() - 0.5) * 0.02
                            plateau_map[ny, nx] = plateau_height + variation
            
            plateaus_added += 1
            
        print(f"Successfully added {plateaus_added} plateau features")
        self.heightmap = plateau_map
        return plateau_map

    def add_rivers(self, river_count: int = 30, min_length: int = 15, meander_factor: float = 0.3):
        """Add rivers flowing from high elevation to the sea with reliable pathing.
        
        Creates scientifically plausible river systems that reliably reach water bodies
        even in challenging terrain by using terrain analysis and river carving.

        Args:
            river_count: Number of major rivers to generate
            min_length: Minimum length for a valid river
            meander_factor: Amount of randomness in river paths (0.0-1.0)
            
        Returns:
            River mask as a 2D numpy array
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before adding rivers")

        print(f"Generating {river_count} major rivers...")

        # Create a river mask (1 where rivers exist, 0 elsewhere)
        river_mask = np.zeros_like(self.heightmap)
        
        # Store river paths for potential later use
        river_paths = []
        
        # Adjust the heightmap for river generation - ensure we have a 0-1 range
        river_heightmap = self.heightmap.copy()
        if np.min(river_heightmap) < 0:
            # Scale negative values to 0 and maintain relative heights above 0
            river_heightmap = river_heightmap - np.min(river_heightmap)
            river_heightmap = river_heightmap / np.max(river_heightmap)

        # Flow accumulation map - helps identify natural drainage patterns
        flow_accumulation = np.zeros_like(river_heightmap)
        
        # Calculate simple flow directions for each cell (8 directions)
        # This simulates water flow across the terrain to find natural channels
        print("Calculating terrain drainage patterns...")
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                if river_heightmap[y, x] <= 0:  # Skip ocean areas
                    continue
                    
                # Find lowest neighbor
                lowest_dir = None
                lowest_h = river_heightmap[y, x]
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        
                        # Handle wrap-around for longitude (x) and clamp latitude (y)
                        nx = (x + dx) % self.size
                        ny = max(0, min(self.size - 1, y + dy))
                        
                        if river_heightmap[ny, nx] < lowest_h:
                            lowest_h = river_heightmap[ny, nx]
                            lowest_dir = (dy, dx)
                
                # If we found a downhill path, add to that cell's flow accumulation
                if lowest_dir:
                    dy, dx = lowest_dir
                    ny = max(0, min(self.size - 1, y + dy))
                    nx = (x + dx) % self.size
                    flow_accumulation[ny, nx] += 1

        # Step 1: Create forced river paths from coastlines inland
        # This ensures major rivers will reach the ocean
        forced_rivers_added = 0
        
        # Find coastlines (where land meets ocean)
        coastline_points = []
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                # Skip if this is already ocean
                if self.heightmap[y, x] < 0:
                    continue
                
                # Check if any neighbor is ocean
                has_ocean_neighbor = False
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        
                        ny = max(0, min(self.size - 1, y + dy))
                        nx = (x + dx) % self.size
                    
                        if self.heightmap[ny, nx] < 0:
                            has_ocean_neighbor = True
                            break
                    if has_ocean_neighbor:
                        break
                    
                if has_ocean_neighbor:
                    coastline_points.append((y, x))
        
        # Choose a sample of coastline points as river mouths
        if coastline_points:
            num_river_mouths = min(len(coastline_points), river_count // 2)
            river_mouths = np.random.choice(len(coastline_points), size=num_river_mouths, replace=False)
            
            # From each river mouth, try to force a path inland and upward
            for mouth_idx in river_mouths:
                y, x = coastline_points[mouth_idx]
                river_path = [(y, x)]
        
                # Start with river mouth and work backward (upstream)
                for _ in range(self.size // 2):  # Limit maximum path length
                    # Find the highest elevation neighboring cell that's higher than current
                    current_height = river_heightmap[y, x]
                    higher_neighbors = []
        
                    # First check immediate neighbors (8-connected)
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                    
                            ny = max(0, min(self.size - 1, y + dy))
                            nx = (x + dx) % self.size
                
                            # Accept any higher neighbor with more favorable scoring
                            if river_heightmap[ny, nx] > current_height:
                                # Weight by elevation difference but be more tolerant
                                elevation_diff = river_heightmap[ny, nx] - current_height
                                # Less penalization for elevation changes
                                score = 1.0 / (1.0 + elevation_diff * 2.0)  
                                # Bonus for cells with high flow accumulation
                                score += flow_accumulation[ny, nx] * 0.01
                                higher_neighbors.append((ny, nx, score))
        
                    # If no immediate higher neighbors, look further (helps cross plateaus)
                    if not higher_neighbors:
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                # Skip immediate neighbors already checked
                                if abs(dy) <= 1 and abs(dx) <= 1:
                                    continue
                                
                                ny = max(0, min(self.size - 1, y + dy))
                                nx = (x + dx) % self.size
                                
                                if river_heightmap[ny, nx] > current_height:
                                    # Score with distance penalty
                                    dist = np.sqrt(dy*dy + dx*dx)
                                    elevation_diff = river_heightmap[ny, nx] - current_height
                                    score = (1.0 / (1.0 + elevation_diff * 2.0)) / dist
                                    higher_neighbors.append((ny, nx, score))
        
                    if not higher_neighbors:
                        # No suitable higher neighbors, end the river
                        break
            
                    # Choose a neighbor, with preference for gentler slopes
                    higher_neighbors.sort(key=lambda n: n[2], reverse=True)
                    # Add some randomness in selection
                    select_idx = min(int(np.random.random() * min(3, len(higher_neighbors))), len(higher_neighbors) - 1)
                    y, x, _ = higher_neighbors[select_idx]
        
                    # Stop if we've reached an existing river cell
                    if river_mask[y, x] > 0:
                        break
            
                    river_path.append((y, x))
        
                    # Stop if we've reached high elevation
                    if river_heightmap[y, x] > 0.8:
                        break
            
                # Add river if it's long enough
                if len(river_path) >= min_length:
                    # Rivers flow downstream, but we built the path upstream
                    river_path.reverse()
                    
                    # Store the river path
                    river_paths.append(river_path)
                
                    # Draw the river with increasing width
                    for i, (py, px) in enumerate(river_path):
                        position_factor = i / len(river_path)
                        river_width = 2 + int(position_factor * 3)
                        river_width = min(river_width, 5)
                    
                        # Apply river with circular brush
                        radius = river_width // 2
                        for wy in range(max(0, py-radius), min(self.size, py+radius+1)):
                            for wx in range(max(0, px-radius), min(self.size, px+radius+1)):
                                wrapped_wx = wx % self.size
                            
                                dy = wy - py
                                dx = min(abs(wrapped_wx - px), self.size - abs(wrapped_wx - px))
                                if (dy*dy + dx*dx) <= radius*radius:
                                    river_mask[wy, wrapped_wx] = 1
                
                    forced_rivers_added += 1
                
                    # Stop if we've added enough forced rivers
                    if forced_rivers_added >= river_count // 2:
                        break

        # Step 2: Now find source points for additional rivers
        # Consider both high peaks AND areas with high flow accumulation
        high_point_candidates = []
        
        # Find both high peaks for river sources
        peak_points = np.where(river_heightmap > 0.6)
        for i in range(len(peak_points[0])):
            y, x = peak_points[0][i], peak_points[1][i]
            # Skip points near the edge
            border = 5
            if border < y < self.size - border and border < x < self.size - border:
                # Score by both elevation and flow potential
                score = river_heightmap[y, x] + (flow_accumulation[y, x] * 0.01)
                high_point_candidates.append((y, x, score))
        
        # Also add points with high flow accumulation (natural drainage channels)
        # These represent areas where multiple small streams would converge
        flow_threshold = np.percentile(flow_accumulation, 95)  # Top 5% of flow cells
        high_flow_points = np.where((flow_accumulation > flow_threshold) & 
                                   (river_heightmap > 0.3) & 
                                   (river_heightmap < 0.6))
        
        for i in range(len(high_flow_points[0])):
            y, x = high_flow_points[0][i], high_flow_points[1][i]
            # Skip points near the edge
            border = 5
            if border < y < self.size - border and border < x < self.size - border:
                # Score emphasizes flow accumulation for these points
                score = 0.6 + (flow_accumulation[y, x] * 0.02)
                high_point_candidates.append((y, x, score))
        
        if not high_point_candidates:
            print("Warning: No suitable high points found for river sources")
            # Fall back to random high points if needed
            for _ in range(50):
                y = np.random.randint(border, self.size - border)
                x = np.random.randint(border, self.size - border)
                if river_heightmap[y, x] > 0.3:
                    high_point_candidates.append((y, x, river_heightmap[y, x]))
        
        # Sort by score (highest first)
        high_point_candidates.sort(key=lambda p: p[2], reverse=True)
        
        # Ensure spatial diversity - don't want all rivers starting in same area
        diverse_sources = []
        min_distance = self.size / 15  # Minimum distance between sources
        
        for y, x, _ in high_point_candidates:
            # Check if this point is far enough from already selected sources
            too_close = False
            for sy, sx, _ in diverse_sources:
                # Calculate distance with wrap-around for longitude
                dx = min(abs(x - sx), self.size - abs(x - sx))
                dy = abs(y - sy)
                distance = np.sqrt(dx*dx + dy*dy)
            
                if distance < min_distance:
                    too_close = True
                    break
        
            if not too_close:
                diverse_sources.append((y, x, river_heightmap[y, x]))
            
                # Stop once we have enough diverse sources
                if len(diverse_sources) >= river_count:
                    break

        rivers_created = 0

        # Step 3: Generate each river from source to sea
        for y, x, _ in diverse_sources:
            river_path = [(y, x)]
            path_length = 0
            
            # Track cells visited to avoid loops
            visited = set([(y, x)])
            
            # Maximum attempts to break through difficult terrain
            punch_through_attempts = 5  # Increased from typical 2 to handle difficult terrain
            has_reached_water = False

            # Generate the river by following the steepest descent
            for _ in range(self.size * 2):  # Maximum river length
                # Find steepest descent neighbor with randomness for meandering
                neighbor_options = []
                current_height = river_heightmap[y, x]
                
                # Check all 8 neighboring cells
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                
                        # Handle wrap-around for x (longitude)
                        nx = (x + dx) % self.size
                        # Clamp y to valid range (latitude)
                        ny = max(0, min(self.size - 1, y + dy))
                        
                        # Skip if already visited (prevents loops)
                        if (ny, nx) in visited:
                            continue
                
                        # Scoring system based on multiple factors
                        score = 0
                        
                        # Prefer downhill flow (most important factor)
                        if river_heightmap[ny, nx] < current_height:
                            # More points for steeper downhill (but don't penalize very steep drops)
                            height_diff = current_height - river_heightmap[ny, nx]
                            # Cap the height difference bonus to prevent avoiding very steep drops
                            capped_diff = min(height_diff, 0.2)
                            score += 10 + (capped_diff * 30)
                        else:
                            # Allow slightly more uphill movement
                            height_diff = river_heightmap[ny, nx] - current_height
                            if height_diff < 0.08:  # Increased from typical 0.05
                                score -= 5 + (height_diff * 80)  # Reduced penalty
                            else:
                                continue  # Skip significant uphills
                        
                        # Prefer high flow accumulation cells (natural drainage)
                        score += flow_accumulation[ny, nx] * 0.1
                        
                        # Add randomness for meandering
                        score += np.random.random() * meander_factor * 5
                        
                        # Avoid diagonal moves slightly (rivers tend to follow cardinal directions more)
                        if dx != 0 and dy != 0:
                            score -= 0.5
                        
                        # Add to options if overall score is positive
                        if score > 0:
                            neighbor_options.append((ny, nx, score))
                
                # No viable options found using regular criteria
                if not neighbor_options:
                    # Try punch-through if we have attempts left (allows crossing difficult terrain)
                    if punch_through_attempts > 0:
                        punch_through_attempts -= 1
                        
                        # Look for ANY lower neighbor within extended range
                        for r in range(1, 6):  # Increased search radius (6 instead of 4)
                            found_exit = False
                            for dy in range(-r, r+1):
                                for dx in range(-r, r+1):
                                    if dy == 0 and dx == 0:
                                        continue
                                    
                                    if abs(dy) == r or abs(dx) == r:  # Only check perimeter
                                        ny = max(0, min(self.size - 1, y + dy))
                                        nx = (x + dx) % self.size
                                        
                                        # Skip if already visited
                                        if (ny, nx) in visited:
                                            continue
                                        
                                        # Accept ANY lower terrain
                                        if river_heightmap[ny, nx] < current_height:
                                            # Mark all cells along the path
                                            # This simulates the river cutting through or having waterfalls
                                            path_points = self.generate_line_points(y, x, ny, nx)
                                            for py, px in path_points:
                                                px = px % self.size  # Handle wrapping
                                                py = max(0, min(self.size - 1, py))
                                                visited.add((py, px))
                                                river_path.append((py, px))
                                            
                                            # Carve a channel into the terrain for this segment
                                            # This ensures future rivers can flow through this path
                                            self.carve_river_segment(path_points, river_heightmap)
                                            
                                            y, x = ny, nx
                                            found_exit = True
                                            break
                                if found_exit:
                                    break
                            if found_exit:
                                break
                        
                        if found_exit:
                            continue
                    
                    # If punch-through failed or we're out of attempts, end the river
                    break
                
                # Choose the best option based on scores
                neighbor_options.sort(key=lambda n: n[2], reverse=True)
                
                # Usually take best path but occasionally take second-best for variety
                choice_idx = 0
                if len(neighbor_options) > 1 and np.random.random() < meander_factor * 0.5:
                    choice_idx = 1
                
                y, x, _ = neighbor_options[choice_idx]
                visited.add((y, x))
                river_path.append((y, x))
                path_length += 1
                
                # Check if we've reached water (existing river, lake, or ocean)
                if (river_mask[y, x] > 0 or 
                    river_heightmap[y, x] <= 0.05 or 
                    self.heightmap[y, x] < 0):
                    has_reached_water = True
                    break
                
                # Stop if we've reached the edge of the map (except for x which wraps)
                if y <= 1 or y >= self.size - 2:
                    break

            # Add river to the mask if it's either long enough or reached water
            valid_river = (path_length >= min_length) or (path_length >= min_length//2 and has_reached_water)
            if valid_river:
                rivers_created += 1
                
                # Store the river path
                river_paths.append(river_path)
            
                # Make rivers wider as they progress downstream
                for i, (py, px) in enumerate(river_path):
                    # Rivers get wider the further downstream they go
                    position_factor = i / len(river_path)
                    # Increase minimum width to 2 for better visibility
                    river_width = 2 + int(position_factor * 3)  # Start with width of 2
                    river_width = min(river_width, 5)  # Max width of 5
                
                    # Apply width with a circular brush
                    radius = river_width // 2
                    for wy in range(max(0, py-radius), min(self.size, py+radius+1)):
                        for wx in range(max(0, px-radius), min(self.size, px+radius+1)):
                            # Handle x wrap-around
                            wrapped_wx = wx % self.size
                        
                            # Use circular distance check
                            dy = wy - py
                            dx = min(abs(wrapped_wx - px), self.size - abs(wrapped_wx - px))
                            if (dy*dy + dx*dx) <= radius*radius:
                                river_mask[wy, wrapped_wx] = 1

        # Step 4: Add springs and minor streams in appropriate areas
        springs_added = 0
        spring_candidates = []
        
        # Identify good spring locations (elevation transitions, high moisture potential)
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                # Skip existing water
                if river_mask[y, x] > 0 or self.heightmap[y, x] < 0:
                    continue
            
                # Look for areas where elevation changes rapidly (foothills)
                local_variation = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                    
                        nx = (x + dx) % self.size
                        ny = max(0, min(self.size - 1, y + dy))
                    
                        local_variation += abs(self.heightmap[y, x] - self.heightmap[ny, nx])
            
                # Good spring locations: mid-elevation + high local variation + flow potential
                if 0.3 < self.heightmap[y, x] < 0.6 and local_variation > 0.2:
                    score = local_variation + (flow_accumulation[y, x] * 0.02)
                    spring_candidates.append((y, x, score))
        
        # Sort by score (best first)
        spring_candidates.sort(key=lambda s: s[2], reverse=True)
        
        # Ensure spatial diversity of springs
        min_spring_distance = self.size / 20
        diverse_springs = []
        
        for y, x, _ in spring_candidates:
            # Check if this spring is far enough from other water features
            too_close = False
            
            # Check distance to rivers
            for ry in range(max(0, y-10), min(self.size, y+11)):
                for rx in range(max(0, x-10), min(self.size, x+11)):
                    rx_wrapped = rx % self.size
                    if river_mask[ry, rx_wrapped] > 0:
                        dy = abs(ry - y)
                        dx = min(abs(rx_wrapped - x), self.size - abs(rx_wrapped - x))
                        distance = np.sqrt(dy*dy + dx*dx)
                        if distance < min_spring_distance * 0.5:  # Springs can be closer to each other
                            too_close = True
                            break
                if too_close:
                    break
                    
            # Check distance to other springs
            if not too_close:
                for sy, sx, _ in diverse_springs:
                    dx = min(abs(x - sx), self.size - abs(x - sx))
                    dy = abs(y - sy)
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance < min_spring_distance:
                        too_close = True
                        break
            
            if not too_close:
                diverse_springs.append((y, x, flow_accumulation[y, x]))
                
                # Create the spring
                radius = 1  # Width of 2
                for wy in range(max(0, y-radius), min(self.size, y+radius+1)):
                    for wx in range(max(0, x-radius), min(self.size, x+radius+1)):
                        wrapped_wx = wx % self.size
                        river_mask[wy, wrapped_wx] = 1
                
                # Extend the spring downhill
                stream_length = np.random.randint(3, 8)  # Variable length
                sy, sx = y, x
                stream_path = [(sy, sx)]
                
                for _ in range(stream_length):
                    lowest_dir = None
                    lowest_h = river_heightmap[sy, sx]
                    
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                                
                            ny = max(0, min(self.size - 1, sy + dy))
                            nx = (sx + dx) % self.size
                            
                            if (ny, nx) not in stream_path and river_heightmap[ny, nx] < lowest_h:
                                lowest_h = river_heightmap[ny, nx]
                                lowest_dir = (dy, dx)
                    
                    if not lowest_dir:
                        break
                        
                    dy, dx = lowest_dir
                    sy = max(0, min(self.size - 1, sy + dy))
                    sx = (sx + dx) % self.size
                    stream_path.append((sy, sx))
                    
                    # Mark the stream on the mask
                    river_mask[sy, sx] = 1
                    
                    # Stop if we hit water
                    if self.heightmap[sy, sx] < 0 or river_mask[sy, sx] > 0:
                        break
                
                springs_added += 1
                
                # Stop if we've added enough springs
                if springs_added >= river_count:
                    break

        # Store the river paths for later use
        self.river_paths = river_paths

        # Verify we have enough water features and add more if needed
        total_water_features = rivers_created + springs_added + forced_rivers_added
        
        # If we have significantly fewer rivers than expected, add more springs
        if total_water_features < river_count * 0.75:
            print("Adding additional water features to ensure sufficient coverage...")
            # Add more springs in suitable locations (less strict criteria)
            additional_springs = 0
            
            for _ in range(river_count):
                # Choose random point in mid-elevations
                attempts = 0
                while attempts < 20:
                    y = np.random.randint(5, self.size - 5)
                    x = np.random.randint(5, self.size - 5)
                    
                    # Check if this is a reasonable location for water
                    if (0.2 < river_heightmap[y, x] < 0.7 and 
                        river_mask[y, x] == 0 and 
                        self.heightmap[y, x] >= 0):
                        
                        # Create a small spring/stream
                        radius = 1
                        for wy in range(max(0, y-radius), min(self.size, y+radius+1)):
                            for wx in range(max(0, x-radius), min(self.size, x+radius+1)):
                                wrapped_wx = wx % self.size
                                river_mask[wy, wrapped_wx] = 1
                        
                        # Simple downhill path
                        cy, cx = y, x
                        for _ in range(np.random.randint(2, 6)):
                            # Find any lower neighbor
                            options = []
                            for dy in [-1, 0, 1]:
                                for dx in [-1, 0, 1]:
                                    if dy == 0 and dx == 0:
                                        continue
                                    
                                    ny = max(0, min(self.size - 1, cy + dy))
                                    nx = (cx + dx) % self.size
                                    
                                    if river_heightmap[ny, nx] < river_heightmap[cy, cx]:
                                        options.append((ny, nx))
                            
                            if options:
                                ny, nx = options[np.random.randint(0, len(options))]
                                river_mask[ny, nx] = 1
                                cy, cx = ny, nx
                            else:
                                break
                        
                        additional_springs += 1
                        break
                        
                    attempts += 1
                
                if additional_springs >= river_count // 4:
                    break
                    
            springs_added += additional_springs
        
        # Apply the river channels to the actual terrain
        # This ensures rivers have properly carved channels in the heightmap
        if river_paths:
            self.carve_river_channels(river_paths)
        
        total_water_features = rivers_created + springs_added + forced_rivers_added
        print(f"Successfully created {rivers_created} rivers from sources, {forced_rivers_added} rivers from coastlines, and {springs_added} springs")
        print(f"Total water features: {total_water_features}")
        return river_mask

    def generate_line_points(self, y0, x0, y1, x1):
        """Generate points in a line between (y0,x0) and (y1,x1) using Bresenham's algorithm.
        
        Args:
            y0, x0: Starting coordinates
            y1, x1: Ending coordinates
            
        Returns:
            List of (y, x) coordinates along the line
        """
        points = []
        steep = abs(y1 - y0) > abs(x1 - x0)
        
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
            
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = dx // 2
        y = y0
        
        if y0 < y1:
            ystep = 1
        else:
            ystep = -1
            
        for x in range(x0, x1 + 1):
            if steep:
                points.append((x, y))
            else:
                points.append((y, x))
                
            error -= dy
            if error < 0:
                y += ystep
                error += dx
                
        return points

    def carve_river_segment(self, path_points, heightmap):
        """Carve a river segment into the terrain to create a channel.
        
        Args:
            path_points: List of points forming the river segment
            heightmap: Heightmap to modify
        """
        # Calculate average height at beginning of segment
        if len(path_points) < 2:
            return
            
        start_y, start_x = path_points[0]
        end_y, end_x = path_points[-1]
        
        # Ensure the path slopes downward
        for i, (y, x) in enumerate(path_points):
            # Skip first point
            if i == 0:
                continue
                
            # Calculate position along the path (0 at start, 1 at end)
            position = i / (len(path_points) - 1)
            
            # Calculate target height (slightly decreasing downstream)
            start_height = heightmap[start_y, start_x]
            end_height = heightmap[end_y, end_x]
            
            # Ensure end is lower than start
            if end_height >= start_height:
                end_height = start_height - 0.05
                
            # Interpolate height
            target_height = start_height * (1-position) + end_height * position
            
            # Lower the terrain to create the channel
            heightmap[y, x] = target_height

    def carve_river_channels(self, river_paths):
        """Physically carve river channels into the terrain.
        
        Creates realistic river valleys that ensure water flows correctly.
        
        Args:
            river_paths: List of river paths, each containing (y, x) coordinates
        """
        if not river_paths:
            return
            
        # Create a copy of the heightmap
        carved_heightmap = self.heightmap.copy()
        
        for path in river_paths:
            # Skip very short paths
            if len(path) < 3:
                continue
                
            # Process each point in the river path
            for i, (y, x) in enumerate(path):
                # Skip ocean cells
                if carved_heightmap[y, x] < 0:
                    continue
                
                # Calculate downstream progression (0 at source, 1 at mouth)
                downstream_factor = i / (len(path) - 1)
                
                # Channel gets wider and deeper downstream
                # Width ranges from 1-5 cells
                width = 1 + int(downstream_factor * 4)
                
                # Depth ranges from slight to significant
                depth_factor = 0.05 + (downstream_factor * 0.15)
                
                # Carve the channel at this point
                for dy in range(-width, width+1):
                    for dx in range(-width, width+1):
                        # Calculate distance from river center
                        dist = np.sqrt(dy*dy + dx*dx)
                        if dist > width:
                            continue
                        
                        ny = max(0, min(self.size-1, y+dy))
                        nx = (x+dx) % self.size
                        
                        # Skip ocean cells
                        if carved_heightmap[ny, nx] < 0:
                            continue
                        
                        # Calculate depth factor based on distance from center
                        # Center is deepest, edges are higher
                        center_factor = 1.0 - (dist / width)
                        cell_depth = depth_factor * center_factor
                        
                        # Apply the carving (make sure river flows downhill)
                        if downstream_factor < 0.05:
                            # Near source: maintain some height
                            carved_heightmap[ny, nx] = min(carved_heightmap[ny, nx], 
                                                      carved_heightmap[ny, nx] * (1.0 - cell_depth * 0.5))
                        else:
                            # Ensure larger rivers carve deeper channels
                            target_height = carved_heightmap[y, x] - (cell_depth * 0.05)
                            carved_heightmap[ny, nx] = min(carved_heightmap[ny, nx], target_height)
                            
                            # Extra smoothing for channel banks
                            if dist > width * 0.7:
                                # Look for adjacent high points and smooth them
                                for bdy in range(-1, 2):
                                    for bdx in range(-1, 2):
                                        bny = max(0, min(self.size-1, ny+bdy))
                                        bnx = (nx+bdx) % self.size
                                        
                                        # If significant height difference, smooth it
                                        if carved_heightmap[bny, bnx] > carved_heightmap[ny, nx] + 0.1:
                                            carved_heightmap[bny, bnx] = carved_heightmap[ny, nx] + 0.05
        
        # Update the heightmap
        self.heightmap = carved_heightmap

    def add_lakes(self, count: int = 20, min_size: int = 8, max_size: int = 80,
                  ocean_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Add inland lakes to depressions in the terrain with improved placement.
        
        Args:
            count: Number of lakes to generate
            min_size: Minimum size for lakes
            max_size: Maximum size for lakes
            ocean_mask: Optional mask of ocean areas to avoid
            
        Returns:
            Lake mask as a 2D numpy array
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before adding lakes")

        print("Generating inland lakes...")

        # Create a lake mask
        lake_mask = np.zeros_like(self.heightmap)

        # Find local minima and suitable lake locations in the terrain
        lake_candidates = []
        
        # First look for depressions (local minima)
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                # Skip ocean areas if mask is provided
                if ocean_mask is not None and ocean_mask[y, x] > 0:
                    continue

                # Skip very low areas (likely ocean) or too high areas
                if self.heightmap[y, x] < 0 or self.heightmap[y, x] > 0.65:
                    continue

                # Check if lower than neighbors - true local minimum
                is_minimum = True
                height = self.heightmap[y, x]

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue

                        # Handle wrap-around for x (longitude)
                        nx = (x + dx) % self.size
                        # Clamp y to valid range (latitude)
                        ny = max(0, min(self.size - 1, y + dy))

                        if height > self.heightmap[ny, nx]:
                            is_minimum = False
                            break
                    if not is_minimum:
                        break

                if is_minimum:
                    # Score by local flatness (better for larger lakes)
                    flatness = 0.0
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            ny = max(0, min(self.size - 1, y + dy))
                            nx = (x + dx) % self.size
                            
                            if abs(self.heightmap[ny, nx] - height) <.02:
                                flatness += 1.0
                                
                    # Store as (y, x, height, score)
                    lake_candidates.append((y, x, height, flatness))
                    
        # Also look for plateaus and valleys that could hold lakes
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                # Skip ocean areas if mask is provided
                if ocean_mask is not None and ocean_mask[y, x] > 0:
                    continue

                # Look for plateau/valley areas (relatively flat terrain)
                if self.heightmap[y, x] < 0.2 or self.heightmap[y, x] > 0.6:
                    continue

                # Check if approximately flat compared to neighbors
                height = self.heightmap[y, x]
                flat_neighbors = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue

                        # Handle wrap-around for x (longitude)
                        nx = (x + dx) % self.size
                        # Clamp y to valid range (latitude)
                        ny = max(0, min(self.size - 1, y + dy))

                        # If height difference is small, count as flat
                        if abs(height - self.heightmap[ny, nx]) < 0.01:
                            flat_neighbors += 1

                # Need at least 5 flat neighbors to be considered a plateau/valley
                if flat_neighbors >= 5:
                    # Also check that we're not near an existing minimum
                    too_close = False
                    for ly, lx, _, _ in lake_candidates:
                        # Handle wrap-around for x (longitude)
                        dx = min(abs(x - lx), self.size - abs(x - lx))
                        dy = abs(y - ly)
                        distance = np.sqrt(dx*dx + dy*dy)

                        if distance < 5:  # Too close to another minimum
                            too_close = True
                            break

                    if not too_close:
                        # Add as potential lake location with flatness score
                        lake_candidates.append((y, x, height, flat_neighbors * 0.5))

        # Sort by score (higher scores first - these make better lakes)
        lake_candidates.sort(key=lambda m: m[3], reverse=True)
        
        # Try to create lakes at each potential location
        lakes_created = 0
        min_lake_distance = self.size / 15  # Minimum distance between lake centers
        created_lake_centers = []
        
        for y, x, height, _ in lake_candidates:
            # Skip if too close to another lake
            too_close = False
            for ly, lx in created_lake_centers:
                dx = min(abs(x - lx), self.size - abs(x - lx))
                dy = abs(y - ly)
                distance = np.sqrt(dx*dx + dy*dy)
                
                if distance < min_lake_distance:
                    too_close = True
                    break
                    
            if too_close:
                continue
                
            # Skip if this area already has a lake
            if lake_mask[y, x] == 1:
                continue

            # Use flood fill to create the lake
            queue = [(y, x)]
            lake_cells = set([(y, x)])

            # Random target size for this lake
            target_size = np.random.randint(min_size, max_size)

            # Maximum height for this lake (higher = larger lake)
            # More variance in lake sizes
            if np.random.random() < 0.3:  # 30% chance of a larger lake
                height_threshold = height + np.random.uniform(0.03, 0.06)
            else:
                height_threshold = height + np.random.uniform(0.01, 0.04)

            while queue and len(lake_cells) < target_size:
                cy, cx = queue.pop(0)

                # Add neighbors if they're below the threshold
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        # Handle wrap-around for x (longitude)
                        nx = (cx + dx) % self.size
                        # Clamp y to valid range (latitude)
                        ny = max(0, min(self.size - 1, cy + dy))

                        if (ny, nx) in lake_cells:
                            continue

                        # Skip if out of bounds
                        if not (0 <= ny < self.size):
                            continue

                        # Skip if this is ocean
                        if ocean_mask is not None and ocean_mask[ny, nx] > 0:
                            continue

                        if self.heightmap[ny, nx] <= height_threshold:
                            lake_cells.add((ny, nx))
                            queue.append((ny, nx))

            # Only create the lake if it's large enough
            if len(lake_cells) >= min_size:
                for ly, lx in lake_cells:
                    lake_mask[ly, lx] = 1

                    # Set the heightmap value to a consistent level for the lake
                    self.heightmap[ly, lx] = max(0, height - 0.01)

                lakes_created += 1
                created_lake_centers.append((y, x))

                # Stop if we've created enough lakes
                if lakes_created >= count:
                    break
                    
        # If we didn't create enough lakes, add some smaller ones in valleys
        if lakes_created < count * 0.75:
            # Find valley bottoms from the valley mask if available
            if hasattr(self, 'valley_mask') and self.valley_mask is not None:
                valley_points = []
                for y in range(1, self.size - 1):
                    for x in range(1, self.size - 1):
                        if (self.valley_mask[y, x] > 0.4 and 
                            self.heightmap[y, x] > 0 and 
                            self.heightmap[y, x] < 0.5 and
                            lake_mask[y, x] == 0):
                            
                            # Check if not too close to existing lakes
                            too_close = False
                            for ly, lx in created_lake_centers:
                                dx = min(abs(x - lx), self.size - abs(x - lx))
                                dy = abs(y - ly)
                                distance = np.sqrt(dx*dx + dy*dy)
                                
                                if distance < min_lake_distance * 0.5:
                                    too_close = True
                                    break
                                    
                            if not too_close:
                                valley_points.append((y, x))
                
                # Add some small valley lakes
                np.random.shuffle(valley_points)
                for y, x in valley_points[:count - lakes_created]:
                    # Create small lake (3-7 cells radius)
                    radius = np.random.randint(3, 8)
                    
                    for dy in range(-radius, radius+1):
                        for dx in range(-radius, radius+1):
                            dist = np.sqrt(dy*dy + dx*dx)
                            if dist > radius:
                                continue
                                
                            ny = max(0, min(self.size - 1, y + dy))
                            nx = (x + dx) % self.size
                            
                            # Check if on land
                            if self.heightmap[ny, nx] <= 0:
                                continue
                                
                            lake_mask[ny, nx] = 1
                            # Level the lake surface
                            self.heightmap[ny, nx] = max(0, self.heightmap[y, x] - 0.01)
                    
                    created_lake_centers.append((y, x))
                    lakes_created += 1
                    
                    if lakes_created >= count:
                        break

        print(f"Created {lakes_created} inland lakes")
        return lake_mask

    def generate_complete_terrain(self) -> np.ndarray:
        """Generate a complete terrain with all features applied.
        
        This is the primary method to call for generating a full terrain map.
        
        Returns:
            Complete heightmap with all features
        """
        print("\n=== Generating Complete Terrain ===")
        
        # Step 1: Generate the continental mask with improved distribution
        print("\n1. Generating continental landmasses with improved distribution...")
        self.create_continental_mask(continent_size=0.35)  # 35% land coverage
        
        # Step 2: Generate base heightmap
        print("\n2. Generating base terrain heightmap...")
        self.generate_heightmap()
        
        # Step 3: Generate valley patterns
        print("\n3. Adding dramatic valleys...")
        self.generate_valley_mask()
        
        # Step 4: Add mountains with deep valleys
        print("\n4. Adding mountain ranges with dramatic relief...")
        self.add_mountains(epic_factor=1.8)
        
        # Step 5: Add ridges and canyons
        print("\n5. Adding ridge lines and canyons...")
        self.add_ridges_and_canyons()
        
        # Step 6: Add plateau features
        print("\n6. Adding plateau features...")
        self.add_plateaus()
        
        # Step 7: Apply erosion to make terrain more realistic
        print("\n7. Applying erosion simulation...")
        self.apply_erosion(iterations=30)
        
        # Step 8: Generate water bodies
        print("\n8. Generating oceans and seas...")
        self.generate_water_bodies(water_coverage=0.65)
        
        # Determine ocean mask for other water features
        ocean_mask = np.zeros_like(self.heightmap)
        ocean_mask[self.heightmap < 0] = 1
        
        # Step 9: Add rivers
        print("\n9. Generating river networks...")
        river_mask = self.add_rivers(river_count=30)
        
        # Step 10: Add lakes
        print("\n10. Adding lakes in depressions and valleys...")
        lake_mask = self.add_lakes(count=20, ocean_mask=ocean_mask)
        
        # Final pass of erosion for smoother transitions
        print("\n11. Applying final terrain smoothing...")
        self.apply_erosion(iterations=10, erosion_strength=0.1)
        
        print("\nTerrain generation complete!")
        return self.heightmap

    def apply_erosion(self, iterations: int = 50, drop_rate: float = 0.05,
                      erosion_strength: float = 0.3) -> np.ndarray:
        """Apply enhanced hydraulic and thermal erosion to the heightmap.

        Args:
            iterations: Number of erosion iterations
            drop_rate: Rate at which water drops are applied
            erosion_strength: Strength of the erosion effect

        Returns:
            Eroded heightmap as a 2D numpy array
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before applying erosion")

        print(f"Applying enhanced erosion with {iterations} iterations...")

        eroded_map = self.heightmap.copy()

        # Generate flow direction map (pointing to lowest neighbor)
        flow_dir = np.zeros((self.size, self.size, 2), dtype=int)
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                # Skip ocean areas (negative values)
                if eroded_map[y, x] < 0:
                    continue

                # Find steepest descent neighbor
                min_height = eroded_map[y, x]
                min_dir = (0, 0)

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = y + dy, x + dx

                        # Handle wrap-around for longitude (east-west)
                        if nx < 0:
                            nx = self.size - 1
                        elif nx >= self.size:
                            nx = 0

                        # Clamp latitude (north-south)
                        ny = max(0, min(self.size - 1, ny))

                        if eroded_map[ny, nx] < min_height:
                            min_height = eroded_map[ny, nx]
                            min_dir = (dy, dx)

                flow_dir[y, x] = min_dir

        # Hydraulic erosion phase
        for _ in range(iterations):
            # Random water drops
            num_drops = int(self.size * self.size * drop_rate)
            for _ in range(num_drops):
                x, y = (np.random.randint(1, self.size - 1),
                        np.random.randint(1, self.size - 1))

                # Skip ocean areas
                if eroded_map[y, x] < 0:
                    continue

                # Water carries sediment
                sediment = 0.0
                path = []

                # Simulate water flow path
                for _ in range(30):  # Maximum path length
                    path.append((y, x))

                    dy, dx = flow_dir[y, x]
                    if dy == 0 and dx == 0:
                        break

                    # Handle wrap-around for x
                    nx = (x + dx) % self.size
                    # Clamp y to valid range
                    ny = max(0, min(self.size - 1, y + dy))

                    # Calculate height difference
                    h_diff = eroded_map[y, x] - eroded_map[ny, nx]

                    # Erode proportional to slope
                    if h_diff > 0:
                        # Erode more on steeper slopes
                        erode_amount = min(h_diff * erosion_strength, 0.01)
                        eroded_map[y, x] -= erode_amount
                        sediment += erode_amount
                    else:
                        # Deposit some sediment when slope decreases
                        deposit = sediment * 0.5
                        eroded_map[y, x] += deposit
                        sediment -= deposit

                    # Move to next position
                    x, y = nx, ny

                    # Stop at ocean or local minimum
                    if eroded_map[y, x] < 0:
                        break

                # Deposit remaining sediment along path
                if sediment > 0 and path:
                    deposit_per_cell = sediment / len(path)
                    for py, px in path:
                        eroded_map[py, px] += deposit_per_cell

        # Thermal weathering phase
        for _ in range(iterations // 2):
            for y in range(1, self.size - 1):
                for x in range(1, self.size - 1):
                    # Skip ocean areas
                    if eroded_map[y, x] < 0:
                        continue

                    # Calculate max slope to neighbors
                    max_slope = 0
                    max_slope_dir = (0, 0)

                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        # Handle wrap-around for x
                        nx = (x + dx) % self.size
                        # Clamp y to valid range
                        ny = max(0, min(self.size - 1, y + dy))

                        slope = eroded_map[y, x] - eroded_map[ny, nx]
                        if slope > max_slope:
                            max_slope = slope
                            max_slope_dir = (dy, dx)

                    # If slope exceeds talus angle, apply thermal erosion
                    talus_angle = 0.05  # Threshold for slope stability
                    if max_slope > talus_angle:
                        # Calculate material to move
                        move_amount = (max_slope - talus_angle) * 0.5

                        # Remove material from current cell
                        eroded_map[y, x] -= move_amount

                        # Deposit at downslope neighbor
                        dy, dx = max_slope_dir
                        ny = max(0, min(self.size - 1, y + dy))
                        nx = (x + dx) % self.size
                        eroded_map[ny, nx] += move_amount

        # Apply additional smoothing to create more gradual transitions
        smoothed_map = eroded_map.copy()
        for y in range(1, self.size-1):
            for x in range(1, self.size-1):
                # Skip water
                if eroded_map[y, x] < 0:
                    continue
                    
                # Calculate local average
                local_sum = 0
                local_count = 0
                max_diff = 0
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                            
                        ny = max(0, min(self.size-1, y+dy))
                        nx = (x+dx) % self.size
                        
                        # Skip water neighbors
                        if eroded_map[ny, nx] < 0:
                            continue
                            
                        local_sum += eroded_map[ny, nx]
                        local_count += 1
                        
                        # Track maximum elevation difference
                        diff = abs(eroded_map[y, x] - eroded_map[ny, nx])
                        max_diff = max(max_diff, diff)
                
                if local_count == 0:
                    continue
                    
                local_avg = local_sum / local_count
                
                # Apply stronger smoothing to cells with large elevation differences
                if max_diff > 0.1:
                    # More aggressive smoothing for steep transitions
                    smoothing_strength = min(0.3, max_diff)
                    smoothed_map[y, x] = eroded_map[y, x] * (1-smoothing_strength) + local_avg * smoothing_strength

        # Normalize if needed
        if np.min(smoothed_map) < -0.5 or np.max(smoothed_map) > 1.0:
            # Keep ocean depths negative
            ocean_mask = smoothed_map < 0

            # Normalize land areas to 0-1
            land = smoothed_map.copy()
            land[ocean_mask] = 0
            land_max = np.max(land)
            if land_max > 0:
                land = land / land_max

            # Normalize ocean areas to -0.5-0
            ocean = smoothed_map.copy()
            ocean[~ocean_mask] = 0
            ocean_min = np.min(ocean)
            if ocean_min < 0:
                ocean = ocean / (2 * abs(ocean_min)) * (-0.5)

            # Combine land and ocean
            smoothed_map = land + ocean

        self.heightmap = smoothed_map
        return smoothed_map

    def generate_water_bodies(self, water_coverage: float = 0.65) -> Tuple[np.ndarray, np.ndarray]:
        """Generate water bodies (oceans, lakes) based on the heightmap."""
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before creating water bodies")

        print(f"Generating water bodies with {water_coverage:.0%} water coverage...")

        # Make a copy of the heightmap for modification
        water_heightmap = self.heightmap.copy()

        # Sort heightmap values to find threshold
        flat_heightmap = water_heightmap.flatten()
        sorted_heights = np.sort(flat_heightmap)
        threshold_index = int(water_coverage * len(sorted_heights))
        sea_level = sorted_heights[threshold_index]

        print(f"Calculated sea level threshold: {sea_level:.4f}")

        # Create water mask (1 where water exists, 0 elsewhere)
        water_mask = np.zeros_like(water_heightmap)
        water_mask[water_heightmap < sea_level] = 1

        # Lower terrain below sea level to create negative values for oceans
        for y in range(self.size):
            for x in range(self.size):
                if water_mask[y, x] > 0:
                    # Normalize depths
                    depth_factor = (sea_level - water_heightmap[y, x]) / sea_level
                    # Make ocean depths negative - but not too extreme
                    water_heightmap[y, x] = -0.3 * depth_factor

        # Update the heightmap
        self.heightmap = water_heightmap

        # Calculate actual water coverage
        actual_coverage = np.sum(water_mask) / water_mask.size
        print(f"Actual water coverage: {actual_coverage:.2%}")

        return self.heightmap, water_mask

    def visualize(self, water_mask: Optional[np.ndarray] = None,
                  title: str = "Terrain Heightmap",
                  show_grid: bool = False):
        """Visualize the current heightmap with improved water representation.

        Args:
            water_mask: Optional mask indicating water bodies
            title: Title for the plot
            show_grid: Whether to show a grid indicating cell sizes
        """
        if self.heightmap is None:
            raise ValueError("No heightmap to visualize")

        plt.figure(figsize=(12, 10))

        # Create a custom colormap that handles negative values for ocean
        if water_mask is not None:
            # Create a copy to avoid modifying the original
            terrain_vis = self.heightmap.copy()

            # Create a visually appealing terrain map
            terrain_rgb = np.zeros((self.size, self.size, 4))

            for y in range(self.size):
                for x in range(self.size):
                    # Get current latitude for information
                    latitude = self.latitudes[y]
                    
                    if water_mask[y, x] > 0:
                        # Ocean depth (negative values)
                        # Map -0.5-0 to blues (darker for deeper)
                        depth = terrain_vis[y, x]
                        if depth >= 0:
                            # Shallow water (light blue)
                            terrain_rgb[y, x] = [0.6, 0.8, 1.0, 1.0]
                        else:
                            # Normalize depth to 0-1 (deeper = higher value)
                            norm_depth = min(1.0, -depth / 0.5)
                            # Darker blue for deeper water
                            blue = 0.7 - norm_depth * 0.5
                            terrain_rgb[y, x] = [0.0, 0.2 + blue * 0.3, 0.5 + blue * 0.5, 1.0]
                    else:
                        # Land elevation (positive values)
                        height = terrain_vis[y, x]
                        
                        # Polar regions (snow-capped)
                        if abs(latitude) > 70 and height > 0.3:
                            # Brighter white for higher elevations
                            snow_factor = min(1.0, (height - 0.3) * 3.0)
                            white_base = 0.8 + snow_factor * 0.2
                            terrain_rgb[y, x] = [white_base, white_base, white_base + 0.05, 1.0]
                            
                        elif height < 0.1:
                            # Coast/Beach (tan/yellow)
                            terrain_rgb[y, x] = [0.9, 0.8, 0.6, 1.0]
                        elif height < 0.3:
                            # Lowlands (green)
                            norm_height = (height - 0.1) / 0.2
                            terrain_rgb[y, x] = [0.2 + norm_height * 0.4, 0.7 - norm_height * 0.2, 0.2, 1.0]
                        elif height < 0.7:
                            # Hills/Highlands (brown/tan)
                            norm_height = (height - 0.3) / 0.4
                            terrain_rgb[y, x] = [0.6 + norm_height * 0.2, 0.5 - norm_height * 0.2, 0.2, 1.0]
                        else:
                            # Mountains (white/gray)
                            norm_height = (height - 0.7) / 0.3
                            white = 0.7 + norm_height * 0.3
                            terrain_rgb[y, x] = [white, white, white, 1.0]

            plt.imshow(terrain_rgb)
        else:
            # Use a standard colormap if no water mask
            plt.imshow(self.heightmap, cmap="terrain")

        # Show scale information
        scale_text = f"World Scale: {self.earth_scale:.2%} of Earth\n"
        scale_text += f"Grid Cell: {self.km_per_cell:.1f} km × "
        scale_text += f"{self.km_per_cell:.1f} km\n"
        scale_text += f"Cell Area: {self.area_per_cell_sqmiles:.1f} sq miles"

        # Place scale info in the upper left
        plt.annotate(scale_text,
                    (0.02, 0.98),
                    xycoords="figure fraction",
                    verticalalignment="top",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white",
                              ec="black",
                              alpha=0.8))

        # If requested, add grid for scale reference
        if show_grid and self.size > 20:
            # Draw grid lines every 20% of the way across
            grid_step = max(1, self.size // 5)
            for i in range(0, self.size + 1, grid_step):
                plt.axhline(i, color="white", alpha=0.3, linestyle=":")
                plt.axvline(i, color="white", alpha=0.3, linestyle=":")

            # Label a few key points
            for i in range(0, self.size + 1, grid_step):
                plt.text(i, -5, f"{int(i * self.km_per_cell)} km",
                        color="black", ha="center", fontsize=8)
                plt.text(-5, i, f"{int(i * self.km_per_cell)} km",
                        color="black", va="center", fontsize=8)
                
                # Also show latitude labels
                lat_val = self.latitudes[min(i, self.size-1)] if i < self.size else -90
                plt.text(self.size + 5, i, f"{lat_val:.0f}°",
                        color="black", va="center", fontsize=8)

        # Add a custom colorbar that shows both ocean depths and land heights
        if water_mask is not None:
            # Create a custom colorbar
            ax = plt.gca()
            cax = plt.colorbar(
                plt.cm.ScalarMappable(
                    norm=plt.Normalize(-0.5, 1.0),
                    cmap=plt.cm.terrain
                ), 
                ax=ax,
                label="Elevation"
            )
            # Add specific tick marks for both ocean and land
            cax.set_ticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
            cax.set_ticklabels(["-4000m", "-2000m", "Sea Level", "+500m", "+2000m", "+4000m", "+8000m"])
        else:
            plt.colorbar(label="Elevation")

        plt.title(title)
        plt.axis("off" if not show_grid else "on")
        plt.tight_layout()
        plt.show()

    def visualize_water_system(self, water_systems: Dict[str, np.ndarray],
                          title: str = "Complete Water System"):
        """Visualize water systems with different colors for each type."""
        if self.heightmap is None:
            raise ValueError("No heightmap to visualize")

        plt.figure(figsize=(12, 10))

        # Create a copy of the heightmap for visualization
        terrain_vis = self.heightmap.copy()

        # Create an RGB image
        rgb_image = np.zeros((self.size, self.size, 4))

        # Base terrain colors (using terrain colormap)
        terrain_colors = plt.cm.gist_earth(terrain_vis)

        # Start with terrain base
        rgb_image = terrain_colors.copy()

        # Apply ocean/sea (deep blue)
        if "ocean" in water_systems:
            for y in range(self.size):
                for x in range(self.size):
                    if water_systems["ocean"][y, x] > 0:
                        # Deep blue for oceans
                        rgb_image[y, x] = [0.0, 0.1, 0.5, 1.0]

        # Apply lakes (lighter blue)
        if "lakes" in water_systems:
            for y in range(self.size):
                for x in range(self.size):
                    if water_systems["lakes"][y, x] > 0:
                        # Lighter blue for lakes
                        rgb_image[y, x] = [0.2, 0.4, 0.8, 1.0]

        # Apply rivers last (bright blue with higher contrast for visibility)
        if "rivers" in water_systems:
            for y in range(self.size):
                for x in range(self.size):
                    if water_systems["rivers"][y, x] > 0:
                        # Brighter, more distinct blue for rivers
                        rgb_image[y, x] = [0.0, 0.7, 1.0, 1.0]  # More vibrant blue

        plt.imshow(rgb_image)

        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=[0.0, 0.1, 0.5, 1.0],
                       markersize=10, label="Ocean"),
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=[0.2, 0.4, 0.8, 1.0],
                       markersize=10, label="Lakes"),
            plt.Line2D([0], [0], marker="s", color="w",
                       markerfacecolor=[0.0, 0.7, 1.0, 1.0],  # Updated river color
                       markersize=10, label="Rivers")
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        # Show scale information
        scale_text = f"World Scale: {self.earth_scale:.2%} of Earth\n"
        scale_text += f"Grid Cell: {self.km_per_cell:.1f} km × "
        scale_text += f"{self.km_per_cell:.1f} km\n"
        scale_text += f"Cell Area: {self.area_per_cell_sqmiles:.1f} sq miles"

        plt.annotate(scale_text, (0.02, 0.98), xycoords="figure fraction",
                    verticalalignment="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="black",
                              alpha=0.8))

        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    def visualize_feature_map(self, title: str = "Terrain Feature Map"):
        """Visualize a colored map showing different terrain features."""
        if self.heightmap is None:
            raise ValueError("No heightmap to visualize")
            
        plt.figure(figsize=(12, 10))
        
        # Create a feature classification map
        feature_map = np.zeros((self.size, self.size, 3))
        
        # Feature detection based on heightmap and derivatives
        for y in range(self.size):
            for x in range(self.size):
                height = self.heightmap[y, x]
                
                # Calculate local slope
                local_slope = 0
                neighbor_count = 0
                
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                            
                        ny = max(0, min(self.size - 1, y + dy))
                        nx = (x + dx) % self.size
                        
                        local_slope += abs(height - self.heightmap[ny, nx])
                        neighbor_count += 1
                
                avg_slope = local_slope / neighbor_count if neighbor_count > 0 else 0
                
                # Feature classification
                if height < 0:
                    # Ocean - darker blue for deeper
                    depth_factor = min(1.0, -height / 0.5)
                    feature_map[y, x] = [0.0, 0.1 + (0.2 * (1-depth_factor)), 0.4 + (0.3 * (1-depth_factor))]
                    
                elif height < 0.05:
                    # Beaches/coastlines - yellow
                    feature_map[y, x] = [0.95, 0.85, 0.55]
                    
                elif height < 0.3:
                    if avg_slope < 0.01:
                        # Plains - light green
                        feature_map[y, x] = [0.4, 0.8, 0.4]
                    else:
                        # Rolling hills - darker green
                        feature_map[y, x] = [0.2, 0.6, 0.3]
                        
                elif height < 0.6:
                    if avg_slope > 0.04:
                        # Rugged highlands - brown
                        feature_map[y, x] = [0.6, 0.4, 0.2]
                    elif hasattr(self, 'valley_mask') and self.valley_mask is not None and self.valley_mask[y, x] > 0.3:
                        # Valleys - olive green
                        feature_map[y, x] = [0.5, 0.5, 0.2]
                    else:
                        # Hills - light brown
                        feature_map[y, x] = [0.7, 0.5, 0.3]
                        
                elif height < 0.8:
                    # Low mountains - dark gray
                    feature_map[y, x] = [0.4, 0.4, 0.4]
                    
                else:
                    # High mountains - white to gray based on height
                    white_factor = (height - 0.8) / 0.2  # 0 to 1
                    base_color = 0.6 + (white_factor * 0.4)  # 0.6 to 1.0
                    feature_map[y, x] = [base_color, base_color, base_color]
        
        plt.imshow(feature_map)
        
        # Add legend for features
        legend_elements = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=[0.0, 0.3, 0.7], markersize=10, label="Ocean"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=[0.95, 0.85, 0.55], markersize=10, label="Beaches"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=[0.4, 0.8, 0.4], markersize=10, label="Plains"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=[0.2, 0.6, 0.3], markersize=10, label="Rolling Hills"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=[0.7, 0.5, 0.3], markersize=10, label="Hills"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=[0.5, 0.5, 0.2], markersize=10, label="Valleys"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=[0.6, 0.4, 0.2], markersize=10, label="Highlands"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=[0.4, 0.4, 0.4], markersize=10, label="Low Mountains"),
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=[1.0, 1.0, 1.0], markersize=10, label="High Peaks")
        ]
        plt.legend(handles=legend_elements, loc="upper right")
        
        # Show scale information
        scale_text = f"World Scale: {self.earth_scale:.2%} of Earth\n"
        scale_text += f"Grid Cell: {self.km_per_cell:.1f} km × "
        scale_text += f"{self.km_per_cell:.1f} km\n"
        scale_text += f"Cell Area: {self.area_per_cell_sqmiles:.1f} sq miles"

        plt.annotate(scale_text, (0.02, 0.98), xycoords="figure fraction",
                    verticalalignment="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.3",
                              fc="white", ec="black",
                              alpha=0.8))
                              
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
    def save_heightmap(self, filename: str = "heightmap.npy") -> None:
        """Save the heightmap to a file.

        Args:
            filename: File to save the heightmap to
        """
        if self.heightmap is None:
            raise ValueError("No heightmap to save")

        np.save(filename, self.heightmap)
        print(f"Heightmap saved to {filename}")

    def load_heightmap(self, filename: str) -> np.ndarray:
        """Load a heightmap from a file.

        Args:
            filename: File to load the heightmap from

        Returns:
            Loaded heightmap
        """
        self.heightmap = np.load(filename)
        print(f"Heightmap loaded from {filename}")
        return self.heightmap
        
    def generate_complete_water_system(self,
                                       ocean_coverage: float = 0.65,
                                       river_count: int = 25,
                                       lake_count: int = 15) -> Tuple[np.ndarray,
                                                                      Dict[str,
                                                                           np.ndarray]]:
        """Generate a complete water system with oceans, rivers, and lakes.

        Args:
            ocean_coverage: Target ocean coverage (0.0-1.0)
            river_count: Number of major rivers to generate
            lake_count: Number of lakes to generate

        Returns:
            Tuple of (combined water mask, individual water feature masks)
        """
        print("\n=== Generating Complete Water System ===")

        # Apply erosion before adding water
        print("\n1. Applying realistic erosion to terrain...")
        self.apply_erosion(iterations=30, erosion_strength=0.2)

        # Generate oceans and seas
        print("\n2. Generating oceans and seas...")
        _, ocean_mask = self.generate_water_bodies(water_coverage=ocean_coverage)

        # Generate rivers using the watershed approach
        print("\n3. Generating rivers using improved water flow analysis...")
        river_mask = self.add_rivers(river_count=river_count, min_length=20)

        # Generate lakes, making sure they don't overlap with oceans
        print("\n4. Generating lakes in terrain depressions...")
        lake_mask = self.add_lakes(count=lake_count, min_size=10, max_size=100,
                                   ocean_mask=ocean_mask)

        # Apply a final erosion pass to smooth out all features
        print("\n5. Applying final terrain smoothing...")
        self.apply_erosion(iterations=10, erosion_strength=0.1)

        # Update the ocean mask based on current heightmap
        ocean_mask = np.zeros_like(self.heightmap)
        ocean_mask[self.heightmap < 0] = 1

        # Combine all water features
        combined_water_mask = np.zeros_like(self.heightmap)
        combined_water_mask[ocean_mask > 0] = 1
        combined_water_mask[river_mask > 0] = 1
        combined_water_mask[lake_mask > 0] = 1

        # Calculate actual water coverage
        total_water = np.sum(combined_water_mask)
        water_percentage = total_water / combined_water_mask.size

        print("\nWater System Statistics:")
        print(f"  Ocean coverage: {np.sum(ocean_mask) / ocean_mask.size:.2%}")
        print(f"  River cells: {np.sum(river_mask)} "
              f"({np.sum(river_mask) / river_mask.size:.2%})")
        print(f"  Lake cells: {np.sum(lake_mask)} "
              f"({np.sum(lake_mask) / lake_mask.size:.2%})")
        print(f"  Total water coverage: {water_percentage:.2%}")

        # Return the combined mask and individual feature masks
        water_systems = {
            "ocean": ocean_mask,
            "rivers": river_mask,
            "lakes": lake_mask
        }

        return combined_water_mask, water_systems