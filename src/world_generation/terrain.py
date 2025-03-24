"""Terrain generation module for EmergenWorld.

This module handles the creation of realistic terrain
using various noise algorithms and provides methods for
terrain manipulation and feature generation.
"""

import numpy as np
from opensimplex import OpenSimplex
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


class TerrainGenerator:
    """Generates and manipulates terrain heightmaps
       for the EmergenWorld simulation.

    Uses noise algorithms to create realistic terrain
    features such as mountains, valleys, and plateaus.
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
                         (doesn't affect physical calculations)
        """
        self.size = size
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.heightmap = None

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
        self.area_per_cell_sqmiles = (self.area_per_cell_sqkm *
                                     0.386102)  # Convert to sq miles

        print(f"Initialized terrain generator "
              f"for world at {earth_scale:.4%} of Earth's size")
        print(f"World radius: {self.scaled_radius_km:.1f} km")
        print(f"Each grid cell represents {self.km_per_cell:.2f} km "
              f"({self.area_per_cell_sqmiles:.2f} sq miles)")
        print(f"Note: Physics calculations use full Earth-sized planet properties")

    def generate_heightmap(self, scale: float = 100.0) -> np.ndarray:
        """Generate a heightmap using simplex noise with proper spherical wrapping.

        Args:
            scale: Scale factor for noise generation

        Returns:
            2D numpy array representing the heightmap
        """
        print(f"Generating simplex noise heightmap of size {self.size}x{self.size}...")

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

        # Apply continental influence to create realistic landmasses
        continental_mask = self.create_continental_mask()

        # Apply the mask to influence heightmap
        for y in range(self.size):
            for x in range(self.size):
                # Blend heightmap with continental influence
                heightmap[y, x] = heightmap[y, x] * 0.4 + continental_mask[y, x] * 0.6

        # Normalize again after continental influence
        heightmap = (heightmap - np.min(heightmap)) / (np.max(heightmap) - np.min(heightmap))

        # Save original heightmap for reference
        original_heightmap = heightmap.copy()

        # Define the sea level threshold (approximately where oceans will be)
        sea_level = 0.3

        # Apply different transformations to different height ranges
        for y in range(self.size):
            for x in range(self.size):
                h = original_heightmap[y, x]

                if h < sea_level:
                    # More dramatic ocean depths
                    # Map 0.0-0.3 to -0.5-0.0 range for ocean depths
                    normalized_depth = h / sea_level
                    # Use exponential curve for deeper ocean basins
                    heightmap[y, x] = -0.5 * (1.0 - normalized_depth**2)
                elif h < 0.7:
                    # Expand the middle range to create more moderate terrain
                    # Map 0.3-0.7 to 0.0-0.6
                    normalized_h = (h - sea_level) / (0.7 - sea_level)
                    heightmap[y, x] = normalized_h * 0.6
                else:
                    # Preserve high peaks (0.7-1.0) but compress slightly
                    # Map 0.7-1.0 to 0.6-1.0 to maintain epic mountains
                    normalized_h = (h - 0.7) / 0.3
                    heightmap[y, x] = 0.6 + normalized_h * 0.4

        self.heightmap = heightmap
        return heightmap

    def create_continental_mask(self, continent_size: float = 0.2,
                                continent_count: int = 3) -> np.ndarray:
        """Generate continent masks to create more realistic landmass distribution.

        Args:
            continent_size: Relative size of continents (0.0-1.0)
            continent_count: Number of major continent blobs

        Returns:
            Continental influence mask (0.0-1.0)
        """
        print("Generating continental influence map...")

        continental_mask = np.zeros((self.size, self.size))
        noise_gen = OpenSimplex(seed=self.seed + 42)  # Different seed for continents

        # Generate base continental noise with different parameters
        for y in range(self.size):
            for x in range(self.size):
                # Convert grid coordinates to cylindrical coordinates for proper wrapping
                lon = (x / self.size) * 2 * np.pi
                lat = ((y / self.size) * np.pi) - (np.pi / 2)

                # Use 3D noise with cylindrical coordinates
                nx = np.cos(lon) * 3.0  # Larger scale for continent shapes
                nz = np.sin(lon) * 3.0
                ny = lat * 3.0

                # Use just 2-3 octaves for broad continental shapes
                value = 0.0
                amplitude = 1.0
                frequency = 1.0

                for _ in range(2):
                    value += noise_gen.noise3(nx * frequency, ny * frequency, nz * frequency) * amplitude
                    amplitude *= 0.5
                    frequency *= 2.0

                continental_mask[y, x] = value

        # Normalize to 0-1
        continental_mask = (continental_mask - np.min(continental_mask)) / (np.max(continental_mask) - np.min(continental_mask))

        # Apply threshold to create distinct continental areas
        # This creates binary continent areas
        binary_continents = (continental_mask > (1.0 - continent_size)).astype(float)

        # Use distance field to create coastal gradients
        from scipy.ndimage import distance_transform_edt
        distance = distance_transform_edt(1 - binary_continents)
        max_distance = np.max(distance)
        if max_distance > 0:
            # Create a gradient that falls off with distance from land
            coastal_gradient = 1.0 - (distance / max_distance)

            # Blend gradient with binary mask
            continental_mask = np.maximum(binary_continents, coastal_gradient * 0.7)

        # Apply a slight overall curve to create higher elevations towards continent centers
        for y in range(self.size):
            for x in range(self.size):
                if binary_continents[y, x] > 0:
                    # Calculate distance from edge of continent
                    dist_from_edge = min(x, self.size - x, y, self.size - y) / (self.size * 0.5)
                    # Add a slight elevation boost towards center of continent
                    continental_mask[y, x] *= (1.0 + 0.2 * dist_from_edge)

        # Final normalization
        continental_mask = (continental_mask - np.min(continental_mask)) / (np.max(continental_mask) - np.min(continental_mask))

        return continental_mask

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

        # Normalize if needed
        if np.min(eroded_map) < -0.5 or np.max(eroded_map) > 1.0:
            # Keep ocean depths negative
            ocean_mask = eroded_map < 0

            # Normalize land areas to 0-1
            land = eroded_map.copy()
            land[ocean_mask] = 0
            land_max = np.max(land)
            if land_max > 0:
                land = land / land_max

            # Normalize ocean areas to -0.5-0
            ocean = eroded_map.copy()
            ocean[~ocean_mask] = 0
            ocean_min = np.min(ocean)
            if ocean_min < 0:
                ocean = ocean / (2 * abs(ocean_min)) * (-0.5)

            # Combine land and ocean
            eroded_map = land + ocean

        self.heightmap = eroded_map
        return eroded_map

    def generate_water_bodies(self,
                              water_coverage: float = 0.65) -> Tuple[np.ndarray, np.ndarray]:
        """Generate water bodies (oceans, lakes) based on the heightmap.

        Args:
            water_coverage: Target water coverage percentage (0.0 - 1.0),
                            default 0.65 for Earth-like coverage

        Returns:
            Tuple of (updated heightmap, water mask)
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before creating water bodies")

        print(f"Generating water bodies with {water_coverage:.0%} water coverage (Earth-like)...")

        # Find sea level threshold that gives the desired water coverage
        # Sort heightmap values and find the threshold
        flat_heightmap = self.heightmap.flatten()
        sorted_heights = np.sort(flat_heightmap)
        threshold_index = int(water_coverage * len(sorted_heights))
        sea_level = sorted_heights[threshold_index]

        print(f"Calculated sea level threshold: {sea_level:.4f}")

        # Create water mask (1 where water exists, 0 elsewhere)
        water_mask = np.zeros_like(self.heightmap)
        water_mask[self.heightmap < sea_level] = 1

        # Generate ocean depth map
        ocean_depths = self.heightmap.copy()
        for y in range(self.size):
            for x in range(self.size):
                if water_mask[y, x] > 0:
                    # Normalize depths based on distance from sea level
                    normalized_depth = (sea_level - ocean_depths[y, x]) / sea_level
                    # Use exponential curve for more realistic ocean basins
                    ocean_depths[y, x] = -0.5 * normalized_depth**1.5
                else:
                    # Leave land elevations as-is
                    pass

        # Update heightmap with ocean depths
        self.heightmap = ocean_depths

        # Calculate actual water coverage
        actual_coverage = np.sum(water_mask) / water_mask.size
        print(f"Actual water coverage: {actual_coverage:.2%}")

        return self.heightmap, water_mask

    def add_mountains(self, mountain_scale: float = 1.2,
                      peak_threshold: float = 0.6,
                      epic_factor: float = 2.0) -> np.ndarray:
        """Add epic fantasy-style mountain ranges to the terrain.

        Args:
            mountain_scale: Scale factor for mountain height
            peak_threshold: Threshold for mountain peaks
                            (lower = more mountains)
            epic_factor: Multiplier for mountain epicness
                         (higher = more dramatic)

        Returns:
            Updated heightmap with epic mountain ranges
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated "
                             "before adding mountains")

        print("Adding EPIC fantasy mountain ranges "
              f"(epicness factor: {epic_factor:.1f})...")

        # Generate a mountain mask using different noise parameters
        mountain_mask = np.zeros((self.size, self.size))
        # Different seed for mountain generation
        mountain_seed = self.seed + 1000
        mountain_noise = OpenSimplex(seed=mountain_seed)

        for y in range(self.size):
            for x in range(self.size):
                nx = x / 200
                ny = y / 200
                # Use 3 octaves for the mountain mask
                value = 0.0
                amplitude = 1.0
                frequency = 1.0

                for _ in range(3):  # Fewer octaves for mountain mask
                    value += mountain_noise.noise2(nx * frequency,
                                                   ny * frequency) * amplitude
                    amplitude *= 0.6  # Different persistence for mountain mask
                    frequency *= 2.5  # Different lacunarity for mountain mask

                mountain_mask[y][x] = value

        # Normalize mountain mask
        mountain_mask = (mountain_mask - np.min(mountain_mask)) / (
            np.max(mountain_mask) - np.min(mountain_mask))

        # Apply mountains where the mask exceeds the threshold
        mountain_terrain = self.heightmap.copy()
        mountain_areas = mountain_mask > peak_threshold

        # For epic fantasy mountains: sharper peaks,
        # more dramatic height changes
        # First, apply basic mountain height increase
        mountain_terrain[mountain_areas] += (
            mountain_scale * (mountain_mask[mountain_areas] - peak_threshold))

        # Then add epic peaks by emphasizing the highest portions
        high_peaks = ((mountain_mask > peak_threshold + 0.2) &
                      (mountain_terrain > 0.7))
        if np.any(high_peaks):
            # Apply exponential height increase for the highest peaks
            peak_height = mountain_terrain[high_peaks]
            mountain_terrain[high_peaks] = (
                peak_height + (epic_factor * (peak_height - 0.7) ** 2))

            print(f"Created {np.sum(high_peaks)} epic mountain peaks!")

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

        # Renormalize if needed
        if np.max(mountain_terrain) > 1.0:
            mountain_terrain = mountain_terrain / np.max(mountain_terrain)

        self.heightmap = mountain_terrain
        return mountain_terrain

    def add_rivers(self, river_count: int = 20,
                   min_length: int = 20,
                   meander_factor: float = 0.3) -> np.ndarray:
        """Add rivers flowing from high elevation to the sea using watershed analysis.

        Args:
            river_count: Number of major rivers to generate
            min_length: Minimum length of rivers to keep
            meander_factor: How much rivers should meander (0.0-1.0)

        Returns:
            River mask where 1 indicates river cells
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before adding rivers")

        print(f"Generating {river_count} major rivers with watershed analysis...")

        # Create a river mask (1 where rivers exist, 0 elsewhere)
        river_mask = np.zeros_like(self.heightmap)

        # Create flow accumulation map to model watersheds
        flow_accum = np.zeros_like(self.heightmap)

        # First calculate flow direction for each cell (D8 algorithm)
        flow_dir = np.zeros((self.size, self.size, 2), dtype=int)

        # Calculate flow directions
        for y in range(self.size):
            for x in range(self.size):
                # Skip ocean areas
                if self.heightmap[y, x] < 0:
                    continue

                # Find steepest downhill neighbor
                min_height = self.heightmap[y, x]
                min_dir = (0, 0)  # No flow

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue

                        # Handle wrap-around for longitude (x)
                        nx = (x + dx) % self.size
                        # Clamp latitude (y) to valid range
                        ny = max(0, min(self.size - 1, y + dy))

                        if self.heightmap[ny, nx] < min_height:
                            min_height = self.heightmap[ny, nx]
                            min_dir = (dy, dx)

                flow_dir[y, x] = min_dir
    
        # Calculate flow accumulation
        # Initialize each cell with 1 unit of water
        flow_accum += 1.0

        # Multiple passes to propagate flow downstream
        for _ in range(5):  # More passes = more accurate watersheds
            # Create a buffer to avoid double-counting within a pass
            flow_accum_buffer = flow_accum.copy()

            for y in range(self.size):
                for x in range(self.size):
                    dy, dx = flow_dir[y, x]
                    if dy != 0 or dx != 0:  # If there's flow from this cell
                        # Handle wrap-around for longitude (x)
                        nx = (x + dx) % self.size
                        # Clamp latitude (y) to valid range
                        ny = max(0, min(self.size - 1, y + dy))

                        # Add this cell's flow to the downstream cell
                        flow_accum_buffer[ny, nx] += flow_accum[y, x]
        
            # Update the accumulation map
            flow_accum = flow_accum_buffer

        # Find high elevation points for river sources (excluding the very edges)
        border = 5
        interior_heightmap = self.heightmap[border:-border, border:-border]
        # We want both high elevation and moderate flow accumulation for sources
        high_points = []

        for y in range(border, self.size - border):
            for x in range(border, self.size - border):
                # Skip ocean areas
                if self.heightmap[y, x] < 0:
                    continue

                # We want high elevation areas with some flow accumulation
                if self.heightmap[y, x] > 0.7 and flow_accum[y, x] > 1.5:
                    high_points.append((y, x, flow_accum[y, x]))

        if not high_points:
            print("Warning: No suitable high points found for river sources")
            return river_mask

        # Sort by flow accumulation and pick the top points
        high_points.sort(key=lambda p: p[2], reverse=True)
        source_points = high_points[:min(len(high_points), river_count*2)]

        # Now apply some spatial diversity - don't want all rivers in same area
        diverse_sources = []
        min_distance = self.size / 10  # Minimum distance between sources

        for y, x, _ in source_points:
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
                diverse_sources.append((y, x, flow_accum[y, x]))

                # Stop once we have enough diverse sources
                if len(diverse_sources) >= river_count:
                    break

        rivers_created = 0

        # Generate each river using the flow direction map
        for y, x, _ in diverse_sources[:river_count]:
            river_path = [(y, x)]

            # Follow the flow direction map downhill
            for _ in range(self.size * 2):  # Maximum river length
                dy, dx = flow_dir[y, x]
                if dy == 0 and dx == 0:
                    # No flow direction, we've reached a sink
                    break

                # Handle wrap-around for longitude (x)
                nx = (x + dx) % self.size
                # Clamp latitude (y) to valid range
                ny = max(0, min(self.size - 1, y + dy))

                # Add meandering based on elevation and random factors
                if np.random.random() < meander_factor:
                    # Try to find an alternative path
                    alternatives = []
                    for mdy in [-1, 0, 1]:
                        for mdx in [-1, 0, 1]:
                            if mdy == 0 and mdx == 0:
                                continue

                            # Handle wrap-around for longitude (x)
                            mnx = (x + mdx) % self.size
                            # Clamp latitude (y) to valid range
                            mny = max(0, min(self.size - 1, y + mdy))

                            # Consider this path if it leads downhill
                            if self.heightmap[mny, mnx] < self.heightmap[y, x]:
                                # Score based on steepness and flow accumulation
                                steepness = self.heightmap[y, x] - self.heightmap[mny, mnx]
                                flow = flow_accum[mny, mnx]
                                score = steepness * 0.5 + flow * 0.5
                                alternatives.append((mny, mnx, score))

                    if alternatives:
                        # Sort by score and pick a good alternative
                        alternatives.sort(key=lambda a: a[2], reverse=True)
                        # Pick one of the top alternatives
                        idx = min(int(np.random.random() * 3), len(alternatives) - 1)
                        ny, nx, _ = alternatives[idx]

                # Check if we've hit an existing river
                if river_mask[ny, nx] > 0:
                    river_path.append((ny, nx))
                    break

                # Check if we've reached the ocean
                if self.heightmap[ny, nx] < 0:
                    river_path.append((ny, nx))
                    break

                # Add to the river path
                river_path.append((ny, nx))

                # Continue downstream
                y, x = ny, nx

            # Add river to mask if it's long enough
            if len(river_path) >= min_length:
                rivers_created += 1

                # Make rivers wider based on flow accumulation
                for i, (py, px) in enumerate(river_path):
                    # Rivers get wider as they flow downstream
                    position_factor = i / len(river_path)
                    flow_factor = min(flow_accum[py, px] / 100, 10)  # Cap the max width
                    river_width = 1 + int(position_factor * flow_factor)

                    # Draw the river with variable width
                    if river_width <= 1:
                        river_mask[py, px] = 1
                    else:
                        # Circular brush for wider rivers
                        half_width = river_width // 2
                        for wy in range(max(0, py-half_width), min(self.size, py+half_width+1)):
                            for wx in range(max(0, px-half_width), min(self.size, px+half_width+1)):
                                # Use wrap-around for x
                                wrapped_wx = wx % self.size
                                # Circular shape
                                if ((wy - py)**2 + min(abs(wrapped_wx - px), self.size - abs(wrapped_wx - px))**2 <= half_width**2):
                                    river_mask[wy, wrapped_wx] = 1

        print(f"Successfully created {rivers_created} major rivers")
        return river_mask

    def add_lakes(self, count: int = 15,
                  min_size: int = 10,
                  max_size: int = 100,
                  ocean_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Add inland lakes to depressions in the terrain with more natural shapes.

        Args:
            count: Target number of lakes to generate
            min_size: Minimum lake size in cells
            max_size: Maximum lake size in cells
            ocean_mask: Optional mask of ocean areas to avoid

        Returns:
            Lake mask where 1 indicates lake cells
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before adding lakes")

        print("Generating inland lakes with natural shapes...")

        # Create a lake mask
        lake_mask = np.zeros_like(self.heightmap)

        # Find local minima in the terrain
        # We use a more robust approach to find proper depressions
        minima = []

        # Calculate flow accumulation to find sinks
        flow_accum = np.ones_like(self.heightmap)  # Start with 1 unit of water per cell
        flow_dir = np.zeros((self.size, self.size, 2), dtype=int)

        # Calculate flow directions
        for y in range(self.size):
            for x in range(self.size):
                # Skip ocean areas
                if ocean_mask is not None and ocean_mask[y, x] > 0:
                    continue

                # Skip very low or high areas
                if self.heightmap[y, x] < 0 or self.heightmap[y, x] > 0.7:
                    continue

                # Find steepest descent neighbor
                min_height = self.heightmap[y, x]
                min_dir = (0, 0)  # No flow

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue

                        # Handle wrap-around for longitude (x)
                        nx = (x + dx) % self.size
                        # Clamp latitude (y) to valid range
                        ny = max(0, min(self.size - 1, y + dy))

                        # Skip ocean areas
                        if ocean_mask is not None and ocean_mask[ny, nx] > 0:
                            continue

                        if self.heightmap[ny, nx] < min_height:
                            min_height = self.heightmap[ny, nx]
                            min_dir = (dy, dx)

                flow_dir[y, x] = min_dir

        # Propagate flow to find sinks
        for _ in range(3):
            # Buffer to avoid double-counting
            flow_buffer = np.zeros_like(flow_accum)

            for y in range(self.size):
                for x in range(self.size):
                    # Skip ocean
                    if ocean_mask is not None and ocean_mask[y, x] > 0:
                        continue

                    dy, dx = flow_dir[y, x]
                    if dy == 0 and dx == 0:
                        # This is a sink
                        flow_buffer[y, x] += flow_accum[y, x]
                    else:
                        # Flow to neighbor
                        ny = max(0, min(self.size - 1, y + dy))
                        nx = (x + dx) % self.size
                        flow_buffer[ny, nx] += flow_accum[y, x]

            flow_accum = flow_buffer + 1.0  # Add 1 for the next iteration

        # Find potential lake depressions - cells with high accumulation and no outflow
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                # Skip ocean areas
                if ocean_mask is not None and ocean_mask[y, x] > 0:
                    continue

                # Skip very low areas (likely ocean)
                if self.heightmap[y, x] < 0:
                    continue

                # Skip high elevations (mountains aren't great lake locations)
                if self.heightmap[y, x] > 0.6:
                    continue

                # This is a sink if it has no outflow
                dy, dx = flow_dir[y, x]
                if dy == 0 and dx == 0 and flow_accum[y, x] > 5.0:
                    # Store as (y, x, height, flow_accumulation)
                    minima.append((y, x, self.heightmap[y, x], flow_accum[y, x]))

        # Sort by flow accumulation (higher is better for lakes)
        minima.sort(key=lambda m: m[3], reverse=True)

        # Try to create lakes at the minima
        lakes_created = 0
        for y, x, height, _ in minima:
            # Skip if this area already has a lake
            if lake_mask[y, x] == 1:
                continue

            # Calculate a size for this lake based on the depression characteristics
            # Larger depressions = larger lakes
            target_size = min(max_size, max(min_size, int(flow_accum[y, x] / 2)))

            # Use flood fill to create the lake
            queue = [(y, x)]
            lake_cells = set([(y, x)])

            # Maximum height for this lake
            # Higher threshold = larger lake
            height_threshold = height + np.random.uniform(0.02, 0.05)

            while queue and len(lake_cells) < target_size:
                cy, cx = queue.pop(0)

                # Add neighbors if they're below the threshold
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        # Handle wrap-around for longitude (x)
                        nx = (cx + dx) % self.size
                        # Clamp latitude (y) to valid range
                        ny = max(0, min(self.size - 1, cy + dy))

                        if (ny, nx) in lake_cells:
                            continue

                        # Skip if this is ocean
                        if ocean_mask is not None and ocean_mask[ny, nx] > 0:
                            continue

                        # Add to lake if below threshold
                        if self.heightmap[ny, nx] <= height_threshold:
                            lake_cells.add((ny, nx))
                            queue.append((ny, nx))

            # Only create the lake if it's large enough
            if len(lake_cells) >= min_size:
                for ly, lx in lake_cells:
                    lake_mask[ly, lx] = 1
                
                    # Optionally set the heightmap value to a consistent level for the lake
                    self.heightmap[ly, lx] = max(0, height - 0.01)

                lakes_created += 1

                # Stop if we've created enough lakes
                if lakes_created >= count:
                    break

        print(f"Created {lakes_created} inland lakes")
        return lake_mask

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

        # First, apply continental mask to ensure proper landmass distribution
        if not hasattr(self, 'continental_mask'):
            continental_mask = self.create_continental_mask()
        
            # Apply to heightmap if not already done
            heightmap_adj = self.heightmap.copy()
            for y in range(self.size):
                for x in range(self.size):
                    # Blend heightmap with continental influence
                    heightmap_adj[y, x] = self.heightmap[y, x] * 0.4 + continental_mask[y, x] * 0.6

            # Normalize
            min_val = np.min(heightmap_adj)
            max_val = np.max(heightmap_adj)
            self.heightmap = (heightmap_adj - min_val) / (max_val - min_val)

        # Generate oceans and seas
        print("\n1. Generating oceans and seas...")
        _, ocean_mask = self.generate_water_bodies(water_coverage=ocean_coverage)

        # Apply improved erosion before adding rivers
        print("\n2. Applying realistic erosion to shorelines and terrain...")
        self.apply_erosion(iterations=30, erosion_strength=0.2)

        # Generate rivers using the watershed approach
        print("\n3. Generating rivers using watershed analysis...")
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
                        if height < 0.1:
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
        """Visualize water systems with different colors for each type.

        Args:
            water_systems: Dictionary of water masks
                           from generate_complete_water_system
            title: Title for the plot
        """
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

        # Apply rivers last (bright blue)
        if "rivers" in water_systems:
            for y in range(self.size):
                for x in range(self.size):
                    if water_systems["rivers"][y, x] > 0:
                        # Bright blue for rivers
                        rgb_image[y, x] = [0.0, 0.5, 1.0, 1.0]

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
                      markerfacecolor=[0.0, 0.5, 1.0, 1.0],
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
