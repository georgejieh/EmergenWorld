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
            earth_scale: Scale factor relative
                         to Earth (default is 0.83% of Earth)
        """
        self.size = size
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.heightmap = None

        # Earth properties for scaling
        self.earth_scale = earth_scale
        self.earth_radius_km = 6371.0  # Earth's radius in km
        self.scaled_radius_km = self.earth_radius_km * np.sqrt(earth_scale)
        self.scaled_circumference_km = 2 * np.pi * self.scaled_radius_km

        # Calculate km per grid cell
        self.km_per_cell = self.scaled_circumference_km / size
        self.area_per_cell_sqkm = self.km_per_cell ** 2
        self.area_per_cell_sqmiles = (self.area_per_cell_sqkm *
                                      0.386102)  # Convert to sq miles

        print(f"Initialized terrain generator "
              f"for world at {earth_scale:.4%} of Earth's size")
        print(f"World radius: {self.scaled_radius_km:.1f} km")
        print(f"Each grid cell represents {self.km_per_cell:.2f} km "
              f"({self.area_per_cell_sqmiles:.2f} sq miles)")

    def generate_heightmap(self, scale: float = 100.0) -> np.ndarray:
        """Generate a heightmap using simplex noise.

        Args:
            scale: Scale factor for noise generation

        Returns:
            2D numpy array representing the heightmap
        """
        print(f"Generating simplex noise heightmap "
              f"of size {self.size}x{self.size}...")

        heightmap = np.zeros((self.size, self.size))
        noise_gen = OpenSimplex(seed=self.seed)

        # Generate base noise with multiple octaves
        for y in range(self.size):
            for x in range(self.size):
                value = 0.0
                amplitude = 1.0
                frequency = 1.0

                for _ in range(self.octaves):
                    nx = x / scale * frequency
                    ny = y / scale * frequency

                    # OpenSimplex noise returns values in range [-1, 1]
                    value += noise_gen.noise2(nx, ny) * amplitude

                    amplitude *= self.persistence
                    frequency *= self.lacunarity

                heightmap[y][x] = value

        # Normalize to 0-1 range
        min_val = np.min(heightmap)
        max_val = np.max(heightmap)
        heightmap = (heightmap - min_val) / (max_val - min_val)

        self.heightmap = heightmap
        return heightmap

    def apply_erosion(self, iterations: int = 50, drop_rate: float = 0.05,
                     erosion_strength: float = 0.3) -> np.ndarray:
        """Apply hydraulic erosion to the heightmap for more realistic terrain.

        Args:
            iterations: Number of erosion iterations
            drop_rate: Rate at which water drops are applied
            erosion_strength: Strength of the erosion effect

        Returns:
            Eroded heightmap as a 2D numpy array
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated "
                             "before applying erosion")

        print(f"Applying hydraulic erosion with {iterations} iterations...")

        eroded_map = self.heightmap.copy()

        # Simple hydraulic erosion simulation
        for _ in range(iterations):
            # Random water drops
            num_drops = int(self.size * self.size * drop_rate)
            for _ in range(num_drops):
                x, y = (np.random.randint(1, self.size - 1),
                        np.random.randint(1, self.size - 1))

                # Find steepest descent
                current_height = eroded_map[y, x]
                neighbors = [
                    (x+1, y), (x-1, y), (x, y+1), (x, y-1),
                    (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)
                ]

                min_height = current_height
                min_pos = (x, y)

                for nx, ny in neighbors:
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if eroded_map[ny, nx] < min_height:
                            min_height = eroded_map[ny, nx]
                            min_pos = (nx, ny)

                # Erode and deposit
                if min_pos != (x, y):
                    # Remove soil from current position
                    soil_removed = (erosion_strength *
                                    (current_height - min_height))
                    eroded_map[y, x] -= soil_removed

                    # Deposit some at the lower position
                    deposit_amount = soil_removed * 0.5
                    nx, ny = min_pos
                    eroded_map[ny, nx] += deposit_amount

        self.heightmap = eroded_map
        return eroded_map

    def generate_water_bodies(self, water_coverage: float = 0.71
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate water bodies (oceans, lakes) based on the heightmap.
        
        Args:
            water_coverage: Target water coverage percentage (0.0 - 1.0),
                            default 0.71 for Earth-like coverage
            
        Returns:
            Tuple of (updated heightmap, water mask)
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated "
                             "before creating water bodies")

        print(f"Generating water bodies with {water_coverage:.0%} "
              f"water coverage (Earth-like)...")

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

    def add_rivers(self,
                  river_count: int = 20,
                  min_length: int = 20,
                  meander_factor: float = 0.3) -> np.ndarray:
        """Add rivers flowing from high elevation to the sea.

        Args:
            river_count: Number of major rivers to generate
            min_length: Minimum length of rivers to keep
            meander_factor: How much rivers should meander (0.0-1.0)

        Returns:
            River mask where 1 indicates river cells
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before adding rivers")

        print(f"Generating {river_count} major rivers...")

        # Create a river mask (1 where rivers exist, 0 elsewhere)
        river_mask = np.zeros_like(self.heightmap)

        # Find high elevation points
        # for river sources (excluding the very edges)
        border = 5
        interior_heightmap = self.heightmap[border:-border, border:-border]
        # Get indices in the interior heightmap
        high_points_y, high_points_x = np.where(interior_heightmap > 0.7)
        # Adjust indices to the original heightmap coordinates
        high_points_y += border
        high_points_x += border

        if len(high_points_x) == 0:
            print("Warning: No suitable high points found for river sources")
            return river_mask

        # Choose random starting points from high elevations
        if len(high_points_x) < river_count:
            print(f"Warning: Only {len(high_points_x)} "
                   "suitable source points found")
            river_count = len(high_points_x)

        source_indices = np.random.choice(len(high_points_x),
                                          size=river_count,
                                          replace=False)

        rivers_created = 0

        # Generate each river
        for idx in source_indices:
            x, y = high_points_x[idx], high_points_y[idx]
            river_path = [(y, x)]

            # Generate the river by following the steepest descent
            for _ in range(self.size * 2):  # Maximum river length
                # Get neighbors (8-connected)
                neighbors = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue

                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.size and 0 <= nx < self.size:
                            # Weight by elevation difference
                            # (steeper = more likely)
                            # Also make straight-ish flow more likely
                            if len(river_path) > 1:
                                prev_y, prev_x = river_path[-2]
                                # Direction from previous to current
                                flow_dy, flow_dx = y - prev_y, x - prev_x
                                # Encourage continuing in roughly
                                # the same direction
                                direction_weight = 1.0
                                # Dot product > 0 means similar direction
                                if dy * flow_dy + dx * flow_dx > 0:
                                    direction_weight = 1.5
                                # Opposite direction
                                elif dy * flow_dy + dx * flow_dx < 0:
                                    direction_weight = 0.5
                            else:
                                direction_weight = 1.0

                            # Add randomness for meandering
                            meander = (1.0 + meander_factor *
                                       (np.random.random() - 0.5))

                            # Calculate the weighted score
                            # lower is better (downhill)
                            # We're looking for cells that
                            # are lower than the current one
                            if self.heightmap[ny, nx] < self.heightmap[y, x]:
                                # If it's a downward slope, compute how steep
                                height_diff = (self.heightmap[y, x] -
                                               self.heightmap[ny, nx])
                                # Lower height diff = higher score
                                score = 1.0 - height_diff
                                # Apply direction and meander weights
                                score /= (direction_weight * meander)
                                neighbors.append((ny, nx, score))

                if not neighbors:
                    # No downhill neighbors, end the river
                    break

                # Choose the lowest neighbor,
                # with some randomness for meandering
                neighbors.sort(key=lambda n: n[2])  # Sort by score
                next_y, next_x, _ = neighbors[0]  # Choose the best neighbor

                # Check if we've reached water or an existing river
                if river_mask[next_y, next_x] == 1:
                    # We've hit an existing river, so we're done
                    river_path.append((next_y, next_x))
                    break

                # Check if we've reached the edge of the map
                if (next_x <= 1 or next_x >= self.size - 2 or
                        next_y <= 1 or next_y >= self.size - 2):
                    # Stop when we get very close to the edge
                    break

                # Add the new point to the river
                river_path.append((next_y, next_x))

                # Update the current position
                y, x = next_y, next_x

                # Check if we've reached low elevation (assumed water level)
                if self.heightmap[y, x] < 0.3:
                    break

            # Only add the river if it's long enough
            if len(river_path) >= min_length:
                rivers_created += 1
                # Mark the river on the mask
                for py, px in river_path:
                    river_mask[py, px] = 1

                    # Add some width to larger rivers
                    # (thicker near the end)
                    # Scale width with length
                    river_width = 1 + int(len(river_path) / 50)
                    if river_width > 1:
                        for wy in range(max(0, py-river_width//2),
                                        min(self.size, py+(river_width+1)//2)):
                            for wx in range(max(0, px-river_width//2),
                                           min(self.size,
                                               px+(river_width+1)//2)):
                                # Circular brush
                                if ((wy - py)**2 + (wx - px)**2 <=
                                        (river_width//2)**2):
                                    river_mask[wy, wx] = 1

        print(f"Successfully created {rivers_created} rivers")
        return river_mask

    def add_lakes(self,
                 count: int = 15,
                 min_size: int = 10,
                 max_size: int = 100,
                 ocean_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Add inland lakes to depressions in the terrain.
    
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

        print("Generating inland lakes...")

        # Create a lake mask
        lake_mask = np.zeros_like(self.heightmap)

        # Find local minima in the terrain
        # (excluding very low areas that are likely ocean)
        # We use a simple approach: a cell is a local minimum
        # if it's lower than its neighbors
        minima = []
        for y in range(1, self.size - 1):
            for x in range(1, self.size - 1):
                # Skip ocean areas if mask is provided
                if ocean_mask is not None and ocean_mask[y, x] > 0:
                    continue

                # Skip very low areas (likely ocean)
                if self.heightmap[y, x] < 0.3:
                    continue

                # Skip high elevations (mountains aren't great lake locations)
                if self.heightmap[y, x] > 0.7:
                    continue

                # Check if lower than neighbors
                is_minimum = True
                height = self.heightmap[y, x]

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        if height > self.heightmap[y + dy, x + dx]:
                            is_minimum = False
                            break
                    if not is_minimum:
                        break

                if is_minimum:
                    # Store as (y, x, height)
                    minima.append((y, x, height))

        # Sort minima by height
        # (shallow depressions first - they make better lakes)
        minima.sort(key=lambda m: m[2])

        # Try to create lakes at the minima
        lakes_created = 0
        for y, x, _ in minima:
            # Skip if this area already has a lake
            if lake_mask[y, x] == 1:
                continue

            # Use flood fill to create the lake
            # Start with the minimum point
            queue = [(y, x)]
            lake_cells = set([(y, x)])

            # Random target size for this lake
            target_size = np.random.randint(min_size, max_size)

            # Maximum height for this lake (higher = larger lake)
            height_threshold = (self.heightmap[y, x] +
                                np.random.uniform(0.02, 0.08))

            while queue and len(lake_cells) < target_size:
                cy, cx = queue.pop(0)

                # Add neighbors if they're below the threshold
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        ny, nx = cy + dy, cx + dx

                        if (ny, nx) in lake_cells:
                            continue

                        # Skip if out of bounds
                        if not (0 <= ny < self.size and 0 <= nx < self.size):
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

                lakes_created += 1

                # Stop if we've created enough lakes
                if lakes_created >= count:
                    break

        print(f"Created {lakes_created} inland lakes")
        return lake_mask

    def generate_complete_water_system(
            self, ocean_coverage: float = 0.65, river_count: int = 25,
            lake_count: int = 15) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate a complete water system with oceans, rivers, and lakes.

        Args:
            ocean_coverage: Target ocean coverage (0.0-1.0)
            river_count: Number of major rivers to generate
            lake_count: Number of lakes to generate

        Returns:
            Tuple of (combined water mask, individual water feature masks)
        """
        # Generate oceans and seas
        _, ocean_mask = self.generate_water_bodies(water_coverage=ocean_coverage)

        # Generate rivers
        river_mask = self.add_rivers(river_count=river_count, min_length=20)

        # Generate lakes, making sure they don't overlap with oceans
        lake_mask = self.add_lakes(count=lake_count, min_size=10, max_size=100,
                                  ocean_mask=ocean_mask)

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
        print(f"  River cells: {np.sum(river_mask)} ({np.sum(river_mask) / river_mask.size:.2%})")
        print(f"  Lake cells: {np.sum(lake_mask)} ({np.sum(lake_mask) / lake_mask.size:.2%})")
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
        """Visualize the current heightmap.

        Args:
            water_mask: Optional mask indicating water bodies
            title: Title for the plot
            show_grid: Whether to show a grid indicating cell sizes
        """
        if self.heightmap is None:
            raise ValueError("No heightmap to visualize")

        plt.figure(figsize=(12, 10))

        # If water mask is provided, apply it to visualization
        if water_mask is not None:
            # Create a copy to avoid modifying the original
            terrain_vis = self.heightmap.copy()

            # More dramatic color scheme for fantasy terrain
            # Deep blues for water, vibrant terrain colors for land
            terrain_colors = plt.cm.gist_earth(terrain_vis)  # More vibrant terrain
            water_colors = plt.cm.ocean(terrain_vis * 0.8)  # Deep blues

            # Combine water and terrain colors using the mask
            for i in range(water_mask.shape[0]):
                for j in range(water_mask.shape[1]):
                    if water_mask[i, j] > 0:
                        terrain_colors[i, j] = water_colors[i, j]

            plt.imshow(terrain_colors)
        else:
            plt.imshow(self.heightmap, cmap="gist_earth")

        # Show scale information
        scale_text = f"World Scale: {self.earth_scale:.2%} of Earth\n"
        scale_text += f"Grid Cell: {self.km_per_cell:.1f} km × {self.km_per_cell:.1f} km\n"
        scale_text += f"Cell Area: {self.area_per_cell_sqmiles:.1f} sq miles"

        # Place scale info in the upper left
        plt.annotate(scale_text, (0.02, 0.98), xycoords="figure fraction",
                    verticalalignment="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

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

        plt.colorbar(label="Elevation")
        plt.title(title)
        plt.axis("off" if not show_grid else "on")
        plt.tight_layout()
        plt.show()

    def visualize_water_system(self, water_systems: Dict[str, np.ndarray],
                              title: str = "Complete Water System"):
        """Visualize water systems with different colors for each type.

        Args:
            water_systems: Dictionary of water masks from generate_complete_water_system
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
                      markerfacecolor=[0.0, 0.1, 0.5, 1.0], markersize=10, label="Ocean"),
            plt.Line2D([0], [0], marker="s", color="w",
                      markerfacecolor=[0.2, 0.4, 0.8, 1.0], markersize=10, label="Lakes"),
            plt.Line2D([0], [0], marker="s", color="w",
                      markerfacecolor=[0.0, 0.5, 1.0, 1.0], markersize=10, label="Rivers")
        ]
        plt.legend(handles=legend_elements, loc="upper right")

        # Show scale information
        scale_text = f"World Scale: {self.earth_scale:.2%} of Earth\n"
        scale_text += f"Grid Cell: {self.km_per_cell:.1f} km × {self.km_per_cell:.1f} km\n"
        scale_text += f"Cell Area: {self.area_per_cell_sqmiles:.1f} sq miles"

        plt.annotate(scale_text, (0.02, 0.98), xycoords="figure fraction",
                    verticalalignment="top", color="black",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

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
