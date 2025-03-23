"""Minimal example demonstrating climate generation for an existing world.

This script shows how to take an existing terrain and quickly generate 
a climate system with basic visualizations.
"""

import matplotlib.pyplot as plt
import numpy as np
from src.world_generation import TerrainGenerator, PlanetarySystem, ClimateSystem


def main():
    """Generate a simple world with climate."""
    # Set parameters
    size = 256
    seed = 42
    
    # 1. Create terrain with water
    print("Generating terrain...")
    terrain = TerrainGenerator(size=size, seed=seed)
    heightmap = terrain.generate_heightmap()
    terrain.add_mountains(epic_factor=1.5)
    water_mask, _ = terrain.generate_water_bodies(water_coverage=0.55)
    
    # 2. Setup planetary system
    print("Setting up planetary system...")
    planet = PlanetarySystem(
        world_size=size,
        axial_tilt_degrees=23.5,
        start_day=80  # Spring
    )
    planet.update_all()
    
    # Create elevation mapping for climate system
    elevation_map = np.zeros_like(heightmap)
    for y in range(size):
        for x in range(size):
            h = heightmap[y, x]
            if h < 0.3:  # Water and coastal areas
                elevation_map[y, x] = h * 2000
            elif h < 0.6:  # Regular land
                normalized_h = (h - 0.3) / 0.3
                elevation_map[y, x] = 600 + normalized_h * 2400
            else:  # Mountain regions
                normalized_h = (h - 0.6) / 0.4
                elevation_map[y, x] = 3000 + normalized_h * 5000
    
    # 3. Generate climate
    print("Generating climate system...")
    climate = ClimateSystem(
        terrain_heightmap=heightmap,
        water_mask=water_mask,
        planetary_system=planet,
        world_size=size,
        random_seed=seed,
        elevation_map=elevation_map
    )
    
    # 4. Visualize climate results
    print("Visualizing results...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Temperature
    climate.visualize_temperature(ax=axes[0, 0], title="Temperature")
    
    # Precipitation
    climate.visualize_precipitation(ax=axes[0, 1], title="Precipitation")
    
    # Wind
    climate.visualize_wind(ax=axes[1, 0], title="Wind Patterns", density=15)
    
    # Climate classification
    climate.visualize_climate_classification(
        ax=axes[1, 1], 
        title="Climate Zones", 
        resolution=4
    )
    
    plt.tight_layout()
    plt.show()
    
    # 5. Show seasonal comparison
    print("Demonstrating seasons...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Summer
    planet.current_day = 170  # Summer
    planet.update_all()
    climate.update_climate(170, 12.0)  # Noon
    climate.visualize_temperature(ax=axes[0], title="Summer Temperature")
    
    # Winter
    planet.current_day = 350  # Winter
    planet.update_all()
    climate.update_climate(350, 12.0)  # Noon
    climate.visualize_temperature(ax=axes[1], title="Winter Temperature")
    
    plt.tight_layout()
    plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()