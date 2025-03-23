"""Example script demonstrating fantasy climate features for EmergenWorld.

This script shows how to use the fantasy climate elements to create
magical and supernatural weather patterns in your world.
"""

import matplotlib.pyplot as plt
import numpy as np

# Direct imports when running in dev mode
from src.world_generation import TerrainGenerator, PlanetarySystem, ClimateSystem


def main():
    """Generate worlds with different fantasy climate features."""
    # Set parameters
    size = 256
    seed = 42
    
    # Generate base terrain and planetary system (shared across examples)
    print("Generating base world...")
    terrain = TerrainGenerator(size=size, seed=seed)
    heightmap = terrain.generate_heightmap()
    terrain.add_mountains(epic_factor=1.5)
    water_mask, _ = terrain.generate_water_bodies(water_coverage=0.55)
    
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
    
    # Create a reference normal climate (no fantasy elements)
    print("Generating reference climate...")
    normal_climate = ClimateSystem(
        terrain_heightmap=heightmap,
        water_mask=water_mask,
        planetary_system=planet,
        world_size=size,
        random_seed=seed,
        elevation_map=elevation_map
    )
    
    # 1. Demonstrate each fantasy element separately
    fantasy_types = [
        "magical_hotspots",
        "elemental_zones",
        "aether_currents",
        "reality_flux"
    ]
    
    plt.figure(figsize=(15, 12))
    
    # First, show reference climate
    ax1 = plt.subplot(2, 3, 1)
    normal_climate.visualize_temperature(ax=ax1, title="Normal Climate (No Fantasy)")
    
    # Show each fantasy element
    for i, feature in enumerate(fantasy_types):
        print(f"Generating climate with {feature}...")
        # Create fantasy features dict with only this element active
        fantasy_features = {f: 0.0 for f in fantasy_types}
        fantasy_features[feature] = 0.8  # High strength
        
        # Create climate with this feature
        fantasy_climate = ClimateSystem(
            terrain_heightmap=heightmap,
            water_mask=water_mask,
            planetary_system=planet,
            world_size=size,
            random_seed=seed,
            fantasy_climate_features=fantasy_features,
            elevation_map=elevation_map
        )
        
        # Plot temperature
        ax = plt.subplot(2, 3, i+2)
        fantasy_climate.visualize_temperature(ax=ax, title=f"{feature.replace('_', ' ').title()}")
    
    plt.tight_layout()
    plt.show()
    
    # 2. Show detailed view of each fantasy feature
    for feature in fantasy_types:
        print(f"Detailed analysis of {feature}...")
        # Create fantasy features dict with only this element active
        fantasy_features = {f: 0.0 for f in fantasy_types}
        fantasy_features[feature] = 0.9  # High strength
        
        # Create climate with this feature
        fantasy_climate = ClimateSystem(
            terrain_heightmap=heightmap,
            water_mask=water_mask,
            planetary_system=planet,
            world_size=size,
            random_seed=seed,
            fantasy_climate_features=fantasy_features,
            elevation_map=elevation_map
        )
        
        # Create figure with multiple visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Fantasy Feature: {feature.replace('_', ' ').title()}", fontsize=16)
        
        # Temperature anomaly
        fantasy_climate.visualize_fantasy_features(ax=axes[0, 0], title="Temperature Anomaly")
        
        # Precipitation
        fantasy_climate.visualize_precipitation(ax=axes[0, 1], title="Precipitation")
        
        # Wind patterns
        fantasy_climate.visualize_wind(ax=axes[1, 0], title="Wind Patterns", density=15)
        
        # Humidity
        fantasy_climate.visualize_humidity(ax=axes[1, 1], title="Humidity")
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for title
        plt.show()
    
    # 3. Create a world with all fantasy features combined
    print("Generating world with all fantasy features...")
    all_features = {
        'magical_hotspots': 0.6,
        'elemental_zones': 0.5,
        'aether_currents': 0.4,
        'reality_flux': 0.3
    }
    
    fantasy_world = ClimateSystem(
        terrain_heightmap=heightmap,
        water_mask=water_mask,
        planetary_system=planet,
        world_size=size,
        random_seed=seed,
        fantasy_climate_features=all_features,
        elevation_map=elevation_map
    )
    
    # Visualize the combined fantasy world
    fantasy_world.visualize_all(figsize=(15, 20))
    plt.suptitle("Fantasy World with All Magical Climate Features", fontsize=16)
    plt.subplots_adjust(top=0.95)  # Make room for title
    plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()