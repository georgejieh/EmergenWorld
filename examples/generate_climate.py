"""Example script demonstrating climate generation for the EmergenWorld simulation.

This script shows how to create terrain, initialize a planetary system,
and generate climate patterns using the climate system module.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from src.world_generation import TerrainGenerator, PlanetarySystem, ClimateSystem


def main():
    """Main function to demonstrate climate generation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate climate patterns for a fantasy world')
    parser.add_argument('--size', type=int, default=256, help='World size (default: 256)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for generation')
    parser.add_argument('--scale', type=float, default=0.0083, help='World scale relative to Earth')
    parser.add_argument('--fantasy', type=float, default=0.0, help='Fantasy elements strength (0.0-1.0)')
    parser.add_argument('--output', type=str, default=None, help='Output file for climate data')
    parser.add_argument('--quick', action='store_true', help='Use faster generation with less detail')
    parser.add_argument('--show-all', action='store_true', help='Show all climate visualizations')
    args = parser.parse_args()
    
    # Adjust parameters based on quick mode
    if args.quick:
        print("Using quick generation mode with less detail")
        args.size = min(args.size, 128)
    
    # 1. Generate terrain
    print("\n=== Generating Terrain ===")
    terrain_gen = TerrainGenerator(
        size=args.size,
        octaves=4 if args.quick else 6,
        seed=args.seed,
        earth_scale=args.scale
    )
    
    # Generate base heightmap
    heightmap = terrain_gen.generate_heightmap()
    
    # Add mountains
    if not args.quick:
        terrain_gen.add_mountains(epic_factor=1.5)
    
    # Create water system
    water_mask, water_systems = terrain_gen.generate_complete_water_system(
        ocean_coverage=0.65,
        river_count=15 if args.quick else 25,
        lake_count=10 if args.quick else 15
    )
    
    # Visualize the terrain with water
    terrain_gen.visualize(water_mask, title="Terrain with Water System")
    
    # 2. Initialize planetary system
    print("\n=== Initializing Planetary System ===")
    planet = PlanetarySystem(
        world_size=args.size,
        axial_tilt_degrees=23.5,  # Earth-like tilt
        earth_scale=args.scale,
        start_day=80  # Around Spring equinox
    )
    
    # Update day/night cycle and solar radiation
    planet.update_all()
    
    # Visualize the planetary system
    plt.figure(figsize=(12, 10))
    
    # Day/night cycle
    ax1 = plt.subplot(2, 1, 1)
    planet.visualize_day_night(ax1)
    
    # Solar radiation
    ax2 = plt.subplot(2, 1, 2)
    planet.visualize_solar_radiation(ax2)
    
    plt.tight_layout()
    plt.show()
    
    # 3. Generate climate system
    print("\n=== Generating Climate System ===")
    
    # Define fantasy climate features if requested
    fantasy_features = None
    if args.fantasy > 0:
        fantasy_features = {
            'magical_hotspots': args.fantasy * 0.8,
            'elemental_zones': args.fantasy * 0.6,
            'aether_currents': args.fantasy * 0.4,
            'reality_flux': args.fantasy * 0.2
        }
        print(f"Adding fantasy climate features with strength {args.fantasy}")

    # Create a modified heightmap for elevation calculation
    elevation_map = np.zeros_like(heightmap)

    # Apply a custom elevation scaling that keeps high peaks but
    # reduces mid-range elevations to more moderate levels
    for y in range(args.size):
        for x in range(args.size):
            h = heightmap[y, x]
            if h < 0.3:  # Water and coastal areas
                elevation_map[y, x] = h * 2000  # Lower elevation for oceans/coast
            elif h < 0.6:  # Regular land
                # Map 0.3-0.6 to 600-3000m (moderate terrain)
                normalized_h = (h - 0.3) / 0.3
                elevation_map[y, x] = 600 + normalized_h * 2400
            else:  # Mountain regions
                # Map 0.6-1.0 to 3000-8000m (dramatic mountains)
                normalized_h = (h - 0.6) / 0.4
                elevation_map[y, x] = 3000 + normalized_h * 5000

    # Now create the ClimateSystem with the modified elevation map
    climate = ClimateSystem(
        terrain_heightmap=heightmap,  # Original heightmap for features
        elevation_map=elevation_map,  # Custom elevation scaling
        water_mask=water_mask,
        planetary_system=planet,
        world_size=args.size,
        base_temperature=14.0,
        seasonal_variation_strength=1.0,
        random_seed=args.seed,
        fantasy_climate_features=fantasy_features
    )
    
    # Visualize all climate elements
    if args.show_all:
        climate.visualize_all(figsize=(15, 20))
        plt.show()
    else:
        # Show just the basic climate maps
        plt.figure(figsize=(15, 12))
        
        # Temperature map
        ax1 = plt.subplot(2, 2, 1)
        climate.visualize_temperature(ax1)
        
        # Precipitation map
        ax2 = plt.subplot(2, 2, 2)
        climate.visualize_precipitation(ax2)
        
        # Wind patterns
        ax3 = plt.subplot(2, 2, 3)
        climate.visualize_wind(ax3)
        
        # Climate classification
        ax4 = plt.subplot(2, 2, 4)
        climate.visualize_climate_classification(ax4, resolution=6)
        
        plt.tight_layout()
        plt.show()
    
    # 4. Demonstrate seasonal changes
    print("\n=== Demonstrating Seasonal Changes ===")
    
    # Show climate for 4 seasons
    season_days = [80, 170, 260, 350]  # Spring, Summer, Fall, Winter
    season_names = ["Spring", "Summer", "Fall", "Winter"]
    
    plt.figure(figsize=(15, 12))
    
    for i, (day, name) in enumerate(zip(season_days, season_names)):
        # Update the planetary system to the specified day
        planet.current_day = day
        planet.update_all()
        
        # Update climate for the new season
        climate.update_climate(day, 12.0)  # Noon
        
        # Plot temperature for this season
        ax = plt.subplot(2, 2, i+1)
        climate.visualize_temperature(ax, title=f"{name} Temperature")
    
    plt.tight_layout()
    plt.show()
    
    # 5. Show fantasy features if enabled
    if args.fantasy > 0:
        print("\n=== Visualizing Fantasy Climate Features ===")
        
        plt.figure(figsize=(12, 10))
        climate.visualize_fantasy_features(title="Fantasy Climate Elements")
        plt.tight_layout()
        plt.show()
    
    # 6. Interactive climate querying
    print("\n=== Interactive Climate Data Querying ===")
    print("Click on the map to see climate data for specific locations")
    
    # Create interactive map
    fig, ax = plt.subplots(figsize=(12, 10))
    climate.visualize_temperature(ax, title="Click to Query Climate Data")
    
    # Define click handler
    def on_click(event):
        if event.inaxes != ax:
            return
        
        # Convert click coordinates to grid indices
        x_ratio = (event.xdata + 180) / 360
        y_ratio = (90 - event.ydata) / 180
        
        x = int(x_ratio * args.size)
        y = int(y_ratio * args.size)
        
        if 0 <= x < args.size and 0 <= y < args.size:
            # Get climate data
            climate_data = climate.get_climate_at_position(x, y)
            climate_type = climate.determine_climate_classification(x, y)
            
            # Print climate information
            print(f"\nLocation: Longitude {event.xdata:.1f}°, Latitude {event.ydata:.1f}°")
            print(f"Climate classification: {climate_type}")
            print(f"Temperature: {climate_data['temperature']:.1f}°C")
            print(f"Precipitation: {climate_data['precipitation']:.1f} mm/day")
            print(f"Humidity: {climate_data['humidity']:.2f}")
            print(f"Wind: {climate_data['wind_speed']:.1f} m/s from {climate_data['wind_direction']:.0f}°")
            print(f"Elevation: {climate_data['elevation']/1000:.2f} km")
            print(f"Is water: {climate_data['is_water']}")
            
            # Mark the clicked location
            ax.plot(event.xdata, event.ydata, 'ko', markersize=8, alpha=0.7)
            fig.canvas.draw()
    
    # Connect the click handler
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    
    # 7. Save climate data if requested
    if args.output:
        climate.save_climate_data(args.output)
        print(f"Climate data saved to {args.output}")
    
    print("\nClimate system generation complete!")


if __name__ == "__main__":
    main()