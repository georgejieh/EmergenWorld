#!/usr/bin/env python3
"""Example script to demonstrate the PlanetarySystem class.

This script shows how to create and visualize a planetary system for
a fantasy world, including day/night cycles, seasons, and solar radiation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import from the world_generation package
from src.world_generation import PlanetarySystem


def demonstrate_day_cycle(planet, steps=24):
    """Demonstrate a full day cycle with animated visualization.
    
    Args:
        planet: PlanetarySystem instance
        steps: Number of steps for the animation
    """
    print("\n=== Day Cycle Demonstration ===")
    
    # Save original state
    original_day = planet.current_day
    original_hour = planet.current_hour
    
    # Reset to dawn
    planet.current_hour = 0
    planet._update_sun_position()
    planet._update_day_night_cycle()
    planet._update_solar_radiation()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Set up the plots
    day_night_img = planet.visualize_day_night(ax=ax1, title="Day/Night Cycle")
    solar_img = planet.visualize_solar_radiation(ax=ax2, title="Solar Radiation")
    
    plt.tight_layout()
    
    # Function to update the plots for each frame
    def update(frame):
        # Advance time by one hour
        hours_per_step = planet.day_length_hours / steps
        planet.advance_time(hours=hours_per_step)
        
        # Update the plots
        day_night_img.set_array(planet.day_night_mask)
        solar_img.set_array(planet.solar_radiation)
        
        # Update titles with current time
        ax1.set_title(f"Day/Night at {planet.get_formatted_date()}")
        ax2.set_title(f"Solar Radiation at {planet.get_formatted_date()}")
        
        return day_night_img, solar_img
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=steps, blit=False, repeat=True)
    
    plt.suptitle("One Day Cycle Animation", fontsize=16)
    plt.show()
    
    # Restore original state
    planet.current_day = original_day
    planet.current_hour = original_hour
    planet._update_sun_position()
    planet._update_day_night_cycle()
    planet._update_solar_radiation()


def demonstrate_seasonal_cycle(planet, steps=12):
    """Demonstrate seasonal changes throughout a year with animated visualization.
    
    Args:
        planet: PlanetarySystem instance
        steps: Number of steps for the animation
    """
    print("\n=== Seasonal Cycle Demonstration ===")
    
    # Save original state
    original_day = planet.current_day
    original_hour = planet.current_hour
    
    # Set time to noon
    planet.current_hour = planet.day_length_hours / 2
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Function to create seasonal maps
    def create_seasonal_maps():
        """Generate maps of temperature factors and day lengths."""
        # Create arrays to hold seasonal data
        world_size = planet.world_size
        temp_factor_map = np.zeros((world_size, world_size))
        day_length_map = np.zeros((world_size, world_size))
        
        # Calculate seasonal factors and day lengths for each latitude
        for y in range(world_size):
            lat = planet.latitudes[y, 0]
            temp_factor = planet.get_seasonal_temperature_factor(lat)
            day_length = planet.get_day_length(lat)
            
            # Apply to all cells at this latitude
            temp_factor_map[y, :] = temp_factor
            day_length_map[y, :] = day_length
        
        return temp_factor_map, day_length_map
    
    # Set up initial plots
    temp_map, day_map = create_seasonal_maps()
    
    temp_img = ax1.imshow(temp_map, cmap='RdBu_r', 
                         extent=[-180, 180, -90, 90], 
                         interpolation='nearest',
                         origin='upper',
                         vmin=-20, vmax=20)
    plt.colorbar(temp_img, ax=ax1, label="Temperature Modifier (°C)")
    
    day_img = ax2.imshow(day_map, cmap='viridis', 
                        extent=[-180, 180, -90, 90], 
                        interpolation='nearest',
                        origin='upper',
                        vmin=0, vmax=planet.day_length_hours)
    plt.colorbar(day_img, ax=ax2, label="Daylight Hours")
    
    # Add grid lines
    ax1.grid(linestyle=':', color='gray', alpha=0.5)
    ax2.grid(linestyle=':', color='gray', alpha=0.5)
    
    # Function to update the plots for each frame
    def update(frame):
        # Set day to evenly spaced points throughout the year
        day = int(frame * (planet.year_length_days / steps))
        planet.current_day = day
        planet._update_sun_position()
        
        # Create new seasonal maps
        temp_map, day_map = create_seasonal_maps()
        
        # Update images
        temp_img.set_array(temp_map)
        day_img.set_array(day_map)
        
        # Update titles with current date and season
        season = planet.get_season()
        month_day = planet.get_formatted_date().split(', ')[0]
        ax1.set_title(f"Temperature Modifiers - {month_day} ({season})")
        ax2.set_title(f"Daylight Hours - {month_day} ({season})")
        
        return temp_img, day_img
    
    # Create the animation
    ani = FuncAnimation(fig, update, frames=steps, blit=False, repeat=True)
    
    plt.suptitle("Seasonal Cycle Throughout the Year", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Restore original state
    planet.current_day = original_day
    planet.current_hour = original_hour
    planet._update_sun_position()
    planet._update_day_night_cycle()
    planet._update_solar_radiation()


def compare_planet_properties():
    """Compare different planetary properties side by side."""
    print("\n=== Planetary Properties Comparison ===")
    
    # Create different planets
    earth_like = PlanetarySystem(
        world_size=128,
        axial_tilt_degrees=23.5,  # Earth-like
        day_length_hours=24.0,
        year_length_days=365.25,
        earth_scale=0.0083
    )
    
    extreme_tilt = PlanetarySystem(
        world_size=128,
        axial_tilt_degrees=45.0,  # Extreme tilt
        day_length_hours=24.0,
        year_length_days=365.25,
        earth_scale=0.0083
    )
    
    no_tilt = PlanetarySystem(
        world_size=128,
        axial_tilt_degrees=0.0,  # No tilt
        day_length_hours=24.0,
        year_length_days=365.25,
        earth_scale=0.0083
    )
    
    # Set all planets to summer solstice at noon
    for planet in [earth_like, extreme_tilt, no_tilt]:
        planet.current_day = int(planet.year_length_days / 2)  # Mid-year
        planet.current_hour = planet.day_length_hours / 2      # Noon
        planet._update_sun_position()
        planet._update_day_night_cycle()
        planet._update_solar_radiation()
    
    # Create a figure with three rows
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # Plot day/night and solar radiation for each planet
    planets = [earth_like, extreme_tilt, no_tilt]
    titles = ["Earth-like (23.5° tilt)", "Extreme Tilt (45°)", "No Tilt (0°)"]
    
    for i, (planet, title) in enumerate(zip(planets, titles)):
        # Day/Night visualization
        planet.visualize_day_night(ax=axes[i, 0], title=f"Day/Night - {title}")
        
        # Solar radiation visualization
        planet.visualize_solar_radiation(ax=axes[i, 1], title=f"Solar Radiation - {title}")
        
        # Day length visualization - custom plot for each planet
        ax = axes[i, 2]
        
        # Calculate day length at different latitudes
        latitudes = [-90, -60, -30, 0, 30, 60, 90]
        day_lengths = []
        
        for lat in latitudes:
            day_lengths.append(planet.get_day_length(lat))
        
        ax.bar(latitudes, day_lengths, width=10, color='skyblue')
        ax.set_ylim(0, planet.day_length_hours)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Hours of Daylight")
        ax.set_title(f"Day Length by Latitude - {title}")
        ax.grid(alpha=0.3)
    
    plt.suptitle("Comparison of Different Planetary Systems", fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def create_fantasy_world():
    """Create and demonstrate a custom fantasy world planetary system."""
    print("\n=== Fantasy World Planetary System ===")
    
    # Create a fantasy planetary system with parameters different from Earth
    fantasy_planet = PlanetarySystem(
        world_size=256,
        axial_tilt_degrees=28.0,     # More extreme tilt
        day_length_hours=32.0,       # Longer days
        year_length_days=310.0,      # Shorter year
        eccentricity=0.028,          # More eccentric orbit
        seasonal_factor=1.2,         # More pronounced seasons
        earth_scale=0.0083           # 0.83% of Earth's size
    )
    
    # Print basic planetary information
    print(f"Planet radius: {fantasy_planet.planet_radius_km:.1f} km")
    print(f"Grid cell size: {fantasy_planet.km_per_cell:.1f} km")
    print(f"Day length: {fantasy_planet.day_length_hours} hours")
    print(f"Year length: {fantasy_planet.year_length_days} days")
    
    # Demonstrate day cycle
    demonstrate_day_cycle(fantasy_planet, steps=16)
    
    # Demonstrate seasonal cycle
    demonstrate_seasonal_cycle(fantasy_planet, steps=12)
    
    return fantasy_planet


if __name__ == "__main__":
    # Create a fantasy world
    fantasy_planet = create_fantasy_world()
    
    # Compare different planetary configurations
    compare_planet_properties()
    
    # Show day length variation throughout the year
    fantasy_planet.visualize_day_length(latitudes=[0, 28, 45, 66.5, 80])
    
    print("Simulation complete.")