#!/usr/bin/env python3
"""Example script to demonstrate the TerrainGenerator class.

This script shows how to create and visualize a fantasy terrain with mountains,
erosion, rivers, lakes, and oceans.
"""

import argparse
import numpy as np
from src.world_generation import TerrainGenerator


def generate_basic_terrain(size=256, seed=42, scale=150.0, show=True):
    """Generate and visualize a basic heightmap.

    Args:
        size: Size of the heightmap (size x size)
        seed: Random seed for reproducible terrain
        scale: Scale factor for the noise generation
        show: Whether to display the visualization

    Returns:
        TerrainGenerator instance with the generated heightmap
    """
    print("\n=== Generating Basic Terrain ===")

    # Initialize terrain generator
    generator = TerrainGenerator(size=size, seed=seed, earth_scale=0.0083)

    # Generate basic heightmap
    generator.generate_heightmap(scale=scale)

    if show:
        # Visualize the heightmap
        generator.visualize(title="Basic Terrain Heightmap", show_grid=True)

    return generator


def apply_terrain_erosion(generator, iterations=30, show=True):
    """Apply erosion to the terrain for more natural features.

    Args:
        generator: TerrainGenerator instance with a heightmap
        iterations: Number of erosion iterations
        show: Whether to display the visualization

    Returns:
        TerrainGenerator instance with the eroded heightmap
    """
    print("\n=== Applying Terrain Erosion ===")

    # Apply hydraulic erosion
    generator.apply_erosion(iterations=iterations)

    if show:
        # Visualize the eroded terrain
        generator.visualize(title="Eroded Terrain")

    return generator


def add_fantasy_mountains(generator, mountain_scale=1.2, peak_threshold=0.6,
                          epic_factor=2.5, show=True):
    """Add epic fantasy-style mountains to the terrain.

    Args:
        generator: TerrainGenerator instance with a heightmap
        mountain_scale: Scale factor for mountain height
        peak_threshold: Threshold for mountain peaks (lower = more mountains)
        epic_factor: Multiplier for mountain epicness
        show: Whether to display the visualization

    Returns:
        TerrainGenerator instance with mountains added
    """
    print("\n=== Adding Fantasy Mountains ===")

    # Add epic mountains for fantasy world
    generator.add_mountains(
        mountain_scale=mountain_scale,
        peak_threshold=peak_threshold,
        epic_factor=epic_factor
    )

    if show:
        # Visualize the terrain with mountains
        generator.visualize(title="Fantasy Terrain with Epic Mountains")

    return generator


def generate_water_system(generator, ocean_coverage=0.65, river_count=25,
                          lake_count=15, show=True):
    """Generate a complete water system including oceans, rivers, and lakes.

    Args:
        generator: TerrainGenerator instance with a heightmap
        ocean_coverage: Target ocean coverage (0.0-1.0)
        river_count: Number of major rivers to generate
        lake_count: Number of lakes to generate
        show: Whether to display the visualization

    Returns:
        Tuple of (TerrainGenerator, water_mask, water_systems)
    """
    print("\n=== Generating Water System ===")

    # Generate complete water system
    water_mask, water_systems = generator.generate_complete_water_system(
        ocean_coverage=ocean_coverage,
        river_count=river_count,
        lake_count=lake_count
    )

    if show:
        # Visualize the water systems
        generator.visualize_water_system(
            water_systems,
            title="Fantasy World Water Systems"
        )

        # Visualize the complete world
        generator.visualize(
            water_mask=water_mask,
            title="Complete Fantasy World",
            show_grid=True
        )

    return generator, water_mask, water_systems


def display_world_statistics(generator, water_mask):
    """Display statistics about the generated world.

    Args:
        generator: TerrainGenerator instance
        water_mask: Water mask from generate_water_system
    """
    print("\n=== World Statistics ===")

    # Calculate land statistics
    total_land_cells = np.sum(water_mask == 0)
    total_land_area_sqkm = total_land_cells * generator.area_per_cell_sqkm
    total_land_area_sqmiles = total_land_cells * generator.area_per_cell_sqmiles

    # Display statistics
    print(f"World scale: {generator.earth_scale:.4%} of Earth's size")
    print(f"World radius: {generator.scaled_radius_km:.1f} km")
    print(f"World circumference: {generator.scaled_circumference_km:.1f} km")
    print(f"Grid size: {generator.size}x{generator.size}")
    print(f"Cell size: {generator.km_per_cell:.2f} km "
          f"({generator.area_per_cell_sqmiles:.2f} sq miles)")
    print(f"Land cells: {total_land_cells}")
    print(f"Water cells: {np.sum(water_mask > 0)}")
    print(f"Land percentage: {(total_land_cells / water_mask.size):.2%}")
    print(f"Water percentage: {(np.sum(water_mask > 0) / water_mask.size):.2%}")
    print(f"Total land area: {total_land_area_sqkm:.0f} sq km "
          f"({total_land_area_sqmiles:.0f} sq miles)")

    # Calculate population estimates
    density_per_sqmile = 10  # Fantasy population density
    est_population = int(total_land_area_sqmiles * density_per_sqmile)
    print(f"Estimated population (at {density_per_sqmile} per sq mile): "
          f"{est_population:,}")


def generate_complete_fantasy_world(size=256, seed=42, show_steps=True):
    """Generate a complete fantasy world with all features.

    Args:
        size: Size of the world grid
        seed: Random seed
        show_steps: Whether to show intermediate steps

    Returns:
        Tuple of (generator, water_mask, water_systems)
    """
    print("\n=== Generating Complete Fantasy World ===")

    # Generate basic terrain
    generator = generate_basic_terrain(
        size=size,
        seed=seed,
        show=show_steps
    )

    # Apply erosion
    generator = apply_terrain_erosion(
        generator,
        iterations=30,
        show=show_steps
    )

    # Add fantasy mountains
    generator = add_fantasy_mountains(
        generator,
        epic_factor=2.5,
        show=show_steps
    )

    # Generate water systems
    generator, water_mask, water_systems = generate_water_system(
        generator,
        ocean_coverage=0.65,
        river_count=25,
        lake_count=15,
        show=show_steps
    )

    # Save the heightmap
    generator.save_heightmap("fantasy_world_heightmap.npy")

    # Display world statistics
    display_world_statistics(generator, water_mask)

    return generator, water_mask, water_systems


def main():
    """Main function to demonstrate terrain generation."""
    # Parse command line arguments (if any)
    parser = argparse.ArgumentParser(description="Generate a fantasy terrain.")
    parser.add_argument("--size", type=int, default=256,
                        help="Size of the terrain grid")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quick", action="store_true",
                        help="Skip intermediate visualizations")
    args = parser.parse_args()

    # Generate a complete fantasy world
    generator, water_mask, _ = generate_complete_fantasy_world(
        size=args.size,
        seed=args.seed,
        show_steps=not args.quick
    )

    # Always show the final world
    if args.quick:
        generator.visualize(
            water_mask=water_mask,
            title="Complete Fantasy World",
            show_grid=True
        )

    print("\nTerrain generation complete!")


if __name__ == "__main__":
    main()
