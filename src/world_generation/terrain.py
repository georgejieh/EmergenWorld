"""Terrain generation module for EmergenWorld.

This module handles the creation of realistic terrain using various noise algorithms
and provides methods for terrain manipulation and feature generation.
"""

import numpy as np
from noise import pnoise2, snoise2
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional


class TerrainGenerator:
    """Generates and manipulates terrain heightmaps for the EmergenWorld simulation.
    
    Uses noise algorithms to create realistic terrain features such as mountains,
    valleys, and plateaus.
    """
    
    def __init__(self, size: int = 1024, octaves: int = 6, 
                 persistence: float = 0.5, lacunarity: float = 2.0,
                 seed: Optional[int] = None):
        """Initialize the TerrainGenerator with configurable parameters.
        
        Args:
            size: Grid size for the heightmap (size x size)
            octaves: Number of octaves for noise generation
            persistence: Persistence value for noise generation
            lacunarity: Lacunarity value for noise generation
            seed: Random seed for reproducible terrain generation
        """
        self.size = size
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.seed = seed if seed is not None else np.random.randint(0, 1000000)
        self.heightmap = None
        
    def generate_heightmap(self, noise_type: str = 'perlin', scale: float = 100.0) -> np.ndarray:
        """Generate a heightmap using the specified noise algorithm.
        
        Args:
            noise_type: Type of noise to use ('perlin' or 'simplex')
            scale: Scale factor for noise generation
            
        Returns:
            2D numpy array representing the heightmap
        """
        print(f"Generating {noise_type} noise heightmap of size {self.size}x{self.size}...")
        
        heightmap = np.zeros((self.size, self.size))
        noise_func = pnoise2 if noise_type == 'perlin' else snoise2
        
        # Generate base noise
        for y in range(self.size):
            for x in range(self.size):
                heightmap[y][x] = noise_func(
                    x / scale, 
                    y / scale,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    repeatx=self.size,
                    repeaty=self.size,
                    base=self.seed
                )
        
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
            raise ValueError("Heightmap must be generated before applying erosion")
            
        print(f"Applying hydraulic erosion with {iterations} iterations...")
        
        eroded_map = self.heightmap.copy()
        
        # Simple hydraulic erosion simulation
        for _ in range(iterations):
            # Random water drops
            num_drops = int(self.size * self.size * drop_rate)
            for _ in range(num_drops):
                x, y = np.random.randint(1, self.size - 1), np.random.randint(1, self.size - 1)
                
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
                    soil_removed = erosion_strength * (current_height - min_height)
                    eroded_map[y, x] -= soil_removed
                    
                    # Deposit some at the lower position
                    deposit_amount = soil_removed * 0.5
                    nx, ny = min_pos
                    eroded_map[ny, nx] += deposit_amount
        
        self.heightmap = eroded_map
        return eroded_map
    
    def generate_water_bodies(self, sea_level: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """Generate water bodies (oceans, lakes) based on the heightmap.
        
        Args:
            sea_level: Threshold for ocean level (0.0 - 1.0)
            
        Returns:
            Tuple of (updated heightmap, water mask)
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before creating water bodies")
            
        print(f"Generating water bodies with sea level at {sea_level}...")
        
        # Create water mask (1 where water exists, 0 elsewhere)
        water_mask = np.zeros_like(self.heightmap)
        water_mask[self.heightmap < sea_level] = 1
        
        return self.heightmap, water_mask
    
    def add_mountains(self, mountain_scale: float = 0.5, 
                      peak_threshold: float = 0.7) -> np.ndarray:
        """Add mountain ranges to the terrain.
        
        Args:
            mountain_scale: Scale factor for mountain height
            peak_threshold: Threshold for mountain peaks
            
        Returns:
            Updated heightmap with mountain ranges
        """
        if self.heightmap is None:
            raise ValueError("Heightmap must be generated before adding mountains")
            
        print("Adding mountain ranges...")
        
        # Generate a mountain mask using different noise parameters
        mountain_mask = np.zeros((self.size, self.size))
        mountain_seed = self.seed + 1000  # Different seed for mountain generation
        
        for y in range(self.size):
            for x in range(self.size):
                mountain_mask[y][x] = snoise2(
                    x / 200, 
                    y / 200,
                    octaves=3,
                    persistence=0.6,
                    lacunarity=2.5,
                    base=mountain_seed
                )
        
        # Normalize mountain mask
        mountain_mask = (mountain_mask - np.min(mountain_mask)) / (np.max(mountain_mask) - np.min(mountain_mask))
        
        # Apply mountains where the mask exceeds the threshold
        mountain_terrain = self.heightmap.copy()
        mountain_areas = mountain_mask > peak_threshold
        
        # Amplify height in mountain areas
        mountain_terrain[mountain_areas] += mountain_scale * (mountain_mask[mountain_areas] - peak_threshold)
        
        # Renormalize if needed
        if np.max(mountain_terrain) > 1.0:
            mountain_terrain = mountain_terrain / np.max(mountain_terrain)
        
        self.heightmap = mountain_terrain
        return mountain_terrain
    
    def visualize(self, water_mask: Optional[np.ndarray] = None, 
                  title: str = "Terrain Heightmap"):
        """Visualize the current heightmap.
        
        Args:
            water_mask: Optional mask indicating water bodies
            title: Title for the plot
        """
        if self.heightmap is None:
            raise ValueError("No heightmap to visualize")
        
        plt.figure(figsize=(10, 8))
        
        # If water mask is provided, apply it to visualization
        if water_mask is not None:
            # Create a copy to avoid modifying the original
            terrain_vis = self.heightmap.copy()
            
            # Blue for water, terrain colors for land
            terrain_colors = plt.cm.terrain(terrain_vis)
            water_colors = plt.cm.Blues(terrain_vis)
            
            # Combine water and terrain colors using the mask
            for i in range(water_mask.shape[0]):
                for j in range(water_mask.shape[1]):
                    if water_mask[i, j] > 0:
                        terrain_colors[i, j] = water_colors[i, j]
            
            plt.imshow(terrain_colors)
        else:
            plt.imshow(self.heightmap, cmap='terrain')
            
        plt.colorbar(label='Elevation')
        plt.title(title)
        plt.axis('off')
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


if __name__ == "__main__":
    # Example usage
    generator = TerrainGenerator(size=512, seed=42)
    heightmap = generator.generate_heightmap(noise_type='simplex', scale=150.0)
    generator.visualize(title="Raw Heightmap")
    
    # Apply erosion
    eroded = generator.apply_erosion(iterations=30)
    generator.visualize(title="Eroded Heightmap")
    
    # Add water bodies
    _, water_mask = generator.generate_water_bodies(sea_level=0.35)
    generator.visualize(water_mask=water_mask, title="Terrain with Water Bodies")
    
    # Add mountains
    generator.add_mountains(mountain_scale=0.6, peak_threshold=0.7)
    generator.visualize(water_mask=water_mask, title="Terrain with Mountains")