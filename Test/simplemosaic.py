import numpy as np
import random
from collections import deque

# --------------------------------------------------------
# TOY MOSAIC CARTOGRAM
# --------------------------------------------------------

class MosaicCartogram:
    def __init__(self, adjacency, weights, grid_size=50):
        """
        adjacency: dict {region: [neighbors]}
        weights:   dict {region: desired_tile_count}
        """
        self.G = adjacency
        self.weights = weights
        self.grid_size = grid_size

        # Grid of region labels, -1 = sea/unassigned
        self.grid = -1 * np.ones((grid_size, grid_size), dtype=int)

        # Region start locations
        self.start_positions = {
            r: (random.randint(0, grid_size-1), random.randint(0, grid_size-1))
            for r in adjacency
        }

        # Place one seed tile per region
        for r, (x, y) in self.start_positions.items():
            self.grid[x, y] = r

    # --------------------------------------------------------

    def neighbors4(self, x, y):
        for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                yield nx, ny

    # --------------------------------------------------------

    def grow(self, max_iter=50000):
        """Greedy BFS-style tile assignment until all regions reach target size."""
        region_sizes = self.compute_sizes()

        queue = deque()

        # Initialize queue with boundary tiles for each region
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                r = self.grid[x, y]
                if r == -1: continue
                for nx, ny in self.neighbors4(x, y):
                    if self.grid[nx, ny] == -1:
                        queue.append((r, nx, ny))

        it = 0
        while queue and it < max_iter:
            it += 1
            r, x, y = queue.popleft()

            # Skip if already owned
            if self.grid[x, y] != -1:
                continue

            # Skip if this region is already “big enough”
            if region_sizes[r] >= self.weights[r]:
                continue

            # Assign tile
            self.grid[x, y] = r
            region_sizes[r] += 1

            # Push new boundary neighbors to queue
            for nx, ny in self.neighbors4(x, y):
                if self.grid[nx, ny] == -1:
                    queue.append((r, nx, ny))

        return self.grid

    # --------------------------------------------------------

    def compute_sizes(self):
        sizes = {r: 0 for r in self.G}
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                r = self.grid[x, y]
                if r != -1:
                    sizes[r] += 1
        return sizes


# --------------------------------------------------------
# EXAMPLE USAGE
# --------------------------------------------------------

if __name__ == "__main__":

    # Simple test graph (triangle of 3 regions)
    adjacency = {
        0: [1,2],
        1: [0,2],
        2: [0,1]
    }

    # Desired tile counts
    weights = {
        0: 60,
        1: 40,
        2: 30
    }

    MC = MosaicCartogram(adjacency, weights, grid_size=40)
    
    # Visualize original map (seed positions before growth)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Original map (initial seed positions)
    original_grid = MC.grid.copy()
    axes[0].imshow(original_grid, cmap="tab20", vmin=-1, vmax=max(adjacency.keys()))
    axes[0].set_title("Original Map (Seed Positions)")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    
    # Add region labels at seed positions
    for region, (x, y) in MC.start_positions.items():
        axes[0].text(y, x, str(region), color='white', 
                    ha='center', va='center', fontweight='bold', fontsize=12)
    
    # Grow the mosaic
    grid = MC.grow()
    
    # Plot 2: Mosaic cartogram (after growth)
    im = axes[1].imshow(grid, cmap="tab20", vmin=-1, vmax=max(adjacency.keys()))
    axes[1].set_title("Mosaic Cartogram (After Growth)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label('Region ID')
    
    # Print statistics
    sizes = MC.compute_sizes()
    print("\nRegion Statistics:")
    print(f"{'Region':<10} {'Target':<10} {'Actual':<10}")
    print("-" * 30)
    for r in sorted(adjacency.keys()):
        print(f"{r:<10} {weights[r]:<10} {sizes[r]:<10}")
    
    plt.tight_layout()
    plt.show()
