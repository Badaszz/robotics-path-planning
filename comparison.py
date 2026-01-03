import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from collections import defaultdict

# Grid setup
grid_size = (20, 20)
grid = np.zeros(grid_size)

# 0 = free space, 0.5 = rough terrain (ash), 1 = obstacles
grid[5:10, 5] = 1
grid[12, 8:15] = 1
grid[3:8, 14] = 1
grid[15:18, 2:8] = 1
grid[8:12, 10:12] = 1
grid[2:4, 10:18] = 1
grid[18:20, 15:20] = 1

# Randomly set half of non-obstacle cells to 0.5 (rough terrain)
mask = (grid == 0)
flat_indices = np.where(mask.flatten())[0]
np.random.seed(42)
ash_indices = np.random.choice(flat_indices, size=len(flat_indices)//2, replace=False)
grid.flat[ash_indices] = 0.5

start = (1, 2)


def get_cost(grid_val):
    """Get traversal cost based on grid value."""
    if grid_val == 0:
        return 1.0
    elif grid_val == 0.5:
        return 1.25  # 25% more costly for rough terrain
    else:
        return float('inf')


def heuristic(a, b):
    """Euclidean distance heuristic."""
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def dijkstra(grid, start, goal):
    """
    Dijkstra's algorithm.
    Returns: (path, total_cost, nodes_considered)
    """
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0, start, [start]))  # (cost, current, path)
    visited = set()
    nodes_considered = 0
    
    while open_set:
        g, current, path = heappop(open_set)
        nodes_considered += 1
        
        if current == goal:
            return path, g, nodes_considered
        
        if current in visited:
            continue
        visited.add(current)
        
        x, y = current
        # 8-connectivity neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] != 1:
                # Prevent diagonal corner-cutting
                if dx != 0 and dy != 0:  # diagonal move
                    if grid[x + dx, y] == 1 or grid[x, y + dy] == 1:
                        continue
                
                if (nx, ny) not in visited:
                    move_cost = np.sqrt(2) if (dx != 0 and dy != 0) else 1
                    cell_cost = get_cost(grid[nx, ny])
                    total_cost = g + move_cost * cell_cost
                    heappush(open_set, (total_cost, (nx, ny), path + [(nx, ny)]))
    
    return None, float('inf'), nodes_considered


def a_star(grid, start, goal):
    """
    A* algorithm.
    Returns: (path, total_cost, nodes_considered)
    """
    rows, cols = grid.shape
    open_set = []
    h = heuristic(start, goal)
    heappush(open_set, (h, 0, start, [start]))  # (f, g, current, path)
    visited = set()
    nodes_considered = 0
    
    while open_set:
        f, g, current, path = heappop(open_set)
        nodes_considered += 1
        
        if current == goal:
            return path, g, nodes_considered
        
        if current in visited:
            continue
        visited.add(current)
        
        x, y = current
        # 8-connectivity neighbors
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] != 1:
                # Prevent diagonal corner-cutting
                if dx != 0 and dy != 0:  # diagonal move
                    if grid[x + dx, y] == 1 or grid[x, y + dy] == 1:
                        continue
                
                if (nx, ny) not in visited:
                    move_cost = np.sqrt(2) if (dx != 0 and dy != 0) else 1
                    cell_cost = get_cost(grid[nx, ny])
                    new_g = g + move_cost * cell_cost
                    h_val = heuristic((nx, ny), goal)
                    heappush(open_set, (new_g + h_val, new_g, (nx, ny), path + [(nx, ny)]))
    
    return None, float('inf'), nodes_considered


def compare_algorithms(grid, start, goal):
    """
    Run both algorithms and return results.
    """
    # Run Dijkstra
    dijkstra_path, dijkstra_cost, dijkstra_nodes = dijkstra(grid, start, goal)
    
    # Run A*
    astar_path, astar_cost, astar_nodes = a_star(grid, start, goal)
    
    return {
        'dijkstra': {
            'path': dijkstra_path,
            'cost': dijkstra_cost,
            'nodes': dijkstra_nodes,
        },
        'astar': {
            'path': astar_path,
            'cost': astar_cost,
            'nodes': astar_nodes,
        }
    }


def visualize_comparison(grid, start, goal, results):
    """
    Visualize both paths on the same grid.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display grid: obstacles in black, rough terrain in gray, free space in white
    display_grid = np.copy(grid)
    display_grid[display_grid == 0.5] = 0.3  # Rough terrain in gray
    ax.imshow(display_grid, cmap='Greys', origin='upper', vmin=0, vmax=1)
    
    # Plot start and goal
    ax.plot(start[1], start[0], 'go', markersize=12, label='Start', zorder=5)
    ax.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal', zorder=5)
    
    # Plot Dijkstra path
    if results['dijkstra']['path']:
        dijkstra_path = results['dijkstra']['path']
        px, py = zip(*dijkstra_path)
        ax.plot(py, px, 'b-', linewidth=2, label='Dijkstra', alpha=0.7, zorder=3)
        ax.plot(py, px, 'bo', markersize=4, alpha=0.5, zorder=2)
    
    # Plot A* path
    if results['astar']['path']:
        astar_path = results['astar']['path']
        px, py = zip(*astar_path)
        ax.plot(py, px, 'c--', linewidth=2, label='A*', alpha=0.7, zorder=3)
        ax.plot(py, px, 'cs', markersize=4, alpha=0.5, zorder=2)
    
    # Create legend and info text
    ax.legend(loc='lower right', fontsize=10)
    
    # Add info box with stats
    info_text = "Algorithm Comparison\n"
    info_text += f"Start: {start}, Goal: {goal}\n\n"
    
    if results['dijkstra']['path']:
        info_text += f"Dijkstra:\n"
        info_text += f"  Path length: {len(results['dijkstra']['path'])}\n"
        info_text += f"  Cost: {results['dijkstra']['cost']:.2f}\n"
        info_text += f"  Nodes considered: {results['dijkstra']['nodes']}\n\n"
    else:
        info_text += "Dijkstra: No path found\n\n"
    
    if results['astar']['path']:
        info_text += f"A*:\n"
        info_text += f"  Path length: {len(results['astar']['path'])}\n"
        info_text += f"  Cost: {results['astar']['cost']:.2f}\n"
        info_text += f"  Nodes considered: {results['astar']['nodes']}\n"
    else:
        info_text += "A*: No path found\n"
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    ax.set_title("A* vs Dijkstra Path Planning Comparison", fontsize=14, fontweight='bold')
    ax.set_xlabel("Y")
    ax.set_ylabel("X")
    plt.tight_layout()
    plt.show()


def interactive_comparison(grid, start):
    """
    Interactive mode: click to set goal and compare algorithms.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display grid
    display_grid = np.copy(grid)
    display_grid[display_grid == 0.5] = 0.3
    ax.imshow(display_grid, cmap='Greys', origin='upper', vmin=0, vmax=1)
    
    start_plot, = ax.plot(start[1], start[0], 'go', markersize=12, label='Start')
    ax.legend()
    
    def onclick(event):
        if event.inaxes != ax:
            return
        goal = (int(round(event.ydata)), int(round(event.xdata)))
        results = compare_algorithms(grid, start, goal)
        
        # Clear previous paths
        ax.clear()
        display_grid = np.copy(grid)
        display_grid[display_grid == 0.5] = 0.3
        ax.imshow(display_grid, cmap='Greys', origin='upper', vmin=0, vmax=1)
        
        # Plot start and goal
        ax.plot(start[1], start[0], 'go', markersize=12, label='Start', zorder=5)
        ax.plot(goal[1], goal[0], 'r*', markersize=20, label='Goal', zorder=5)
        
        # Plot paths
        if results['dijkstra']['path']:
            px, py = zip(*results['dijkstra']['path'])
            ax.plot(py, px, 'b-', linewidth=2, label='Dijkstra', alpha=0.7, zorder=3)
        
        if results['astar']['path']:
            px, py = zip(*results['astar']['path'])
            ax.plot(py, px, 'c--', linewidth=2, label='A*', alpha=0.7, zorder=3)
        
        # Info text
        info_text = "Algorithm Comparison\n"
        info_text += f"Start: {start}, Goal: {goal}\n\n"
        
        if results['dijkstra']['path']:
            info_text += f"Dijkstra:\n"
            info_text += f"  Cost: {results['dijkstra']['cost']:.2f}\n"
            info_text += f"  Nodes: {results['dijkstra']['nodes']}\n\n"
        else:
            info_text += "Dijkstra: No path\n\n"
        
        if results['astar']['path']:
            info_text += f"A*:\n"
            info_text += f"  Cost: {results['astar']['cost']:.2f}\n"
            info_text += f"  Nodes: {results['astar']['nodes']}\n"
        else:
            info_text += "A*: No path\n"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        
        ax.legend(loc='lower right')
        ax.set_title("Click to set new goal", fontsize=12)
        fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    ax.set_title("Click to set goal (comparing A* vs Dijkstra)", fontsize=12)
    plt.show()


if __name__ == '__main__':
    # Static comparison with a fixed goal
    # goal = (15, 15)  # Far corner, likely reachable
    # results = compare_algorithms(grid, start, goal)
    # visualize_comparison(grid, start, goal, results)
    
    # Uncomment for interactive mode
    interactive_comparison(grid, start)
