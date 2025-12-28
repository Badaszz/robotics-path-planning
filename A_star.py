import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

# --- Grid setup ---
grid_size = (20, 20)
grid = np.zeros(grid_size)

# 0 represents free space, 1 represents obstacles

# Obstacles
grid[5:10, 5] = 1
grid[12, 8:15] = 1
grid[3:8, 14] = 1

start = (1, 2)  # fixed start point

# --- A* algorithm ---
def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
    visited = set()
    
    while open_set:
        f, g, current, path = heappop(open_set)
        if current == goal:
            return path
        if current in visited:
            continue
        visited.add(current)
        x, y = current
        # neighbors (4-connectivity)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]==0:
                heappush(open_set, (g+1 + heuristic((nx,ny), goal), g+1, (nx,ny), path+[(nx,ny)]))
    return None

# --- Matplotlib interactive plot ---
fig, ax = plt.subplots()
ax.imshow(grid, cmap='Greys', origin='upper')

# Plot start point
start_plot, = ax.plot(start[1], start[0], 'go', markersize=10)

# Plot path placeholder
path_plot, = ax.plot([], [], 'r-', linewidth=2)

def onclick(event):
    if event.inaxes != ax:
        return
    goal = (int(round(event.ydata)), int(round(event.xdata)))
    path = astar(grid, start, goal)
    if path:
        px, py = zip(*path)
        path_plot.set_data(py, px)
        fig.canvas.draw()
    else:
        print("No path found!")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Click anywhere to plan path from green start point")
plt.show()
