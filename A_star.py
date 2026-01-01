import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop

#  Grid setup 
grid_size = (20, 20)
grid = np.zeros(grid_size)

# 0 represents free space, 1 represents obstacles

# Obstacles
grid[5:10, 5] = 1
grid[12, 8:15] = 1
grid[3:8, 14] = 1
grid[15:18, 2:8] = 1
grid[8:12, 10:12] = 1
grid[2:4, 10:18] = 1
grid[18:20, 15:20] = 1

# Randomly set half of non-obstacle cells to 0.5 (ash)
mask = (grid == 0)
flat_indices = np.where(mask.flatten())[0]
np.random.seed(42)  # for reproducibility
rough_terrain = np.random.choice(flat_indices, size=len(flat_indices)//2, replace=False)
grid.flat[rough_terrain] = 0.5

start = (1, 2)  # fixed start point

#  A* algorithm 
def heuristic(a, b, current_pos=None, grid_ref=None):
    base_h = np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) # Heuristic function for cost estimation (Euclidean distance)
    # Increase heuristic cost by 1/4 if current position is in rough terrain (0.5)
    if current_pos is not None and grid_ref is not None and grid_ref[current_pos] == 0.5:
        return base_h * 1.25
    return base_h

def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heappush(open_set, (0 + heuristic(start, goal, start, grid), 0, start, [start])) # add the currrnt path and its elements to the heap Open_set
    visited = set() # create a set pf visited nodes, to prevent revisiing them
    
    while open_set:
        f, g, current, path = heappop(open_set) # remove top most element and puth its features in variables
        if current == goal:
            # If currrent node is the goal node, return the path
            return path
        if current in visited:
            # if we have visited this node, continue from the top
            continue
        visited.add(current) # add the current node to the set of visited nodes
        x, y = current # put the x and y coordinates of the current node in variables
        # neighbors (8-connectivity with diagonals)
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x+dx, y+dy # the new x and y coordinates based on available movements
            cost = 1 if (dx == 0 or dy == 0) else np.sqrt(2)  # diagonal moves g_cost is √2 
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]!=1 and (dx == 0 or dy == 0 or (grid[x+dx, y] != 1 and grid[x, y+dy] != 1)): # add another condition to avoid diagonal movements around edges
                # if path is traversible, add to the open set
                heappush(open_set, (g+cost + heuristic((nx,ny), goal, (nx,ny), grid), g+cost, (nx,ny), path+[(nx,ny)])) 
    return None

#  Matplotlib interactive plot 
fig, ax = plt.subplots()
ax.imshow(grid, cmap='Greys', origin='upper')

# Plot start point
start_plot, = ax.plot(start[1], start[0], 'go', markersize=10)

# Plot path placeholder
path_plot, = ax.plot([], [], 'r-', linewidth=2)

def onclick(event):
    global start
    if event.inaxes != ax: 
        # if click is outside the axes
        return
    goal = (int(round(event.ydata)), int(round(event.xdata)))
    path = a_star(grid, start, goal)
    if path:
        # if traversible path is available, move to the goal using the path
        px, py = zip(*path)
        path_plot.set_data(py, px)
        start = goal  # update start point to the clicked location
        start_plot.set_data([goal[1]], [goal[0]])  # update the visual marker
        fig.canvas.draw()
        fig.canvas.draw()
    else:
        print("No path found!")

fig.canvas.mpl_connect('button_press_event', onclick)
plt.title("Click anywhere to plan path from green start point")
plt.show()

### There are things that are not taken into account here
### The cost of steering, this doesnt apply here because we are using a point
### The fact that there might be some uknown regions neither 0 or 1. 
### Local pah planning, whihc actually gives the controls for the movements
