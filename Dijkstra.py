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

start = (0, 0)  # fixed start point

def dijkstra(grid, start, goal):
    rows,cols = grid.shape
    open_set = []
    heappush(open_set, (0, start, [start])) # add the current path and its elements to the heap Open_set
    visited = set() # create a set of visited nodes, to prevent revisiting them
    
    while open_set:
        cost, current_pos, path = heappop(open_set)
        if current_pos == goal:
            # check if current node is the goal node, return the path
            return path
        if current_pos in visited:
            # if we have visited this node, continue from the top
            continue
        visited.add(current_pos) # add the current node to the set of visited nodes
        x, y = current_pos
        # neighbors (8-connectivity with diagonals) check through all possible movements
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x+dx, y+dy # the new x and y coordinates based on available movements
            move_cost = 1 if (dx == 0 or dy == 0) else np.sqrt(2)  # diagonal moves cost is √2 pythagoras did eventually come in handy
            if 0<=nx<rows and 0<=ny<cols and grid[nx,ny]!=1 and (dx == 0 or dy == 0 or (grid[x+dx, y] != 1 and grid[x, y+dy] != 1)): # add another condition to avoid diagonal movements through edges
                # if path is traversable, add to the open set
                heappush(open_set, (cost + move_cost, (nx,ny), path + [(nx,ny)]))
    return None # if no path is found

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
    path = dijkstra(grid, start, goal)
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
plt.title("Click anywhere to plan path from green start point (Djikstra's Algorithm)")
plt.show()