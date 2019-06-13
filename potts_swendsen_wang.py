import numpy as np

# implement the swendsen wang algorithm
def runIter_SW(grid, beta, n_states = 4):
    N = grid.shape[0]
    # sample observed edges 
    # with probability 1 - exp(-beta)
    p = 1 - np.exp(-beta)
    # this a N by N by 4 (top, bottom, left right) array
    edges = np.random.choice(a=[True, False], size=(N, N, 4), p=[p, 1-p])
    # make sure edges are only between observations of the same class
    # check current position with one to the ...
    # 
    edges[np.roll(a=grid, shift=1, axis=1) != grid, 0] = False
    # 
    edges[np.roll(a=grid, shift=-1, axis=1) != grid, 1] = False
    # 
    edges[np.roll(a=grid, shift=1, axis=0) != grid, 2] = False
    # 
    edges[np.roll(a=grid, shift=-1, axis=0) != grid, 3] = False
    
    # randomly initialize seed for cluster
    # intial draw
    # xy_loc = np.array([0, 4])
    xy_loc = np.random.randint(N, size = (1, 2))[0]
    # initialize cluster
    xylist = [xy_loc]
    reached_end = False
    i = 0
    while (not reached_end) and (i < N*N - 1):
      # find neighbors
      this_x = xylist[i][0]
      this_y = xylist[i][1]
      neighbors = np.where(edges[this_x, this_y,:])
      if np.any(edges[this_x, this_y,:]):
        for neighbor in np.nditer(neighbors):
          # print(neighbor)
          # 
          if neighbor == 3:
            new_x = this_x + 1
            new_xy = np.array([new_x%N, this_y])
            if not any((new_xy == loc).all() for loc in xylist):
              xylist.append(new_xy)
          # 
          elif neighbor == 2:
            new_x = this_x - 1
            new_xy = np.array([new_x%N, this_y])
            if not any((new_xy == loc).all() for loc in xylist):
              xylist.append(new_xy)
          # 
          elif neighbor == 1:
            new_y = this_y + 1
            new_xy = np.array([this_x, new_y%N])
            if not any((new_xy == loc).all() for loc in xylist):
              xylist.append(new_xy)
          # 
          elif neighbor == 0:
            new_y = this_y - 1
            new_xy = np.array([this_x, new_y%N])
            if not any((new_xy == loc).all() for loc in xylist):
              xylist.append(new_xy)
          # print(xylist)
      # print("i = ", i)
      i += 1
      if i == len(xylist):
        reached_end = True
    # resulting cluster    
    # print(xylist)
    
    # flip entire cluster
    new_state = np.int64(np.random.randint(n_states))
    while new_state == grid[this_x, this_y]:
      new_state = np.int64(np.random.randint(n_states))
    # all x and y indices
    idx = [xy[0] for xy in xylist]
    idy = [xy[1] for xy in xylist]
    grid[idx, idy] = new_state
    return(grid)
    
    
    # # compute neighbors for resulting cluster
    # all_neighbors = []
    # all_neighbors_xylist = list(xylist)
    # for idx, idy in xylist:
    #   print(np.array([idx, idy]))
    #   # x + 1
    #   xp1 = np.array([(idx + 1)%N, idy])
    #   if not any((xp1 == loc).all() for loc in all_neighbors_xylist):
    #     all_neighbors.append(grid[(idx + 1)%N, idy])
    #     all_neighbors_xylist.append(xp1)
    #   # x - 1
    #   xm1 = np.array([(idx - 1)%N, idy])
    #   if not any((xm1 == loc).all() for loc in all_neighbors_xylist):
    #     all_neighbors.append(grid[(idx - 1)%N, idy])
    #     all_neighbors_xylist.append(xm1)
    #   # y + 1
    #   yp1 = np.array([idx, (idy + 1)%N])
    #   if not any((yp1 == loc).all() for loc in all_neighbors_xylist):
    #     all_neighbors.append(grid[idx, (idy + 1)%N])
    #     all_neighbors_xylist.append(yp1)
    #   # y - 1
    #   ym1 = np.array([idx, (idy - 1)%N])
    #   if not any((ym1 == loc).all() for loc in all_neighbors_xylist):
    #     all_neighbors.append(grid[idx, (idy - 1)%N])
    #     all_neighbors_xylist.append(ym1)
    #   # print(all_neighbors)
      
      
  
# B = 10
# grid = initialize(N)
# for b in range(B):
#   grid = runIter_SW(grid=grid, beta=0.01)
#   
# # Visualization of clustering algorithm
# plot_grid = grid.copy()
# for idx, idy in xylist:
#   plot_grid[idx, idy] = -1
# import matplotlib.colors as pltcols
# colors = ['yellow','blue','green', 'red']
# miss_colors = ['grey', 'yellow','blue','green', 'red']
# cMap = pltcols.ListedColormap(colors, N = n_states)
# cMapmiss = pltcols.ListedColormap(miss_colors)
# 
# plt.figure()
# plt.matshow(A=plot_grid, cmap = cMapmiss)
# plt.show()
# 
# plt.matshow(A=grid, cmap = cMap, vmin=0, vmax=n_states)
# plt.show()
# # plt.matshow(A=grid, cmap = cMap, vmin=0, vmax=4)
# # plt.show()
# # plt.matshow(A=grid, cmap = cMap, vmin=0, vmax=4)
# # plt.show()
# 

