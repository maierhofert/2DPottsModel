
import numpy as np
import matplotlib.colors as pltcols
import matplotlib.pyplot as plt 

# implement the swendsen wang algorithm
from potts_missing import *
from potts_swendsen_wang import *
# from potts_estimate_beta_missing_values import *
from potts_estimate_beta import *


colors = ['yellow','blue','green', 'red']
miss_colors = ['grey', 'yellow','blue','green', 'red']
cMap = pltcols.ListedColormap(colors, N = n_states)
cMapmiss = pltcols.ListedColormap(miss_colors)


def runIter_SW_missing(grid, miss, beta, n_states = 4):
    N = grid.shape[0]
    # sample observed edges 
    # with probability 1 - exp(-beta)
    p = 1 - np.exp(-beta)
    # this a N by N by 4 (top, bottom, left right) array
    edges = np.random.choice(a=[True, False], size=(N, N, 4), p=[p, 1-p])
    # make sure edges are only between observations of the same class
    # 
    edges[np.roll(a=grid, shift=1, axis=1) != grid, 0] = False
    # that would forbid clusters that contain observed values
    # edges[np.invert(np.roll(a=miss, shift=1, axis=1)), 0] = False
    # 
    edges[np.roll(a=grid, shift=-1, axis=1) != grid, 1] = False
    # 
    edges[np.roll(a=grid, shift=1, axis=0) != grid, 2] = False
    # 
    edges[np.roll(a=grid, shift=-1, axis=0) != grid, 3] = False
    
    # randomly initialize seed for cluster
    # only draw from missing values
    missing_coordinates = np.where(miss)
    length = missing_coordinates[0].shape[0]
    missing_xy_loc = np.transpose(missing_coordinates)
    missing_xyloc_array = [np.array(xy) for xy in missing_xy_loc]
    # sample from the indices of missing observations
    loc = np.random.randint(missing_xyloc_array.__len__())
    xy_loc = missing_xyloc_array[loc]
    
    # initialize cluster
    xylist = [xy_loc]
    reached_end = False
    contain_observed = False
    i = 0
    while (not reached_end) and (i < N*N - 1) and (not contain_observed):
      # find neighbors
      this_x = xylist[i][0]
      this_y = xylist[i][1]
      neighbors = np.where(edges[this_x, this_y,:])
      if np.any(edges[this_x, this_y,:]):
        for neighbor in np.nditer(neighbors):
          # print(neighbor)
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
      # stop if end of cluster is reached
      if i == len(xylist):
        reached_end = True
      # stop if cluster contains observed values
      if not all([any((a == xy).all() for xy in missing_xyloc_array) for a in xylist]):
        contain_observed = True
        # print("Break: contained observed value")
    # resulting cluster
    # print(xylist)
    #
    # flip entire cluster
    if not contain_observed:
      new_state = np.int64(np.random.randint(n_states))
      while new_state == grid[this_x, this_y]:
        new_state = np.int64(np.random.randint(n_states))
      # all x and y indices
      idx = [xy[0] for xy in xylist]
      idy = [xy[1] for xy in xylist]
      grid[idx, idy] = new_state
    return(grid)

# create grid
n_states = 4
N = 20
beta = 1

# set seed
np.random.seed(123)

# initialize grid
grid = initialize(N, n_states)
# visualization
plt.figure()
plt.matshow(A=grid, cmap = cMap, vmin=0, vmax=n_states)
# plt.show()
# save figure
plt.savefig("plots/grid_initial.png", bbox_inches='tight')

# run 200 iterations for convergence
for i in range(401):
    grid = runIter_SW(grid, beta, n_states)
# visualization
plt.figure()
plt.matshow(A=grid, cmap = cMap, vmin=0, vmax=n_states)
# plt.show()
# save figure
plt.savefig("plots/grid_b1_it401.png", bbox_inches='tight')

# run one more iteration for showing SW
grid = runIter_SW(grid, beta)
# visualization
plt.figure()
plt.matshow(A=grid, cmap = cMap, vmin=0, vmax=n_states)
# plt.show()
# save figure
plt.savefig("plots/grid_b1_it402.png", bbox_inches='tight')


# sample missingness
miss = mcar(grid, 0.5)
# Visualization
plt.matshow(A=miss, cmap = cMap)
# plt.show()

# randomly initialize missing values
grid[miss] = np.random.randint(n_states, size=miss.sum())

# Visualization
plot_grid = grid.copy()
plot_grid[miss] = -1

plt.figure()
plt.matshow(A=plot_grid, cmap = cMapmiss)
# plt.show()
# save figure
plt.savefig("plots/grid_b1_it402_missing.png", bbox_inches='tight')

# step with missing values 
# and true beta
grid_true_beta = grid.copy()
# burn in needed
for i in range(50):
    grid_true_beta = runIter_SW_missing(grid_true_beta, miss, beta)
# Visualization
plt.matshow(A=grid_true_beta, cmap = cMap, vmin=0, vmax=n_states)
# plt.show()

# set up Gibbs Sampler that estimates beta from data
# then fills missing values using estimated beta
grid_est_beta = grid.copy()
# beta_path = [0.5]
B = 10000
beta_path = np.zeros((B))
# longer burn in needed
for i in range(B):
  # estimate beta using MPL
  est_beta = estimate_beta_MPL(grid=grid_est_beta, betas=np.arange(0.1, 1.5, 0.01), n_states=n_states)
  # beta_path.append(est_beta.copy())
  beta_path[i] = est_beta.copy()
  # estimate missing values given beta
  grid_est_beta = runIter_SW_missing(grid_est_beta, miss, est_beta)

# Visualize path of estimated beta
plt.figure()
plt.plot(beta_path)
plt.axhline(beta, color="green")
# plt.show()
# save figure
plt.savefig("plots/beta_b1_it500_missing.png", bbox_inches='tight')


