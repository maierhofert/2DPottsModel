# ugly copy and paste from potts_SW_missing.py

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

#########################################################
# # animations

import numpy as np
import matplotlib.colors as pltcols
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# implement the swendsen wang algorithm
from potts_missing import *
from potts_swendsen_wang import *
# from potts_estimate_beta_missing_values import *
from potts_estimate_beta import *

# colors = ['yellow','blue','green', 'red']
# miss_colors = ['grey', 'yellow','blue','green', 'red']
colors = ['palegoldenrod','palegreen','lightcoral','lightblue']
miss_colors = ['grey', 'palegoldenrod','palegreen','lightcoral','lightblue']

cMap = pltcols.ListedColormap(colors, N = n_states)
cMapmiss = pltcols.ListedColormap(miss_colors)



# recreate grid
n_states = 4
N = 20
beta = 1

# set seed
np.random.seed(123)

# initialize grid
grid = initialize(N, n_states)
# run 200 iterations for convergence
for i in range(401):
    grid = runIter_SW(grid, beta, n_states)
# run one more iteration for showing SW
grid = runIter_SW(grid, beta)

# sample missingness
miss = mcar(grid, 0.5)
# randomly initialize missing values
grid[miss] = np.random.randint(n_states, size=miss.sum())


#########################################################
# # animations
def update(data):
    mat.set_data(data)
    return mat

def data_gen():
    while True:
      global grid_est_beta
      # estimate beta using MPL
      est_beta = estimate_beta_MPL(grid=grid_est_beta, betas=np.arange(0.6, 1.4, 0.01), n_states=n_states)
      # estimate missing values given beta
      grid_est_beta = runIter_SW_missing(grid_est_beta, miss, est_beta)
      yield grid_est_beta

fig, ax = plt.subplots()

# set up Gibbs Sampler that estimates beta from data
# then fills missing values using estimated beta
grid_est_beta = grid.copy()

mat = ax.matshow(grid_est_beta)
# plt.show(mat)
ani = animation.FuncAnimation(fig, update, data_gen, interval=10, repeat = True, repeat_delay = 500,
                              save_count=500)
# plt.show()

ani.save('plots/grid_missing_burnin.gif', dpi=200, writer='imagemagick')



####################################################################
# run the update a bunch of times to get chain in equilibrium
for i in range(1000):
  # est_beta = estimate_beta_MPL(grid=grid_est_beta, betas=np.arange(0.1, 1.5, 0.01), n_states=n_states)
  # estimate missing values given beta
  grid_est_beta = runIter_SW_missing(grid=grid_est_beta, miss=miss, beta=1)


# plot grid working with missing values
ani = animation.FuncAnimation(fig, update, data_gen, interval=10, repeat_delay = 500,
                              save_count=500)
# plt.show()

ani.save('plots/grid_missing_equilibrium.gif', dpi=200, writer='imagemagick')


#########################################################
# # SW in equlibrium without missing values
def update(data):
    mat.set_data(data)
    return mat

def data_gen():
    while True:
      global grid_beta
      grid_beta = runIter_SW(grid=grid_beta, beta=1, n_states=4)
      yield grid_beta

fig, ax = plt.subplots()

# set up Gibbs Sampler that estimates beta from data
# then fills missing values using estimated beta
grid_beta = grid.copy()

for i in range(500):
  # est_beta = estimate_beta_MPL(grid=grid_est_beta, betas=np.arange(0.1, 1.5, 0.01), n_states=n_states)
  # estimate missing values given beta
  grid_beta = runIter_SW(grid=grid_beta, beta=1)


mat = ax.matshow(grid_beta)
# plt.show(mat)
ani = animation.FuncAnimation(fig, update, data_gen, interval=10, repeat = True, repeat_delay = 500,
                              save_count=500)
# plt.show()

ani.save('plots/grid_SW_equilibrium.gif', dpi=200, writer='imagemagick')

# ###############################################################
# # Gibbs sampler in equlibrium without missing values
def runIter(grid, beta, n_states = 4):
    N = grid.shape[0]
    x, y = np.random.randint(0, N, 2)
    # current point s
    s = grid[x, y]
    new_state = np.int64(np.random.randint(n_states))
    # current energy state
    neighbors = [grid[(x+1)%N, y], grid[x, (y+1)%N],
          grid[(x-1)%N, y], grid[x, (y-1)%N]]
    H = -sum(neighbors == s)
    # new energy state
    altH = -sum(neighbors == new_state)
    #Compute dE: (H = -  \sum_{<i,j>} S_i S_j )
    dE = altH - H 
    if dE <= 0:
        grid[x, y] = new_state
    elif np.random.rand() < np.exp(-beta * dE): #high energy --> maybe flip?
        grid[x, y] = new_state
    return grid

def update(data):
    mat.set_data(data)
    return mat 

def data_gen():
    while True:
      global grid_beta
      grid_beta = runIter(grid=grid_beta, beta=1, n_states=4)
      yield grid_beta

fig, ax = plt.subplots()
    
# set up Gibbs Sampler that estimates beta from data
# then fills missing values using estimated beta
grid_beta = grid.copy()

for i in range(10000):
  # estimate missing values given beta
  grid_beta = runIter(grid=grid_beta, beta=1)


mat = ax.matshow(grid_beta)
# plt.show(mat)
ani = animation.FuncAnimation(fig, update, data_gen, interval=5, repeat = True, repeat_delay = 500,
                              save_count=500)
# plt.show()

ani.save('plots/grid_Gibbs_equilibrium.gif', dpi=200, writer='imagemagick')








