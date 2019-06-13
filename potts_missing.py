# potts for filling missing values
import numpy as np

n_states = 4

def initialize(N, n_states = 4): 
	grid = np.random.randint(n_states, size=(N,N))
	return grid
	
def mcar(grid, pmiss):
  N = grid.shape[0]
  missing = np.random.choice([True, False], size=(N,N), p = [pmiss, 1 - pmiss])
  return missing

def runIter(grid, T):
  N = grid.shape[0]
  for x in range(N):
    for y in range(N): 
      # current point s
      s = grid[x, y]
      new_state = np.random.randint(n_states)
      # current energy state
      H = 0
      H -= (s == grid[(x+1)%N, y])
      H -= (s == grid[x, (y+1)%N])
      H -= (s == grid[(x-1)%N, y])
      H -= (s == grid[x, (y-1)%N])
      #Check later
      # new energy state
      altH = 0
      altH -= (new_state == grid[(x+1)%N, y])
      altH -= (new_state == grid[x, (y+1)%N])
      altH -= (new_state == grid[(x-1)%N, y])
      altH -= (new_state == grid[x, (y-1)%N])
      #Compute dE: (H = -  \sum_{<i,j>} S_i S_j )
      dE = altH - H
      # update grid
      if dE < 0:
        grid[x, y] = new_state
      elif np.random.rand() < np.exp(-dE/T): #high energy --> maybe flip?
        grid[x, y] = new_state
  return grid

# N = 30
# grid = initialize(N)
# 
# for i in range(100): 
#  	grid = runIter(grid, 5)
# 
# # Visualization
# import matplotlib.pyplot as plt 
# X, Y = np.meshgrid(range(N), range(N))
# plt.pcolormesh(X, Y, grid)
# plt.show()


# second step:
# in this somewhat organized grid, introduce missing values
def runIter_missing(grid, miss, T):
  N = grid.shape[0]
  for x in range(N):
    for y in range(N):
      # if missing, sample new observation
      # if observed, do nothing
      if (miss[x, y]):
        # current point s
        s = grid[x, y]
        new_state = np.random.randint(n_states)
        # current energy state
        H = 0
        H -= (s == grid[(x+1)%N, y])
        H -= (s == grid[x, (y+1)%N])
        H -= (s == grid[(x-1)%N, y])
        H -= (s == grid[x, (y-1)%N])
        # new energy state
        altH = 0
        altH -= (new_state == grid[(x+1)%N, y])
        altH -= (new_state == grid[x, (y+1)%N])
        altH -= (new_state == grid[(x-1)%N, y])
        altH -= (new_state == grid[x, (y-1)%N]) #Check later
        #Compute dE: (H = -  \sum_{<i,j>} S_i S_j )
        dE = altH - H
        if dE < 0:
          s = new_state
        elif np.random.rand() < np.exp(-dE/T): #high energy --> maybe flip?
          s = new_state
        grid[x, y] = s
  return grid


# # create a very structured grid
# grid = np.ones(shape = [N,N])
# grid[:(N/3), :] = 0
# grid[(2*N/3):, :] = 2
# 
# # plot initial data
# X, Y = np.meshgrid(range(N), range(N))
# plt.pcolormesh(X, Y, grid)
# plt.show()
# 
# # sample missing values
# miss = mcar(grid, 0.1)
# 
# # randomly impute missing values
# grid[miss] = np.random.randint(n_states, size=miss.sum())
# # plot imputed data
# X, Y = np.meshgrid(range(N), range(N))
# plt.pcolormesh(X, Y, grid)
# plt.show()
# 
# 
# # fill missing values
# for i in range(100): 
#  	grid = runIter_missing(grid, miss, 0.1)
# 
# # plot final data
# import matplotlib.pyplot as plt 
# X, Y = np.meshgrid(range(N), range(N))
# plt.pcolormesh(X, Y, grid)
# plt.show()



