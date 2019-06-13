import numpy as np

n_states = 2
def initialize(N):
    grid = 2*np.random.randint(n_states, size=(N,N))-1
    return grid

def pm_equal(x, y):
  return((x == y) * 2 - 1)

def runIter(grid, beta):
    N = grid.shape[0]
    for x in range(N):
        for y in range(N):
            # current point s
            s = grid[x, y]
            new_state = 2*np.random.randint(n_states)-1
            # current energy state
            H = 0
            H -= pm_equal(s, grid[(x+1)%N, y])
            H -= pm_equal(s, grid[x, (y+1)%N])
            H -= pm_equal(s, grid[(x-1)%N, y])
            H -= pm_equal(s, grid[x, (y-1)%N]) 
            # Check later
            # new energy state
            altH = 0
            altH -= pm_equal(new_state, grid[(x+1)%N, y])
            altH -= pm_equal(new_state, grid[x, (y+1)%N])
            altH -= pm_equal(new_state, grid[(x-1)%N, y])
            altH -= pm_equal(new_state, grid[x, (y-1)%N]) 
            #Compute dE: (H = -  \sum_{<i,j>} S_i S_j )
            dE = altH - H
            if dE < 0:
                s = new_state
            elif np.random.rand() < np.exp(-beta * dE): #high energy --> maybe flip?
                s = new_state
            grid[x, y] = s
    return grid

# # test functionality
# N = 30
# grid = initialize(N)
# for i in range(500):
#     grid = runIter(grid, 0.1)
# 
# # Visualization
# import matplotlib.pyplot as plt 
# plt.figure()
# X, Y = np.meshgrid(range(N), range(N))
# plt.pcolormesh(X, Y, grid)
# plt.show()
