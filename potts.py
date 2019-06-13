# reticulate::repl_python()
import numpy as np

n_states = 4
def initialize(N):
    grid = np.random.randint(n_states, size=(N,N))
    return grid

def pm_equal(x, y):
  return((x == y))

def runIter(grid, beta):
    N = grid.shape[0]
    for x in range(N):
        for y in range(N):
            # current point s
            s = grid[x, y]
            new_state = np.int64(np.random.randint(n_states))
            # current energy state
            neighbors = [grid[(x+1)%N, y], grid[x, (y+1)%N],
                  grid[(x-1)%N, y], grid[x, (y-1)%N]]
            H = -sum(neighbors == s)
            # Check later
            # new energy state
            altH = -sum(neighbors == new_state)
            #Compute dE: (H = -  \sum_{<i,j>} S_i S_j )
            dE = altH - H 
            dE
            if dE <= 0:
                s = new_state
            elif np.random.rand() < np.exp(-beta * dE): #high energy --> maybe flip?
                s = new_state
            grid[x, y] = s
    return grid

# # run test
# beta = 0.5
# N = 30
# grid = initialize(N)
# for i in range(300):
#     grid = runIter(grid, beta)
# 
# # Visualization
# import matplotlib.pyplot as plt 
# plt.figure()
# X, Y = np.meshgrid(range(N), range(N))
# plt.pcolormesh(X, Y, grid)
# plt.show()
