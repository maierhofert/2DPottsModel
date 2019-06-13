import numpy as np

n_states = 4
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


N = 30
grid = initialize(N)
for i in range(300):
    grid = runIter(grid, 0.3)

# Visualization
import matplotlib.pyplot as plt 
plt.figure()
X, Y = np.meshgrid(range(N), range(N))
plt.pcolormesh(X, Y, grid)
plt.show()


# ############################################################
########################
# estimate beta using MPLE in Ising model
# estimate H_n, the energy of the current state
this_H_n = H_n(grid, 1)
# for x in range(N):
#   for y in range(N):
#     # current point s
#     s = grid[x, y]
#     # current energy state
#     neighbors = [grid[(x+1)%N, y], grid[x, (y+1)%N],
#                 grid[(x-1)%N, y], grid[x, (y-1)%N]]
#     # total
#     H_n += np.sum(np.dot(neighbors, s))
# H_n = 0.5 * H_n
this_H_n

# ###############################################################
# WARNING, THIS ONLY WORKS FOR THE ISING MODEL, I.E. n_states = 2
# estimate energy
power = np.arange(-5, 1, 0.1)

total_sum = np.zeros_like(power, "float")
for b in range(len(power)):
  this_beta = np.exp(power[b])
  total_sum[b] = 0
  for x in range(N):
    for y in range(N):
      # current point s
      s = grid[x, y]
      # current energy state
      neighbors = [grid[(x+1)%N, y], grid[x, (y+1)%N],
                  grid[(x-1)%N, y], grid[x, (y-1)%N]]
      tanh_arg = np.dot(this_beta, neighbors)
      tanh_sum = np.tanh(np.sum(tanh_arg))
      # total
      total_sum[b] += np.sum(np.dot(neighbors, tanh_sum))
  total_sum[b] = 0.5 * total_sum[b]
  # break if larger than needed
  if total_sum[b] >= this_H_n:
    break
total_sum

# checked betas
np.exp(power)

# estimated beta
beta_hat = np.exp(power[b - 1])
beta_hat


# plot
plt.figure()
plt.plot(np.exp(power)[:(b+1)], total_sum[:(b+1)])
plt.axhline(y=this_H_n, color='r', linestyle='-')
plt.axvline(x=beta_hat, color='g', linestyle='-')
plt.show()





##########################################################################
########################
# estimate beta using MCMC

# estimate energy
betas = np.arange(0.1, 0.5, 0.03)

def H_n(grid, beta):
  L_beta_prop = 0
  for x in range(N):
    for y in range(N):
      # current point s
      s = grid[x, y]
      # local energy state
      neighbors = [grid[(x+1)%N, y], grid[x, (y+1)%N],
                  grid[(x-1)%N, y], grid[x, (y-1)%N]]
      equal_neighbors = [pm_equal(neighbor, s) for neighbor in neighbors]
      L_beta_prop -= beta*np.sum(equal_neighbors)
  return np.exp(L_beta_prop)
  
  
L_beta_prop = np.zeros(len(betas))
for b in range(len(betas)):
  L_beta_prop[b] = H_n(grid, betas[b])
L_beta_prop

# create M samples of a 2D Potts model with beta=beta0
beta_0 = 0.32
grid_0 = initialize(N)
# burn in of 500 samples
for i in range(100):
    grid_0 = runIter(grid_0, beta_0)
# # Visulaization
# plt.figure()
# X, Y = np.meshgrid(range(N), range(N))
# plt.pcolormesh(X, Y, grid_0)
# plt.show()
# create M = 100 samples of the grid
M = 10
grid_0_samples = [grid_0]
for i in range(M):
  grid_0_samples.append(runIter(grid_0_samples[i], beta_0).copy())

# estimate constant
H_nm = np.zeros([len(betas),M], "float")
for b in range(len(betas)):
  for m in range(M):
    H_nm[b, m] = H_n(grid_0_samples[m], betas[b] - beta_0)
hat_c_beta = np.mean(H_nm, axis=1)
hat_c_beta

# estimate Likeilihood
L_beta = L_beta_prop/hat_c_beta
L_beta

# estimated beta
beta_hat = betas[np.argmax(L_beta)]
beta_hat


# plot
plt.figure()
plt.plot(betas, L_beta)
plt.axvline(x=beta_hat, color='g', linestyle='-')
plt.axvline(x=beta, color='r', linestyle='-')
plt.show()



# # estimate initial value beta_0 as if it were 1D Ising model
# # define logit function
# def logit(x):
#   return np.log(x / (1 - x))
# 
# 
# def create_flip_grid(grid):
#     N = grid.shape[0]
#     flip_grid = np.zeros([N, N])
#     for x in range(N):
#         for y in range(N):
#             # current point s
#             s = grid[x, y]
#             # current energy state
#             neighbors_x = np.array([1, -1, 0, 0])
#             neighbors_y = np.array([0, 0, 1, -1])
#             neighbors = grid[(x + neighbors_x)%N, (y + neighbors_y)%N]
#             flip_grid[x, y] = np.mean(neighbors != s)
#     return flip_grid
# flip_grid = create_flip_grid(grid)
# prob = np.mean(flip_grid)
# prob
# beta0_hat = -2 * logit(prob)
# beta0_hat


