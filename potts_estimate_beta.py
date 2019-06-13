# reticulate::repl_python()
import numpy as np
np.random.seed(123)

n_states = 4
def initialize(N, n_states=4):
    grid = np.random.randint(n_states, size=(N,N))
    return grid

# def pm_equal(x, y):
#   return((x == y) * 2 - 1)

def pm_equal(x, y):
  return((x == y))

def runIter(grid, beta, n_states = 4):
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
            # new energy state
            altH = -sum(neighbors == new_state)
            #Compute dE: (H = -  \sum_{<i,j>} S_i S_j )
            dE = altH - H 
            if dE <= 0:
                s = new_state
            elif np.random.rand() < np.exp(-beta * dE): #high energy --> maybe flip?
                s = new_state
            grid[x, y] = s
    return grid

# run test
beta = 0.5
N = 30
grid = initialize(N, 2)
for i in range(10):
    grid = runIter(grid, beta)
# 
# # Visualization
# import matplotlib.pyplot as plt 
# plt.figure()
# X, Y = np.meshgrid(range(N), range(N))
# plt.pcolormesh(X, Y, grid)
# plt.show()

# ############################################################
########################
# estimate beta using MPLE
# functionalized



# estimate beta using MPLE
# ############################################################
########################
# estimate beta using MPLE in Potts model
# estimate H_n, the energy of the current state
def H_n(grid, beta):
  N = grid.shape[0]
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
  return L_beta_prop
# # test
# this_H_n = H_n(grid, -1)
# this_H_n

def estimate_beta_MPL(grid, betas, n_states=4):
  N = grid.shape[0]
  # sum LHS
  sum_Ui = H_n(grid, -1)
  # sum RHS
  L = n_states
  sum_RHS = np.zeros_like(betas, "float")
  for b in range(len(betas)):
    this_beta = -betas[b]
    for x in range(N):
      for y in range(N):
        s = grid[x, y]
        neighbors = [grid[(x+1)%N, y], grid[x, (y+1)%N],
                    grid[(x-1)%N, y], grid[x, (y-1)%N]]
        Uil = [neighbors.count(l) for l in range(L)]
        sum_RHS[b] += np.dot(np.exp(np.dot(-this_beta, Uil)), Uil) / np.exp(np.dot(-this_beta, Uil)).sum()
    # break if larger than needed
    if sum_RHS[b] >= sum_Ui:
      break
  # estimated beta
  beta_hat_MPL = betas[b]
  return beta_hat_MPL
# test
beta_hat_MPL = estimate_beta_MPL(grid=grid, betas=np.arange(0.1, 2, 0.1), n_states=n_states)
beta_hat_MPL


# # ###############################################################
# # pseudolikelihood estimator for beta in potts model
# 
# # sum_Ui = 0
# # for x in range(N):
# #   for y in range(N):
# #     s = grid[x, y]
# #     neighbors = [grid[(x+1)%N, y], grid[x, (y+1)%N],
# #                 grid[(x-1)%N, y], grid[x, (y-1)%N]]
# #     neighbor_counts = [neighbors.count(l) for l in range(L)]
# #     Ui = neighbor_counts[s]
# #     sum_Ui += Ui
# # sum_Ui
# sum_Ui = H_n(grid, -1)
# sum_Ui
# 
# betas = np.arange(0.01, 2, 0.01)
# L = n_states
# 
# sum_RHS = np.zeros_like(betas, "float")
# for b in range(len(betas)):
#   this_beta = -betas[b]
#   for x in range(N):
#     for y in range(N):
#       s = grid[x, y]
#       neighbors = [grid[(x+1)%N, y], grid[x, (y+1)%N],
#                   grid[(x-1)%N, y], grid[x, (y-1)%N]]
#       Uil = [neighbors.count(l) for l in range(L)]
#       sum_RHS[b] += np.dot(np.exp(np.dot(-this_beta, Uil)), Uil) / np.exp(np.dot(-this_beta, Uil)).sum()
#   # break if larger than needed
#   if sum_RHS[b] >= sum_Ui:
#     break
# sum_RHS
# 
# # estimated beta
# beta_hat_MPL = betas[b]
# beta_hat_MPL
# 
# 
# # plot
# plt.figure()
# plt.plot(betas[:(b+1)], sum_RHS[:(b+1)])
# plt.axhline(y=sum_Ui, color='r', linestyle='-')
# plt.axvline(x=beta_hat_MPL, color='g', linestyle='--')
# plt.axvline(x=beta, color='g', linestyle='-')
# plt.show()


##########################################################################
########################
# estimate beta using MCMC
def estimate_beta_MCMC(grid, betas, beta_0, M):
  N = grid.shape[0]
  # Likelihood up to proportionality
  L_beta_prop = np.zeros(len(betas))
  for b in range(len(betas)):
    L_beta_prop[b] = H_n(grid, betas[b])
    
  # create M samples of a 2D Potts model with beta=beta0
  beta_0 = beta_0
  grid_0 = initialize(N)
  # burn in of 100 samples
  for i in range(100):
      grid_0 = runIter(grid_0, beta_0)
  # throw away D iterations between samples
  D = 1
  grid_0_samples = [grid_0]
  for i in range(M):
    for d in range(D):
      grid_0_samples[i] = runIter(grid_0_samples[i], beta_0)
    grid_0_samples.append(runIter(grid_0_samples[i], beta_0).copy())
  # estimate constant
  H_nm = np.zeros([len(betas),M], "float")
  for b in range(len(betas)):
    for m in range(M):
      H_nm[b, m] = H_n(grid_0_samples[m], betas[b] - beta_0)
  hat_c_beta = np.mean(H_nm, axis=1)
  
  # estimate Likeilihood
  L_beta = L_beta_prop/hat_c_beta
  
  # estimated beta
  beta_hat = betas[np.argmax(L_beta)]
  return beta_hat

# test
betas=np.arange(beta_hat_MPL - 0.5, beta_hat_MPL + 0.5, 0.01)
estimate_beta_MCMC(grid=grid, betas=betas, beta_0=beta_hat_MPL, M=10)
  
  

# # estimate energy
# betas = np.arange(beta_hat_MPL - 0.5, beta_hat_MPL + 0.5, 0.01)
# 
# L_beta_prop = np.zeros(len(betas))
# for b in range(len(betas)):
#   L_beta_prop[b] = H_n(grid, betas[b])
# L_beta_prop
# 
# # create M samples of a 2D Potts model with beta=beta0
# beta_0 = beta_hat_MPL
# grid_0 = initialize(N)
# # burn in of 500 samples
# for i in range(100):
#     grid_0 = runIter(grid_0, beta_0)
# # # Visulaization
# # plt.figure()
# # X, Y = np.meshgrid(range(N), range(N))
# # plt.pcolormesh(X, Y, grid_0)
# # plt.show()
# # create M = 100 samples of the grid
# M = 10
# # throw away D iterations between samples
# D = 1
# grid_0_samples = [grid_0]
# for i in range(M):
#   for d in range(D):
#     grid_0_samples[i] = runIter(grid_0_samples[i], beta_0)
#   grid_0_samples.append(runIter(grid_0_samples[i], beta_0).copy())
#   
# 
# # estimate constant
# H_nm = np.zeros([len(betas),M], "float")
# for b in range(len(betas)):
#   for m in range(M):
#     H_nm[b, m] = H_n(grid_0_samples[m], betas[b] - beta_0)
# hat_c_beta = np.mean(H_nm, axis=1)
# hat_c_beta
# 
# # estimate Likeilihood
# L_beta = L_beta_prop/hat_c_beta
# L_beta
# 
# # estimated beta
# beta_hat = betas[np.argmax(L_beta)]
# beta_hat
# 
# # plot
# plt.figure()
# plt.plot(betas, L_beta)
# plt.axvline(x=beta_hat, color='g', linestyle='-')
# plt.axvline(x=beta, color='r', linestyle='-')
# plt.show()
