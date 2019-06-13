# estimate beta with missing values
n_states = 4

# include missing values
def mcar(grid, pmiss):
  N = grid.shape[0]
  missing = np.random.choice([True, False], size=(N,N), p = [pmiss, 1 - pmiss])
  return missing
# test
miss = mcar(grid, 0.1)

# randomly initialize missing values
grid[miss] = np.random.randint(n_states, size=miss.sum())

# given observed and missing values make a step
def runIter_missing(grid, miss, beta, n_states = 4):
  N = grid.shape[0]
  for x in range(N):
		for y in range(N):
		  # if missing, sample new observation
		  # if observed, do nothing
		  if (miss[x, y]):
  		  # current point s
  			s = grid[x, y]
  			new_state = np.int64(np.random.randint(n_states))
  			# current energy state
  			neighbors = [grid[(x+1)%N, y], grid[x, (y+1)%N], grid[(x-1)%N, y], grid[x, (y-1)%N]]
  			H = -sum(neighbors == s)
  			# new energy state
  			altH = -sum(neighbors == new_state)
  			#Compute dE: (H = -  \sum_{<i,j>} S_i S_j )
  			dE = altH - H 
  			# flip
  			if dE < 0: 
  			  grid[x, y] = new_state
  			elif np.random.rand() < np.exp(-beta * dE): #high energy --> maybe flip?
  			  grid[x, y] = new_state
  return grid


# create grid
n_states = 4
N = 20
beta = 1

# initialize grid
grid = initialize(N)
for i in range(10):
    grid = runIter(grid, beta)

# Visualization
import matplotlib.pyplot as plt 
plt.figure()
X, Y = np.meshgrid(range(N), range(N))
plt.pcolormesh(X, Y, grid)
plt.show()

# sample missingness
miss = mcar(grid, 0.5)
# Visualization
plt.pcolormesh(X, Y, miss)
plt.show()

# randomly initialize missing values
grid[miss] = np.random.randint(n_states, size=miss.sum())

# Visualization
plot_grid = grid.copy()
plot_grid[miss] = -1
import matplotlib.colors as pltcols
colors = ['yellow','blue','green', 'red']
miss_colors = ['grey', 'yellow','blue','green', 'red']
cMap = pltcols.ListedColormap(miss_colors)

plt.figure()
plt.pcolormesh(X, Y, plot_grid, cmap=cMap)
plt.show()

# step with missing values 
# and true beta
grid_true_beta = grid.copy()
# burn in needed
for i in range(10):
    grid_true_beta = runIter_missing(grid_true_beta, miss, beta)
# Visualization
plt.pcolormesh(X, Y, grid_true_beta)
plt.show()

# set up Gibbs Sampler that estimates beta from complete data
# then fills missing values using estimated beta
grid_est_beta = grid.copy()
beta_path = [0]
# longer burn in needed
for i in range(50):
  # estimate beta using MPL
  est_beta = estimate_beta_MPL(grid=grid_est_beta, betas=np.arange(0.5, 1.5, 0.01), n_states=n_states)
  beta_path.append(est_beta.copy())
  # estimate missing values given beta
  grid_est_beta = runIter_missing(grid_est_beta, miss, est_beta)
  # # Visualization
  # plt.pcolormesh(X, Y, grid_true_beta)
  # plt.show()
  
# Visualize path of estimated beta
plt.figure()
plt.plot(beta_path)
plt.axhline(beta, color="green")
plt.show()




