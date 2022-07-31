################################################################
#             cl-MDS example on a hypersphere
#                (using default settings)
################################################################
import numpy as np
import cluster_mds as clmds
from sklearn import metrics
import matplotlib.pyplot as plt

# (n-1)-dimensional sphere
N = 10000
n = 10
X = np.random.normal(0,1,(N, n))
r = np.sqrt(np.sum(X**2, axis=1))
S = X/r[:,None]

# distance matrix (input)
D = metrics.pairwise_distances(S)

# initialize clMDS class
data = clmds.clMDS(dist_matrix = D, 
                   sparsify='random', n_sparse=1000)

# set the n. of clusters (n0) or cluster hierarchy ([n0, n1, ..., 1])
n_clusters = 30
# obtain 2-dim. embedding (Y[:,:2]) and clustering (Y[:,2]) of the sparse set
Y = data.get_sparse_coordinates(n_clusters)
# estimate results for the other points
Y_estim = data.get_estim_coordinates(n_steps=2)

# save to file
data.save_to_file()





