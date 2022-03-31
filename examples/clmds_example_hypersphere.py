################################################################
#             cl-MDS example on a hypersphere
#                (using default settings)
################################################################
import numpy as np
import cluster_mds as clmds
from sklearn import metrics


# (n-1)-dimensional sphere
N = 2000
n = 10
X = np.random.normal(0,1,(N, n))
r = np.sqrt(np.sum(X**2, axis=1))
S = X/r[:,None]

# distance matrix (input)
D = metrics.pairwise_distances(S)

# initialize clMDS class
method = clmds.clMDS(dist_matrix = D)
# set the n. of clusters or cluster hierarchy ([n0, n1, ..., 1])
n_clusters = 20
# obtain 2-dim. embedding (Y[:,:1]) and clustering (Y[:,2])
Y = method.get_sparse_coordinates(n_clusters)

