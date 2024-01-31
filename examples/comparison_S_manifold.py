#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#               S manifold comparison
# this script is a modified version of Jake Vanderplas' example:
#    https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py
# to include cl-MDS, kernel PCA and UMAP
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from time import time
from functools import partial
from sklearn import manifold, datasets, metrics, decomposition
import umap
import cluster_mds as clmds

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

## Main parameters
n_points = 1000   # total number of data points
n_components = 2  # reduced dimension
n_neighbors = 15  # hyperparameter for several methods

# 1. generate dataset
X, color = datasets.make_s_curve(n_points, random_state=0)

# 2. generate a dictionary with all the techniques
methods = {}
LLE = partial( manifold.LocallyLinearEmbedding, n_neighbors=n_neighbors,
               n_components=n_components, eigen_solver="auto" )
methods["LLE"] = LLE(method="standard")
methods["LTSA"] = LLE(method="ltsa")
methods["Hessian LLE"] = LLE(method="hessian")
methods["Modified LLE"] = LLE(method="modified")
methods["Isomap"] = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components)
methods["LE"] = manifold.SpectralEmbedding(n_components=n_components, n_neighbors=n_neighbors)
methods["t-SNE"] = manifold.TSNE(n_components=n_components, init="pca", random_state=0)
methods['UMAP'] = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, metric='euclidean')
methods['PCA'] = decomposition.PCA(n_components=n_components)
methods['kPCA (RBF)'] = decomposition.KernelPCA(n_components=n_components, eigen_solver="auto", kernel="rbf")
methods["MDS"] = manifold.MDS(n_components, max_iter=100, n_init=1, normalized_stress="auto")
methods["cl-MDS"] = [42,1]

# 3. create figure
fig = plt.figure(figsize=(10,6))

# plot data points in their original representation
ax = fig.add_subplot(351, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.get_cmap('viridis'), s=10)
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
ax.zaxis.set_major_formatter(NullFormatter())
ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])
ax.view_init(4, -62)
ax.dist = 9

# plot the 2-dimensional coordinates of the data
for i, (label, method) in enumerate(methods.items()):
    t0 = time()
    if 'cl-MDS' in label:
        D = metrics.pairwise_distances(X)
        data_clmds = clmds.clMDS(dist_matrix=D, verbose=False)
        data_clmds.cluster_MDS(method, iter_med=1, n_init_mds_anchor=1, n_init_mds_cluster=1, 
                          weight_cluster_mds=2, weight_anchor_mds=None, param_anchor=[90,90,90])
        Y = data_clmds.get_sparse_coordinates(method)[:,:2]
    else:
        Y = method.fit_transform(X)
    t1 = time()
    print("%s: %.2g sec" % (label, t1 - t0))
    ax = fig.add_subplot(3, 5, 2 + i + (i > 3) + (i > 7))
    ax.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.get_cmap('viridis'), s=10)
    ax.set_title(r'%s (%.2g sec)' % (label, t1 - t0), fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis("tight")


fig.tight_layout(pad=1.)
fig.savefig('clmds_S_curve_comparison.pdf', format='pdf', dpi=600)
