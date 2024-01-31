#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                   Gruyere example
# toy model of a 2-dimensional dataset with "holes" to test
# cl-MDS preservation of the original metric (in this case,
# the Euclidean distance to those holes)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from sklearn.metrics import pairwise_distances
import cluster_mds as clmds

#______Suplementary functions_________________________________

# swiss cheese slice
def gruyere(X, holes):
    # only works with meshgrid arrays
    mask = np.ones(np.shape(X[0]))
    L = np.ones(len(np.shape(X)), dtype=int)
    L[0] = np.shape(X)[0]
    for x0, r in holes:
        x0 = np.reshape(x0, L)
        out_hole = ( np.sum( (X - x0)**2, axis=0) > r**2 )
        mask *= out_hole
    return np.where(mask, 1, 0)

# feature vectors based on distances to holes
def descriptor_holes(X, holes, grid=False):
    Q = np.empty((len(X),len(holes)))
    n = 0
    for x0, r in holes:
        Q[:,n] = (pairwise_distances(X, [x0])).T - r
        n += 1
    return Q

#______Data and embedding_____________________________________

# 1. generate a cube
n_dim = 2
N = 1000
cube = np.random.rand(N, n_dim) 

# 2. define the holes
n_holes = 12
X = cube
holes = []
for i in range(0, n_holes):
    x_hole = np.random.rand(1, n_dim)
    r_hole = np.random.rand(1)*0.15
    out_hole = ( np.sum((X - x_hole)**2, axis=1) > r_hole**2 )
    holes.append( (x_hole[0], r_hole) )
    X = X[out_hole]

# 3. get the Euclidean distance (in the "dist. to holes" space)
Q = descriptor_holes(X, holes) 
D = pairwise_distances(Q)

# 4. compute clMDS
data = clmds.clMDS(dist_matrix=D)
data.cluster_MDS([n_holes,1], weight_anchor_mds=None)
Y = data.sparse_coordinates
C = data.sparse_cluster_indices
M = data.sparse_medoids

#______Plots__________________________________________________
fig = plt.figure(figsize=(12,4))

# colour palette
colours = np.array(['gold','#dcbeff','#42d4f4','#9A6324','blue',
                    'green','orange','#f032e6','#e6194B','limegreen',
                    '#aaffc3','#800000','#fabed4','cyan','#911eb4',
                    '#bfef45','rosybrown','coral','peru','darkgreen',  
                    'olive','#ffd8b1','navy','gray'])
colour_cl = np.array([colours[i] for i in C])

# plot original data points and holes
ax = fig.add_subplot(1,3,1)
ax.scatter(X[:,0], X[:,1], color='gold', s=10)
x0 = np.append(np.arange(500)/500, 1.)
Z = np.meshgrid(x0, x0, indexing='ij')
G = gruyere(Z, holes)
ax.contourf(Z[0], Z[1], G, levels=1, colors=['black','white'], alpha=0.2)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('off')

# plot cl-MDS results
ax = fig.add_subplot(1,3,2)
ax.scatter(Y[:,0], Y[:,1], c=colour_cl, s=20)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('off')

## Voronoi partition of the medoids
#
# coloured Voronoi partition was generated using the function
# colorized_voronoi from:
#     https://gist.github.com/pv/8036995 from Pauli Virtanen
#
# if you want to calculate it:
#      1. download the previous python script in the same folder
#      2. uncomment the code below 
"""
from scipy.spatial import Voronoi, voronoi_plot_2d
from colorized_voronoi import voronoi_finite_polygons_2d

# compute the voronoi partition
vor = Voronoi(Y[M])

# plot
ax = fig.add_subplot(1,3,3)

# associate colour to each region
regions, vertices = voronoi_finite_polygons_2d(vor)
for i, region in enumerate(regions):
    polygon = vertices[region]
    ax.fill(*zip(*polygon), color=colours[i], alpha=0.1)

voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='white',
              line_width=2, line_alpha=1, point_size=2)    
ax.scatter(Y[:,0], Y[:,1], c=colour_cl, s=20)

ax.set_xlim(np.min(Y[:,0]) - 0.05, np.max(Y[:,0]) + 0.05)
ax.set_ylim(np.min(Y[:,1]) - 0.05, np.max(Y[:,1]) + 0.05)
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('off')
"""

fig.tight_layout(pad=1.5)
fig.savefig('clmds_swiss_cheese.pdf', format='pdf', dpi=300)

