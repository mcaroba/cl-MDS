#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#               S manifold test
# toy example to show the effect of the main hyperparameter of
# cl-MDS (hierarchy) in its performance and results
#
# it uses the S-curve dataset from scikit-learn
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from sklearn import manifold, datasets, metrics
from time import time
import cluster_mds as clmds

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

## Main parameters
n_points = 1000    # total number of data points

# 1. generate dataset
X, color = datasets.make_s_curve(n_points, random_state=0)

# 2. generate a dictionary with different hierarchies
methods = {}
methods[r'$N_{cl} = 5$'] = [5,1]
methods[r'$N_{cl} = 15$'] = [15,1]
methods[r'$N_{cl} = 30$'] = [30,1]
methods[r'$N_{cl} = 43$'] = [42,1]
methods[r'$N_{cl} = 53$'] = [55,1]

# 3. create figure
fig = plt.figure(figsize=(8,4))

# color palette
custom_col = np.array([[45,124,8], [254,169,14], [22,5,185], [209,187,247], [30,203,241],
                       [235,117,242], [119,8,191], [47,193,61], [207,20,25], [99,32,6],
                       [135,153,38], [247,220,63], [158,82,6], [9,87,237], [141,254,251],
                      [119,7,71], [239,246,135], [11,111,124], [237,117,2], [185,254,178],
                      [3,73,19], [50,251,152], [97,52,196], [252,205,255], [182,46,75],
                      [187,0,235], [41,55,140], [180,73,27], [190,178,44], [252,68,20],
                      [41,178,144], [11,3,113], [241,142,87], [5,57,172], [237,3,7],
                      [93,162,217], [255,17,219], [140,102,16], [214,84,84], [206,250,26],
                      [176,6,1], [38,231,34], [255,209,0], [255,222,125], [86,25,75],
                      [51,116,209], [211,3,211]])
colors = [tuple(custom_col[i]/255) for i in range(0, len(custom_col))]
colors = np.concatenate((colors, colors))

# plot the 2-dimensional coordinates obtained using each hierarchy
D = metrics.pairwise_distances(X)
clMDS = clmds.clMDS(dist_matrix= D, verbose=False)

for i, (label, method) in enumerate(methods.items()):
#   compute cl-MDS coordinates
    t0 = time()
    clMDS.cluster_MDS(method, iter_med=2, n_init_mds_cluster=1, max_iter_cluster=100,
                      n_init_mds_anchor=1, max_iter_anchor=100, weight_cluster_mds=4,
                      weight_anchor_mds=None, param_anchor=[90,90,90])
    Y = clMDS.get_sparse_coordinates(method)[:,:2]
    t1 = time()
    cl = clMDS.sparse_cluster_indices
    print("MDS stress: ", clMDS.MDS_stress)
    print(r'%s: %.2g sec' % (label, t1 - t0))
#   plot 2-dimensional results
    ax = fig.add_subplot(2, 5, 6 + i)
    ax.scatter(Y[:, 0], Y[:, 1], c=colors[cl], s=10)
    ax.set_title(r'%s ($%.2g$ sec)' % (label, t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.axis("tight")
#   plot original 3-dimensional S-curve with the clustering
    ax = fig.add_subplot(2, 5, 1 + i, projection="3d")
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors[cl], s=10)
    ax.w_xaxis.set_pane_color((0, 0, 0, 0.05))
    ax.w_yaxis.set_pane_color((0, 0, 0, 0.03))
    ax.w_zaxis.set_pane_color((0, 0, 0, 0.01))
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    ax.set_zticks([])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.zaxis.set_major_formatter(NullFormatter())
    ax.view_init(4, -62)
    ax.dist = 9
    ax.set_axis_off()
    

fig.tight_layout(pad=1.)
fig.savefig('test_clMDS_S_curve.pdf', format='pdf', dpi=600)
