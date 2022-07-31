##############################################################
#            cl-MDS atomic-structure example
#                         (simple)
##############################################################
import numpy as np
import cluster_mds as clmds

# Initialize clMDS class
data = clmds.clMDS(atoms='qm9_F_struct.xyz',
                   descriptor="quippy_soap_turbo",
                   cutoff=[3., 3.5], do_species=['F'],
                   sparsify='random', n_sparse=1000)

# Compute 2-dim. coordinates for the sparse set
n_clusters = 12
Y = data.get_sparse_coordinates(n_clusters)  # Y := [x coord., y coord., cluster label]
C = Y[:,2].astype(int)
M = data.sparse_medoids

# Estimate the coordinates for the rest of the database
Y_estim = data.get_estim_coordinates()
C_estim = Y_estim[:,2].astype(int)

# Save to file
# (1) in the original xyz file (only 2-dim. coord. and clustering)
# data.write_xyz(filename='qm9_struct.xyz')

# (2) in a new file
data.save_to_file()

# Plot the results:
# 1. Generate the atomic structures corresponding to the medoids (uses ovito)
dir_medoids = './medoids/'
data.medoids_to_xyz(dir=dir_medoids, carve_radius=3.5, render=True)

# 2. Create the plot
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig, ax = plt.subplots()
ax.scatter(Y_estim[:, 0], Y_estim[:, 1], c=C_estim, cmap='nipy_spectral',
            alpha=0.3)
ax.scatter(Y[:, 0], Y[:, 1], c=C, cmap='nipy_spectral')
ax.scatter(Y[M, 0], Y[M, 1], color='black', label='medoids')
ax.set_xlabel(r'cl-MDS coordinate 1')
ax.set_ylabel(r'cl-MDS coordinate 2')

for i in range(0, n_clusters):
    arr_img = plt.imread(dir_medoids + 'medoid_%i.png' % i, format='png')
    imagebox = OffsetImage(arr_img, zoom=0.3)
    imagebox.image.axes = ax
    cl = data.sparse_clusters[i]
    if Y[M[i],0] < 0:
        a = np.min(Y[cl,0]) - 20
    else:
        a = np.max(Y[cl,0]) + 30
    if Y[M[i],1] < 0:
        b = np.min(Y[cl,1]) - 30
    else:
        b = np.max(Y[cl,1]) + 20
    if i == 0:
        xy = np.array([a,b])[None,:]
    else:
        dist_med = np.sqrt(np.sum((xy - np.array([[a,b]]))**2, axis=1))
        if (dist_med < 0.1).any():
            b = -b
            if a < 0:
                a += 10
            else:
                a -= 10
        xy = np.concatenate((xy, np.array([a,b])[None,:]) )
    ab = AnnotationBbox(imagebox, Y[M[i],:2], xybox=(a, b),
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0, frameon=False, arrowprops=dict(arrowstyle="-",
                        connectionstyle="angle,angleA=0,angleB=90,rad=3"))
    ax.add_artist(ab)

plt.legend()
plt.savefig('clmds_plot_simple.png', format='png', dpi=300)
plt.show()


