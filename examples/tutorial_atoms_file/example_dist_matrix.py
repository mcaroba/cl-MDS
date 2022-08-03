##############################################################
#            cl-MDS atomic-structure example
#                  (distance matrix)
##############################################################
import numpy as np
import cluster_mds as clmds

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a custom distance matrix using a linear combination of kernels
data = clmds.clMDS(atoms='qm9_F_struct.xyz',
                   descriptor="quippy_soap_turbo",
                   cutoff=[2.5, 3], do_species=['F'])
data.build_descriptor()
Q1 = data.descriptor
K1 = np.matmul(Q1, Q1.T)**data.zeta

data = clmds.clMDS(atoms='qm9_F_struct.xyz',
                   descriptor="quippy_soap_turbo",
                   cutoff=[4.5, 5], do_species=['F'])
data.build_descriptor()
Q2 = data.descriptor
K2 = np.matmul(Q2, Q2.T)**data.zeta

K = 0.4*K1 + 0.6*K2
K[np.where(K > 1)] = 1. # avoids numerical round-off problems
D = np.sqrt(1 - K)      # distance matrix associated to the kernel K
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Initialize clMDS class
data = clmds.clMDS(dist_matrix=D, sparsify='random', n_sparse=1000)

# Compute 2-dim. coordinates for the sparse set
n_clusters = 12
Y = data.get_sparse_coordinates(n_clusters)
C = Y[:,2].astype(int)
M = data.sparse_medoids

# Estimate the coordinates for the rest of the database
Y_estim = data.get_estim_coordinates()
C_estim = Y_estim[:,2].astype(int)

# Save to file
dirname = './results_dist_matrix/'
data.save_to_file(dir=dirname)

# Plot the results
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

fig, ax = plt.subplots()
ax.scatter(Y_estim[:, 0], Y_estim[:, 1], c=C_estim, cmap='nipy_spectral',
            alpha=0.3)
ax.scatter(Y[:, 0], Y[:, 1], c=C, cmap='nipy_spectral')
ax.scatter(Y[M, 0], Y[M, 1], color='black', label='medoids')
ax.set_xlabel(r'cl-MDS coordinate 1')
ax.set_ylabel(r'cl-MDS coordinate 2')

plt.legend()
plt.savefig(dirname + 'clmds_plot_dist_matrix.png', format='png', dpi=300)
plt.show()


