##############################################################
#            cl-MDS atomic-structure example
#                         (advanced)
##############################################################
import numpy as np
import cluster_mds as clmds

#~~~~~~~~~~ Pre-pocessing steps ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Custom sparse set
custom_sparse = np.loadtxt('improved_sparse_set.txt', dtype=int)

#~~~~~~~~~~ cl-MDS calculations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize clMDS class passing our own descriptor string
Z = ['H','C','N','O','F']
nmax = 8; lmax = 8
rcut_hard = 3.5; rcut_soft = rcut_hard - 0.5
at_sr = [0.2, 0.2, 0.2, 0.2, 0.2]; at_srs = [0.1, 0.1, 0.1, 0.1, 0.1]
at_st = [0.2, 0.2, 0.2, 0.2, 0.2]; at_sts = [0.1, 0.1, 0.1, 0.1, 0.1]
desc_string = {z:'soap_turbo alpha_max={%i %i %i %i %i} l_max=%i rcut_soft=%.2f \
                 rcut_hard=%.2f atom_sigma_r={%.2f %.2f %.2f %.2f %.2f} \
                 atom_sigma_t={%.2f %.2f %.2f %.2f %.2f} \
                 atom_sigma_r_scaling={%.2f %.2f %.2f %.2f %.2f} \
                 atom_sigma_t_scaling={%.2f %.2f %.2f %.2f %.2f} radial_enhancement=1 \
                 amplitude_scaling={1. 1. 1. 1. 1.} basis="poly3gauss" \
                 scaling_mode="polynomial" species_Z={1 6 7 8 9} n_species=5 \
                 central_index=%i central_weight={1. 1. 1. 1. 1.}'
                 % (nmax, nmax, nmax, nmax, nmax, lmax, rcut_soft, rcut_hard, at_sr[0],
                   at_sr[1], at_sr[2], at_sr[3], at_sr[4], at_st[0], at_st[1], at_st[2],
                   at_st[3], at_st[4], at_srs[0], at_srs[1], at_srs[2], at_srs[3],
                   at_srs[4], at_sts[0], at_sts[1], at_sts[2], at_sts[3], at_sts[4], i+1)
                 for i, z in enumerate(Z)}

data = clmds.clMDS(atoms='qm9_F_struct.xyz', descriptor="quippy_soap_turbo",
                   descriptor_string=desc_string, do_species=['F'],
                   sparsify=custom_sparse)

# Compute 2-dim. representation of the sparse set
# Parameters available: k-medoids initialization, MDS-related weights
hierarchy = [12,1]
Y = data.get_sparse_coordinates(hierarchy, init_medoids="isolated", n_iso_med=1,
                                weight_anchor_mds=None, eta=0)
C = Y[:,2].astype(int)
M = data.sparse_medoids

"""
# More advanced
# Needed to calculate cl-MDS again with same hierarchy
# Includes param. related to computational performance (the code is NOT optimized for this!)
data.cluster_MDS(hierarchy, iter_med=100, init_medoids="isolated", n_iso_med=1,
                 n_init_mds_cluster=100, n_jobs_cluster=1, weight_cluster_mds=10,
                 param_anchor=[70,80,90], n_init_mds_anchor=3500, n_jobs_anchor=1,
                 n_jobs_cluster=8, n_jobs_anchor=8, weight_anchor_mds=None, eta=0)
Y = data.sparse_coordinates
C = data.sparse_cluster_indices
M = data.sparse_medoids
"""

# Estimate the coordinates for the rest of the database
Y_estim = data.get_estim_coordinates(n_steps=10)
C_estim = Y_estim[:,2].astype(int)

#~~~~~~~~~~ Post-processing steps ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save to file
data.save_to_file()

# Plot the results
# 1. Generate the atomic structures corresponding to the medoids (uses ovito)
dir_medoids = './medoids/'
data.medoids_to_xyz(dir=dir_medoids, carve_radius=3.5, render=True)
"""
You need to manually modify these structures for better results.
Open the generated medoids xyz files (with the desired radius cutoff) in
any visualization/analysis program, e.g., ASE (rendering with POVRAY/Blender)
or VMD.
"""

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

for i in range(0, hierarchy[0]):
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
plt.savefig('clmds_plot_advanced.png', format='png', dpi=300)
plt.show()



