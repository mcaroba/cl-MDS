###################################################################
#            cl-MDS atomic-structure example
#                         (advanced)
#
#       uncomment ## lines to use different options
###################################################################
import numpy as np
import cluster_mds as clmds

#~~~~~~~~~~~ Initialization parameters ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# n. of clusters
hierarchy = [12,1]

# xyz file with atomic structures
atoms_file = 'qm9_F_struct.xyz'
descriptor = "quippy_soap_turbo"
# selection of atomic species
do_species = ['F']
# custom descriptor string
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

# alternative initialization using custom descriptor matrix
descriptor = np.loadtxt("descriptor.dat") #< custom array of descriptors with shape=(n. desc., dim. desc.) >
atoms_file = None # default, added to run this file
do_species = None # default, added to run this file (note that you need to pass 
                     # descriptors of only those species you want)
desc_string = None # default, added to run this file

# directories to store the results
dirname = './results_advanced/'
dir_medoids = dirname + 'medoids/'

#~~~~~~~~~~ Advanced sparse set options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1) Pass your own custom list/array of sparse indices
## custom_sparse = np.loadtxt('improved_sparse_set.txt', dtype=int)

# 2) Compute a custom sparse set using sparsify_module
import sparsify_module as spmod
n_sparse = 100 # reference size of the sparse set
               # (some methods increase/decrease it a bit)
P_med = 40 # approx. percentage of medoids in the sparse set
           # lower it (15-20) for databases with more than 30000 atoms
# (2.1) only medoids in the sparse set
## custom_sparse = spmod.sparsify_kmedoids( atoms=atoms_file, descriptor=descriptor, do_species=do_species,
##                 descriptor_string=desc_string, max_n_sparse=n_sparse, percentage_med=P_med )

# (2.2) medoids + random points in the sparse set
custom_sparse = spmod.sparsify_rand_and_kmedoids( n_sparse, atoms=atoms_file, descriptor=descriptor,
                 descriptor_string=desc_string, do_species=do_species, percentage_med=P_med )

# (2.3) optimized version of (2.1), (2.2) for a given number of clusters (hierarchy[0]) 
#       it ensures a minimum number of points per cluster in the sparse set, improving cl-MDS performance
#custom_sparse = spmod.sparsify_cluster_size( n_sparse, hierarchy[0], atoms=atoms_file, 
#                descriptor=descriptor, descriptor_string=desc_string, do_species=do_species, 
#                percentage_med=P_med, min_cluster_size=5, max_iter=15 )

# It is recommended to save the sparse set for further uses/testing parameters
np.savetxt(dirname + 'improved_sparse_set.txt', custom_sparse, fmt='%i')

#~~~~~~~~~~ cl-MDS calculations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize clMDS class passing our own descriptor string
data = clmds.clMDS(atoms=atoms_file, descriptor=descriptor,
                   descriptor_string=desc_string, do_species=do_species,
                   sparsify=custom_sparse)
# Compute 2-dim. representation of the sparse set
# Parameters available: k-medoids initialization, MDS-related weights
Y = data.get_sparse_coordinates(hierarchy, init_medoids="isolated", n_iso_med=1,
                                weight_anchor_mds=2, eta=0)
np.savetxt("descriptor.dat", data.descriptor)
C = Y[:,2].astype(int)
M = data.sparse_medoids

"""
# More advanced
# Needed to calculate cl-MDS again with same hierarchy but different parameters
# for better clustering, MDS embedding and/or computational speed
data.cluster_MDS(hierarchy, iter_med=1000, init_medoids="isolated", n_iso_med=2,
                 n_init_mds_cluster=100, n_jobs_cluster=1, weight_cluster_mds=8,
                 param_anchor=[80,90,90], n_init_mds_anchor=3500, n_jobs_anchor=1,
                 n_jobs_cluster=8, n_jobs_anchor=8, weight_anchor_mds=2, eta=0)
Y = data.sparse_coordinates
C = data.sparse_cluster_indices
M = data.sparse_medoids
"""

# Estimate the coordinates for the rest of the database
Y_estim = data.get_estim_coordinates(n_steps=10)
C_estim = Y_estim[:,2].astype(int)

#~~~~~~~~~~ Post-processing steps ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Save to file
data.save_to_file(dir=dirname)

# Plot the results
# 1. Generate the atomic structures corresponding to the medoids (uses ovito)
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
plt.savefig(dirname + 'clmds_plot_advanced.png', format='png', dpi=300)
plt.show()



