##############################################################
#            cl-MDS atomic-structure example
#                (sparse set selection)
##############################################################
import numpy as np
import cluster_mds as clmds

atoms_file = 'qm9_F_struct.xyz'
do_z = 'F'
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

# Proportion of medoids vs. random points in the sparse set
n_sparse_med = 400
n_sparse_rand = 600

print("________________________________________________________")
print(" ")
print("                 OBTAINING SPARSE SET")
print("________________________________________________________")
# Selecting sparse points using k-medoids:
#  1. Obtain the complete distance matrix
#     If the total number is too big, you may prefer using a big random chunk instead

data = clmds.clMDS(atoms=atoms_file, descriptor="quippy_soap_turbo",
                   descriptor_string=desc_string, do_species=[do_z])
data.build_dist_matrix()

# 2. Compute the medoids (if it takes too long, lower n_inits)
import kmedoids as km
M = km.kMedoids( data.dist_matrix, n_sparse_med, n_inits=20, init_Ms='isolated',
                 n_iso=50 )[0]
M = data.do_species_list[M] # we are restricting k-medoids to F atoms only, but the indices
                            # of the sparse set are given respect to the whole dataset

# Third, get the random atoms
non_sparse_ind = np.setdiff1d(data.do_species_list, M)
np.random.shuffle(non_sparse_ind)
rand_ind = non_sparse_ind[:n_sparse_rand]

# Complete sparse set
sparse_ind = np.sort( np.concatenate((M, rand_ind)) )
np.savetxt('original_sparse_set.txt', sparse_ind, fmt='%i')

assert set(np.array(data.species_list)[sparse_ind]) == {do_z}


# Now, you can directly use it OR improve it as follow
print("________________________________________________________")
print(" ")
print("                 IMPROVING SPARSE SET")
print("________________________________________________________")

# set the minimum number of elements per cluster (not lower than 4) 
n_min = 5
print_info = True
# init. param.
original_ind = sparse_ind
improved_sparse = False
n = 0
while not improved_sparse:
    data = clmds.clMDS(atoms=atoms_file, descriptor="quippy_soap_turbo",
                       descriptor_string=desc_string, do_species=[do_z],
                       sparsify=sparse_ind, verbose=False)
    # We are only interested in the clusters, so use bad parameters where
    # possible for faster calculations
    data.cluster_MDS([12,1], iter_med=10**4, weight_cluster_mds=1,
                     n_init_mds_cluster=1, param_anchor=[90,90,90],
                     n_init_mds_anchor=1, n_jobs_cluster=8, n_jobs_anchor=8)
    C = data.sparse_cluster_indices
    # Classify all F atoms within the sparse clustering
    C_all = data.assign_atoms_to_cluster()
    # Information about the clustering
    if print_info:
        print("---------------------------------------------")
        print('       Final clustering (iteration %i)' % n)
        print("---------------------------------------------")
        print('n_cl | size (sparse) | size (complete)')
        for i in set(C):
            cl_i = len(np.where(C == i)[0])
            cl_i_all = len(np.where(C_all == i)[0])
            print(' %2i       %6i        %6i' % (i, cl_i, cl_i + cl_i_all))
        print(" ")
    # Improve the sparse set by increasing the smallest clusters
    new_ind = sparse_ind
    for i in set(C):
        cl_i = sparse_ind[ np.where(C == i)[0] ]
        cl_i_all = data.do_species_list[ np.where(C_all == i)[0] ]
        if len(cl_i) == len(cl_i_all):
            continue
        elif len(cl_i) < n_min:
            l = n_min - len(cl_i)
            temp = np.setdiff1d(cl_i_all, cl_i)
            np.random.shuffle(temp)
            new_ind = np.concatenate((new_ind, temp[:l]))
    if set(sparse_ind) == set(new_ind):
        improved_sparse = True
        print('Total number of iterations: %i' % n)
    else:
        n += 1
        sparse_ind = new_ind

new_ind = np.unique(new_ind)
print('Length original sparse set:', len(original_ind))
print('Length new sparse set:', len(new_ind))

np.savetxt('improved_sparse_set.txt', new_ind, fmt='%i')







