#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This file is a supporting module with advanced sparsify methods,
# based on cluster MDS and fast-kmedoids functionalities.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import kmedoids as km
import cluster_mds as clmds
import sys


# This method divides the descriptor matrix and apply kmedoids to each slice
# The original code was written by Jan Kloppenburg and adapted by Patricia Hernandez-Leon
def kmedoids_divider(descriptor, pts_in_slices=2*10**4, n_medoids=None, percentage_med=15, 
                     output="descriptor", verbose=1):
    """
    INPUT-----------
      (i) a descriptor matrix with shape = (n. of descriptors, dim. of descriptors)

    Other parameters:
       *pts_in_slices: an estimate of the size of the slices
       *n_medoids: number of medoids computed per slice (takes precedence over percentage_med)
       *percentage_med: percentage of sparse points that are computed as medoids
       *output: if "descriptor", returns the sparse descriptor matrix; 
                if "indices", returns an array with the sparse indices.

    OUTPUT----------
       (i, ii) array with the sparse indices of length <= max_n_sparse


    (*) implemented_descriptors = ["quippy_soap","quippy_soap_turbo","quippy_soap_turbo_compress"]
    """
    L = len(descriptor)
    n_slices = int(np.ceil( L / pts_in_slices))
    final_medoids = list()
    for i in range(n_slices):
        if verbose:
            sys.stdout.write('\rComputing medoids of chunk %i' % (i + 1))
            sys.stdout.flush()
#       Obtain slice of descriptors
        if i == n_slices - 1:
            index = [i * int(L / n_slices), L]
        else:
            index = [i * int(L / n_slices), (i+1) * int(L / n_slices)]
        work_desc = list()
        for bit in range(index[0], index[1]):
            work_desc.append(descriptor[bit])
        work_desc = np.array(work_desc)
#       The parameter n_medoids takes precedence over percentage_med
        if n_medoids is None:
            n_medoids = int(len(work_desc) * percentage_med / 100)
#       Compute dist_matrix and pass it to kmedoids
        data = clmds.clMDS(descriptor=work_desc, verbose=verbose)
        data.build_dist_matrix()
        ind_medoids = km.kMedoids( data.dist_matrix, n_medoids, init_Ms="isolated", n_iso=n_medoids )[0]
        for med in sorted(ind_medoids):
            final_medoids.append(med + index[0])
    if verbose:
        print(" ")
    if output == "descriptor":
        return descriptor[final_medoids,:]
    elif output == "indices":
        return np.array(final_medoids)
 

# This method obtains a sparse set containing only medoids, using kmedoids_divider 
# for bigger descriptor matrices (atoms files)
def sparsify_kmedoids(atoms=None, descriptor=None, descriptor_string=None, do_species=None,
                      max_n_sparse=3000, run_divider=3*10**4, pts_in_slices=2*10**4, 
                      percentage_med=20):
    """
    INPUT-----------
      (i) an atoms_file + implemented descriptor name(*), or
      (ii) a descriptor matrix with shape = (n. of descriptors, dim. of descriptors)

    Other parameters:
       *max_n_sparse: maximum number of descriptors in the sparse set
       *run_divider: gives the maximum amount of descriptors used in a single kmedoids calculation.
                     If the total amount is bigger, kmedoids_divider is used.
       *pts_in_slices: an estimate of the size of the slices
       *percentage_med: percentage of sparse points that are computed as medoids.

    OUTPUT----------
       (i, ii) array with the sparse indices of length <= max_n_sparse


    (*) implemented_descriptors = ["quippy_soap","quippy_soap_turbo","quippy_soap_turbo_compress"]
    """
    data = clmds.clMDS(atoms=atoms, descriptor=descriptor, do_species=do_species,
                       descriptor_string=descriptor_string)
    if isinstance(descriptor, str):
        data.build_descriptor()    
    Q = data.descriptor
#   The maximum size for a single kmedoids calculation is limited, to avoid memory-issues
    if len(Q) > run_divider:
        sparse_med = kmedoids_divider(Q, pts_in_slices=pts_in_slices, percentage_med=percentage_med, 
                                      output="indices" )
#       Check that the amount of medoids (sparse pts.) is below user's limit
        if len(sparse_med) > max_n_sparse:
            print("** Total number of medoids per slice is bigger than max_n_sparse.")
            print("** Final medoids are computed from those.")
            sparse_med = kmedoids_divider(Q[sparse_med,:], pts_in_slices=len(sparse_med),
                                          n_medoids=max_n_sparse, output="indices" )
    else:
        data.build_dist_matrix()
        sparse_med = km.kMedoids( data.dist_matrix, max_n_sparse, init_Ms='isolated', n_iso=max_n_sparse)[0]
    if do_species is not None:
        sparse_med = data.do_species_list[sparse_med]

    return sparse_med


# This method computes a sparse set combining medoids and random descriptors, with a ratio
# given by percentage_med (percentage of medoids included in the sparse set).
def sparsify_rand_and_kmedoids(n_sparse, atoms=None, descriptor=None, descriptor_string=None,
                               do_species=None, percentage_med=20):
    """
    INPUT-----------
      (i) an atoms_file + implemented descriptor name(*), or
      (ii) a descriptor matrix with shape = (n. of descriptors, dim. of descriptors)
    together with n_sparse (total number of descriptors in the sparse set)

    Other parameters:
       *percentage_med: percentage of sparse points that are computed as medoids.

    OUTPUT----------
       (i, ii) array with the sparse indices of length = n_sparse


    (*) implemented_descriptors = ["quippy_soap","quippy_soap_turbo","quippy_soap_turbo_compress"]
    """
#   Obtain medoids for the sparse set
    n_sparse_med = n_sparse*percentage_med // 100
    sparse_med = sparsify_kmedoids(atoms=atoms, descriptor=descriptor, do_species=do_species,
                                   descriptor_string=descriptor_string, max_n_sparse=n_sparse_med)
#   Fill the rest of the sparse set with random descriptors
    data = clmds.clMDS(atoms=atoms, descriptor=descriptor, do_species=do_species,
                     descriptor_string=descriptor_string)
    I = np.arange(0, data.n_env, 1)
    if do_species is not None:
        I = data.do_species_list
    non_sparse = np.setdiff1d(I, sparse_med)
    np.random.shuffle(non_sparse)
    sparse_rand = non_sparse[ :(n_sparse - len(sparse_med))]
    print("Sparse points: n. medoids = %i, n. random = %i" % (len(sparse_med), len(sparse_rand)))
#   Full sparse set
    sparse = np.sort( np.concatenate((sparse_med, sparse_rand)) )
    return sparse



# This method optimizes the sparse set from previous methods (see sparsify_rand_and_kmedoids)
# by guaranteeing a minimum cluster size for a specific sparse clustering.
# *** Especially helpful to optimize cluster MDS results ***
def sparsify_cluster_size(n_sparse, n_clusters, atoms=None, descriptor=None, descriptor_string=None,
                          do_species=None, percentage_med=20, min_cluster_size=5, max_iter=15):
    """
    INPUT-----------
      (i) an atoms_file + implemented descriptor name(*), or
      (ii) a descriptor matrix with shape = (n. of descriptors, dim. of descriptors)
    together with n_sparse (total number of descriptors in the sparse set) and 
                  n_clusters (total number of clusters for later processing, e.g., with cl-MDS)

    Other parameters:
        *percentage_med: percentage of sparse points that are computed as medoids.
        *min_cluster_size: minimum amount of members expected in each sparse cluster, as long
                           as there are as many (or more) members in the full cluster.
        *max_iter: maximum number of iterations performed to improved the sparse set.

    OUTPUT----------
       (i, ii) array with the sparse indices of length = n_sparse


    (*) implemented_descriptors = ["quippy_soap","quippy_soap_turbo","quippy_soap_turbo_compress"]
    """
    print("________________________________________________________")
    print("                                                        ")
    print("              OBTAINING INITIAL SPARSE SET              ")
    print("________________________________________________________")
    sparse = sparsify_rand_and_kmedoids(n_sparse, atoms=atoms, descriptor=descriptor, 
                                        do_species=do_species, descriptor_string=descriptor_string,
                                        percentage_med=percentage_med)
    L_init = len(sparse)
    print("________________________________________________________")
    print("                                                        ")
    print("                 IMPROVING SPARSE SET                   ")
    print("________________________________________________________")
    improved_sparse = False
    n = 0
    while not improved_sparse and n < max_iter:
#       Compute a new clustering for the updated sparse set
        data = clmds.clMDS(atoms=atoms, descriptor=descriptor, do_species=do_species,
                           descriptor_string=descriptor_string, sparsify=sparse, verbose=0)
        data.build_dist_matrix()
        M, C = km.kMedoids( data.dist_matrix, n_clusters, init_Ms='isolated', n_iso=n_clusters)
#       Check how descriptors outside the sparse set are assigned to the clustering    
        data.hierarchy = [n_clusters]
        data.sparse_medoids = M
        data.sparse_cluster_indices = -np.ones(len(data.dist_matrix), dtype=int)
        for i in range(0, n_clusters):
            data.sparse_cluster_indices[ C[i] ] = i
        C_all = data.assign_atoms_to_cluster()
#       Print results
        print("---------------------------------------------")
        print("    New clustering (iteration %i)          " % n)
        print("---------------------------------------------")
        print(" n_cl | size (sparse) | size (complete)      ")
        for i in C.keys():
            cl_i = len(C[i])
            cl_i_all = len(np.where(C_all == i)[0])
            print(" %2i       %6i        %6i" % (i, cl_i, cl_i + cl_i_all))
        print(" ")
#       Improve the sparse set by adding members of the smallest clusters
        new_sparse = sparse
        for i in C.keys():
            cl_i = sparse[C[i]]
            if do_species is not None:
                cl_i_all = data.do_species_list[ np.where(C_all == i)[0] ]
            else:
                cl_i_all = np.where(C_all == i)[0]
            if len(cl_i) == len(cl_i_all):
                continue
            elif len(cl_i) < min_cluster_size:
                l = min_cluster_size - len(cl_i)
                temp = np.setdiff1d(cl_i_all, cl_i)
                np.random.shuffle(temp)
                new_sparse = np.concatenate((new_sparse, temp[:l]))
#       Check if there is room for improvement
        if set(sparse) == set(new_sparse):
            improved_sparse = True
            print('Total number of iterations: %i' % n)
        else:
            n += 1
            sparse = new_sparse
            if n == max_iter:
                print('Stopped at iteration %i' % n)

#   Save improved sparse set
    new_sparse = np.unique(new_sparse)
    print('Length initial sparse set:', L_init)
    print('Length improved sparse set:', len(new_sparse))

    return new_sparse









