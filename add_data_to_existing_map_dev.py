#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This script computes the cl-MDS coordinates of new data given an
# already existing cl-MDS map. 
#
# Note that it assumes that:
#   (1) the user has the descriptors too, at least for the sparse set,
#   (2) all cl-MDS data (coord., cluster, medoids, sparse) is on a 
#       txt/dat file
#
# "Original data" refers to those atoms part of the sparse set used to
# get the cl-MDS map. This is the baseline map, any new data is added
# on top of it and uses it as reference.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import cluster_mds as clmds
from ase.io import read, write
import time
import info_atomic_data as iad

##################################################################
### Main parameters
clmds_file = ""       # file with clMDS info (i_atoms, clmds_coords, cluster, ind_medoids, sparse)
only_sparse = False   # set True if sparse column is missing from clmds_file
descriptors_file = "" # file with descriptors
zeta = 6
dirname_out = "./results/" # directory where the output file will be placed

# change if you have the transformations and local sparse coordinates
has_transformations = False # if False, also check lines 132-135 below
transformations_file = ""

# pass same hierarchy used for the original cl-MDS map
hierarchy = [,1]
n_chunks = 10 # set the amount of chunks in which the full database will be
              # divided to compute the estimation of coordinates
              # (increase this number to reduce memory consumption per step)
n_cpus = 1 # max. jobs available for MDS
print_timing = True

# if you do not have all the descriptors, change these
atoms_file = ""     # xyz filename with the original data
new_data_file = ""  # add here the xyz filename with new data points
new_data_tag = 1    # label (str or int) added to the combined xyz file (atoms_file + new_data_file)
                    # to keep track of different iterations of new data
desc_string = None
Z = None  # it considers all species present in the database
          # change to [z1, z2, ...] if the cl-MDS maps is only for
          # certain chemical species (z1, z2, ...)

####################################################################
#~~~~~~~~~~~~~~~~~~~~~~~~~ Functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This method reads the input file (txt, xyz) to extract all data
# related to cl-MDS (clustering, cl-MDS coordinates). 
def read_clmds_data(filename, has_local=False, only_sparse=False):
    """
    filename = txt/dat file with cl-MDS information
        Compatible with clMDS.save_to_file methods (see cluster_mds.py).
        In particular, the file should include (in this order):
         i_atoms, clmds_coords (XY), cluster_number (C), i_medoids (M), sparse
    has_local = bool
        If True, the file includes the first local MDS coordinates (XY_local) too
    only_sparse = bool 
        If True, the sparse column is missing from file. That is, all
        data is sparse data.
    """
#   Check first the extension
    filetype = filename.split('.')[-1]
    if filetype in ['txt', 'dat']:
#       Get all info from file
        with open(filename, 'r') as f:
            header = f.readline()
            N=[]; XY=[]; C=[]; M=[]; XY_local=[]; sparse=[]
            for line in f:
                N.append( int( line.split()[0] ) )
                XY.append( [float( line.split()[1] ), float( line.split()[2] )] )
                C.append( int( line.split()[3]) )
                M.append( int( line.split()[4]) )
                if only_sparse:
                    sparse.append( 1 )
                else:
                    sparse.append( int( line.split()[5] ) )
                if has_local:
                    XY_local.append( [float( line.split()[6] ), float( line.split()[7] )] )
        N = np.array(N, dtype=int)
        XY = np.array(XY, dtype=float)
        C = np.array(C, dtype=int)
        M = np.array(M, dtype=int)
        sparse = np.array(sparse, dtype=bool)
        if has_local:
            XY_local = np.array(XY_local, dtype=float)
            return N, XY, C, M, sparse, XY_local
    else:
        raise Exception('I do not recognize the file extension, you need to \
                         pass a txt/dat file.')
    return N, XY, C, M, sparse

# Get the clustering information organized as a dictionary of indices
def get_clustering_indices(C, M):
    """
    Given a list/array with a cluster label per datapoint (C[i] = cluster_j), 
    get the clustering information organized as a dict. of members per cluster
    C_ind = {cluster_j: [indices datapoints]}
    """
    C_ind = {i: np.where(np.array(C, dtype=int) == i)[0] for i in np.unique(C)}
    M_ind = np.where(np.array(M, dtype=int) == 1)[0]
    M_ind = M_ind[np.argsort(C[M_ind])]

    return C_ind, M_ind

# Given indices of atoms inside an xyz file, returns which structure they are
# part of and additional info. (optional)
def get_struct_indices(ind_atoms, atoms_file, add_info=None):
    """
    Given a list/array of atom indices inside a database (where first atom is 0,
    second atom is 1, and so on), returns
    (1) a list of indices regarding the structure number for each atom
    OR
    (2) a dict. with (1) plus any additional info requested (e.g., config types)

    ind_atoms = list/array with atoms indices
    atoms_file = xyz filename of the complete dataset
    add_info = None(default), list/array with the desired labels from extended
               xyz file structure header (e.g., "config_type", "energy")
    """
    all_atoms = read(atoms_file, index=':')
    ind_struct = []
    I = {}
    for i, ats in enumerate(all_atoms):
        ind_struct += [i]*len(ats)
        if add_info:
            for label in add_info:
                I.setdefault(label, [])
                I[label] += [ats.info[label]]*len(ats)
    if add_info:
        I["i_struct"] = ind_struct
        ind_struct = I
    return ind_struct

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Put together all data files when new data is in a separated xyz file
if new_data_file:
    atoms = read(atoms_file, index=':')
    new_atoms = read(new_data_file, index=':')
#   keep track of the indexing (where does the original data ends and new data starts?)
    i_start_new = 1
    for ats in atoms:
        if not 'n_iter' in ats.info:
            ats.info['n_iter'] = 0
        i_start_new += len(ats)
    i_end_new = i_start_new
#   add a label to keep track of which iteration of new data it is
    i_end_new = 0
    if new_data_tag:
        for ats in new_atoms:
            ats.info['n_iter'] = new_data_tag
            i_end_new += len(ats)
#   combined atoms_file
    write('all_atoms_data.xyz', atoms + new_atoms)
    atoms_file = 'all_atoms_data.xyz'

# Gather all the information
if has_transformations:
    indices, clmds_coords, C, M, sparse, local_coords = read_clmds_data(clmds_file, has_local=True, only_sparse=only_sparse)
else:
    indices, clmds_coords, C, M, sparse = read_clmds_data(clmds_file, only_sparse=only_sparse)

C_ind, M_ind = get_clustering_indices(C[sparse], M[sparse])
N = len(sparse)
Q = np.loadtxt(descriptors_file)
print(M_ind, indices[sparse][M_ind])

if print_timing:
    print('Initializing cl-MDS class ...')
t0 = time.time()
# Start clMDS class
t00 = time.time()
data = clmds.clMDS(atoms=atoms_file, descriptor=Q, descriptor_string=desc_string,
                   sparsify=indices[sparse], n_sparse=N, do_species=Z)
data.build_dist_matrix(zeta=zeta)

# Retrieve info about transformations from the original map
if has_transformations:
#   descriptors and distance matrix
    data.descriptor = Q
#   original clMDS map
    data.has_clmds = True
    data.hierarchy = hierarchy
    data.sparse_coordinates = clmds_coords[sparse]
    data.sparse_cluster_indices = C[sparse]
    data.sparse_clusters, data.sparse_medoids = C_ind, M_ind
    data.local_sparse_coordinates = local_coords[sparse]
    data.all_transformations = np.load(transformations_file, allow_pickle='TRUE').item()
else:
    M_ind = indices[sparse][M_ind]
#   Add here same parameters as the one used in the original clMDS calculation, EXCEPT those related to
#   medoids (any necessary modifications are already included below).
#   That is, do not add: iter_med, n_iso_med, n_init_mds_cluster, max_iter_cluster, n_jobs_cluster
    data.cluster_MDS(hierarchy, tmax=0, init_medoids=M_ind, iter_med=0, 
                     n_jobs_cluster=n_cpus, weight_cluster_mds=8,
                     param_anchor=[70,80,90], n_init_mds_anchor=3500, n_jobs_anchor=n_cpus,
                     weight_anchor_mds=2, eta=0)
#   Save all the information in case you need to add more data in the future
#   This will allow you to set has_transformations=True
    data.save_to_file(dir=dirname_out, save_all=True)

t01 = time.time()

if print_timing:
    print('Retrieving all necessary elements of cl-MDS class: ', t01-t00)
    print('Estimating coordinates ...')

# Get the estimated coordinates for new datapoints
t10 = time.time()
estim_clmds_coords = data.get_estim_coordinates(n_steps=n_chunks)
t11 = time.time()
if print_timing:
    print('Time estimation cl-MDS: ', t11-t10)
    print('Saving results ...')

# Save results
t20 = time.time()
# Get the structure indices and config. types for all the data points.
# We need to pass the indices of that data (in this case, atoms) within the database
if data.do_species:
#   Only certain chemical elements where used, the indices list is already computed by clMDS
    ind_atoms = data.do_species_list
else:
    ind_atoms = np.arange(0, data.n_env, 1)
info_struct = get_struct_indices(ind_atoms, atoms_file, add_info=["config_type"])
# Save all
data.save_to_file(dir=dirname_out, add_label=info_struct)
t21 = time.time()
if print_timing:
    print('Time saving file: ', t21-t20)
    t1 = time.time()
    print('--------------------------------------')
    print('Total time: ', t1-t0)


