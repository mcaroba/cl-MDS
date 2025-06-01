#************************************************************************************************************
# This is the cluster-based MultiDimensional Scaling code for dimensionality reduction data analysis.       #
#                                                                                                           #
#                                                cl-MDS                                                     #
#                                                                                                           #
# This code has been written and is copyright (c) 2018-2022 of the following authors:                       #
#                                                                                                           #
# *) Patricia Hernandez-Leon                                                                                #
# *) Miguel A. Caro
#                                                                                                           #
# from the Department of Electrical Engineering and Automation, Aalto University, Finland                   #
#                                                                                                           #
#                                                                                                           #
# See the file LICENSE.md for license information and the README.md file for practical installation         #
# instructions and usage examples. The official code repository is                                          #
#                                                                                                           #
#                                   https://github.com/mcaroba/cl-MDS                                       #
#                                                                                                           #
# Visit the repository for the latest version of this distribution.                                         #
#                                                                                                           #
#                                                                                                           #
# If you use cl-MDS for the compilation of academic/scientific/technical work, please cite, as appropriate: #
#                                                                                                           #
# P. Hernandez-Leon and M.A. Caro, XXX, YYY (2022)                                                          #
#                                                                                                           #
# Please cite the following reference too:                                                                  #
#                                                                                                           #
# P. Hernandez-Leon and M.A. Caro, Phys. Scr. 99, 066004 (2024)                                             #
#                                                                                                           #
#************************************************************************************************************
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This file is a module for supporting advanced uses of the
# estimation of coordinates functionality, part of cluster MDS main
# class. It is based on cluster MDS application to atomic databases,
# especially thanks to Rina Ibragimova's thorough feedback.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import cluster_mds as clmds
from ase.io import read, write


# This method reads the input file (txt, xyz) to extract all data
# related to cl-MDS (clustering, cl-MDS coordinates). 
def read_clmds_data(filename, has_local=False, only_sparse=False):
    """
    INPUT----------
      (i) filename = txt/dat file with cl-MDS information
            Compatible with clMDS.save_to_file methods (see cluster_mds.py).
            In particular, the file should include (in this order):
             i_atoms, clmds_coords (XY), cluster_number (C), i_medoids (M), sparse

    Other parameters:
      *has_local: True if the file includes the first local MDS coordinates (XY_local)
      *only_sparse: True if the sparse column is missing from file.
          That is, all data is sparse data.

    OUTPUT----------
      (i, ii) arrays with the following information: 
                N: atom indices within the original xyz file,
               XY: 2-dimensional coordinates for each atom, 
                C: integer cluster label for each atom, 
                M: whether each atom is a medoid or not,
           sparse: whether each atom was in the sparse set,
       (XY_local): first local embedding performed by cl-MDS.
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



# Get the clustering information organized as a dictionary of indices (members per cluster)
def get_clustering_indices(C, M):
    """
    INPUT----------
      (i) C: list/array with a cluster label per datapoint 
               C[i] = cluster_j
      (ii) M: list/array with 0/1 values depending on whether 
              each datapoint is a medoid or not

    OUTPUT----------
      (i) dict. of members per cluster
               C_ind = {cluster_j: [indices datapoints]}
          and array of medoids indices (M).
    """
    C_ind = {i: np.where(np.array(C, dtype=int) == i)[0] for i in np.unique(C)}
    M_ind = np.where(np.array(M, dtype=int) == 1)[0]
    M_ind = M_ind[np.argsort(C[M_ind])]

    return C_ind, M_ind



# Given indices of atoms inside an xyz file, returns which structure
# they are part of and (optional) additional info.
def get_struct_indices(ind_atoms, atoms_file, add_info=None):
    """
    INPUT----------
      (i) ind_atoms: list/array of atom indices inside a database
                     (where first atom is 0, second atom is 1, and so on)
      (ii) atoms_file: xyz filename of the complete dataset

    Other parameters:
      *add_info: None(default), list/array with the desired labels from extended
               xyz file structure header (e.g., "config_type", "energy")

    OUTPUT----------
      (i) list of indices regarding the structure number for each atom,
          plus any additional info requested (e.g., config types)
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



# Put together all data files when there is new data in different xyz file(s)
def join_xyz_files(new_atoms_file, original_atoms_file, all_atoms_file='all_atoms_data.xyz', new_atoms_tag=None):
    """
    INPUT----------


    Other parameters:
      *new_atoms_tag: label(s) added to the output xyz file to
                      keep track of different iterations of new data.
                      It can be str/int or a list/array of str/int of same length as new_atoms_file


    OUTPUT---------

    """
    atoms = read(atoms_file, index=':')
    all_atoms = atoms.copy()
#   keep track of the indexing 
#   where does the original data ends and new data starts?
    i_start_new = 1
    for ats in atoms:
        if not 'n_iter' in ats.info:
            ats.info['n_iter'] = 0
        i_start_new += len(ats)
    i_end_new = i_start_new

    for new_data in new_atoms_file:
        new_atoms = read(new_data, index=':')
#       add a label to keep track of which iteration of new data it is
        i_end_new = 0
        if new_data_tag:
            for ats in new_atoms:
                ats.info['n_iter'] = new_data_tag
                i_end_new += len(ats)
#       combine all Atoms objects
        all_atoms += new_atoms

    write(all_atoms_file, all_atoms )


