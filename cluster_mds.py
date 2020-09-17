#************************************************************************************************************
# This is the cluster-based MultiDimensional Scaling code for dimensionality reduction data analysis.       #
#                                                                                                           #
#                                                cl-MDS                                                      #
#                                                                                                           #
# This code has been written and is copyright (c) 2018-2020 of the following authors:                       #
#                                                                                                           #
# *) Patricia Hernandez-Leon                                                                                #
# *) Miguel A. Caro                                                                                         #
#                                                                                                           #
# from the Department of Electrical Engineering and Automation, Aalto University, Finland                   #
#                                                                                                           #
#                                                                                                           #
# See the file LICENSE.md for license information and the README.md file for practical installation         #
# instructions and usage examples. The official code repository is                                          #
#                                                                                                           #
#                                   https://github.com/mcaroba/cl-MDS/                                       #
#                                                                                                           #
# Visit the repository for the latest version of this distribution.                                         #
#                                                                                                           #
#                                                                                                           #
# If you use cl-MDS for the compilation of academic/scientific/technical work, please cite, as appropriate:  #
#                                                                                                           #
# P. Hernandez-Leon and M.A. Caro, XXX, YYY (2020)                                                          #
#                                                                                                           #
#************************************************************************************************************


# Import dependencies
import numpy as np
from sklearn import manifold
# We should include kmedoids in the installation                                                              <--- comment
import kmedoids
import random
import sys


#************************************************************************************************************
class clMDS:
    """
    This is the main clMDS class. It can take a distance matrix and/or an ASE compatible
    (e.g., an xyz file) atomic structure. It can also take concatenated xyz files or any trajectory that
    can be imported with ASE.
    If the user chooses an implemented descriptor, clMDS will make an educated guess and assign some sensible
    defaults. If the user wants more control over the choice of hyperparameters, they should pass a
    distance matrix instead.
    """

#   Initialize the class:
    def __init__(self, dist_matrix=None, atoms=None, descriptor=None, descriptor_string=None,
                 sparsify=None, n_sparse=None, average_kernel=False, cutoff=None, verbose=True):
#       This is the list of implemented atomic descriptors (it typically requires external
#       programs)
        implemented_descriptors = ["quippy_soap"]
        sparse_options = ["random"]
        self.verbose = verbose
        self.sparsify = sparsify
        self.n_sparse = n_sparse
        self.is_clustered = False
        self.has_clmds = False
        self.cutoff = cutoff
        self.average_kernel = average_kernel
        self.compute_non_sparse = False
        if sparsify is not None:
            if isinstance(sparsify, (list, np.ndarray)):
                self.n_sparse = len(sparsify)
                self.sparsify = sparsify
            elif sparsify not in sparse_options:
                raise Exception("The sparsify option you chose is not available. Choose one of the following: ",
                                sparse_options)
            else:
                if n_sparse is None:
                    raise Exception("If you chose a sparsification option, you need to pass the n_sparse parameter")
                else:
                    self.sparsify = sparsify
                    self.n_sparse = n_sparse
#       Implement the CUR decomposition as a sparsify option                                                   <--- comment
#       BE CAREFUL WITH POSSIBLE REPEATED ENTRIES (should I make the core of the code more robust?)    
                    
#       The descriptor is not attached until build_descriptor() is run
        self.has_descriptor = False

#       Read in the distance matrix
        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
            self.has_dist_matrix = True
        else:
            self.has_dist_matrix = False

#       Read in the atoms file and create an ASE atoms object
        if atoms is not None:
            try:
                from ase.io import read, write
                self.atoms = read(atoms, index=":")
            except:
#               This error appears also when atoms has a wrong file extension, should we mention this too?      <-- comment
                raise Exception("I couldn't find an ASE installation; you need ASE to pass an atoms filename!")
        else:
            self.atoms = None

#       Check if the user wants to use a descriptor
        if descriptor is not None:
            if descriptor not in implemented_descriptors:
                raise Exception("The descriptor you chose is not implemented; the options are: ",
                                implemented_descriptors)
            if dist_matrix is not None:
                raise Exception("You can't define a distance matrix and a descriptor; choose one or the other!")
            if atoms is None:
                raise Exception("If you define a descriptor, you must also provide an atoms filename")
            self.descriptor_type = descriptor
            self.descriptor_string = descriptor_string


#   This method takes care of adding a descriptor to the clMDS class:
    def build_descriptor(self):
        if not hasattr(self, 'descriptor_type'):
            raise Exception("You must define a descriptor, check the implemented options")

        descriptor = self.descriptor_type
        descriptor_string = self.descriptor_string
        if descriptor == "quippy_soap":
            from ase.data import atomic_numbers
            from quippy.descriptors import Descriptor
            from quippy.convert import ase_to_quip
            self.zeta = 4
#           This uses some default SOAP parameters
            if descriptor_string is None:
                species_list = []
                for ats in self.atoms:
                    for at in ats:
                        if at.symbol not in species_list:
                            species_list.append(at.symbol)
                species_string = ""
                for species in species_list:
                    species_string += " " + str(atomic_numbers[species])
                n_Z = len(species_list)
                if self.cutoff == None:
                    cutoff = 3.0
                    self.cutoff = cutoff
                else:
                    cutoff = self.cutoff
                quippy_string = "soap n_max=8 l_max=8 cutoff=" + str(cutoff) + " atom_sigma=0.5 Z={" + \
                                species_string + "} species_Z={" + species_string + "} n_Z=" + str(n_Z) + \
                                " n_species=" + str(n_Z) 
                if self.average_kernel:
                    quippy_string = quippy_string + " average=T" 
#           This uses a user-defined SOAP
            else:
                quippy_string = self.descriptor_string
                if self.cutoff is not None:
                    cutoff = self.cutoff
#               Cumbersome code to get the cutoff from a string:
                else:
                    a = quippy_string
                    for i in range(0, len(a.split())):
                        b = a.split()[i]
                        if b[0:6] != "cutoff":
                            continue
                        if len(b) == 6:
                            c = a.split()[i+1]
                            if len(c) == 1:
                                cutoff = float(a.split()[i+2])
                                break
                            else:
                                cutoff = float(c[1:])
                                break
                        elif len(b) == 7:
                            c = a.split()[i+1]
                            cutoff = float(c)
                            break
                        else:
                            cutoff = float(b[7:])
                            break
                    self.cutoff = cutoff
#               Check if both quippy_string and self.average_kernel have equal True/False values              <-- comment
#               (otherwise there could be undetected errors, the same can happen with the cutoff)

            d = Descriptor(quippy_string)
            if self.average_kernel:
                n_env = len(self.atoms)
            else:
                n_env = sum(len(ats) for ats in self.atoms)
            if self.sparsify is not None:
                if not self.compute_non_sparse:
                    if self.sparsify == "random":
                        sparse_list = list(range(n_env))
                        np.random.shuffle(sparse_list)
                        sparse_list = sparse_list[0:self.n_sparse]
                        self.sparse_list = sorted(sparse_list)
                    else:
                        if len(self.sparsify) > n_env:
                            raise Exception("The sparse set can't be bigger than the complete dataset")
                        else: 
                            sparse_list = sorted(self.sparsify)
                            self.sparse_list = sparse_list
                else:
                    sparse_list = [i for i in range(0, n_env) if i not in self.sparse_list]
                    self.sparse_list = sparse_list
                    self.all_env = n_env
            else:    
#               Added to avoid possible crushes in other methods
                self.sparse_list = list(range(n_env))

            n = 0
            descriptor = []
            species_list = []
            config_type_list = []
            if self.verbose:
                print("")
            for ats in self.atoms:
                if self.verbose:
                    sys.stdout.write('\rComputing descriptors:%6.1f%%' % (float(n)*100./float(n_env)) )
                    sys.stdout.flush()
                species_list.append(ats.symbols)
                if not self.average_kernel:
                    if "config_type" in ats.info:
                        config_type_list.append([ats.info["config_type"]]*len(ats))
                    else:
                        config_type_list.append([None]*len(ats))
                else:
                    if "config_type" in ats.info:
                        config_type_list.append(ats.info["config_type"])
                    else:
                        config_type_list.append(None)
                a = ase_to_quip(ats)
                a.set_cutoff(cutoff)
                a.calc_connect()
                qs = d.calc_descriptor(a)
                for q in qs:
                    if self.sparsify is not None:
                        if n in sparse_list:
                            descriptor.append(q)                       
                    else:
                        descriptor.append(q)
                    n += 1
            if self.verbose:
                sys.stdout.write('\rComputing descriptors:%6.1f%%' % 100. )
                sys.stdout.flush()
                print("")

            if not self.average_kernel:
                self.config_type_list = np.concatenate([c for c in config_type_list])
            else:
                self.config_type_list = np.array(config_type_list)
            self.n_env = len(descriptor)
            self.species_list = np.concatenate([z for z in species_list])
            self.descriptor = np.array(descriptor)
            self.has_descriptor = True
        else:
            raise Exception("You must choose among the implemented descriptors: ", implemented_descriptors)


#   This method takes care of building a distance matrix:
    def build_dist_matrix(self):
        if self.descriptor_type is not None:
#           If the descriptors have not been computed, we need to do so
            if not self.has_descriptor:
                self.build_descriptor()

            n_env = self.n_env
            dist_matrix = np.zeros([n_env, n_env])
            n = 0
            if self.verbose:
                print("")
            for i in range(0, n_env):
                if self.verbose:
                    sys.stdout.write('\rComputing dist_matrix:%6.1f%%' % (float(n)*100./float(n_env*(n_env+1)/2)) )
                    sys.stdout.flush()
#               Do this to remove numerical round-off problems
                dist_matrix[i,i] = 0.
                for j in range(i+1, n_env):
                    prod = np.dot(self.descriptor[i], self.descriptor[j])**self.zeta
                    if prod <= 1.:
                        dist_matrix[i][j] = np.sqrt(1. - prod)
                        dist_matrix[j][i] = np.sqrt(1. - prod)
                    else:
                        dist_matrix[i][j] = 0.
                        dist_matrix[j][i] = 0.
                    n += 1
            if self.verbose:
                sys.stdout.write('\rComputing dist_matrix:%6.1f%%' % 100. )
                sys.stdout.flush()
                print("")
            self.dist_matrix = dist_matrix
            self.has_dist_matrix = True
        else:
            raise Exception("No descriptor defined: nothing to do!")


#   This method clusters the data and produces the embedded 2-dimensional coordinates
#   Make sure these are sensible defaults!!!!!!                                                              <-- comment
    def cluster_MDS(self, hierarchy, iter_med=10000, t_max=100, init_medoids="random", n_iso_med=None,
                    n_init_mds_cluster=10, max_iter_cluster=200, n_jobs_cluster=1, verbose_cluster=0,
                    n_anchorpts=3, n_init_mds_anchorpts=3500, max_iter_anchorpts=300, n_jobs_anchorpts=1,
                    verbose_anchorpts=0):
        """
        NOTE: Use hierarchy = [n_clusters, n_level1, n_level2, ... , 1] to perform hierarchical
        embedding and clustering. There, n_clusters refers to the finest clustering (computed 
        in the data n-dimensional space) and 1 refers to the final embedded 2d-space.
        Depending on the chosen hierarchy, different local structures of the dataset can be 
        weighted more during the computation. The simplest hierarchy system is [n_clusters, 1], 
        where only one level of clustering is considered.
        """
        if len(hierarchy) == 1:
            hierarchy.append([1])

        if not self.has_dist_matrix:
            self.build_dist_matrix()

        self.hierarchy = hierarchy

#       Finest clustering (initial hierarchy level)
        try:
            n_clusters = hierarchy[0]
            assert type(n_clusters) == int
        except:
            raise Exception("You need to define a hierarchy of cluster levels by providing the \
                            hierarchy parameter, e.g., hierarchy=[8,3,1] or hierarchy=[8,1]")
        if self.verbose:
            print("")
        for t in range(0, iter_med):
            if self.verbose:
#               This print is insufficient, make sure all the other tasks within this funtion get printed out   <-- comment
                sys.stdout.write('\rClustering data:%6.1f%%' % (float(t)*100./float(iter_med)) )
                sys.stdout.flush()
            M, C = kmedoids.kMedoids( self.dist_matrix, n_clusters, tmax=t_max, init_Ms=init_medoids, 
                                      n_iso=n_iso_med )
#           Obtain total intra-cluster incoherence
            temp_I = 0.
            for i in range(0, n_clusters):
                temp_I += np.sum(self.dist_matrix[M[i]][C[i]])
            if t == 0:
                I_tot = temp_I
#           Minimize this value
            if temp_I <= I_tot:
                I_tot = temp_I
                ind_medoids, ind_clusters = M, C
        if self.verbose:
            sys.stdout.write('\rClustering data:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")

#       Compute the distance matrix per cluster
        dist_clusters = [self.dist_matrix[np.ix_(ind_clusters[i], ind_clusters[i])]
                         for i in range(0, n_clusters)]
#       MDS calculation minimizing the stress
        embedding = manifold.MDS( n_components = 2, dissimilarity = "precomputed",
                                  n_init = n_init_mds_cluster, max_iter = max_iter_cluster,
                                  n_jobs = n_jobs_cluster, verbose = verbose_cluster )
        mds_clusters = np.zeros((len(self.dist_matrix),2))
        if self.verbose:
            print("")
        for i in range(0, n_clusters):
            if self.verbose:
                sys.stdout.write('\rEmbedding data:%6.1f%%' % (float(i)*100./float(n_clusters)) )
                sys.stdout.flush()
            if len(ind_clusters[i]) > 1:
                mds_clusters[ind_clusters[i]] = embedding.fit_transform(dist_clusters[i])
            else:
                mds_clusters[ind_clusters[i]] = np.zeros((1,2)) # avoid sklearn RuntimeWarning
        if self.verbose:
            sys.stdout.write('\rEmbedding data:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")
        
        self.mds_sparse_clusters = mds_clusters

#       Hierarchy levels
        n_levels = len(hierarchy)
        M_prev = ind_medoids
        C_prev = ind_clusters

        embedding_h = manifold.MDS( n_components = 2, dissimilarity = "precomputed",
                                    n_init = n_init_mds_anchorpts, max_iter = max_iter_anchorpts,
                                    n_jobs = n_jobs_anchorpts, verbose = verbose_anchorpts )
        C_int = {}
        linear_transformation = {}
        for level in range(1, n_levels):
            if self.verbose:
                print("")
#           Check the data reorganization needed for this new hierarchy level
            if hierarchy[level] > 1:
#               Assign the clusters of previous level to the current ones
                for t in range(0, 500):
                    temp_m, temp_c = kmedoids.kMedoids(self.dist_matrix[np.ix_(M_prev, M_prev)], hierarchy[level],
                                                       init_Ms="isolated", n_iso=hierarchy[level])
                    temp_I = 0 
                    for i in range(hierarchy[level]):
                        temp_I += np.sum(self.dist_matrix[temp_m[i]][temp_c[i]])/len(temp_c[i])
                    if t == 0:
                        I_rel = temp_I
                    if temp_I <= I_rel:
                        I_rel = temp_I
                        m = temp_m
                        c = temp_c
#               Obtain a dictionary with all the indexes of each new cluster
                C_new = {newcl: np.concatenate( [C_prev[i] for i in c[newcl]] ) for newcl in range(0, hierarchy[level])}
                C_int[level] = C_new
            elif hierarchy[level] == 1:
#               No clustering (consider all data points)
                c = { 0: np.arange(hierarchy[level-1]) }
                C_new = { 0: np.arange(len(self.dist_matrix)) }
            else:
                raise Exception("There is a wrong entry in the hierarchy parameter, it must have \
                                 non-zero integers only (e.g. hierarchy=[8,1])")

#           Obtention of anchor points
#           Consider anchor points AND medoids separately
            mds_M_prev = mds_clusters[M_prev]
            mds_A = []
            ind_A = []
            for i in range(0, hierarchy[level-1]):
                if self.verbose:
                    sys.stdout.write( '\rHierarchy level %i (anchor points):%6.1f%%' 
                                      % (level-1, float(i)*100./float(hierarchy[level-1])) )
                    sys.stdout.flush()
#               Exclude the medoid as a possible anchor point
                C_prev_nomed = np.setdiff1d(C_prev[i], M_prev[i])
#               Choose the number of random iterations needed depending on cluster length
                if len(C_prev[i]) - 1 < 4:
                    n_rand = 1
                elif 4 <= len(C_prev[i]) -1 < 40:
#                   Exact solution: binomial coefficient (len(C_prev[i])-1; n_anchorpts)
                    bin_coef_denom = np.math.factorial(n_anchorpts)*np.math.factorial(len(C_prev[i])-1-n_anchorpts)
                    n_rand = np.math.factorial(len(C_prev[i])-1)/bin_coef_denom
                else:
#                   Optimize the anchor points selection                                                        <-- comment
                    n_rand = 10000
#               MDS of anchor points in previous level
                mds_A.append(anchor_points(n_anchorpts, mds_clusters[C_prev_nomed], n_rand))
#               indexes of anchor points in previous level
                ind_A.append( [np.where(mds_clusters == mds_A[-1][j])[0][0] for j in range(len(mds_A[-1]))] )
            if self.verbose:
                sys.stdout.write('\rHierarchy level %i (anchor points):%6.1f%%' % (level-1, 100.) )
                sys.stdout.flush()
                print("")
            if level == 1:
                self.anchor_indices = ind_A
                               
#           MDS of anchor points and transformation (from previous level to the new one)
            A = {}
            linear_transf = {}
            transf_coordinates = np.zeros((len(self.dist_matrix),2))
            n = 0
            for newcl in range(0, hierarchy[level]):
                if self.verbose:
                    sys.stdout.write( '\rHierarchy level %i (transformation):%6.1f%%' 
                                      % (level-1, float(n)*100./float(hierarchy[level-1])) )
                    sys.stdout.flush()
                temp_A = np.concatenate( [ind_A[i] for i in c[newcl]] ).astype('int32')
                A[newcl] = np.concatenate( (temp_A, M_prev[c[newcl]]) )
#               metric matrix for the anchor points + medoids
                dist_anchor = self.dist_matrix[np.ix_(A[newcl], A[newcl])]
                mds_anchor = embedding_h.fit_transform(dist_anchor)

#               transformation matrix T ( X·T = X')
                total_n_anchor = np.zeros((len(c[newcl])+1,), dtype=int)
                for l, i in enumerate(c[newcl]):
                    n += 1
                    total_n_anchor[l+1] = total_n_anchor[l] + len(ind_A[i])
                    diff_X_prev = mds_A[i] - mds_M_prev[i]
                    diff_X_new = mds_anchor[total_n_anchor[l]:total_n_anchor[l+1]] \
                                 - mds_anchor[l-len(c[newcl])]
                    T = np.linalg.lstsq(diff_X_prev, diff_X_new, rcond=None )[0]
#                   Transform and translate each cluster to the origin of its transf. matrix T (i.e. its medoid)
                    correction = mds_anchor[l-len(c[newcl])] 
                    transf_coordinates[C_prev[i]] = np.dot(mds_clusters[C_prev[i]]
                                                            - mds_M_prev[i], T) + correction

#                   Save the transformation and its correction for testing and coordinate estimations
                    linear_transf[i] = T
            linear_transformation[level-1] = linear_transf
            if self.verbose:
                sys.stdout.write('\rHierarchy level %i (transformation):%6.1f%%' % (level-1, 100.) )
                sys.stdout.flush()
                print("")

            if hierarchy[level] > 1:
#               Reassign the label "previous" to the new results
                M_prev = M_prev[m]
                C_prev = C_new
                mds_clusters = transf_coordinates

#       These indices refer to the dist_matrix; we need to make sure that the information required to retrieve  <-- comment
#       the atomic structures from the original data base are consistent with the sparsification technique used <-- comment
#       We should also give the option to output the mds coordinates to the xyz file and generate carved xyz    <-- comment
#       structures around the medoids for plotting                                                              <-- comment
        self.has_clmds = True
        self.sparse_coordinates = transf_coordinates
        self.sparse_clusters = ind_clusters
        self.sparse_medoids = ind_medoids
        self.linear_transformation = linear_transformation
        self.correction_medoids = correction

        sparse_cluster_indices = np.empty(len(self.dist_matrix), dtype=int)
        for i in range(0, hierarchy[0]):
             cluster = self.sparse_clusters[i]
             sparse_cluster_indices[cluster] = i

        self.sparse_cluster_indices = sparse_cluster_indices

        sparse_int_cluster_indices = {}
        if n_levels > 2:
            for level in range(1, n_levels-1):
                int_indices = np.empty(len(self.dist_matrix), dtype=int)
                for i in range(0, hierarchy[level]):
                    cluster = C_int[level][i]
                    int_indices[cluster] = i                    
                sparse_int_cluster_indices[level] = int_indices

            self.sparse_int_cluster_indices = sparse_int_cluster_indices



#   This is a user friendly function that returns the clusters and medoids of the sparse set
    def get_sparse_coordinates(self, hierarchy):
        if not self.has_clmds:
            self.cluster_MDS(hierarchy = hierarchy)

        ext_coordinates = np.empty([self.n_env,3])
        ext_coordinates[0:self.n_env, 0:2] = self.sparse_coordinates
        ext_coordinates[0:self.n_env, 2] = self.sparse_cluster_indices

        return ext_coordinates
              

#   This method gives a "cheap" estimation of the MDS coordinates of the points not included in the sparse set
    def compute_estim_coordinates(self, hierarchy):
        if not self.has_clmds:
            self.cluster_MDS(hierarchy = hierarchy)

        hierarchy = self.hierarchy
        sparse_list = self.sparse_list
        sparse_descriptor = self.descriptor

#       Compute the descriptors of all the atoms left out of the sparse set
        self.compute_non_sparse = True
        self.build_descriptor()
        all_env = self.all_env
        n_env = self.n_env
        non_sparse_list = self.sparse_list
#       Classify them considering the clustering of the sparse set
        cluster_indices = np.zeros(all_env, dtype=int)
        cluster_indices[sparse_list] = self.sparse_cluster_indices
        if self.verbose:
            print("")
        for i in range(0, n_env):
            if self.verbose:
                sys.stdout.write('\rAssigning each point to cluster:%6.1f%%' % (float(i)*100./float(n_env)) )
                sys.stdout.flush()
            dist_med = np.zeros(hierarchy[0])
            for j, med in enumerate(self.sparse_medoids):
                prod = np.dot(self.descriptor[i], sparse_descriptor[med])**self.zeta 
                if prod < 1.:
                    dist_med[j] =  np.sqrt(1. - prod)
                else:
                    dist_med[j] = 0.
            cluster_indices[non_sparse_list[i]] = np.argmin(dist_med)           
        if self.verbose:
            sys.stdout.write('\rAssigning each point to cluster:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")

#       Compute the transformation from kernel space to 2D
        local_coordinates = np.zeros((all_env,2))       
        transf_coordinates = np.zeros((all_env,2))
        dist_transf = {}
        for i in range(0, hierarchy[0]):
            if self.verbose:
                print("")
#           distance matrix per cluster
            M = self.sparse_list[self.sparse_medoids[i]]
            C = np.where(cluster_indices == i)[0] 
            C_sparse = self.sparse_clusters[i]
            N_c = len(C)
            N_sparse = len(C_sparse)
            dist_cluster = np.empty([N_c, N_sparse])
            for j in range(0, N_c):
                if self.verbose:
                    sys.stdout.write('\rEmbedding all data (cluster %i):%6.1f%%' % (i, float(j)*100./float(N_c)) )
                    sys.stdout.flush()
                if C[j] in sparse_list:
                    ind_sparse = np.where(sparse_list == C[j])[0]
                    assert ind_sparse in C_sparse
                    dist_cluster[j,:] = self.dist_matrix[ind_sparse, C_sparse]  
                    continue
                ind = np.where(non_sparse_list == C[j])[0]
                for k in range(0, N_sparse):
                    ind_sparse = C_sparse[k]
                    prod = np.dot(self.descriptor[ind], sparse_descriptor[ind_sparse])**self.zeta
                    if prod < 1.:
                        dist_cluster[j][k] = np.sqrt(1. - prod)
                    else:
                        dist_cluster[j][k] = 0.          

            dist_sparse_cluster = self.dist_matrix[np.ix_(C_sparse, C_sparse)]
            dist_medoid = self.dist_matrix[self.sparse_medoids[i], C_sparse]
            local_mds_sparse = self.mds_sparse_clusters[C_sparse,:]
            local_mds_medoid = self.mds_sparse_clusters[self.sparse_medoids[i],:]
            global_mds_medoid = self.sparse_coordinates[self.sparse_medoids[i], :] 
#           transformation matrix T from distance space to 2D ( X·T = Y)
            X_anchor = dist_sparse_cluster
            Y_anchor = local_mds_sparse
            T = np.linalg.lstsq(X_anchor, Y_anchor, rcond=None)[0]
            dist_transf[i] = T
            local_coordinates[C] = np.dot(dist_cluster - dist_medoid, T)
#           Obtain the coordinates, considering previous linear transformations
            T0 = self.linear_transformation[0][i]
            transf_coordinates[C] = np.dot(local_coordinates[C], T0)
            for level in range(1, len(hierarchy)-1):
                cluster_ind = self.sparse_int_cluster_indices[level][C_sparse][0]
                T1 = self.linear_transformation[level][cluster_ind]
                transf_coordinates[C] = np.dot(transf_coordinates[C], T1)
            local_coordinates[C] = local_coordinates[C] + local_mds_medoid
            transf_coordinates[C] = transf_coordinates[C] + global_mds_medoid
            if self.verbose:
                sys.stdout.write('\rEmbedding all data (cluster %i):%6.1f%%' % (i, 100.) )
                sys.stdout.flush()
                print("")
                                           
        self.sparse_list = list(sparse_list) 
        self.descriptor = sparse_descriptor
        self.distance_transformation = dist_transf 
        self.all_local_coordinates = local_coordinates

        self.all_cluster_indices = cluster_indices
        self.all_coordinates = transf_coordinates



#   This is a user friendly function that returns the clusters and medoids of the complete set
    def get_all_coordinates(self, hierarchy):
        if not self.compute_non_sparse:
            self.compute_estim_coordinates(hierarchy = hierarchy)

        ext_coordinates = np.empty([self.all_env,3])
        ext_coordinates[0:self.all_env, 0:2] = self.all_coordinates
        ext_coordinates[0:self.all_env, 2] = self.all_cluster_indices

        return ext_coordinates



#   This method writes an extended xyz file with the cl-MDS coordinates
    def write_xyz(self, filename=None):
        if filename == None:
            raise Exception("You must define a filename to write to disk")

        if self.atoms == None:
            raise Exception("You need to provide an input atoms file if you want to write the coordinates \
                             to an xyz file.")

        if not self.has_clmds:
            raise Exception("You haven't run a cl-MDS coordinate calculation yet!")

        from ase.io import write

        new_atoms = self.atoms.copy()
        n = 0
        for ats in new_atoms:
            natoms = len(ats)
            coords = np.empty([natoms, 2], dtype=float)
            cluster = np.empty(natoms, dtype=int)
            if not self.compute_non_sparse:
                coords[:,:] = np.NaN
                cluster[:] = -1
            for j in range(0, natoms):
                if not self.compute_non_sparse:
                    if n in self.sparse_list:
                        i = self.sparse_list.index(n)
                        coords[j] = self.sparse_coordinates[i]
                        cluster[j] = self.sparse_cluster_indices[i]
                else:
                    coords[j] = self.all_coordinates[n]
                    cluster[j] = self.all_cluster_indices[n]
                n += 1

            ats.new_array("clmds_coords", coords)
            ats.new_array("cluster_number", cluster)

        write(filename, new_atoms)

#   This method exports carved medoid environments to xyz files (ONLY for average_kernel=False)
    def medoids_to_xyz(self, dir=None, carve_radius=None, render=False, bond_cutoff=1.9, gnuplot=False):
        if dir == None:
            raise Exception("You must define a directory to write medoid's xyz files to")

        if self.atoms == None:
            raise Exception("You need to provide an input atoms file if you want to write the coordinates \
                             to an xyz file.")

        if not self.has_clmds:
            raise Exception("You haven't run a cl-MDS coordinate calculation yet!")

        if carve_radius == None:
            cutoff = self.cutoff
        else:
            cutoff = carve_radius

        from ase.io import write
        from ase import Atoms
        import os

        if not os.path.exists(dir):
            os.mkdir(dir)

        n = 0
        for ats in self.atoms:
            natoms = len(ats)
            for j in range(0, natoms):
                if n in self.sparse_list:
                    i = self.sparse_list.index(n)
                    if i in self.sparse_medoids:
                        pos = ats.get_positions()
                        cell = ats.get_cell()
                        new_pos = []
                        neighbors = []
                        site = Atoms()
                        for j2 in range(0, natoms):
                            d = ats.get_distance(j, j2, mic=True)
                            if d < cutoff:
                                neighbors.append(j2)
                                new_pos.append(pos[j2])
                                site += ats[j2]

                        site.set_cell(cell)
                        site.set_positions(new_pos)
                        site.set_pbc(True)
                        shift = np.array([cutoff, cutoff, cutoff]) - pos[j]
                        site.translate(shift); site.wrap()
                        site.set_cell([2.*cutoff, 2.*cutoff, 2.*cutoff])
                        site.set_pbc(False)
                        i2 = np.where(self.sparse_medoids == i)[0][0]
                        write(dir + "/medoid_%i.xyz" % i2, site)
                n += 1

        if render:
            try:
                from ovito.io import import_file
                from ovito.vis import Viewport
#                from ovito.modifiers import CreateBondsModifier
            except:
                raise Exception("You need Ovito (pip3 install ovito) to use the rendering capability")

            for i in range(0, len(self.sparse_medoids)):
                atoms = import_file(dir + "/medoid_%i.xyz" % i)
#               I didn't manage to get bonds to render                                                          <-- comment
#                modifier = CreateBondsModifier(cutoff = bond_cutoff)
#                modifier.vis.enabled = True
#                modifier.vis.width = 0.3
#                atoms.modifiers.append(modifier)
#                atoms.compute()
                atoms.source.data.cell.vis.render_cell = False
                atoms.add_to_scene()
                vp = Viewport()
                vp.type = Viewport.Type.Perspective
                vp.zoom_all()
                vp.render_image(filename=dir+"/medoid_%i.png" % i, size=(400,400), alpha=True)
                atoms.remove_from_scene()

        if gnuplot:
            if not self.compute_non_sparse:
                ext_coords = self.get_sparse_coordinates(hierarchy=self.hierarchy)
            else:
                ext_coords = self.get_all_coordinates(hierarchy=self.hierarchy)
            f = open(dir + "/xy.dat", "w+")
            for i in ext_coords:
                print(i[0], i[1], i[2], file=f)

            f.close()
            f = open(dir + "/gnuplot.script", "w+")
            print("set term pngcairo size 640,640; set output 'clmds_map.png'", file=f)
            print("set size ratio -1", file=f)
            print("set xlabel 'MDS coordinate 1'", file=f)
            print("set ylabel 'MDS coordinate 2'", file=f)
            if render:
                print("plot 'xy.dat' u 1:2:3 lc var pt 7 not, \\", file=f)
                for i in range(0, len(self.sparse_medoids)):
                    x, y = self.sparse_coordinates[self.sparse_medoids[i]]
                    print("     'medoid_" + str(i) + ".png' binary filetype=png dx=0.0005" + \
                          "center=(" + str(x) + "," + str(y) + ") w rgbalpha not, \\", file=f)
            else:
                print("plot 'xy.dat' u 1:2:3 lc var not", file=f)

            f.close()

            try:
                os.system("cd " + dir + "; gnuplot gnuplot.script")
            except:
                print("I tried running gnuplot but it failed!")
#************************************************************************************************************





#************************************************************************************************************
# Suporting functions 

#  Given a dataset, this method choose the N points corresponding to the N vertices of the polygon containing
#  the highest number of data points. 
#  It uses random-chosen sequences (less accurate but faster in general).
def anchor_points(N, points, n_random):
    """
    NOTE: only valid with N=3,4 !!!
    """
    # Only implemented for N = 3, 4
    if not (N == 3 or N == 4):
        raise Exception("Only N=3 and N=4 anchor points implemented!")

    # Check if the number of samples given is enough to build at least 2 N-gons
    if len(points) > N:
        s_opt = 0
        anchor_p = points[:N]
        for m in np.arange(n_random):
            shuffle_points = random.sample(list(points), len(points))
            temp_vertices = shuffle_points[:N]
            if N == 3:
                s = points_in_triang(temp_vertices, shuffle_points[3:])
            else:
               # ONLY VALID FOR N=4
               s = points_in_quad(temp_vertices,shuffle_points[N:])
            if s > s_opt:
                s_opt = s
                anchor_p = temp_vertices
    else:
        anchor_p = points

    return np.array(anchor_p)


# Computation of the number of points from a given set lying within a triangle whose vertices are known.
def points_in_triang(vertices, other_points):
    s=0
    for point in other_points:
        # Sign point, vertex 1, vertex 2
        b0 = (point[0]-vertices[1][0])*(vertices[0][1]-vertices[1][1]) \
            - (vertices[0][0]-vertices[1][0])*(point[1]-vertices[1][1])
        # Sign point, vertex 2, vertex 3
        b1 = (point[0]-vertices[2][0])*(vertices[1][1]-vertices[2][1]) \
            - (vertices[1][0]-vertices[2][0])*(point[1]-vertices[2][1])
        # Sign point, vertex 3, vertex 1
        b2 = (point[0]-vertices[0][0])*(vertices[2][1]-vertices[0][1]) \
            - (vertices[2][0]-vertices[0][0])*(point[1]-vertices[0][1])
        if (b0*b1 > 0) & (b1*b2 > 0):
            s += 1

    return s


# Computation of the number of points from a given set lying within a quadrilateral whose vertices are known.
def points_in_quad(vertices, other_points):
    """
    NOTE: Double counting of the points lying over the diagonal 
    (we only need an estimation, not the exact count)
    """
    # Get one of its diagonal
    # Option 1: (vertex 1, vertex 2) , (vertex 3, vertex 4)
    sum1 = np.linalg.norm(vertices[1]-vertices[0]) + np.linalg.norm(vertices[3]-vertices[2])
    # Option 2: (vertex 1, vertex 3) , (vertex 2, vertex 4)
    sum2 = np.linalg.norm(vertices[2]-vertices[0]) + np.linalg.norm(vertices[3]-vertices[1])
    # Option 3: (vertex 1, vertex 4) , (vertex 2, vertex 3)
    sum3 = np.linalg.norm(vertices[3]-vertices[0]) + np.linalg.norm(vertices[2]-vertices[1])
    diag_opt = np.argmax([sum1,sum2,sum3])
    diag_1 = [vertices[0],vertices[diag_opt+1]]
    diag_2 = np.delete(vertices,[0,diag_opt+1],0)
    # Divide the quadrilateral in two triangles
    triangle_1 = np.concatenate((diag_1, [diag_2[0]]),axis=0)
    triangle_2 = np.concatenate((diag_1, [diag_2[1]]),axis=0)
    s1 = points_in_triang(triangle_1, other_points)
    s2 = points_in_triang(triangle_2, other_points)

    return s1 + s2
#************************************************************************************************************
