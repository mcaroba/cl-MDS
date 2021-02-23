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
from kmedoids import kmedoids
from cur import cur
import random
import sys
import itertools
from scipy.special import comb
from scipy.spatial import ConvexHull


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
        sparse_options = ["random", "cur"]
        self.verbose = verbose
        self.sparsify = sparsify
        self.n_sparse = n_sparse
        self.is_clustered = False
        self.has_clmds = False
        self.cutoff = cutoff
        self.average_kernel = average_kernel
        self.compute_non_sparse = False
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

#       Check if the user wants to use sparsification       
        if sparsify is not None:
            if isinstance(sparsify, (list, np.ndarray)):
                sparsify = list(set(sparsify)) # Avoid repeated entries
                self.n_sparse = len(sparsify)
                self.sparsify = sparsify
            elif sparsify not in sparse_options:
                raise Exception("The sparsify option you chose is not available. Choose one of the following: ",
                                sparse_options, " or provide a list of indexes")
            else:
                if n_sparse is None:
                    raise Exception("If you choose a sparsify option, you need to pass the n_sparse parameter")
                else:
                    self.sparsify = sparsify
                    self.n_sparse = n_sparse
                if sparsify == "cur" and self.has_dist_matrix:
                    self.sparse_list = list(set(cur.cur_decomposition(self.dist_matrix, n_sparse)[-1]))
                    self.dist_matrix = dist_matrix(np.ix_(self.sparse_list, self.sparse_list))
#               Implement the "optimized sparse set" as a sparsify option                                           <--- comment  


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
                    if isinstance(self.sparsify, (list, np.ndarray)):
                        if len(self.sparsify) > n_env:
                            raise Exception("The sparse set can't be larger than the complete dataset")
                        else: 
                            sparse_list = self.sparsify
                            self.sparse_list = sparse_list
                    elif self.sparsify == "random":
                        sparse_list = list(range(n_env))
                        np.random.shuffle(sparse_list)
                        sparse_list = sparse_list[0:self.n_sparse]
                        self.sparse_list = sorted(sparse_list)
                    elif self.sparsify == "cur":
                        sparse_list = list(range(n_env))
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
                    if self.sparsify is None:
                        descriptor.append(q)
                    else:
                        if n in sparse_list:
                            descriptor.append(q)                       
                    n += 1
            descriptor = np.array(descriptor)
            if not self.compute_non_sparse and (self.sparsify == "cur"):
                self.sparse_list = list(set(cur.cur_decomposition(descriptor, self.n_sparse)[-1]))
                descriptor = descriptor[self.sparse_list,:]
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
            self.descriptor = descriptor
            self.has_descriptor = True
        else:
            raise Exception("You must choose among the implemented descriptors: ", implemented_descriptors)


#   This method takes care of building a distance matrix:
    def build_dist_matrix(self, precision=1.e-8):
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
                    if prod <= 1. - precision:
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
    def cluster_MDS(self, hierarchy, iter_med=10000, tmax=100, init_medoids="isolated", n_iso_med=1,
                    n_init_mds_cluster=10, max_iter_cluster=200, n_jobs_cluster=1, verbose_cluster=0,
                    n_anchor=4, criterion_anchor="area", n_init_mds_anchor=3500, max_iter_anchor=300, 
                    n_jobs_anchor=1, verbose_anchor=0, precision_qhull=1e-7):
        """
        Parameters:

        * Hierarchy
        Use hierarchy = [n_clusters, n_level1, n_level2, ... , 1] to perform hierarchical embedding
        and clustering, where n_clusters refers to the finest clustering (computed in the data
        n-dimensional space) and 1 refers to the final embedded 2d-space.
        Depending on the chosen hierarchy, different local structures of the dataset can be weighted 
        more during the computation. The simplest hierarchy system is [n_clusters, 1], where only
        one level of clustering is considered.

        * Clustering (iter_med, tmax, init_medoids, n_iso_med)
        Check kmedoids for further information.

        * Embedding (Initial clusters: n_init_cluster, max_iter_cluster, n_jobs_cluster, verbose_cluster;
                     Anchor points: n_init_mds_anchor, max_iter_anchor, n_jobs_anchor, verbose_anchor)
        Check sklearn.manifold.MDS for additional information

        * Anchor points (n_anchor, criterion_anchor, precision_qhull)
        This method only supports n_anchor=3,4 (the anchor point selection process and later
        transformations won't make sense with other values).
        Check cluster_mds (anchor_points()) and scipy.spatial.ConvexHull for further information.

        """
        if isinstance(hierarchy, list):
            try:
                n_clusters = hierarchy[0]
                assert type(n_clusters) == int
            except:
                raise Exception("You need to define a hierarchy of cluster levels by providing the \
                                 hierarchy parameter, e.g., [8,1] or [8,3,1]")
            if len(hierarchy) == 1:
                hierarchy.append(1)
        elif isinstance(hierarchy, int):
            n_clusters = hierarchy
            hierarchy = [hierarchy, 1]
        else:
            raise Exception("You need to define a hierarchy of cluster levels by providing the \
                            hierarchy parameter, e.g., [8,1] or [8,3,1]")

        if not self.has_dist_matrix:
            self.build_dist_matrix()

        self.hierarchy = hierarchy

#       Finest clustering (initial hierarchy level) 
        ind_medoids, ind_clusters = optim_kmedoids( self.dist_matrix, n_clusters, incoherence="rel",
                                                    n_iter=iter_med, tmax=tmax, init_Ms=init_medoids,
                                                    n_iso=n_iso_med, verbose=self.verbose )
        dist_clusters = [self.dist_matrix[np.ix_(ind_clusters[i], ind_clusters[i])]
                         for i in range(0, n_clusters)]
#       MDS calculation minimizing the stress
        embedding = manifold.MDS( n_components=2, dissimilarity="precomputed",
                                  n_init=n_init_mds_cluster, max_iter=max_iter_cluster,
                                  n_jobs=n_jobs_cluster, verbose=verbose_cluster )
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
        
        self.local_sparse_coordinates = mds_clusters

#       Hierarchy levels
        n_levels = len(hierarchy)
        M_prev = ind_medoids
        C_prev = ind_clusters

        embedding_h = manifold.MDS( n_components=2, dissimilarity="precomputed", n_init=n_init_mds_anchor,
                                    max_iter=max_iter_anchor, n_jobs=n_jobs_anchor, verbose=verbose_anchor )                       
        H = {}                  
        C_hierarchy = {}
        T_hierarchy = {i:{} for i in range(0, n_clusters)}
        for level in range(1, n_levels):
            if self.verbose:
                print("")
                print( '\rHierarchy level %i (%i ---> %i clusters)' % (level-1, hierarchy[level-1], 
                        hierarchy[level]) )
#           Check the data reorganization needed for this new hierarchy level
            if hierarchy[level] > 1:
#               Assign the clusters of previous level to the current ones
                D = self.dist_matrix[np.ix_(M_prev, M_prev)]
                M, C = optim_kmedoids( D, hierarchy[level], incoherence="rel", n_iter=500, init_Ms="isolated",
                                       n_iso=hierarchy[level], verbose=self.verbose )
#               Obtain a dictionary with all the indexes of each new cluster
                C_new = { newcl: np.concatenate( [C_prev[i] for i in C[newcl]] ) 
                                                            for newcl in range(0, hierarchy[level]) }
                C_hierarchy[level] = C_new
            elif hierarchy[level] == 1:
#               No clustering (consider all data points)
                C = { 0: np.arange(hierarchy[level-1]) }
                C_new = { 0: np.arange(len(self.dist_matrix)) }
            else:
                raise Exception("There is a wrong entry in the hierarchy parameter, it must have \
                                 non-zero integers only (e.g. hierarchy=[8,1])")
            if level == 1:
                H[level-1] = {i: np.array([i]) for i in range(0, n_clusters)}
            H[level] = {i: np.concatenate( ([H[level-1][j] for j in C[i]]) ) for i in C}

#           Obtention of anchor points (consider anchor points AND medoids separately)
            mds_M_prev = mds_clusters[M_prev]
            mds_A = []
            ind_A = []
            for i in range(0, hierarchy[level-1]):
                if self.verbose:
                    sys.stdout.write( '\rObtaining anchor points:%6.1f%%' 
                                      % (float(i)*100./float(hierarchy[level-1])) )
                    sys.stdout.flush()
                if len(C_prev[i]) <= n_anchor:
                    mds_A.append( mds_clusters[C_prev[i],:] )
                    ind_A.append( C_prev[i] )
                else:
#                   Exclude the medoid as a possible anchor point
                    C_prev_nomed = np.setdiff1d(C_prev[i], M_prev[i])
#                   Choose the procedure for the anchor points calculations depending on cluster length
                    if len(C_prev[i]) - 1 < 20:
                        method_anchor = None
                    else:                                                       
                        method_anchor = "optimized"
#                   MDS and indexes of anchor points in previous level
                    mds = anchor_points( n_anchor, mds_clusters[C_prev_nomed,:], method=method_anchor, 
                                         criterion=criterion_anchor )
                    indexes = [np.where(mds_clusters == mds[j])[0][0] for j in range(0, len(mds))]
#                   Order of the anchor points on the previous level
                    h = ConvexHull(mds_clusters[indexes])
                    mds_A.append( mds[h.vertices] )
                    ind_A.append( np.array(indexes)[h.vertices] )
            if self.verbose:
                sys.stdout.write('\rObtaining anchor points:%6.1f%%' % 100. )
                sys.stdout.flush()
                print("")
                               
#           MDS of anchor points on the new level and their transformations
            self.sparse_coordinates = np.zeros((len(self.dist_matrix), 2))
            for newcl in range(0, hierarchy[level]):
                if self.verbose:
                    print( '\rResult for new cluster %i' % newcl  )
                temp_A = [ind_A[i] for i in C[newcl]] 
                A = np.concatenate( temp_A ).astype('int32')
                dist_anchor = self.dist_matrix[np.ix_(A, A)]
                if len(A) == 1:
                    self.sparse_coordinates[A,:] = np.zeros((1,2)) # avoid sklearn RuntimeWarning   
                    self.order_anchor = [0]
                    self.transformation = [0]
                else:
                    mds_anchor = embedding_h.fit_transform(dist_anchor)
#                   Convexity check per cluster for their new MDS
                    embedding_h.set_params(n_init=1)
                    prev_clusters = [C_prev[i] for i in C[newcl]]
                    self.convexity_check( C[newcl], prev_clusters, temp_A, dist_anchor, mds_anchor,
                                          embedding_h, precision = precision_qhull ) 
                    embedding_h.set_params(n_init=n_init_mds_anchor)
#                   Transformation from previous level to the new one
                    self.transform_2d( C[newcl], prev_clusters, temp_A, mds_clusters )

                for i, cl in enumerate(C[newcl]):
                    for j in H[level-1][cl]:
                        T_hierarchy[j].setdefault(level,{})["cluster"] = newcl
                        T_hierarchy[j].setdefault(level,{})["anchor"] = ind_A[cl][self.order_anchor[i]]
                        T_hierarchy[j].setdefault(level,{})["transf"] = self.transformation[i]
                
            if hierarchy[level] > 1:
#               Reassign the label "previous" to the new results
                M_prev = M_prev[M]
                C_prev = C_new
                mds_clusters = self.sparse_coordinates

#       These indices refer to the dist_matrix; we need to make sure that the information required to retrieve  <-- comment
#       the atomic structures from the original data base are consistent with the sparsification technique used <-- comment
#       We should also give the option to output the mds coordinates to the xyz file and generate carved xyz    <-- comment
#       structures around the medoids for plotting                                                              <-- comment
        self.has_clmds = True
        self.sparse_clusters = ind_clusters
        self.sparse_medoids = ind_medoids
        self.all_transformations = T_hierarchy

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
                    cluster = C_hierarchy[level][i]
                    int_indices[cluster] = i                    
                sparse_int_cluster_indices[level] = int_indices

            self.sparse_int_cluster_indices = sparse_int_cluster_indices



#   This method checks the presence of pathological arrangements of anchor points in the MDS (i.e. non-convex
#   and self-intersecting results) and improves the final MDS solution (free of pathologies)
    def convexity_check(self, clusters, prev_clusters, ind_anchor, dist_anchor, 
                        mds_anchor, embedding, max_perm=6, precision=None): 
        no_pathologies = 0
        final_vertices = []
        N_anchor = np.zeros((len(clusters)+1, ), dtype=int)
        if precision:
            precision = 'E' + str(precision)
        else:
            precision = 'QbB'
        for i in range(0, len(clusters)):
            if self.verbose:
                sys.stdout.write( '\rChecking convexity:%6.1f%%' % (float(i)*100./float(len(clusters)))  )
                sys.stdout.flush()
            N_anchor[i+1] = N_anchor[i] + len(ind_anchor[i])
            if len(prev_clusters[i]) <= 3:
                final_vertices.append(np.arange(0, len(prev_clusters[i]), 1))
                no_pathologies += 1
                continue
#           Check if there is a pathological quadrilateral (non-convex)
            hull = ConvexHull( mds_anchor[N_anchor[i]:N_anchor[i+1], :], qhull_options=precision )
            vertices = hull.vertices
            if len(vertices) == 3:
                final_vertices.append(vertices)
                no_pathologies += 0.5
            elif len(vertices) == 4:
#               Check if there is a pathological quadrilateral (self-intersecting)
                go_ahead = checkpermutation( ind_anchor[i][vertices], ind_anchor[i], verbose=False )
                if go_ahead:
                    final_vertices.append(vertices)
                    no_pathologies += 1
                else:
                    n_perm = 0       
                    temp_vertices = np.copy(vertices)
                    temp_mds = np.copy(mds_anchor)
                    while not go_ahead and (n_perm != max_perm):
#                       We need to permute the anchor coordinates for this cluster
                        perm_cluster = temp_mds[N_anchor[i]:N_anchor[i+1], :]
                        n_cycle = np.where(vertices == 0)[0][0]
                        perm_0 = np.roll(vertices, -n_cycle)
                        if (perm_0[1] == 2) & (n_perm <= 3): 
                            n_perm+=1
                            perm_cluster[[0,3]] = perm_cluster[[3,0]]
                        elif (perm_0[1] == 2): 
                            n_perm+=1
                            perm_cluster[[1,2]] = perm_cluster[[2,1]]
                        else:
                            n_perm+=1
                            perm_cluster[[0,1]] = perm_cluster[[1,0]]
                        init_embed = temp_mds
                        init_embed[N_anchor[i]:N_anchor[i+1], :] = perm_cluster
                        new_embed = embedding.fit(dist_anchor, init=init_embed) 
                        temp_mds = new_embed.embedding_
                        hull = ConvexHull( temp_mds[N_anchor[i]:N_anchor[i+1], :], qhull_options=precision )
                        temp_vertices = hull.vertices
#                       Check if the convex hull is a triangle now
                        if len(temp_vertices) == 3:
                            go_ahead = True
                            temp_pathologies = 0.5
                        elif len(temp_vertices) == 4:
                            go_ahead = checkpermutation( ind_anchor[i][temp_vertices], 
                                                         ind_anchor[i], verbose=False )        
                            temp_pathologies = 1
                        else: 
#                           Improve this error message                                                          <-- check this
                            raise Warning("This is a pathological choice of anchor points, check cluster ", i) 
                    if n_perm == max_perm:
#                       We reached the maximum number of permutations
#                       Improve this choice (maybe triangle with maximum area in MDS local?)                    <-- comment
#                       BE CAREFUL with the MDS (do I need to remove the 4th vertex?)                           <-- comment
                        final_vertices.append(vertices[:3]) 
#                       Improve this print                                                                      <-- comment
                        if self.verbose:
                            print("\rSelf-intersecting quadrilateral on cluster %i, a linear transformation \
                                     will be used instead with anchor points " % i, final_vertices[i])
                    else:
#                       The current cluster is now non-pathological
#                       Check the effects of the new MDS on the convexity of the previous clusters
                        new_vertices = []
                        for j in range(0, i):
                            if len(ind_anchor[j]) <= 3:
                                temp_pathologies += 1
                                new_vertices.append(final_vertices[j])
                            else:
                                h = ConvexHull( temp_mds[N_anchor[j]:N_anchor[j+1], :], qhull_options=precision )
                                if len(h.vertices) == 3:
                                    temp_pathologies += 0.5
                                    new_vertices.append(h.vertices)
                                else:
                                    if checkpermutation( ind_anchor[j][h.vertices], ind_anchor[j], 
                                                         verbose=False ):
                                        temp_pathologies += 1
                                        new_vertices.append(h.vertices)
                                    else:
                                        new_vertices.append(h.vertices[:3])
#                                       Improve this print                                                       <-- comment
                                        if self.verbose:
                                            print("\rSelf-intersecting quadrilateral on cluster %i, a linear \
                                                     transf. will be used instead with anchor points " % j, 
                                                   ind_anchor[j][h.vertices[:3]])
                        if temp_pathologies > no_pathologies:
                            no_pathologies = temp_pathologies
                            mds_anchor = temp_mds
                            final_vertices = new_vertices + [temp_vertices]
                        else:
#                           We keep the previous MDS
#                           Improve this choice (maybe triangle with maximum area in MDS local?)                 <-- comment
#                           BE CAREFUL with the MDS (do I need to remove the 4th vertex?)                        <-- comment
                            final_vertices.append(vertices[:3])             
            else:
#               We have a problem here (the code is assigning a wrong number of anchor points or its 
#               global MDS is a perfect line)                                                               
#               Improve this error message                                                                       <-- check this
                raise Warning("This is a highly pathological choice of anchor points, check cluster ", i) 
        if self.verbose:
            sys.stdout.write( '\rChecking convexity:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")

        self.final_n_anchor = N_anchor 
        self.order_anchor = final_vertices
        self.mds_anchor = mds_anchor


#   This method transforms each cluster from one 2D space (previous) to another (new)
#   The choice of transformation (linear or homography) depends on the anchor points given for each cluster
    def transform_2d(self, clusters, prev_clusters, ind_anchor, mds_clusters):
        N_anchor = self.final_n_anchor
        self.transformation = []
        for i in range(0, len(clusters)):
            if self.verbose:
                sys.stdout.write( '\rPerforming transformations:%6.1f%%' % (float(i)*100./float(len(clusters))) )
                sys.stdout.flush()
#           CASE 1: cluster with 1 anchor point (translation)
            if len(self.order_anchor[i]) == 1:
                self.sparse_coordinates[ind_anchor[i],:] = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :]
                self.transformation.append( [0] )
                continue
            indexes = ind_anchor[i][self.order_anchor[i]]
            X_prev = mds_clusters[indexes,:]
            X_new = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :][self.order_anchor[i]]
            diff_X_prev = X_prev - X_prev[1,:]
            diff_X_new = X_new - X_new[1,:]
#           CASE 2: cluster with 2 or 3 anchor points (linear transformation)
            if len(self.order_anchor[i]) in [2,3]:
                T = np.linalg.lstsq(diff_X_prev, diff_X_new, rcond=None )[0]
                self.transformation.append(T)
                if len(prev_clusters[i]) in [2,3]:
#                   Small clusters
                    self.sparse_coordinates[ind_anchor[i],:] = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :]
                else:
#                   Transform and translate each cluster to the origin of its transf. matrix T
                    product = np.dot(mds_clusters[prev_clusters[i], :] - X_prev[1,:], T)
                    self.sparse_coordinates[prev_clusters[i], :] = product + X_new[1,:]
#           CASE 3: cluster with 4 non-pathological anchor points (homography transf.)
            elif len(self.order_anchor[i]) == 4:
                axis = np.array([[1,0],[0,0],[0,1]])     
                T_prev = np.linalg.lstsq(diff_X_prev[:3,:], axis, rcond=None)[0]
                T_new = np.linalg.lstsq(diff_X_new[:3,:], axis, rcond=None)[0]
                T_new_inv = np.linalg.lstsq(axis, diff_X_new[:3,:], rcond=None)[0]
                a, b = np.dot(diff_X_prev[-1,:], T_prev)
                c, d = np.dot(diff_X_new[-1,:], T_new)
                s = a + b - 1
                t = c + d - 1
#               Decide if we should keep these warnings                                                         <-- comment 
                try:
                    assert (s > 0) & (t > 0)    
                except:
                    raise Warning('The anchor points of cluster %i form a non-convex quadrilateral' % i)
                F = np.array([[b*c*s,     0, b*(c*s - a*t)],
                              [    0, a*d*s, a*(d*s - b*t)],
                              [    0,     0,         a*b*t]])
                self.transformation.append( [X_prev[1,:], T_prev, F, T_new_inv, X_new[1,:]] )
                if len(prev_clusters[i]) == 4:
#                   Small clusters
                    self.sparse_coordinates[ind_anchor[i],:] = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :]
                else:
                    transf_prev = np.dot(mds_clusters[prev_clusters[i]]- X_prev[1,:], T_prev)
                    transf_prev_homog = np.concatenate((transf_prev, np.ones((len(transf_prev),1))), axis=1)
                    perspective_homog =  np.dot(transf_prev_homog, F)
                    perspective = perspective_homog/perspective_homog[:,-1][:,None] 
                    self.sparse_coordinates[prev_clusters[i]] = np.dot(perspective[:,:2], T_new_inv) + X_new[1,:]
            else:
                raise Warning("There must be something wrong, check the list of anchor points: ", ind_anchor)

        if self.verbose:
            sys.stdout.write( '\rPerforming transformations:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")            


#   This is a user friendly function that returns the clusters and medoids of the sparse set
    def get_sparse_coordinates(self, hierarchy):
        if not self.has_clmds:
            self.cluster_MDS(hierarchy = hierarchy)

        ext_coordinates = np.empty([self.n_env,3])
        ext_coordinates[0:self.n_env, 0:2] = self.sparse_coordinates
        ext_coordinates[0:self.n_env, 2] = self.sparse_cluster_indices

        return ext_coordinates


#   This is a user friendly function that returns the anchor points, the cluster or/and 
#   the transformations in each level of the hierarchy
    def extract_transf_info(self, info=None):
#       Extract specific information ("cluster", "anchor", "transf")
        if info: 
            n_levels = len(self.hierarchy)
            I = {}
            for level in range(1, n_levels):
                I[level] = {j: [] for j in range(0, self.hierarchy[level])}
                for i in range(0, self.hierarchy[0]):                        
                    cl = self.all_transformations[i][level]["cluster"]
                    if info == "cluster":
                        I[level][cl].append( i )
                    else:
                        temp = self.all_transformations[i][level][info]
                        if np.array_equal(temp, np.eye(2)):
                            I[level][cl].append(temp)
                            continue
                        already_there = False
                        for k in I[level][cl]:
                            if type(k) == type(temp) and len(k) == len(temp):
                                if isinstance(temp, np.ndarray) and np.allclose(temp, k):
                                    already_there = True
                                    break
                                elif isinstance(temp, list) and next( np.allclose(temp[j], k[j]) for j in range(0, 3) ):
                                    already_there = True  
                                    break   
                        if not already_there:
                            I[level][cl].append(temp)
            return I
#       Extract ALL the information in different arrays
        H = self.extract_transf_info(info="cluster") 
        A = self.extract_transf_info(info="anchor") 
        T = self.extract_transf_info(info="transf") 
        return H, A, T


#   This method gives a "cheap" estimation of the MDS coordinates of the points not included in the sparse set
    def compute_estim_coordinates(self, hierarchy, precision=1e-8):
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
                if prod <= 1. - precision:
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
        for i in range(0, hierarchy[0]):
            if self.verbose:
                print("")
#           distance matrix per cluster
            C = np.where(cluster_indices == i)[0] 
            C_sparse = self.sparse_clusters[i]
            dist_cluster = np.empty([len(C), len(C_sparse)])
            for j in range(0, len(C)):
                if self.verbose:
                    sys.stdout.write('\rEmbedding all data (cluster %i):%6.1f%%' % (i, float(j)*100./float(len(C))) )
                    sys.stdout.flush()
                if C[j] in sparse_list:
                    ind_sparse = np.where(sparse_list == C[j])[0]
                    assert ind_sparse in C_sparse
                    dist_cluster[j,:] = self.dist_matrix[ind_sparse, C_sparse]  
                    continue
                ind = np.where(non_sparse_list == C[j])[0]
                for k in range(0, len(C_sparse)):
                    ind_sparse = C_sparse[k]
                    prod = np.dot(self.descriptor[ind], sparse_descriptor[ind_sparse])**self.zeta
                    if prod < 1.:
                        dist_cluster[j][k] = np.sqrt(1. - prod)
                    else:
                        dist_cluster[j][k] = 0.          

#           Transformation matrix T from distance space to 2D ( X·T = Y)
            dist_sparse = self.dist_matrix[np.ix_(C_sparse, C_sparse)]
            local_mds_sparse = self.local_sparse_coordinates[C_sparse,:]
            T = np.linalg.lstsq(dist_sparse, local_mds_sparse, rcond=None)[0]
            self.all_transformations[i]["dist"] = T
            local_coordinates[C] = np.dot(dist_cluster, T)

#           Transform the coordinates from local to global space 
            ref = self.all_transformations[i][1]["anchor"]
            if len(ref) > 1:
                ref = ref[1]
            local_mds_ref = self.local_sparse_coordinates[ref,:]
            global_mds_ref = self.sparse_coordinates[ref, :]
            transf_coordinates[C] = local_coordinates[C] - local_mds_ref
            for level in range(1, len(hierarchy)):
                T_2d = self.all_transformations[i][level]["transf"]
                if isinstance(T_2d, np.ndarray):
#                   Lineal transformation
                    transf_coordinates[C] = np.dot(transf_coordinates[C], T_2d)         
                elif T_2d == [0]:
#                   Sparse cluster with 1 point (translation)
                    continue
                else:
#                   Homography
                    ref_prev, T_prev, F, T_new_inv, ref_new = T_2d
                    if level > 1:
#                       We need to keep track of the reference anchor point in each step
                        transf_coordinates[C] = transf_coordinates[C] - ref_prev
                    X_prev = np.dot(transf_coordinates[C], T_prev)
                    X_prev_homog = np.concatenate((X_prev, np.ones((len(X_prev),1))), axis=1)
                    X_new_homog =  np.dot(X_prev_homog, F)
                    X_new = X_new_homog/X_new_homog[:,-1][:,None]
                    transf_coordinates[C] = np.dot(X_new[:,:2], T_new_inv)
                    if hierarchy[level] > 1:
#                       We need to keep track of the reference anchor point in each step
                        transf_coordinates[C] = transf_coordinates[C] + ref_new
            transf_coordinates[C] = transf_coordinates[C] + global_mds_ref
            if self.verbose:
                sys.stdout.write('\rEmbedding all data (cluster %i):%6.1f%%' % (i, 100.) )
                sys.stdout.flush()
                print("")
                                           
        self.sparse_list = list(sparse_list) 
        self.descriptor = sparse_descriptor
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
#******************************************* Suporting functions ********************************************

# This method chooses the kmedoids clustering with minimum intra-cluster incoherence (relative or total)
def optim_kmedoids(D, n_clusters, incoherence="rel", n_iter=100,  tmax=100, 
                   init_Ms="random", n_iso=None, verbose=False):
    if verbose:
        print("")
    for t in range(0, n_iter):
        if verbose:
            sys.stdout.write('\rClustering data:%6.1f%%' % (float(t)*100./float(n_iter)) )
            sys.stdout.flush()
        temp_M, temp_C = kmedoids.kMedoids( D, n_clusters, tmax=tmax, 
                                            init_Ms=init_Ms, n_iso=n_iso )
#       Relative intra-cluster incoherence
        temp_I = 0.
        for i in range(0, n_clusters):
            if incoherence == "rel":
                temp_I += np.sum(D[temp_M[i]][temp_C[i]])/len(temp_C[i])
            elif incoherence == "tot":
                temp_I += np.sum(D[temp_M[i]][temp_C[i]])
            else:
                raise Exception("Wrong incoherence option, choose between: rel, tot")
#       Minimize this value
        if (t == 0) or (temp_I <= I):
            M, C, I = temp_M, temp_C, temp_I
    if verbose:
        sys.stdout.write('\rClustering data:%6.1f%%' % 100. )
        sys.stdout.flush()
        print("")

    return M, C


# Given a dataset, this method choose the N points corresponding to the N vertices of the polygon that fulfils  
# the selected criterion (area, number of points)
def anchor_points(N, points, method=None, n_random=None, criterion="area", precision=1.e-8):
    """
    3 available methods: 
        None (default) = use all possible combinations (without repetition) of the points given 
        "optimized" = consider all the combinations of the 30 furthest points in the dataset (or the 70% for small datasets)
        "random" = N random-chosen sequences, where n_random=N (less accurate)

    2 possible criteria:
        "area" (default) = choose the polygon with the largest area 
        "points" = choose the polygon including more data points
    """
#   Check if the number of samples given is enough to build at least 2 N-gons
    if len(points) <= N:
        return points
    s_opt = 0
    anchor_p = points[:N]
    n_comb = comb(len(points), N, exact=True)
    indexes = np.arange(0, len(points),1)
#   Generate vertices and their possible combinations
    precision_opt = 'E'+str(precision)
    h = ConvexHull(points, qhull_options=precision_opt)
    external_ind = h.vertices
    if len(external_ind) <= N:
        return points[external_ind]
    if method == "optimized":
        h = ConvexHull(points, qhull_options=precision_opt)
        external_ind = h.vertices
        while len(external_ind) < 0.7*len(points) and len(external_ind) < 30:
            temp = points[ ~np.isin(indexes, external_ind), :]
            h = ConvexHull(temp, qhull_options=precision_opt)
            temp_vert = [np.where(pt == points)[0][0] for pt in temp[h.vertices,:]]
            external_ind = np.concatenate((external_ind, temp_vert))
        vertices = itertools.combinations(np.sort(external_ind), N)
    elif method == "random":
        n_random = int(n_random)
        vertices = itertools.combinations(indexes, N)
        if n_random < n_comb:
            rand_mask = np.zeros(n_comb, dtype=int)
            rand_vertices = random.sample(range(0, n_comb), n_random)
            rand_mask[rand_vertices] = 1
            vertices = list(itertools.compress(vertices, rand_mask))
        elif not n_random:
            raise Exception("You need to provide a number of random combinations of vertices (n_random)")
    else:
        vertices = itertools.combinations(indexes, N)
#   Obtain the best polygon considering the chosen criterion
    for vert in vertices:
        temp_anchor = points[vert,:]
        if criterion == "area":
            try:
                h = ConvexHull(temp_anchor, qhull_options=precision_opt)
            except:
                continue
            s = h.volume
            temp_anchor = temp_anchor[h.vertices] 
        elif criterion == "points":
            try:
                h = ConvexHull(temp_anchor, qhull_options=precision_opt)
            except:
                continue
            temp_ind = indexes[ ~np.isin(indexes, vert) ]
            other_points = points[temp_ind,:]
            s = points_in_polygon(N, temp_anchor, other_points, qhull_opt=precision_opt)
        else:
            raise Exception("Choose a criterion included in the options: area, points")
        if s > s_opt:
            s_opt = s
            anchor_p = temp_anchor

    return anchor_p


# Computation of the number of points of a given set lying within a polygon with N vertices
def points_in_polygon(N, vertices, other_points, qhull_opt='QbB'):
    s=0
    if N != len(vertices):
        print("You need to provide %i vertices exactly" % N)
    for point in other_points:
        temp = np.concatenate((vertices, point[None,:]), axis=0)
        h = ConvexHull(temp, qhull_options=qhull_opt)
        if len(h.vertices) == N:
            if set(range(0,N)) <= set(h.vertices):  
                s += 1
    return s
    
    
# Find if a set of points is a cyclic permutation or a reflection (or both) of another set
# Should we keep the verbose?                                                                                   <-- comment
def checkpermutation(x, y, verbose=False):
    if len(x) != len(y):
        verdict = False
        if verbose:
            print("The arrays have different length, x can't be a permutation of y")
            print("x: ", x, ", y: ", y)
    else:
        cyclic_group = np.concatenate((y,y), axis=0)
        inv_cyclic_group = cyclic_group[::-1]
        i = np.where(cyclic_group == x[0])[0][0]
        i_inv = np.where(inv_cyclic_group == x[0])[0][0]
        if (cyclic_group[i:i+len(x)] == x).all():
            verdict = True
            if verbose:
                print(x, " is a cyclic permutation of ", y)
        elif (inv_cyclic_group[i_inv:i_inv+len(x)] == x).all():
            verdict = True
            if verbose:
                print(x, " is an inverted cyclic permutation of ", y)
        else:
            verdict = False
    return verdict

#************************************************************************************************************
