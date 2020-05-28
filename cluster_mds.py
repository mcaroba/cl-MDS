#************************************************************************************************************
# This is the cluster-based MultiDimensional Scaling code for dimensionality reduction data analysis.       #
#                                                                                                           #
#                                                cMDS                                                       #
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
#                                   https://github.com/mcaroba/cMDS/                                        #
#                                                                                                           #
# Visit the repository for the latest version of this distribution.                                         #
#                                                                                                           #
#                                                                                                           #
# If you use cMDS for the compilation of academic/scientific/technical work, please cite, as appropriate:   #
#                                                                                                           #
# P. Hernandez-Leon and M.A. Caro, XXX, YYY (2020)                                                          #
#                                                                                                           #
#************************************************************************************************************


# Import dependencies
import numpy as np
#import quippy
#from quippy import Atoms, descriptors
from sklearn import manifold
# We should include kmedoids in the installation
import kmedoids
import random
import sys


#************************************************************************************************************
class cMDS:
    """
    This is the main cMDS class. It can take a distance matrix and/or an ASE compatible
    (e.g., an xyz file) atomic structure. It can also take concatenated xyz files or any trajectory that
    can be imported with ASE.

    If the user chooses an implemented descriptor, cMDS will make an educated guess and assign some sensible
    defaults. If the user wants more control over the choice of hyperparameters, they should pass a
    distance matrix instead.
    """

#   Initialize the class:
    def __init__(self, dist_matrix=None, atoms=None, descriptor=None, descriptor_string=None,
                 sparsify=None, n_sparse=None, verbose=True):
#       This is the list of implemented atomic descriptors (it typically requires external
#       programs)
        implemented_descriptors = ["quippy_soap"]
        sparse_options = ["random"]
        self.verbose = verbose
        self.sparsify = sparsify
        self.n_sparse = n_sparse
        self.is_clustered = False
        self.has_cmds = False
        if sparsify is not None:
            if sparsify not in sparse_options:
                raise Exception("The sparsify option you chose is not available. Choose one of the following: ",
                                sparse_options)
            else:
                if n_sparse is None:
                    raise Exception("If you chose a sparsification option, you need to pass the n_sparse parameter")
                else:
                    self.sparsify = sparsify
                    self.n_sparse = n_sparse

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
                raise Exception("I couldn't find an ASE installation; you need ASE to pass an atoms filename!")

#       Check if the user wants to use a descriptor
        if descriptor is not None:
            if descriptor not in implemented_descriptors:
                raise Exception("The descriptor you chose is not implemented; the options are: ",
                                implemented_descriptors)
            if dist_matrix is not None:
                raise Exception("You can't define a distance matrix and a descriptor; choose one or the other!")
            if atoms is None:
                raise Exception("If you define a descriptor, you must also provide an atoms filename")
            self.descriptor = descriptor
            self.descriptor_string = descriptor_string


#   This method takes care of adding a descriptor to the cMDS class:
    def build_descriptor(self):
        descriptor = self.descriptor
        descriptor_string = self.descriptor_string
        if descriptor == "quippy_soap":
            from ase.data import atomic_numbers
            from quippy.descriptors import Descriptor
            from quippy.convert import ase_to_quip
            self.zeta = 4
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
                cutoff = 3.0
                quippy_string = "soap n_max=8 l_max=8 cutoff=" + str(cutoff) + " atom_sigma=0.5 Z={" + \
                                species_string + "} species_Z={" + species_string + "} n_Z=" + str(n_Z) + \
                                " n_species=" + str(n_Z)
            else:
                quippy_string = self.descriptor_string
#               Cumbersome code to get the cutoff from a string:
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

            d = Descriptor(quippy_string)
            n_env = sum(len(ats) for ats in self.atoms)
            if self.sparsify is not None:
                if self.sparsify == "random":
                    sparse_list = list(range(n_env))
                    np.random.shuffle(sparse_list)
                    sparse_list = sparse_list[0:self.n_sparse]
                    self.sparse_list = sparse_list
            n = 0
            descriptor = []
            if self.verbose:
                print("")
            for ats in self.atoms:
                if self.verbose:
                    sys.stdout.write('\rComputing descriptors:%6.1f%%' % (float(n)*100./float(n_env)) )
                    sys.stdout.flush()
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

            self.n_env = len(descriptor)
            self.descriptor = np.array(descriptor)
            self.has_descriptor = True
        else:
            raise Exception("You must choose among the implemented descriptors: ", implemented_descriptors)


#   This method takes care of building a distance matrix:
    def build_dist_matrix(self):
        if self.descriptor is not None:
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


#   This method clusters the data and produces the embedded coordinates
#   Make sure these are sensible defaults!!!!!!                                                             <-- comment here
    def cluster_MDS(self, hierarchy, iter_med=100, t_max=100, init_medoids="random", n_iso_med="None",
                    n_init_mds_cluster=10, max_iter_cluster=100, n_jobs_cluster=1, verbose_cluster=0,
                    n_anchorpts=3, n_init_mds_anchorpts=500, max_iter_anchorpts=100, n_jobs_anchorpts=1,
                    verbose_anchorpts=0):
        """
        This method carries out multidimensional scaling (MDS) dimensionality
        reduction for a given metric matrix preserving both the local and global
        structure of the dataset. Local estructure is given by a clustering
        process.

        The hierarchy parameter is defined by a list containing levels of
        clustering, [n_clusters, n_level1, n_level2, ... , 1], where n_clusters
        refers to the finest clustering (computed in the data n-dimensional space)
        and 1 refers to the final MDS 2d-space. Depending on the chosen hierarchy,
        different local information of the dataset can be weighted more during
        the computation. The simplest hierarchy system is [n_clusters, 1], where 
        only that local structure is considered.
        """
        if len(hierarchy) == 1:
            hierarchy.append([1])

        if not self.has_dist_matrix:
            self.build_dist_matrix()

#       Finest clustering (initial hierarchy level)
        try:
            n_clusters = hierarchy[0]
        except:
            raise Exception("You need to define a hierarchy of cluster levels by providing the hierarchy \
                            parameter, e.g., hierarchy=[8,3,1] or hierarchy=[8,1]")
        I_rel = 10**4
        n = 0
        if self.verbose:
            print("")
        for t in range(iter_med):
            n += 1
            if self.verbose:
#               This printing is insufficient, make sure all the other tasks within this funtion get printed out <-- comment here
                sys.stdout.write('\rClustering data:%6.1f%%' % (float(n)*100./float(iter_med)) )
                sys.stdout.flush()
            M, C = kmedoids.kMedoids( self.dist_matrix, n_clusters, tmax=t_max,
                                      init_Ms=init_medoids, n_iso=n_iso_med )
#           Obtain relative intercluster (in)coherence
            temp_I = 0
            for i in range(n_clusters):
                temp_I += np.sum(self.dist_matrix[M[i]][C[i]])/len(C[i])
#           Minimize this value
            if temp_I <= I_rel:
                I_rel = temp_I
                ind_medoids, ind_clusters = M, C
        if self.verbose:
            sys.stdout.write('\rClustering data:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")

#       Compute the distance matrix per cluster
        dist_clusters = [self.dist_matrix[np.ix_(ind_clusters[i], ind_clusters[i])]
                         for i in range(n_clusters)]
#       MDS calculation minimizing the stress
        embedding = manifold.MDS( n_components = 2, dissimilarity = "precomputed",
                                  n_init = n_init_mds_cluster, max_iter = max_iter_cluster,
                                  n_jobs = n_jobs_cluster, verbose = verbose_cluster )
        mds_clusters = np.zeros((len(self.dist_matrix),2))
        if self.verbose:
            print("")
        for i in range(n_clusters):
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
#       Hierarchy levels
        n_levels = len(hierarchy)
        M_prev = ind_medoids
        C_prev = ind_clusters
        embedding_h = manifold.MDS( n_components = 2, dissimilarity = "precomputed",
                                    n_init = n_init_mds_anchorpts, max_iter = max_iter_anchorpts,
                                    n_jobs = n_jobs_anchorpts, verbose = verbose_anchorpts )

        C_int = {}
        for level in range(1, n_levels):
#           Check the data reorganization needed for this new hierarchy level
            if hierarchy[level] > 1:
#               Assign the clusters of previous level to the current ones
                m, c = kmedoids.kMedoids(self.dist_matrix[np.ix_(M_prev, M_prev)], hierarchy[level])
#               Obtain a dictionary with all the indexes of each new cluster
                C_new = { newcl: np.concatenate( [C_prev[i] for i in c[newcl]] )
                          for newcl in range(hierarchy[level]) }
                C_int[level] = C_new
            elif hierarchy[level] == 1:
#               No clustering (consider all data points)
                c = { 0: np.arange(hierarchy[level-1]) }
                C_new = { 0: np.arange(len(self.dist_matrix)) }
            else:
#               Can this be more specific?                                                                   <-- comment here
                raise Exception("Error in the given hierarchy, wrong entries")

#           Obtention of anchor points
#           Consider anchor points AND medoids separately
            mds_M_prev = mds_clusters[M_prev]
            mds_A = []
            ind_A = []
            for i in range(hierarchy[level-1]):
#               Exclude the medoid as a possible anchor point
                C_prev_nomed = np.setdiff1d(C_prev[i], M_prev[i])
#               Choose the number of random iterations needed depending on cluster length
                n_rand = 10*len(C_prev[i]) if len(C_prev[i])-1 > n_anchorpts else 1
#               TO DO: improve the choosing method                                                            <-- comment here

#               MDS of anchor points in previous level
                mds_A.append(anchor_points(n_anchorpts, mds_clusters[C_prev_nomed], n_rand))
#               indexes of anchor points in previous level
                ind_A.append( [np.where(mds_clusters == mds_A[-1][j])[0][0]
                               for j in range(len(mds_A[-1]))] )

#           MDS of anchor points and transformation (from previous level to the new one)
            A = {}
            mds_clusters_transf = np.zeros((len(self.dist_matrix),2))
            for newcl in range(hierarchy[level]):
                temp_anchors = np.concatenate( [ind_A[i] for i in c[newcl]] ).astype('int32')
                A[newcl] = np.concatenate( (temp_anchors, M_prev[c[newcl]]) )
#               metric matrix for the anchor points + medoids
                dist_anchor = self.dist_matrix[np.ix_(A[newcl], A[newcl])]
                mds_anchor = embedding_h.fit_transform(dist_anchor)

#               transformation matrix ( X·T = X')
                real_n_anchor = np.zeros((len(c[newcl])+1,), dtype=int)
                for n, i in enumerate(c[newcl]):
                    real_n_anchor[n+1] = real_n_anchor[n] + len(ind_A[i])
                    diff_X_prev = mds_A[i] - mds_M_prev[i]
                    diff_X_new = mds_anchor[real_n_anchor[n]:real_n_anchor[n+1]] \
                                 - mds_anchor[n-len(c[newcl])]
                    T = np.linalg.lstsq(diff_X_prev, diff_X_new, rcond=-1 )[0]
#                   Translate each cluster to the origin of its transf. matrix T (i.e. its medoid)
                    correction_med = mds_anchor[n-len(c[newcl])] - np.dot([0,0], T)
#                   Transform their coordinates
                    mds_clusters_transf[C_prev[i]] = np.dot(mds_clusters[C_prev[i]]
                                                            - mds_M_prev[i], T) + correction_med
            if hierarchy[level] > 1:
#               Reassign the label "previous" to the new results
                M_prev = M_prev[m]
                C_prev = C_new
                mds_clusters = mds_clusters_transf

#       These indices refer to the dist_matrix; we need to make sure that the information required to retrieve  <-- comment here
#       the atomic structures from the original data base are consistent with the sparsification technique used <-- comment here
#       We should also give the option to output the mds coordinates to the xyz file and generate carved xyz    <-- comment here
#       structures around the medoids for plotting                                                              <-- comment here
        self.has_cmds = True
#        return mds_clusters_transf, ind_clusters, C_int, ind_medoids
        self.sparse_coordinates = mds_clusters_transf
        self.sparse_clusters = ind_clusters
        self.sparse_medoids = ind_medoids


#   This is a user friendly function that returns the clusters and medoids of the sparse set
    def get_sparse_coordinates(self, hierarchy):
        if not self.has_cmds:
            self.cluster_MDS(hierarchy = hierarchy)

        ext_coordinates = np.zeros([self.n_env,3])
        ext_coordinates[0:self.n_env, 0:2] = self.sparse_coordinates
        for i in range(0, self.n_env):
             for cluster in range(0, hierarchy[0]):
                 if i in self.sparse_clusters[cluster]:
                     ext_coordinates[i][2] = cluster
                     break

        return ext_coordinates


#************************************************************************************************************





#************************************************************************************************************
############################ Suporting functions #############################
def anchor_points(N, points, n_random):
    """
    Given a dataset, this method chooses the N points ("anchor points")
    corresponding to the N vertices of the polygon containing the highest 
    number of data points. It uses random-chosen sequences, being less 
    accurate (depending on the number of iterations) but faster in general.

    ONLY VALID WITH N=3,4 !!!!!!!!!
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


def points_in_triang(vertices, other_points):
    """
    Computation of the number of points from a given set lying within a
    triangle whose vertices are known.
    """
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


def points_in_quad(vertices, other_points):
    """
    Computation of the number of points from a given set lying within a
    quadrilateral whose vertices are known.

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
