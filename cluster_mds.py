import numpy as np
#import quippy
#from quippy import Atoms, descriptors
from sklearn import manifold
import kmedoids
import random


def new_MDS(dist_matrix, hierarchy, iter_med=100, t_max=100, init_medoids="random", n_iso_med="None", 
            n_init_mds_cluster=10, max_iter_cluster=100, n_jobs_cluster=1, verbose_cluster=0,
            n_anchorpts=3, n_init_mds_anchorpts=500, max_iter_anchorpts=100, 
            n_jobs_anchorpts=1, verbose_anchorpts=0):
    """
    This method compute the multidimensional scaling (MDS) dimensionality 
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

    TO DO: Include an option to allow both dist_matrix and .xyz file, changing
           return accordingly.
    """
    ### Finest clustering (initial hierarchy level) ###
    n_clusters = hierarchy[0]
    I_rel = 10**4
    for t in range(iter_med):
        M, C = kmedoids.kMedoids( dist_matrix, n_clusters, tmax=t_max,
                                  init_Ms=init_medoids, n_iso=n_iso_med )
        # Obtain relative intercluster (in)coherence
        temp_I = 0 
        for i in range(n_clusters):
            temp_I += np.sum(dist_matrix[M[i]][C[i]])/len(C[i])
        # Minimize this value
        if temp_I <= I_rel:
            I_rel = temp_I
            ind_medoids, ind_clusters = M, C

    # Compute the distance matrix per cluster
    dist_clusters = [dist_matrix[np.ix_(ind_clusters[i], ind_clusters[i])] 
                     for i in range(n_clusters)]
    # MDS calculation minimizing the stress
    embedding = manifold.MDS( n_components = 2, dissimilarity = "precomputed",
                              n_init = n_init_mds_cluster, max_iter = max_iter_cluster, 
                              n_jobs = n_jobs_cluster, verbose = verbose_cluster )
    mds_clusters = np.zeros((len(dist_matrix),2))
    for i in range(n_clusters):
        if len(ind_clusters[i]) > 1:
            mds_clusters[ind_clusters[i]] = embedding.fit_transform(dist_clusters[i])
        else:             
            mds_clusters[ind_clusters[i]] = np.zeros((1,2)) # avoid sklearn RuntimeWarning

  
    ### Hierarchy levels ###
    n_levels = len(hierarchy)
    M_prev = ind_medoids
    C_prev = ind_clusters
    embedding_h = manifold.MDS( n_components = 2, dissimilarity = "precomputed", 
                                n_init = n_init_mds_anchorpts, max_iter = max_iter_anchorpts, 
                                n_jobs = n_jobs_anchorpts, verbose = verbose_anchorpts )
    C_int = {} 
    for level in range(1, n_levels):
        ## Check the data reorganization needed for this new hierarchy level
        if hierarchy[level] > 1:
            # Asign the clusters of previous level to the current ones 
            m, c = kmedoids.kMedoids(dist_matrix[np.ix_(M_prev, M_prev)], hierarchy[level])
            # Obtain a dictionary with all the indexes of each new cluster
            C_new = { newcl: np.concatenate( [C_prev[i] for i in c[newcl]] ) 
                      for newcl in range(hierarchy[level]) }
            C_int[level] = C_new
        elif hierarchy[level] == 1:
            # No clustering (consider all data points)
            c = { 0: np.arange(hierarchy[level-1]) }
            C_new = { 0: np.arange(len(dist_matrix)) }           
        else:
            raise Exception("error in the given hierarchy, wrong entries")
       
        ## Obtention of anchor points
        # Consider anchor points AND medoids separately
        mds_M_prev = mds_clusters[M_prev] 
        mds_A = []
        ind_A = []    
        for i in range(hierarchy[level-1]):
            # Exclude the medoid as a possible anchor point
            C_prev_nomed = np.setdiff1d(C_prev[i], M_prev[i])
            # Choose the number of random iterations needed depending on cluster length
            n_rand = 10*len(C_prev[i]) if len(C_prev[i])-1 > n_anchorpts else 1
                     # TO DO: improve the choosing method

            # MDS of anchor points in previous level
            mds_A.append(anchor_points(n_anchorpts, mds_clusters[C_prev_nomed], n_rand))
            # indexes of anchor points in previous level
            ind_A.append( [np.where(mds_clusters == mds_A[-1][j])[0][0] 
                           for j in range(len(mds_A[-1]))] )

        ## MDS of anchor points and transformation (from previous level to the new one)
        A = {}
        mds_clusters_transf = np.zeros((len(dist_matrix),2))
        for newcl in range(hierarchy[level]):
            temp_anchors = np.concatenate( [ind_A[i] for i in c[newcl]] ).astype('int32')
            A[newcl] = np.concatenate( (temp_anchors, M_prev[c[newcl]]) ) 
            # metric matrix for the anchor points + medoids
            dist_anchor = dist_matrix[np.ix_(A[newcl], A[newcl])] 
            mds_anchor = embedding_h.fit_transform(dist_anchor)

            # transformation matrix ( X·T = X')
            real_n_anchor = np.zeros((len(c[newcl])+1,), dtype=int)
            for n, i in enumerate(c[newcl]):
                real_n_anchor[n+1] = real_n_anchor[n] + len(ind_A[i])
                diff_X_prev = mds_A[i] - mds_M_prev[i]
                diff_X_new = mds_anchor[real_n_anchor[n]:real_n_anchor[n+1]] \
                             - mds_anchor[n-len(c[newcl])]
                T = np.linalg.lstsq(diff_X_prev, diff_X_new, rcond=None)[0]
                # Translate each cluster to the origin of its transf. matrix T (i.e. its medoid)
                correction_med = mds_anchor[n-len(c[newcl])] - np.dot([0,0], T)
                # Transform their coordinates
                mds_clusters_transf[C_prev[i]] = np.dot(mds_clusters[C_prev[i]]
                                                        - mds_M_prev[i], T) + correction_med

        if hierarchy[level] > 1:
            # Reasign the label "previous" to the new results
            M_prev = M_prev[m]
            C_prev = C_new
            mds_clusters = mds_clusters_transf

            
    return mds_clusters_transf, ind_clusters, C_int, ind_medoids
  

############################ Suporting functions #############################

def anchor_points(N, points, n_random):
    """
    Given a dataset, this method chooses the N points ("anchor points")
    corresponding to the N vertices of the polygon containing the highest 
    number of data points. It uses random-choosen secuencies, being less 
    accurate (depending on the number of iterations) but faster in general.

    ONLY VALID WITH N=3,4 !!!!!!!!!
    """
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


###################### cMDS applied to Atomic Physics #######################

#! CHECK THIS (update of the quippy module + new SOAP)
def kernel_dist(file, cut_off, nmax, lmax, sigma, n_species, species_Z, zeta):
    """
    Obtaining a distance matrix from the SOAP kernels of the given structure.
    n_max, l_max, n_species, zeta = integers (lmax <= nmax)
    cut_off, sigma = floats
    species_Z = string ("{Z1 Z2 ... Zn_species}")
    """  
    aC = quippy.Atoms(file)
    desc = descriptors.Descriptor(" soap cutoff={} l_max={} n_max={} atom_sigma={} n_Z={} Z={}".format(cut_off, 
                                  lmax,nmax,sigma,n_species,species_Z))
    aC.set_cutoff(desc.cutoff())
    aC.calc_connect()
    q_SOAP = desc.calc(aC).descriptor # Normalized SOAP descriptor
    # Similarity matrix (SOAP kernels)
    k_SOAP = np.array([[np.dot(q_SOAP[i],q_SOAP[j])**zeta if j!=i else 1. 
                        for j in range(0,aC.n)] for i in range(0,aC.n)])

    return np.sqrt(1-k_SOAP) # Distance matrix


