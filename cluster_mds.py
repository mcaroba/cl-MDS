import numpy as np
#import quippy
#from quippy import Atoms, descriptors
from sklearn import manifold
import kmedoids
import random


def new_MDS(dist_matrix, n_medoids, t_max=100, init_medoids="random", n_iso_med="None", 
            n_init_mds_cluster=10, max_iter_cluster=100, n_jobs_cluster=1, verbose_cluster=0,
            n_anchorpts=3, n_init_mds_anchorpts=500, max_iter_anchorpts=100, 
            n_jobs_anchorpts=1, verbose_anchorpts=0):
    """
    This method compute the multidimensional scaling (MDS) dimensionality 
    reduction for a given metric matrix preserving both the local and global
    structure of the dataset. Local estructure is given by a clustering
    process.
    """
    ## Clusters
    N = 10**2
    I_rel = 10**4
    for n in range(N):
        M, C = kmedoids.kMedoids( dist_matrix, n_medoids, init_Ms=init_medoids, n_iso=n_iso_med )
        # Obtain relative intercluster (in)coherence
        temp_I = 0 
        for i in range(n_medoids):
            temp_I += np.sum(dist_matrix[M[i]][C[i]])/len(C[i])
        # Minimize this value
        if temp_I <= I_rel:
            I_rel = temp_I
            ind_medoids, ind_clusters = M, C

    # Compute the distance matrix per cluster
    dist_clusters = [[dist_matrix[ind_clusters[i]][j][ind_clusters[i]] 
                      for j in range(len(ind_clusters[i]))] for i in range(n_medoids)]
    # MDS calculation minimizing the stress
    embedding = manifold.MDS( n_components = 2, dissimilarity = "precomputed",
                              n_init = n_init_mds_cluster, max_iter = max_iter_cluster, 
                              n_jobs = n_jobs_cluster, verbose = verbose_cluster )
    mds_clusters = [embedding.fit_transform(dist_clusters[i]) for i in range(n_medoids)]

    ## Anchor points
    n_rand = [10*len(ind_clusters[i]) if len(ind_clusters[i])-1 > n_anchorpts else 1 
              for i in range(0,n_medoids)] 
             # Choose a different number of iterations depending on the length of the cluster
    medoids_in_clusters = []
    N_anchor = []
    total_anchor_points = []
    ind_anchor_global = []
    for i,j in zip(range(n_medoids),n_rand):
        # medoids coordinates in the MDS per cluster
        ind_medoid_cl = np.where(ind_medoids[i] == ind_clusters[i])[0][0]
        medoids_in_clusters.append(mds_clusters[i][ind_medoid_cl])
        # calculate the anchor points for each cluster
        anchor_cluster = anchor_points(n_anchorpts, np.delete(mds_clusters[i],ind_medoid_cl,axis=0), j)
        N_anchor.append(len(anchor_cluster))
        total_anchor_points.append(anchor_cluster)
        # indexes in both the local- (clusters) and global-space matrices
        for k in range(N_anchor[i]):
            ind_anchor_local = np.where(anchor_cluster[k,:] == mds_clusters[i])[0][0]
            ind_anchor_global.append(ind_clusters[i][ind_anchor_local])
    ind_anchor_global = np.append(ind_anchor_global, ind_medoids,axis=0) # Add the medoids
    # metric matrix for the anchor points
    dist_anchor = dist_matrix[np.ix_(ind_anchor_global,ind_anchor_global)] 
    # MDS calculation minimizing the stress
    embedding = manifold.MDS( n_components = 2, dissimilarity = "precomputed", 
                              n_init = n_init_mds_anchorpts, max_iter = max_iter_anchorpts, 
                              n_jobs = n_jobs_anchorpts, verbose = verbose_anchorpts )
    mds_anchor = embedding.fit_transform(dist_anchor)

    ## Transformation (from clusters space to MDS-anchor-points space)
    mds_clusters_transf = []
    mds_anchor_transf = []   
    m = 0
    for i in range(n_medoids):
        diff_X = total_anchor_points[i] - medoids_in_clusters[i]
        diff_Y = mds_anchor[m:m+N_anchor[i]] - mds_anchor[i-n_medoids]
        A = np.linalg.lstsq(diff_X, diff_Y, rcond=None)[0]
        # We need to translate each cluster to the origin of its transf. matrix A,
        # which corresponds to its medoid.
        correction_medoid = mds_anchor[i-n_medoids] - np.dot([0,0], A)
        mds_clusters_transf.append(np.dot(mds_clusters[i] - medoids_in_clusters[i], A) 
                                   + correction_medoid)
        mds_anchor_transf.append(np.dot(total_anchor_points[i] - medoids_in_clusters[i], A)
                                   + correction_medoid)
        m = int(np.sum(N_anchor[:i+1]))
            
    return ind_clusters, ind_medoids, embedding, np.array(mds_clusters_transf), np.array(mds_anchor_transf), mds_anchor[-n_medoids:]
  

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


