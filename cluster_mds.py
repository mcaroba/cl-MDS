import numpy as np
import quippy
from quippy import Atoms, descriptors
from sklearn import manifold
import kmedoids
import random


def new_MDS(dist_matrix, n_medoids):
    """
    This method compute the multidimensional scaling (MDS) dimensionality 
    reduction for a given metric matrix preserving both the local and global
    structure of the dataset. Local estructure is given by a clustering
    process.
    """
    ## Clusters
    N = 10**4
    I_rel = np.max(dist_matrix)
    for n in range(0,N):
        M, C = kmedoids.kMedoids(dist_matrix, n_medoids, init_Ms='isolated', n_iso=4)
        # Obtain relative intercluster (in)coherence
        temp_I = 0 
        for i in range(0, n_medoids):
            temp_I += np.sum(dist_matrix[M[i]][C[i]])/len(C[i])
        # Minimize this value
        if temp_I < I_rel:
            I_rel = temp_I
            ind_medoids, ind_clusters = M, C
    print(ind_medoids)                       
    # Compute the distance matrix per cluster
    dist_clusters = [[dist_matrix[ind_clusters[i]][j][ind_clusters[i]] 
                      for j in range(0,len(ind_clusters[i]))] for i in range(0,n_medoids)]
    # MDS calculation minimizing the stress
    embedding = manifold.MDS(n_components=2, dissimilarity='precomputed')
    mds_clusters = []
    for i in range(0, n_medoids):
        stress = 1
        for j in range(0,100):
            temp_mds = embedding.fit_transform(dist_clusters[i])
            temp_s = embedding.stress_
            if temp_s < stress:
                stress = temp_s
                final_mds = temp_mds
        mds_clusters.append(final_mds)

    ## Anchor points
    n_rand = [10000 for i in range(0,n_medoids)] # we could choose a different number of iterations
                                                 # depending on the length of the cluster
    medoids_in_clusters = []
    total_anchor_points = []
    n_anchor = []
    ind_anchor_global = []
    for i,j in zip(range(0,n_medoids),n_rand):
        # medoids coordinates in the MDS per cluster
        ind_medoid_cl = np.where(ind_medoids[i] == ind_clusters[i])[0][0]
        medoids_in_clusters.append(mds_clusters[i][ind_medoid_cl])
        # calculate the anchor points for each cluster
        anchor_cluster = anchor_points(np.delete(mds_clusters[i],ind_medoid_cl,axis=0), j)
        n_anchor.append(len(anchor_cluster))
        total_anchor_points.append(anchor_cluster)
        # indexes in both the local- (clusters) and global-space matrices
        for k in range(0,n_anchor[i]):
            ind_anchor_local = np.where(anchor_cluster[k,:] == mds_clusters[i])[0][0]
            ind_anchor_global.append(ind_clusters[i][ind_anchor_local])
    ind_anchor_global = np.append(ind_anchor_global,ind_medoids,axis=0) # Add the medoids
    # metric matrix for the anchor points
    dist_anchor = dist_matrix[np.ix_(ind_anchor_global,ind_anchor_global)] 
    # MDS calculation minimizing the stress
    stress = 1000
    for i in range(0,1000):
        temp_mds = embedding.fit_transform(dist_anchor)
        temp_s = embedding.stress_
        if temp_s < stress:
            stress = temp_s
            mds_anchor = temp_mds

    ## Transformation (from clusters space to MDS-anchor-points space)
    mds_clusters_transf = []   
    n=0
    for i in range(0,n_medoids):
        diff_X = total_anchor_points[i] - medoids_in_clusters[i]
        diff_Y = mds_anchor[n:n+n_anchor[i]] - mds_anchor[i-n_medoids]
        A = np.linalg.lstsq(diff_X, diff_Y, rcond=None)[0]
        # We need to translate each cluster to the origin of its transf. matrix A,
        # which corresponds to its medoid.
        correction_medoid = mds_anchor[i-n_medoids] - np.dot([0,0], A)
        mds_clusters_transf.append(np.dot(mds_clusters[i] - medoids_in_clusters[i], A) 
                                   + correction_medoid)
        n = int(np.sum(n_anchor[:i+1]))
            
    return mds_clusters_transf
  

############################ Suporting functions #############################

def anchor_points(points, n_rand):
    """
    Given a dataset, this method chooses the three points ("anchor points")
    corresponding to the vertices of the triangle containing the highest 
    number of data points. It uses random-choosen secuencies, being less 
    accurate (depending on the number of iterations) but faster in general.
    """
    # Check if the number of samples given is enough to build 2 triangles or more
    if len(points) > 3:
        s_opt = 0
        anchor_p = points[0:3]
        for m in np.arange(n_rand):
            shuffle_points = random.sample(list(points), len(points))
            temp_vertices = shuffle_points[0:3]
            s = points_in_triang(temp_vertices, shuffle_points[3:]) 
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


###################### cMDS applied to Atomic Physics #######################

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
    q = desc.calc(aC).descriptor
    # Normalized SOAP descriptor
    q_SOAP = [q[i]/np.linalg.norm(q[i]) for i in range(0, aC.n)] 
    
    # Similarity matrix (SOAP kernels)
    k_SOAP = np.array([[np.dot(q_SOAP[i],q_SOAP[j])**zeta if j!=i else 1. 
                        for j in range(0,aC.n)] for i in range(0,aC.n)])

    return np.sqrt(1-k_SOAP) # Distance matrix


