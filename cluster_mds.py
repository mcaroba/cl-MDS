#!/usr/bin/env python
"""
This script computes an improved multidimensional scaling for a dataset with dimension n, keeping the information related to both global and local structure.   
"""

import numpy as np
import quippy
from quippy import Atoms, descriptors
from sklearn import manifold
import kmedoids



def new_MDS(dist_matrix, n_medoids):
    """
    This method compute the multidimensional scaling for a given distance matrix 
    preserving the internal structure of each cluster
    n_medoids = number of clusters, integer

    """
    embedding = manifold.MDS(n_components=2, dissimilarity = 'precomputed')
    # We obtain the specific matrix distance for each cluster and their MDS
    ind_medoids, ind_clusters = kmedoids.kMedoids(dist_matrix, n_medoids)
    dist_clusters = [[dist_matrix[ind_clusters[i]][j][ind_clusters[i]] for j in range(0,len(ind_clusters[i]))] for i in range(0,n_medoids)]
    mds_clusters = [embedding.fit_transform(dist_clusters[i]) for i in range(0,n_medoids)]
    """
    # MDS of the complete distance matrix
    MDS = []
    stress = []
    for i in range(0,3):
        MDS.append(embedding.fit_transform(dist_matrix,embedding))
        stress.append(embedding.stress_)
    mds_kernel = MDS[np.argmin(stress)]
    """
    ## Anchor points
    n_rand = [10000 for i in range(0,n_medoids)]
    # medoids coord in the MDS per cluster
    medoids_in_cl = [mds_clusters[i][np.where(ind_medoids[i] == ind_clusters[i])[0][0]] for i in range(0,n_medoids)] 
    total_anchor_points = [ np.append(anchor_points(mds_clusters[i], j), [medoids_in_cl[i]], axis=0) for i,j in zip(range(0,n_medoids),n_rand) ]
    ind_anchor_local = [[np.where(total_anchor_points[i][j,:] == mds_clusters[i])[0][0] for j in range(0,4)] for i in range(0,n_medoids)]
    ind_anchor_global = [ind_clusters[i][ind_anchor_local[i]] for i in range(0,n_medoids)]
    dist_anchor = np.reshape( [ np.reshape(dist[ind_anchor_global[i]][:,ind_anchor_global], (4,4*n_medoids,)) for i in range(0,n_medoids) ], (4*n_medoids,4*n_medoids))    

    ## Minimizing the stress
    MDS = []
    stress = []
    for i in range(0,3):
        MDS.append(embedding.fit_transform(dist_points))
        stress.append(embedding.stress_)        
    mds_points = MDS[np.argmin(stress)]
    
    # Translation of each cluster (M_i=M'_i / M, M' in medoids)       
    dist_medoids=[]
    mds_clusters_trans = []
    for i in range(0,n_medoids):
        dist_medoids.append(mds_points[5*i]-mds_clusters[i][ind_points_clusters[i][0]])
        mds_clusters_trans.append(mds_clusters[i] + dist_medoids[i])
    
    # Transforming the MDS clusters into the original space (MDS anchor points)
    mds_points_in_clusters = [] 
    A = []
    mds_clusters_orig = []
    
    for i in range(0,n_medoids):
        temp_ind1 = np.where(ind_clusters[i]==ind_medoids[i])[0][0]
        for j in range(0,5):
            temp_ind2 = np.where(points[i][j]==ind_clusters[i])[0][0]
            mds_points_in_clusters.append(mds_clusters_trans[i][temp_ind2])
        A.append(np.linalg.lstsq( mds_points_in_clusters[5*i:5*(i+1)], mds_points[5*i:5*(i+1)],rcond=None )[0])
        correction_medoids = mds_points[5*i] - np.dot(mds_clusters_trans[i][temp_ind1],A[i])
        mds_clusters_orig.append(np.dot(mds_clusters_trans[i],A[i]) + correction_medoids)
        
    mds_points_in_clusters = np.array(mds_points_in_clusters)
    
    return
  
###################################################################
##################### Suporting functions #########################

def points_in_triang(vertices, other_points):
	"""
	It computes the number of points from a dataset lying 		withing a triangle whose vertices are given.
	"""
	s=0
	for point in other_points:
		b0 = (point[0] - vertices[1][0])*(vertices[0][1] - vertices[1][1]) - (vertices[0][0] - vertices[1][0])*(point[1] - vertices[1][1]) #sign point, v1, v2
		b1 = (point[0] - vertices[2][0])*(vertices[1][1] - vertices[2][1]) - (vertices[1][0] - vertices[2][0])*(point[1] - vertices[2][1]) #sign point, v2, v3
		b2 = (point[0] - vertices[0][0])*(vertices[2][1] - vertices[0][1]) - (vertices[2][0] - vertices[0][0])*(point[1] - vertices[0][1]) #sign point, v1, v3
		if (b0*b1>0) & (b1*b2>0):
			s += 1
	return s



def anchor_points(points, n_rand):
	"""
	Determination of the three points ("anchor points") which 		form the triangle with the highest amount of points from the 		dataset lying within it. 
	It uses random-choosen secuencies, being less accurate 		(depending on the value of n_rand) but faster in general.
	"""
	s_opt = 0
	anchor_p = points[0:3]
	for m in np.arange(n_rand):
		shuffle_points = random.sample(list(points), len(points))
		temp_vertices = shuffle_points[0:3]
		s = points_in_triang(temp_vertices, shuffle_points[3:]) 
		if s > s_opt:
			s_opt = s				
			anchor_p = temp_vertices
	return np.array(anchor_p)
    

####################################################################
############ Functions for MDS aplications purposes ################

def kernel_dist(file, cut_off, nmax, lmax, sigma, zeta):
    """
    Obtaining a distance matrix from the SOAP kernels of the given   	 structure.
    n_max, l_max, zeta = integers (lmax <= nmax)
    cut_off, sigma = floats
    
    (need to be updated for more than one specie)
    """
    aC = quippy.Atoms(file)
    desc = descriptors.Descriptor(" soap cutoff=" + str(cut_off) + " l_max=" + str(lmax) +" n_max=" + str(nmax) + " atom_sigma=" + str(sigma) + " n_Z=1 Z={6} ")
    aC.set_cutoff(desc.cutoff())
    aC.calc_connect()

    q = desc.calc(aC).descriptor
    q_SOAP = [q[i]/np.linalg.norm(q[i]) for i in range(0, aC.n)] # normaliz$
    # Similarity matrix (SOAP kernels)
    k_SOAP = np.array([[np.dot(q_SOAP[i],q_SOAP[j])**zeta if j!=i else 1. for j in range(0,aC.n)] for i in range(0,aC.n)])
    # Distance matrix using SOAP kernel
    return np.sqrt(1-k_SOAP)

