import numpy as np
import cluster_mds as clmds

# Basic examples using default settings

# INPUT 1: Atoms file  
# (cl-MDS computes the descriptors and their dissimilarity matrix)
data_1 = clmds.clMDS(atoms='basic_example.xyz', descriptor="quippy_soap")
# Clustering and embedding
XYC_1 = data_1.get_sparse_coordinates([3,1])
Z = data_1.species_list

# INPUT 2: Dissimilarity matrix
D = data_1.dist_matrix
data_2 = clmds.clMDS(dist_matrix = D)
# Clustering and embedding
XYC_2 = data_2.get_sparse_coordinates([3,1])

# RESULTS
print('Points | Clusters 1  Embedded coord. 1 | Clusters 2  Embedded coord. 2  ')
for i in range(0, len(D)):
    print( '   %s         %i       [%s]       %i       [%s]' % 
           (Z[i], XYC_1[i,2], ' '.join('% .4f' % j for j in XYC_1[i,:2]), XYC_2[i,2], 
           ' '.join('% .4f' % j for j in XYC_2[i,:2])) )

# Extend the Atoms file with the cl-MDS coordinates and cluster indices
data_1.write_xyz(filename='basic_example_ext.xyz')

# Export carved medoid environments and include them in a plot (using gnuplot)
data_1.medoids_to_xyz(dir='basic_example', carve_radius=1.9, render=True, gnuplot=True)

