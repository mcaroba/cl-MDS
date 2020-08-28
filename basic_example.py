import numpy as np
import cluster_mds as cmds

# Basic examples using default settings

# INPUT 1: Atoms file  
# Include the computation of descriptors and their dissimilarity matrix
data_1 = cmds.cMDS(atoms='basic_example.xyz', descriptor="quippy_soap") 

# Clustering and embedding
data_1.cluster_MDS([3,1], init_medoids='isolated', n_iso_med=3) 
XY_1 = data_1.sparse_coordinates
C_1 = data_1.sparse_cluster_indices
Z = data_1.species_list[data_1.sparse_list]

# INPUT 2: Dissimilarity matrix
D = data_1.dist_matrix
data_2 = cmds.cMDS(dist_matrix=D)

# Clustering and embedding
data_2.cluster_MDS([3,1], init_medoids='isolated', n_iso_med=3) 
XY_2 = data_2.sparse_coordinates
C_2 = data_2.sparse_cluster_indices


# RESULTS
print('Points | Clusters 1  Embedded coord. 1 | Clusters 2  Embedded coord. 2  ')
for i in range(0, len(D)):
    print( '   %s         %i       [%s]       %i       [%s]' % 
           (Z[i], C_1[i], ' '.join('% .4f' % j for j in XY_1[i,:]), C_2[i], 
           ' '.join('% .4f' % j for j in XY_2[i,:])) )

# Extend the Atoms file with the cMDS coordinates and cluster indices
data_1.write_xyz(filename='basic_example_ext.xyz')

# Export carved medoid environments and include them in a plot (using gnuplot)
data_1.medoids_to_xyz(dir='basic_example', carve_radius=1.9, render=True, gnuplot=True)

