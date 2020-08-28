# cMDS
Cluster-based multidimensional scaling embedding tool for data visualization

### Basic Example  
```python
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
```

Output:
```
Points | Clusters 1  Embedded coord. 1 | Clusters 2  Embedded coord. 2
   C         1       [ 0.4821 -0.1902]       1       [-0.1993 -0.5469]
   C         1       [ 0.4348  0.1461]       1       [ 0.0294 -0.4909]
   C         1       [ 0.9155 -0.2807]       1       [-0.6042 -0.1560]
   C         1       [ 0.5778 -0.2330]       1       [-0.3163 -0.4654]
   C         1       [ 0.2482  0.4091]       1       [ 0.3081 -0.5489]
   C         1       [ 0.5500  0.1310]       1       [-0.0941 -0.3808]
   C         1       [ 0.3552  0.1874]       1       [ 0.1234 -0.5511]
   C         1       [ 0.3148  0.3925]       1       [ 0.2361 -0.4958]
   C         1       [ 0.5499  0.1311]       1       [-0.0941 -0.3808]
   C         1       [ 0.5421 -0.2115]       1       [-0.2768 -0.4913]
   O         2       [-0.0200 -0.6565]       2       [-0.4184  0.4731]
   O         2       [-0.6283 -0.0975]       2       [ 0.3630  0.5091]
   O         2       [-0.3798 -0.4738]       2       [-0.0587  0.5831]
   H         0       [-0.5044  0.5121]       0       [ 0.6952  0.0211]
```

Detailed example:
```python
import numpy as np
import cluster_mds as cmds
import matplotlib.pyplot as plt
import time

# More detailed example using an Atoms file

## 1. Complete dataset
data = cmds.cMDS(atoms="example_structures.xyz", descriptor="quippy_soap", cutoff=3.0,
                 descriptor_string="soap n_max=8 l_max=8 cutoff=3.0 atom_sigma=0.5 Z={1 6 8} \
                 species_Z={1 6 8} n_Z=3 n_species=3") 

# Compute the clusters and embedded coordinates
data.cluster_MDS([7, 1], init_medoids='isolated', n_iso_med=4, n_init_mds_anchorpts=4000)
coord_complete = data.sparse_coordinates
C_complete = data.sparse_cluster_indices


## 2. Additional features

# 2.1 Taking advantage of sparsification
data_sparse = cmds.cMDS(atoms="example_structures.xyz", descriptor="quippy_soap", cutoff=3.0,
                 descriptor_string="soap n_max=8 l_max=8 cutoff=3.0 atom_sigma=0.5 Z={1 6 8} \
                 species_Z={1 6 8} n_Z=3 n_species=3", sparsify="random", n_sparse=500) 

# Obtain the dimensionality reduction of the sparse set
data_sparse.cluster_MDS([7, 1], init_medoids='isolated', n_iso_med=4)
coord_sparse = data_sparse.sparse_coordinates

# Estimate the embedded coordinates of the remaining data
data_sparse.compute_estim_coordinates([7, 1])
coord_estim = data_sparse.all_coordinates
C_estim = data_sparse.all_cluster_indices

# 2.2 Hierarchical embedding
# Useful when different levels of local features are known
# In this case, there are two explicit structural levels:
#   - 7 types of structures  (check data.config_type_list)
#   - 3 atomic species       (check data.species_list)
data_sparse.cluster_MDS([7, 3, 1], init_medoids='isolated', n_iso_med=4)
coord_sparse_h = data_sparse.sparse_coordinates

data_sparse.compute_estim_coordinates([7, 3, 1])
coord_hierarchy = data_sparse.all_coordinates
C_hierarchy = data_sparse.all_cluster_indices


## 3. Average kernel
data_struct = cmds.cMDS(atoms="example_structures.xyz", descriptor="quippy_soap", cutoff=3.0,
                        descriptor_string="soap n_max=8 l_max=8 cutoff=3.0 atom_sigma=0.5 \
                        Z={1 6 8} species_Z={1 6 8} n_Z=3 n_species=3 average=T",
                        average_kernel=True)

# Obtain the embedded coordinates and clusters (trivial in this case) 
data_struct.cluster_MDS([7, 1])
coord_struct = data_struct.sparse_coordinates
C_struct = data_struct.sparse_cluster_indices
```

### Main parameters

### Installation


