# cl-MDS
Cluster-based multidimensional scaling (**cl-MDS**) embedding tool for data visualization

**cl-MDS** is copyright (c) 2018-2022 by Patricia Hernández-León and Miguel A. Caro. It is
distributed under the GNU General Public License version 3. **cl-MDS** is shipped with other
codes it relies on for some of its functionalities, but which are not copyright of the
**cl-MDS** authors. They are shipped for the user's convenience in accordance with their
own licenses/terms of usage. See the LICENSE.md file for detailed information on this
software's license.

## Installation

### Prerrequisites

- Numpy
- Sklearn
- A Fortran compiler (successfully tested with `gfortran`)

### Building the libraries

Clone the **cl-MDS** repository *recursively*:

    git clone --recursive http://github.com/mcaroba/cl-MDS.git

Execute the build script:

    cd cl-MDS/
    ./build_libraries.sh

Add the root directory to your Python path:

    echo "export PYTHONPATH=$(pwd):\$PYTHONPATH" >> ~/.bashrc
    source ~/.bashrc

### Basic Example  
```python
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
```

Output:
```
Points | Clusters 1  Embedded coord. 1 | Clusters 2  Embedded coord. 2
   C         2       [-0.5579 -0.0920]       2       [ 0.4902  0.2944]
   C         2       [-0.4895 -0.3194]       2       [ 0.6035  0.0403]
   C         2       [-0.4451  0.4241]       2       [ 0.0357  0.6153]
   C         2       [-0.5837 -0.0574]       2       [ 0.4817  0.3537]
   C         2       [-0.3134 -0.4948]       2       [ 0.5498 -0.1522]
   C         2       [-0.2987 -0.2134]       2       [ 0.3513  0.0677]
   C         2       [-0.4601 -0.3567]       2       [ 0.5997 -0.0049]
   C         2       [-0.3334 -0.4578]       2       [ 0.5457 -0.1211]
   C         2       [-0.2987 -0.2134]       2       [ 0.3513  0.0677]
   C         2       [-0.5755 -0.0741]       2       [ 0.4857  0.3315]
   O         1       [ 0.6293 -0.0469]       1       [-0.4211 -0.4700]
   O         1       [ 0.1424  0.6296]       1       [-0.5421  0.3613]
   O         1       [ 0.4960  0.3519]       1       [-0.6001 -0.0877]
   H         0       [ 0.3732 -0.5931]       0       [ 0.1448 -0.6882]
```


### Main parameters



