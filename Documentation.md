# What is cl-MDS?

cl-MDS is a cluster-based multidimensional scaling tool for visualizing high-dimensional data in low-dimensional space. (refer to the paper for a detailed description of the embedding technique)

Cluster MDS or cl-MDS is a general dimensionality reduction tool for embedding high-dimensional data in 2-dimensional space, developed for visualization purposes.  
For further details on the embedding technique, please check the original paper: [P. Hernández-León, M. A. Caro, *Phys. Scr.* **99** 066004 (2024)](https://iopscience.iop.org/article/10.1088/1402-4896/ad432e).


The `clmds` package (**not true (yet)**) includes a wide range of features related to: 
- *clustering*: 
- *sparsification*: 
- *dimensionality reduction*: generation of 
- *visualization*: the motivation behind cl-MDS. It includes plot generation, visualization of representative data (i.e., carving medoids out)
- *functionality specific to atomic databases*: reading/writing in extended xyz files, computation of atomic descriptors, 

This package relies on the `fast-kmedoids` repository and a slightly modified version of the `sklearn` package to build an algorithm that combines the k-medoids clustering technique with multidimensional scaling (MDS). This enhances the performance of MDS for bigger and more complex databases, such as (but not limited to) atomistic ones.

---

# Installation and Prerequisites

You have this already, but we can hopefully make it easier by preparing a conda environment with all the prerequisites.

---

# Typical Workflow

## 1. Input Data Preparation

Data can be anything, but this technique is designed with atomistic data in mind (more on descriptors, similarity kernels, and relevant atomistic parameters).

- distance metric

### 1.1. Sparsification Options

Mention here the sparsification module. It is a standalone module, meaning it can be used for clustering and sparsification purposes independently of cl-MDS.

See [Section 3](#3-adding-data-points-to-the-embedding) below.

## 2. Running cl-MDS

Explain here main parameters:
- hierarchy
- k-medoids related param.

Other (hyper)parameters should be included in [Inside cl-MDS](#inside-cl-mds-algorithm-details).

## 3. Adding Data Points to the Embedding

In this step, we incorporate those points not included in the sparse set to the cl-MDS embedding that we just computed (skip if sparsification was not used).

## 4. Saving cl-MDS information

## 5. Visualization and Further Use of the Embedding

Possible contents:
 - Saving medoids information: carving the atomic environment, generating a png image for each medoid
 - Plotting with gnuplot, matplotlib
 - Adding new data points to an existing cl-MDS embedding afterwards (in a later session, only using the saved information)

---

# Tutorials

More on this once we finish improving the tutorials. I think any example showing how this can be used is helpful here (could reference Miguel's tutorial?).
- Jupyter tutorial(s) -- should contain a summarized and interactive version of most of [Typical Workflow](#typical-workflow)
- [TurboGAP tutorial ref.](https://turbogap.fi/wiki/index.php/Energetic_and_structural_analysis_of_a_database_of_PtAu_nanoclusters)

---

# Advanced Tips

- Keeping track of the data: labeling and indexing sparse sets, extended data and so on. 
- Beware of indexing, especially when passing custom medoids.
- Does it make sense to use larger hierarchies? That is, how many levels `[N_levels,...., N_2, N_1, 1]` should one choose?
- useful but hidden hyperparameters for not-so-nice plots: weights per clusters, weights per anchor similarity

---

# Inside cl-MDS: Algorithm Details

Add here core details regarding the cl-MDS class (lighter version than the paper, focused more on the ideas and less on the math?)

## 1. Cluster Hierarchy

## 2. Algorithm Steps
### 2.1. K-medoids Clustering

Tiny intro to k-medoids:

Data is partitioned into clusters, and each cluster has a representative datapoint that summarizes its features.

### 2.2. Local Embedding

### 2.3. Anchor Points

### 2.4. Global Embedding

### 2.5. The Output: a single 2-Dimensional Embedding

## 3. Estimation of cluster membership and embedding coordinates for additional data

(change this title)

---

# Supporting Functions

It is useful to at least describe what can be done with the supporting functions (also help users locate them to learn how to use them).  
100% agree, we could expand this section and make it a "quick list" to find out the name and a small description of useful functions inside the cl-MDS class too, probably mentioned or related to things explained above.

## Matrix Processing

- **unique_rows_matrix**: removes repeated rows within a given tolerance  
- **remove_zero_entries**: simplifies the matrix by removing data points that are effectively identical.

## Clustering and Sparsification

- **optim_kmedoids**: Runs k-medoids several times to optimize the intra-cluster incoherence — this is a very useful function people should know about.  
We should check if this is already included in fast-kmedoids and can be used directly (I think it was added there later, so maybe this function is obsolete).
- Fucntions from the sparsification module

## Anchor Point Selection

- **select_further_points**: helps select extreme points that are farther from a reference point
-  **anchor_points_ndim**: I do not understand this one yet :)
- **points_in_polygon**: Computation of the number of points of a given set lying within a polygon with N vertices

## Check Permutation

Identifies the permutation type for a set of points — I do not yet understand the use case of this one.

