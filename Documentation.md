# What is cl-MDS?

cl-MDS is a cluster-based multidimensional scaling tool for visualizing high-dimensional data in low-dimensional space. (refer to the paper for a detailed description of the embedding technique)

---

# Installation and Prerequisites

You have this already, but we can hopefully make it easier by preparing a conda environment with all the prerequisites.

---

# Typical Workflow

## 1. Input Data Preparation

Data can be anything, but this technique is designed with atomistic data in mind (more on descriptors, similarity kernels, and relevant atomistic parameters).

- distance metric

## 2. K-medoid Clustering

Data is partitioned into clusters, and each cluster has a representative datapoint that summarizes its features.

## 3. Global Embedding

## 4. Local Embedding

## 5. Sparsification Option

## 6. The Final Low-Dimensional Embedding

## 7. Visualization and Further Use of the Embedding

---

# Supporting Functions

It is useful to at least describe what can be done with the supporting functions (also help users locate them to learn how to use them).

## Matrix Processing

1. **unique_rows_matrix**: removes repeated rows within a given tolerance  
2. **remove_zero_entries**: simplifies the matrix by removing data points that are effectively identical.

## optim_kmedoids

Runs k-medoids several times to optimize the intra-cluster incoherence — this is a very useful function people should know about.

## Point Selection

1. **select_further_points**: helps select extreme points that are farther from a reference point  
2. **anchor_points_ndim**: I do not understand this one yet :)

## Check Permutation

Identifies the permutation type for a set of points — I do not yet understand the use case of this one.

## points_in_polygon

Computation of the number of points of a given set lying within a polygon with N vertices.

---

# Tutorials

More on this once we finish improving the tutorials. I think any example showing how this can be used is helpful here (could reference Miguel's tutorial?).

