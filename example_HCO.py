import numpy as np
import cluster_mds as cmds
from sklearn import manifold
from matplotlib import pyplot as plt
import random
import time


species = ['H','C','O']
species_Z = [1,6,8]
n_medoids = 11

# Read the distance matrix saved in a file
f = open('dist_matrix_HCO.dat', 'r')
dim = int(f.readline())
dist = np.zeros(dim*dim)
for i in range(0, dim*dim):
    dist[i] = float(f.readline().split()[0])
dist_matrix = np.reshape(dist,(dim,dim))
f.close()
# Read filenames and atom numbers from file
f = open("fps_list_HCO.dat", "r")
Z_data = np.zeros(dim)
for n_line, line in enumerate(f):
    Z_data[n_line] = int(line.split()[2])
f.close()

# Usual MDS
start_u = time.time()
embedding = manifold.MDS(n_components = 2, dissimilarity = "precomputed",
                         n_init = 1, n_jobs = 1, max_iter = 100)
mds_global = embedding.fit_transform(dist_matrix)
end_u = time.time()
print("Time usual MDS: ",end_u-start_u)

plt.figure(1)
colors =['red','cyan','purple']
for n, z in enumerate(species_Z):
    temp_ind = np.where(Z_data == z)[0]
    plt.scatter(mds_global[temp_ind,0], mds_global[temp_ind,1], color=colors[n],
                label='{}'.format(species[n]))
plt.title('usual MDS', fontsize=18)
plt.legend()

# Cluster MDS
start_n = time.time()
ind_clusters, ind_med, embedding, cmds_matrix, cmds_anchor, cmds_med = cmds.new_MDS(dist_matrix, n_medoids, 
                                                                                    init_medoids = "isolated", n_iso_med = 8,
                                                                                    n_init_mds_anchorpts = 2000,
                                                                                    max_iter_anchorpts = 200)
end_n = time.time()
print("Time cluster MDS: ",end_n-start_n)
# Reshape cMDS matrix and reorder Z_data according to clusters
cmds = cmds_matrix[0]
indexes = ind_clusters[0]
for i in range(n_medoids-1):
    cmds = np.concatenate((cmds,cmds_matrix[i+1])) 
    indexes = np.append(indexes,ind_clusters[i+1])
Z_data_clusters = Z_data[indexes]

plt.figure(2)
for n,z in enumerate(species_Z):
    temp_ind = np.where(Z_data_clusters == z)[0]
    plt.scatter(cmds[temp_ind,0], cmds[temp_ind,1], color=colors[n], label='{}'.format(species[n]))
plt.scatter(cmds_med[:,0], cmds_med[:,1], marker='*', color='black', label='kmedoids')
plt.title('cluster MDS', fontsize=18)
plt.legend()


plt.show()

