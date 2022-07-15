##############################################################
#                      cl-MDS demo
##############################################################
import numpy as np
import cluster_mds as clmds
import time
print("________________________________________________________")
print("________________________________________________________")
print(" ")
print("                 cl-MDS demo")
print("________________________________________________________")

data = clmds.clMDS(atoms='qm9_demo.xyz',
                   descriptor="quippy_soap_turbo",
                   cutoff=[2.5, 3.], do_species=['F'],
                   sparsify='random', n_sparse=300)


print("________________________________________________________")
print(" ")
print("                  SPARSE SET")
print("________________________________________________________")
print("--------------------------------------------------------")
print("---------     User-friendly way     --------------------")
print("--------------------------------------------------------")
t0u = time.time()
Y1 = data.get_sparse_coordinates([12,1])
C1 = Y1[:,2].astype(int)
t1u = time.time()
print("--------------------------------------------------------")
print("--------------------------------------------------------")
print("-----------     'Advanced' way      --------------------")
print("--------------------------------------------------------")
t0a = time.time()
data.cluster_MDS([12,1], weight_cluster_mds=2, iter_med=100, 
                 n_jobs_cluster=8, n_jobs_anchor=8)
t1a = time.time()

Y2 = data.sparse_coordinates
C2 = data.sparse_cluster_indices
print("--------------------------------------------------------")


print("________________________________________________________")
print(" ")
print("                 COMPLETE SET ")
print("________________________________________________________")
print("--------------------------------------------------------")
print("------------ Estimating their coord. -------------------")
t0e = time.time()
all_F = np.where(np.array(data.species_list) == 'F')[0]
ind_F = np.setdiff1d(all_F, data.sparse_list)
data.compute_atoms_estim_coordinates(indices=ind_F)
t1e = time.time()

Y_estim = data.estim_coordinates
C_estim = data.estim_cluster_indices
print("--------------------------------------------------------")
print(" ")
print('Time clMDS (sparse): %.2g sec'% (t1u-t0u))
print('Time clMDS (sparse): %.2g sec'% (t1a-t0a))
print('Time clMDS (estimation): %.2g sec' % (t1e-t0e))
print(" ")


#############################################################
## Save file
# data.write_xyz(filename='qm9_struct.xyz')

clmdsfile = './clmds_qm9_demo.dat'
f = open(clmdsfile, 'w+')
sparse = data.sparse_list
M2 = data.sparse_medoids
print('n_sparse X_clmds Y_clmds C M', file=f)
for i in range(0, len(sparse)):
    if i in M2:
        ind_M = 1
    else:
        ind_M = 0
    print('%i %f %f %i %i' % (sparse[i], Y2[i,0], Y2[i,1],
          C2[i], ind_M), file=f)
f.close()



## Plotting
import matplotlib.pyplot as plt
color_list=np.array(['gold','limegreen','green','red','brown',
                     'cyan','darkgoldenrod','magenta',
                     'deepskyblue', 'navy','orange','purple',
                     'pink','gray','darkgreen','peru',
                     'rosybrown', 'coral', 'blue', 'olive'])
plt.suptitle("cl-MDS map of F atoms (QM9 database)")
plt.subplot(1,3,1)
plt.title("sparse (easy)")
plt.scatter(Y1[:, 0], Y1[:, 1], c=color_list[C1], alpha=0.4)

plt.subplot(1,3,2)
plt.title("sparse (param.)")
plt.scatter(Y2[:, 0], Y2[:, 1], c=color_list[C2], alpha=0.4)

plt.subplot(1,3,3)
plt.title("complete (estimation)")
plt.scatter(Y_estim[:, 0], Y_estim[:, 1], c=color_list[C_estim],
            alpha=0.4)
plt.scatter(Y2[:, 0], Y2[:, 1], c='black', marker='.',
            label='sparse set')
plt.legend()

plt.show()

