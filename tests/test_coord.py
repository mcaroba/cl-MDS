
import numpy as np
import cluster_mds as clmds
import time

print(" ")
print("********************************************")
print("Test final coordinates within [-1,1]")
print(" ")

hierarchy = [[5,1],[5,3,1]]

for N in [100, 500, 1000]:
    D = np.random.rand(N,N)
    np.fill_diagonal(D, 0)
    D = (D + D.T)/2
    data = clmds.clMDS(dist_matrix=D, verbose=False)
    for h in hierarchy:
        print(" ")
        print(" Hierarchy ", h)
        print("-----------------------------")
        for i in range(0, 5):
            print("Dataset size %i, result %i" % (N, i))
            t0 = time.time()
            data.cluster_MDS(h, iter_med=10)
            t1 = time.time()
            print("Time calculations: %.2f" % (t1-t0))
            X = data.sparse_coordinates
            pos_cond = np.where(X > 1)[0]
            neg_cond = np.where(X < -1)[0]
            if pos_cond.size > 0 or neg_cond.size > 0:
                I = data.all_transformations
                if pos_cond.size > 0:
                    print( "FAIL! coordinates bigger than 1: ")
                    for i in set(pos_cond):
                        cl = data.sparse_cluster_indices[i]
                        print("Point ", i, ", clustering: ", cl)
                        for level  in range(1, len(h)):
                            cl_h = I[cl][level]["cluster"]
                            T_h = ("lineal" if isinstance(I[cl][level]["transf"], np.ndarray) else "homography")
                            a_h = I[cl][level]["anchor"]
                            if i in a_h:
                                position = "anchor point"
                            elif clmds.points_in_polygon(len(a_h), X[a_h,:], [X[i,:]]):
                                position = "in"
                            else:
                                position = "out"
                            print(" ---> ", cl_h, ", transf. ", T_h, ", position regarding anchor points: ", position)    
                if neg_cond.size > 0:
                    print( "FAIL! coordinates smaller than -1: ")
                    for i in set(neg_cond):
                        cl = data.sparse_cluster_indices[i]
                        print("Point ", i, ", clustering: ", cl)
                        for level  in range(1, len(h)):
                            cl_h = I[cl][level]["cluster"]
                            T_h = ("lineal" if isinstance(I[cl][level]["transf"], np.ndarray) else "homography")
                            a_h = I[cl][level]["anchor"]
                            if i in a_h:
                                position = "anchor point"
                            elif clmds.points_in_polygon(len(a_h), X[a_h,:], [X[i,:]]):
                                position = "in"
                            else:
                                position = "out"
                            print(" ---> ", cl_h, ", transf. ", T_h, ", position regarding anchor points: ", position)  
            else:
                print("All right")
            print(" ")


print(" ")
print("********************************************")


