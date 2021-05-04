#************************************************************************************************************
# This is the cluster-based MultiDimensional Scaling code for dimensionality reduction data analysis.       #
#                                                                                                           #
#                                                cl-MDS                                                      #
#                                                                                                           #
# This code has been written and is copyright (c) 2018-2020 of the following authors:                       #
#                                                                                                           #
# *) Patricia Hernandez-Leon                                                                                #
# *) Miguel A. Caro                                                                                         #
#                                                                                                           #
# from the Department of Electrical Engineering and Automation, Aalto University, Finland                   #
#                                                                                                           #
#                                                                                                           #
# See the file LICENSE.md for license information and the README.md file for practical installation         #
# instructions and usage examples. The official code repository is                                          #
#                                                                                                           #
#                                   https://github.com/mcaroba/cl-MDS/                                       #
#                                                                                                           #
# Visit the repository for the latest version of this distribution.                                         #
#                                                                                                           #
#                                                                                                           #
# If you use cl-MDS for the compilation of academic/scientific/technical work, please cite, as appropriate:  #
#                                                                                                           #
# P. Hernandez-Leon and M.A. Caro, XXX, YYY (2020)                                                          #
#                                                                                                           #
#************************************************************************************************************


# Import dependencies
import numpy as np
from sklearn import manifold
from kmedoids import kmedoids
from cur import cur
import random
import sys
import itertools
from scipy.special import comb
from scipy.spatial import ConvexHull


#************************************************************************************************************
class clMDS:
    """
    This is the main clMDS class. It can take a distance matrix and/or an ASE compatible
    (e.g., an xyz file) atomic structure. It can also take concatenated xyz files or any trajectory that
    can be imported with ASE.
    If the user chooses an implemented descriptor, clMDS will make an educated guess and assign some sensible
    defaults. If the user wants more control over the choice of hyperparameters, they should pass a
    distance matrix instead.
    """

#   Initialize the class:
    def __init__(self, dist_matrix=None, atoms=None, descriptor=None, descriptor_string=None,
                 sparsify=None, sparsify_per_cluster=False, n_sparse=None, max_n_sparse=None,
                 average_kernel=False, cutoff=None, verbose=True):
#       This is the list of implemented atomic descriptors (it typically requires external
#       programs)
        implemented_descriptors = ["quippy_soap","quippy_soap_turbo"]
        sparse_options = ["random", "cur"]
        self.verbose = verbose
        self.sparsify = sparsify
        self.sparsify_per_cluster = sparsify_per_cluster
        self.n_sparse = n_sparse
        self.max_n_sparse = max_n_sparse
        self.is_clustered = False
        self.has_clmds = False
        self.cutoff = cutoff
        self.average_kernel = average_kernel
        self.has_estimation = False
#       The descriptor is not attached until build_descriptor() is run
        self.has_descriptor = False

#       Read in the distance matrix
        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
            self.has_dist_matrix = True
        else:
            self.has_dist_matrix = False

#       Read in the atoms file and create an ASE atoms object
        if atoms is not None:
            try:
                from ase.io import read, write
                self.atoms = read(atoms, index=":")
            except:
#               This error appears also when atoms has a wrong file extension, should we mention this too?      <-- comment
                raise Exception("I couldn't find an ASE installation; you need ASE to pass an atoms filename!")
        else:
            self.atoms = None

#       Check if the user wants to use a descriptor
        if descriptor is not None:
            if descriptor not in implemented_descriptors:
                raise Exception("The descriptor you chose is not implemented; the options are: ",
                                implemented_descriptors)
            if dist_matrix is not None:
                raise Exception("You can't define a distance matrix and a descriptor; choose one or the other!")
            if atoms is None:
                raise Exception("If you define a descriptor, you must also provide an atoms filename")
            self.descriptor_type = descriptor
            self.descriptor_string = descriptor_string

#       Check if the user wants to use sparsification       
        if sparsify is not None:
            if isinstance(sparsify, (list, np.ndarray)):
                sparsify = list(set(sparsify)) # Avoid repeated entries                                     <--- check this
                self.n_sparse = len(sparsify)
                self.sparsify = sparsify
                if self.has_dist_matrix:
                    self.sparse_list = sorted(sparsify)
                    self.dist_matrix = dist_matrix(np.ix_(self.sparse_list, self.sparse_list))
            elif sparsify not in sparse_options:
                raise Exception("The sparsify option you chose is not available. Choose one of the following: ",
                                sparse_options, " or provide a list of indexes")
            else:
                if n_sparse is None:
                    raise Exception("If you choose a sparsify option, you need to pass the n_sparse parameter")
                else:
                    self.sparsify = sparsify
                    self.n_sparse = n_sparse
                if sparsify == "cur" and self.has_dist_matrix:
                    self.sparse_list = list(set(cur.cur_decomposition(self.dist_matrix, n_sparse)[-1]))
                    self.dist_matrix = dist_matrix(np.ix_(self.sparse_list, self.sparse_list))
#               Implement the "optimized sparse set" as a sparsify option                                           <--- comment  



#   This method takes care of adding a descriptor to the clMDS class:
    def build_descriptor(self):
        if not hasattr(self, 'descriptor_type'):
            raise Exception("You must define a descriptor, check the implemented options")

        descriptor = self.descriptor_type
        descriptor_string = self.descriptor_string
        if descriptor in ["quippy_soap", "quippy_soap_turbo"]:
            from ase.data import atomic_numbers
            from quippy.descriptors import Descriptor
            from quippy.convert import ase_to_quip
            self.zeta = 4
            species_list = []
            for ats in self.atoms:
                for at in ats:
                    species_list.append(at.symbol)
            self.species_list = species_list
            species_set = list(set(species_list))
#           This uses some default SOAP parameters
            if descriptor_string is None:
                species_string = ""
                for species in species_set:
                    species_string += " " + str(atomic_numbers[species])
                n_Z = len(species_set)
                if descriptor == "quippy_soap":
                    if self.cutoff == None:
                        cutoff = 5.0
                        self.cutoff = cutoff
                    else:
                        cutoff = self.cutoff
                    quippy_string = "soap n_max=8 l_max=8 cutoff=" + str(cutoff) + " atom_sigma=0.5 Z={" + \
                                    species_string + "} species_Z={" + species_string + "} n_Z=" +  \
                                    str(n_Z) + " n_species=" + str(n_Z) 
                    if self.average_kernel:
                        quippy_string = quippy_string + " average=T" 
                elif descriptor == "quippy_soap_turbo":
                    if not self.cutoff:
                        rcut_hard = 5.0
                        rcut_soft = rcut_hard - 0.5
                        self.cutoff = [rcut_soft, rcut_hard]*n_Z
                        cutoff = self.cutoff
                    elif len(self.cutoff) == 2:
                        cutoff = sorted(self.cutoff)*n_Z
                        rcut_soft = self.cutoff[0]
                        rcut_hard = self.cutoff[1]
                    elif len(self.cutoff) == 2*n_Z:
                        cutoff = self.cutoff
                    else:
                        raise Exception("soap_turbo uses two cutoff (soft and hard cutoff) per specie. \
                                         Please give a list with 2 or 2*n_species values.")
                    alpha = ""; at_s = ""; at_ss = ""
                    amplitude = ""; central_w = ""
                    for i in range(0, n_Z):
                        alpha += " 8"
                        at_s += " 0.2"
                        at_ss += " 0.1"
                        amplitude += " 1."
                        central_w += " 1."
                    quippy_string = {}
                    for i, z in enumerate(species_set):
                        rcut_soft = min(cutoff[2*i:2*(i+1)])
                        rcut_hard = max(cutoff[2*i:2*(i+1)])
                        quippy_string[z] = 'soap_turbo alpha_max={' + alpha + '} l_max=8 rcut_soft=' + \
                                           str(rcut_soft) + ' rcut_hard=' + str(rcut_hard) + ' atom_sigma_r={' \
                                           + at_s + '} atom_sigma_t={' + at_s + '} atom_sigma_r_scaling={' \
                                           + at_ss + '} atom_sigma_t_scaling={' + at_ss + '} radial_enhancement=1 \
                                           amplitude_scaling={' + amplitude + '} basis="poly3gauss" \
                                           scaling_mode="polynomial" species_Z={' + species_string + '} \
                                           n_species=' + str(n_Z) + ' central_index=' + str(i+1) + \
                                           ' central_weight={' + central_w + '}' 
#                        if self.average_kernel:
#                            quippy_string[z] = quippy_string[z] + " average=T" 

#           This uses a user-defined SOAP/SOAP_TURBO
            else:
                quippy_string = self.descriptor_string
                if descriptor == "quippy_soap":
#                   Check string
                    n_Z = self.get_info_string(quippy_string, label="n_Z", type_label=int)
                    if n_Z != len(species_set):
                        raise Exception("Your database has a different amount of species than the \
                                         given on the descriptor string.")
#                   Take cutoff and average kernel from descriptor string
                    cutoff = self.get_info_string(quippy_string, label="cutoff", type_label=float)
                    average = self.get_info_string(quippy_string, label="average")     
                    self.cutoff = cutoff
                    self.average_kernel = average
                elif descriptor == "quippy_soap_turbo":
#                   Check string
                    n_Z = self.get_info_string(quippy_string[species_set[0]], label="n_species", type_label=int)
                    if n_Z != len(species_set):
                        raise Exception("Your database has a different amount of species than the \
                                         given on the descriptor string.")
                    elif n_Z == 1:
                        if isinstance(quippy_string, str):
                            quippy_string = {species_set[0]: quippy_string}
                    else:
                        if isinstance(quippy_string, str) or len(quippy_string) != n_Z:
                            raise Exception("You need to give as many descriptor strings as n_species for ",
                                             descriptor)
                        elif not isinstance(quippy_string, dict):
#                           Check this part (I assume the string always has ordered Z)                               <-- comment
                            indices = [self.get_info_string(quippy_string[i], label="central_index", type_label=int)-1
                                       for i in range(0, n_Z)]
                            quippy_string = {species_set[i]: quippy_string[indices[i]] for i in range(0, n_Z)}
#                   Take cutoff from descriptor string
                    cutoff = []
                    for z in species_set:
                        rsoft = self.get_info_string(quippy_string[z], label="rcut_soft", type_label=float)
                        rhard = self.get_info_string(quippy_string[z], label="rcut_hard", type_label=float)
                        if not rsoft or not rhard:
                            raise Exception("soap_turbo uses two cutoff (soft and hard cutoff) per specie, \
                                             please include both in the descriptor string.")
                        cutoff.append(rsoft)
                        cutoff.append(rhard) 
                    self.cutoff = cutoff

#           Get the number of environments 
            if self.average_kernel:
                n_env = len(self.atoms)
            else:
                n_env = sum(len(ats) for ats in self.atoms)
            self.all_env = n_env
#           Check if there is sparsification
            if self.sparsify is not None:
                if isinstance(self.sparsify, (list, np.ndarray)):
                    if len(self.sparsify) > n_env:
                        raise Exception("The sparse set can't be larger than the complete dataset")
                    else: 
                        sparse_list = np.array(self.sparsify, dtype=int)
                        self.sparse_list = sorted(sparse_list)
                elif self.sparsify == "random":
                    sparse_list = np.array(range(n_env), dtype=int)
                    np.random.shuffle(sparse_list)
                    sparse_list = sparse_list[0:self.n_sparse]
                    self.sparse_list = sorted(sparse_list)
                elif self.sparsify == "cur":
                    sparse_list = list(range(n_env))
            else:
#               Added to avoid possible crushes in other methods
                self.sparse_list = list(range(n_env))

#           Descriptors
            if descriptor == "quippy_soap":
                d = Descriptor(quippy_string)
            elif descriptor == "quippy_soap_turbo":
                d = {z: Descriptor(quippy_string[z]) for z in species_set}
            n = 0
            descriptor_list = []
            config_type_list = []
            if self.verbose:
                print("")
            for ats in self.atoms:
                if self.verbose:
                    sys.stdout.write('\rComputing descriptors:%6.1f%%' % (float(n)*100./float(n_env)) )
                    sys.stdout.flush()
                if not self.average_kernel:
                    if "config_type" in ats.info:
                        config_type_list.append([ats.info["config_type"]]*len(ats))
                    else:
                        config_type_list.append([None]*len(ats))
                else:
                    if "config_type" in ats.info:
                        config_type_list.append(ats.info["config_type"])
                    else:
                        config_type_list.append(None)
                if self.sparsify is not None and not self.sparsify_per_cluster: 
                    if isinstance(self.sparsify, (list, np.ndarray)) or self.sparsify=="random":
                        in_sparse = (sparse_list >= n) & (sparse_list < n + len(ats))
                        if not (in_sparse).any():
                            n += len(ats)
                            continue
                if descriptor == "quippy_soap":                
                    a = ase_to_quip(ats)
                    a.set_cutoff(cutoff)
                    a.calc_connect()
                    qs = d.calc_descriptor(a)
                    for q in qs:
                        if not self.sparsify_per_cluster:
                            if isinstance(self.sparsify, (list,np.ndarray)) or self.sparsify == "random":
                                if n in sparse_list:       
                                    descriptor_list.append(q) 
                            else:    
                                descriptor_list.append(q)                             
                        else:
                            descriptor_list.append(q) 
                        n += 1   
                elif descriptor == "quippy_soap_turbo":   
                    q = {}
                    N = {z: 0 for z in species_set}
                    for i, z in enumerate(species_set):   
                        rcut_hard = max(cutoff[2*i:2*(i+1)])     
                        a = ase_to_quip(ats)
                        a.set_cutoff(rcut_hard)    
                        a.calc_connect()
                        q[z] = d[z].calc_descriptor(a)   
                    for at in ats:
                        symb = at.symbol
                        if not self.sparsify_per_cluster:
                            if isinstance(self.sparsify, (list,np.ndarray)) or self.sparsify == "random":
                                if n in sparse_list:       
                                    descriptor_list.append(q[symb][N[symb]])  
                            else:    
                                descriptor_list.append(q[symb][N[symb]])                                 
                        else:
                            descriptor_list.append(q[symb][N[symb]]) 
                        N[symb] += 1 
                        n += 1                
            descriptor_list = np.array(descriptor_list)
            if isinstance(self.sparsify, str):
                if self.sparsify == "cur":
                    self.sparse_list = sorted(set(cur.cur_decomposition(descriptor_list, self.n_sparse)[-1]))
            if self.verbose:
                sys.stdout.write('\rComputing descriptors:%6.1f%%' % 100. )
                sys.stdout.flush()
                print("")

            if not self.average_kernel:
                self.config_type_list = np.concatenate([c for c in config_type_list])
            else:
                self.config_type_list = np.array(config_type_list)
            self.n_env = len(self.sparse_list)
            self.sparse_list = list(self.sparse_list)
            self.descriptor = descriptor_list
            self.has_descriptor = True
        else:
            raise Exception("You must choose among the implemented descriptors: ", implemented_descriptors)


#   Cumbersome code to get a specific label (only those with a single value) from a descriptor string
    def get_info_string(self, quippy_string, label, type_label=str):
        list_labels = {"quippy_soap": ["n_max", "l_max","cutoff", "atom_sigma", "n_Z", "n_species", "average"], 
                       "quippy_soap_turbo": ["alpha_max","l_max","rcut_hard","rcut_soft", "radial_enhancement",
                                             "n_species", "central_index", "central_weight"] }
        list_descriptors = list(list_labels.keys())
        if not self.descriptor_type in list_descriptors:
            raise Exception("The code can't get any label from a string of the given descriptor_type (yet). \
                             Available options are: ", list_descriptors)
#       Check if it is label with a single value (not arrays) 
        if not label in list_labels[self.descriptor_type]:
            raise Exception("This method can't extract the chosen label, the available ones for %s are:" 
                             % self.descriptor_type,  list_labels[self.descriptor_type])
        a = quippy_string.split()
        if a[0] != self.descriptor_type[7:]:
            raise Exception("The descriptor string doesn't correspond to the descriptor type, check this.")
        N = len(label)
        for i in range(0, len(a)):
            b = a[i]
            if b[0:N] != label:
                continue
            if len(b) == N:
                c = a[i+1]
                if len(c) == 1:
                    param = a[i+2]
                    break
                else:
                    param = c[1:]
                    break
            elif len(b) == N+1:
                c = a[i+1]
                param = c
                break
            else:
                param = b[N+1:]
                break
        else:
            param = False

        if param:
            if type_label == float:
                param = float(param)
            elif type_label == int:
                param = int(param)

        return param


#   This method takes care of building a distance matrix:
    def build_dist_matrix(self, precision=1.e-8):
        if self.descriptor_type is not None:
#           If the descriptors have not been computed, we need to do so
            if not self.has_descriptor:
                self.build_descriptor()
            
            n_env = self.n_env
            descriptor = self.descriptor
            if self.sparsify is not None:
                if len(descriptor) > n_env:
                    descriptor = descriptor[self.sparse_list]
            dist_matrix = np.zeros([n_env, n_env])
            n = 0
            if self.verbose:
                print("")
            for i in range(0, n_env):
                if self.verbose:
                    sys.stdout.write('\rComputing dist_matrix:%6.1f%%' % (float(n)*100./float(n_env*(n_env+1)/2)) )
                    sys.stdout.flush()
#               Do this to remove numerical round-off problems
                dist_matrix[i,i] = 0.
                for j in range(i+1, n_env):
                    prod = np.dot(descriptor[i], descriptor[j])**self.zeta
                    if prod <= 1. - precision:
                        dist_matrix[i][j] = np.sqrt(1. - prod)
                        dist_matrix[j][i] = np.sqrt(1. - prod)
                    else:
                        dist_matrix[i][j] = 0.
                        dist_matrix[j][i] = 0.
                    n += 1
            if self.verbose:
                sys.stdout.write('\rComputing dist_matrix:%6.1f%%' % 100. )
                sys.stdout.flush()
                print("")

            self.dist_matrix = dist_matrix
            self.has_dist_matrix = True
        else:
            raise Exception("No descriptor defined: nothing to do!")


#   This function ensures a minimum amount of points per cluster in the sparse set
    def cluster_sparsification(self, n_clusters, init_medoids="random", n_iso_med=None, 
                               iter_med=1000, tmax=100):
        if not self.sparsify_per_cluster:
            raise Exception("You haven't set sparsify_per_cluster=True. Please check if you want \
                             that kind of sparsification and set it right.")
        n_sparse = self.n_sparse
        self.n_sparse = self.n_sparse*n_clusters
        if not self.has_descriptor:
            self.build_descriptor()
        elif len(self.descriptor) < self.all_env:
            self.build_descriptor()
        self.build_dist_matrix()

        sparse_list = self.sparse_list
#       Compute the clustering with the initial sparse set
        M, C, I = optim_kmedoids( self.dist_matrix, n_clusters, incoherence="rel", n_iter=iter_med, 
                                  tmax=tmax, init_Ms=init_medoids, n_iso=n_iso_med, verbose=self.verbose)
        n_C = []
        C_incomplete = []
        C_indices = np.zeros(self.n_env, dtype=int)
        for i in range(0, n_clusters):
            n_C.append(len(C[i]))
            C_indices[C[i]] = i
            if n_C[i] < n_sparse:
                C_incomplete.append(i)
        if C_incomplete:
            non_sparse_list = [i for i in range(0, self.all_env) if i not in sparse_list]
            np.random.shuffle(non_sparse_list)
            for i in non_sparse_list:
#               Assign each non_sparse point to a cluster
                dist_med = np.zeros(n_clusters)
                for j in range(0, n_clusters):
                    med = sparse_list[M[j]]
                    prod = np.dot(self.descriptor[i], self.descriptor[med])**self.zeta 
                    if prod < 1.:
                        dist_med[j] =  np.sqrt(1. - prod)
                    else:
                        dist_med[j] = 0.
#               Add it to its cluster if it has less than n_sparse points
                ind_cluster = np.argmin(dist_med)
                if ind_cluster in C_incomplete:
                    sparse_list.append(i)
                    C_indices = np.append(C_indices, ind_cluster)
                    n_C[ind_cluster] += 1
                    C_incomplete = [i for i in range(0, n_clusters) if (n_C[i] < n_sparse)]
#               Check if all the clusters has at least n_sparse elements
                if not C_incomplete:
                    break
            add_points = 0
            if C_incomplete:
                for i in C_incomplete:
                    add_points += n_sparse - n_C[i]
                    print('Only %i elements where found for cluster %i in the complete database' 
                          % (n_C[i], i))
            if self.max_n_sparse:
                max_n_sparse = self.max_n_sparse
                C_overfull = [i for i in range(0, n_clusters) if n_C[i] > max_n_sparse]
                if add_points:
                    max_n_sparse = max_n_sparse + add_points//len(C_overfull)
                    C_overfull = [i for i in range(0, n_clusters) if n_C[i] > max_n_sparse]
                if C_overfull:
                    out_sparse = set()
                    for i in C_overfull:
                        np.random.shuffle(C[i])
                        out_sparse.update(C[i][max_n_sparse:])
                    sparse_list = [point for i, point in enumerate(sparse_list) if i not in out_sparse]
                    C_indices = np.array([ind for i, ind in enumerate(C_indices) if i not in out_sparse])
            new_indices = np.argsort(sparse_list)
            C_indices = C_indices[ new_indices ]
            M = np.array([np.where(new_indices == med)[0][0] for med in M])
            C = {i: np.where(C_indices == i)[0] for i in range(0, n_clusters) } 
            self.n_env = len(sparse_list)
            self.sparse_list = sorted(sparse_list)
            self.build_dist_matrix()
            self.n_sparse = n_sparse
        else:
            print("The sparse set already fulfils your n_sparse request.")  

        return M, C, I



#   Method that prints a small analysis of intra-cluster incoherence and silhouette values for   
#   a given dataset, range of number of clusters and k-medoids parameters (no plot, only values)                                  
#   Improve how are shown the results                                                                     <-- comment 
    def clustering_analysis(self, n_cluster_min=2, n_cluster_max=10, specific_Ms="isolated",
                            specific_n_iso=None, I_min="tot", n_ranking=15):
        if not self.has_dist_matrix:
            self.build_dist_matrix()

        from sklearn.metrics import silhouette_samples, silhouette_score

        print(" N_cl  n_iso  |  I_rel     I_tot   |  average silhouette   min/max. silhouette  |  cluster  size ")
        print("-------------------------------------------------------------------------------------------------")
        N_cl = np.arange(n_cluster_min, n_cluster_max+1, 1)
        if specific_Ms=="random":
            param_iso = [0]
        elif isinstance(specific_n_iso, int):
            param_iso = [specific_n_iso]
        elif isinstance(specific_n_iso, (list, np.ndarray)):
            param_iso = specific_n_iso
        else:
            param_iso = list(range(1, n_cluster_max+1))

        C_ind = np.empty( len(self.dist_matrix), dtype=int )
        I_rel = 10*np.ones( (len(N_cl), len(param_iso)) )
        I_tot = np.ones( (len(N_cl), len(param_iso)) )*10**4
        average_sil = -10*np.ones( (len(N_cl), len(param_iso)) )
        for n in N_cl:
            ind = n - n_cluster_min
            for m, param in enumerate(param_iso):
                if param > n:
                    break
                M, C, I_rel[ind,m] = optim_kmedoids( self.dist_matrix, n, incoherence="rel", 
                                             init_Ms=specific_Ms, n_iso=param)
                I_tot[ind,m] = 0.
                for i in range(0, n):
                    C_ind[C[i]] = i
                    I_tot[ind,m] += np.sum(self.dist_matrix[M[i]][C[i]])
                average_sil[ind,m] = silhouette_score(self.dist_matrix, C_ind, metric="precomputed")
                sil_values = silhouette_samples(self.dist_matrix, C_ind, metric="precomputed")
                print(" %2i     %2i    |  %2.4f  %2.4f  |    %2.4f" % (n, param, I_rel[ind,m], I_tot[ind,m],
                                                                             average_sil[ind,m]))
                for i in range(0, n):
                    sil_cluster = sil_values[C[i]]
                    print("              |                    |      %2.6f        % 2.6f/% 2.6f   |  %4i     %4i "  
                          % (np.mean(sil_cluster), np.min(sil_cluster), np.max(sil_cluster), i, len(C[i])))
        print("----------------------------------------------------------------------------------------------")
        print("n.         silhuoette                    incoherence            ")
        print("        N_cl  n_iso    s        N_cl  n_iso    I_tot    I_rel   ")
        score_sil = np.argsort( average_sil.flatten() )
        if I_min == "tot":
            score_I = np.argsort( I_tot.flatten() )
        elif I_min == "rel":
            score_I = np.argsort( I_rel.flatten() )
        for i in range(1, n_ranking+1):
            row_sil, row_I = score_sil[-i]//len(param_iso), score_I[i-1]//len(param_iso)
            col_sil, col_I = score_sil[-i]%len(param_iso), score_I[i-1]%len(param_iso)
            print("%2i       %2i    %2i    %2.3f      %2i    %2i       %2.3f  %2.3f" % (i, N_cl[row_sil], param_iso[col_sil], 
                   average_sil[row_sil, col_sil], N_cl[row_I], param_iso[col_I], I_tot[row_I, col_I], I_rel[row_I, col_I]))


#   This method clusters the data and produces the embedded 2-dimensional coordinates
#   Make sure these are sensible defaults!!!!!!                                                              <-- comment
    def cluster_MDS(self, hierarchy, iter_med=10000, tmax=100, init_medoids="isolated", n_iso_med=1,
                    n_init_mds_cluster=10, max_iter_cluster=200, n_jobs_cluster=1, verbose_cluster=0,
                    n_anchor=4, criterion_anchor="area", n_init_mds_anchor=3500, max_iter_anchor=300, 
                    n_jobs_anchor=1, verbose_anchor=0, precision_qhull=1e-7, eta=0.):
        """
        Parameters:

        * Hierarchy
        Use hierarchy = [n_clusters, n_level1, n_level2, ... , 1] to perform hierarchical embedding
        and clustering, where n_clusters refers to the finest clustering (computed in the data
        n-dimensional space) and 1 refers to the final embedded 2d-space.
        Depending on the chosen hierarchy, different local structures of the dataset can be weighted 
        more during the computation. The simplest hierarchy system is [n_clusters, 1], where only
        one level of clustering is considered.

        * Clustering (iter_med, tmax, init_medoids, n_iso_med)
        Check kmedoids for further information.

        * Embedding (n_init_*, max_iter_*, n_jobs_*, verbose_*; *=(initial) clusters, anchor (points))
        Check sklearn.manifold.MDS for additional information

        * Anchor points (n_anchor, criterion_anchor, precision_qhull)
        This method only supports n_anchor=3,4 (the anchor point selection process and later
        transformations won't make sense with other values).
        Check anchor_points (from cluster_mds) and scipy.spatial.ConvexHull for further information.

        * Kernel weight for the distance matrix (eta)

        """
        if isinstance(hierarchy, list):
            try:
                n_clusters = hierarchy[0]
                assert type(n_clusters) == int
            except:
                raise Exception("You need to define a hierarchy of cluster levels by providing the \
                                 hierarchy parameter, e.g., [8,1] or [8,3,1]")
            if len(hierarchy) == 1:
                hierarchy.append(1)
        elif isinstance(hierarchy, int):
            n_clusters = hierarchy
            hierarchy = [hierarchy, 1]
        else:
            raise Exception("You need to define a hierarchy of cluster levels by providing the \
                            hierarchy parameter, e.g., [8,1] or [8,3,1]")

        self.hierarchy = hierarchy

#       Finest clustering (initial hierarchy level)
        if not self.sparsify_per_cluster: 
            if not self.has_dist_matrix:
                self.build_dist_matrix() 
            ind_medoids, ind_clusters, I = optim_kmedoids( self.dist_matrix, n_clusters, incoherence="rel",
                                                           n_iter=iter_med, tmax=tmax, init_Ms=init_medoids,
                                                           n_iso=n_iso_med, verbose=self.verbose )
        else: 
            ind_medoids, ind_clusters, I = self.cluster_sparsification(n_clusters, init_medoids=init_medoids,
                                                                       n_iso_med=n_iso_med, iter_med=iter_med,
                                                                       tmax=tmax)
        dist_clusters = [self.dist_matrix[np.ix_(ind_clusters[i], ind_clusters[i])]
                         for i in range(0, n_clusters)]      
#       MDS calculation minimizing the stress
        embedding = manifold.MDS( n_components=2, dissimilarity="precomputed",
                                  n_init=n_init_mds_cluster, max_iter=max_iter_cluster,
                                  n_jobs=n_jobs_cluster, verbose=verbose_cluster )
        mds_clusters = np.zeros((len(self.dist_matrix),2))
        if self.verbose:
            print("")
        for i in range(0, n_clusters):
            if self.verbose:
                sys.stdout.write('\rEmbedding data:%6.1f%%' % (float(i)*100./float(n_clusters)) )
                sys.stdout.flush()
            if len(ind_clusters[i]) > 1:
                mds_clusters[ind_clusters[i]] = embedding.fit_transform(dist_clusters[i])
            else:
                mds_clusters[ind_clusters[i]] = np.zeros((1,2)) # avoid sklearn RuntimeWarning
        if self.verbose:
            sys.stdout.write('\rEmbedding data:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")
        
        self.cluster_incoherence = I
        self.local_sparse_coordinates = mds_clusters

#       Hierarchy levels
        n_levels = len(hierarchy)
        M_prev = ind_medoids
        C_prev = ind_clusters

        embedding_h = manifold.MDS( n_components=2, dissimilarity="precomputed", n_init=n_init_mds_anchor,
                                    max_iter=max_iter_anchor, n_jobs=n_jobs_anchor, verbose=verbose_anchor )                       
        H = {}                  
        C_hierarchy = {}
        T_hierarchy = {i:{} for i in range(0, n_clusters)}
        for level in range(1, n_levels):
            if self.verbose:
                print("")
                print( '\rHierarchy level %i (%i ---> %i clusters)' % (level-1, hierarchy[level-1], 
                        hierarchy[level]) )
#           Check the data reorganization needed for this new hierarchy level
            if hierarchy[level] > 1:
#               Assign the clusters of previous level to the current ones
                D = self.dist_matrix[np.ix_(M_prev, M_prev)]
                M, C = optim_kmedoids( D, hierarchy[level], incoherence="rel", n_iter=500, init_Ms="isolated",
                                       n_iso=hierarchy[level], verbose=self.verbose )[:2]
#               Obtain a dictionary with all the indexes of each new cluster
                C_new = { newcl: np.concatenate( [C_prev[i] for i in C[newcl]] ) 
                                                            for newcl in range(0, hierarchy[level]) }
                C_hierarchy[level] = C_new
            elif hierarchy[level] == 1:
#               No clustering (consider all data points)
                C = { 0: np.arange(hierarchy[level-1]) }
                C_new = { 0: np.arange(len(self.dist_matrix)) }
            else:
                raise Exception("There is a wrong entry in the hierarchy parameter, it must have \
                                 non-zero integers only (e.g. hierarchy=[8,1])")
            if level == 1:
                H[level-1] = {i: np.array([i]) for i in range(0, n_clusters)}
            H[level] = {i: np.concatenate( ([H[level-1][j] for j in C[i]]) ) for i in C}

#           Obtention of anchor points (consider anchor points AND medoids separately)
            mds_M_prev = mds_clusters[M_prev]
            mds_A = []
            ind_A = []
            for i in range(0, hierarchy[level-1]):
                if self.verbose:
                    sys.stdout.write( '\rObtaining anchor points:%6.1f%%' 
                                      % (float(i)*100./float(hierarchy[level-1])) )
                    sys.stdout.flush()
                if len(C_prev[i]) <= 3:
                    mds_A.append( mds_clusters[C_prev[i],:] )
                    ind_A.append( C_prev[i] )
                elif len(C_prev[i]) == 4:
                    mds = mds_clusters[C_prev[i],:]
                    h = ConvexHull(mds)
                    mds_A.append( mds[h.vertices] )
                    ind_A.append( C_prev[i][h.vertices] )
                else:
#                   Exclude the medoid as a possible anchor point
                    C_prev_nomed = np.setdiff1d(C_prev[i], M_prev[i])
#                   Choose the procedure for the anchor points calculations depending on cluster length
                    if len(C_prev[i]) - 1 < 20:
                        method_anchor = None
                    else:                                                       
                        method_anchor = "optimized"
#                   MDS and indexes of anchor points in previous level
                    mds = anchor_points( n_anchor, mds_clusters[C_prev_nomed,:], method=method_anchor, 
                                         criterion=criterion_anchor )
                    indexes = np.array([np.where(mds_clusters == mds[j])[0][0] for j in range(0, len(mds))])
#                   Order of the anchor points on the previous level
                    h = ConvexHull(mds)
                    mds_A.append( mds[h.vertices] )
                    ind_A.append( indexes[h.vertices] )
            if self.verbose:
                sys.stdout.write('\rObtaining anchor points:%6.1f%%' % 100. )
                sys.stdout.flush()
                print("")
                               
#           MDS of anchor points on the new level and their transformations
            self.sparse_coordinates = np.zeros((len(self.dist_matrix), 2))
            for newcl in range(0, hierarchy[level]):
                if self.verbose:
                    print( '\rResult for new cluster %i' % newcl  )
                temp_A = [ind_A[i] for i in C[newcl]] 
                A = np.concatenate( temp_A ).astype('int32')
                if len(A) == 1:
                    self.sparse_coordinates[A,:] = np.zeros((1,2)) # avoid sklearn RuntimeWarning   
                    self.order_anchor = [0]
                    self.transformation = [0]
                else:
                    dist_anchor = self.dist_matrix[np.ix_(A, A)]
                    if hasattr(self, 'descriptor_type') and eta != 0:
#                       Distances can have an additional weigth using medoids kernel
                        weight_med = np.eye(len(A))
                        l=0
                        new_ind=[]
                        for a in temp_A:
                            new_ind.append(np.arange(l, l+len(a), 1))
                            l+=len(a)
                        for i in range(0, len(temp_A)):
                            weight_med[np.ix_(new_ind[i], new_ind[i])] = 1
                            for j in range(i+1, len(temp_A)):
                                med_i = M_prev[C[newcl][i]]
                                med_j = M_prev[C[newcl][j]]
                                prod = np.dot(self.descriptor[med_i], self.descriptor[med_j])**self.zeta
                                weight_med[np.ix_(new_ind[i], new_ind[j])] = prod
                                weight_med[np.ix_(new_ind[j], new_ind[i])] = prod
                        dist_anchor = np.sqrt(1 + (dist_anchor**2 -1)*weight_med**eta)

                    mds_anchor = embedding_h.fit_transform(dist_anchor)
                    self.MDS_stress = embedding_h.stress_
#                   Convexity check per cluster for their new MDS
                    embedding_h.set_params(n_init=1)
                    prev_clusters = [C_prev[i] for i in C[newcl]]
                    self.convexity_check( C[newcl], prev_clusters, temp_A, dist_anchor, mds_anchor,
                                          embedding_h, precision = precision_qhull ) 
                    embedding_h.set_params(n_init=n_init_mds_anchor)
#                   Transformation from previous level to the new one
                    self.transform_2d( C[newcl], prev_clusters, temp_A, mds_clusters )

                for i, cl in enumerate(C[newcl]):
                    for j in H[level-1][cl]:
                        T_hierarchy[j].setdefault(level,{})["cluster"] = newcl
                        T_hierarchy[j].setdefault(level,{})["anchor"] = ind_A[cl][self.order_anchor[i]]
                        T_hierarchy[j].setdefault(level,{})["transf"] = self.transformation[i]
                
            if hierarchy[level] > 1:
#               Reassign the label "previous" to the new results
                M_prev = M_prev[M]
                C_prev = C_new
                mds_clusters = self.sparse_coordinates

#       These indices refer to the dist_matrix; we need to make sure that the information required to retrieve  <-- comment
#       the atomic structures from the original data base are consistent with the sparsification technique used <-- comment
#       We should also give the option to output the mds coordinates to the xyz file and generate carved xyz    <-- comment
#       structures around the medoids for plotting                                                              <-- comment
        self.has_clmds = True
        self.sparse_clusters = ind_clusters
        self.sparse_medoids = ind_medoids.astype(int)
        self.all_transformations = T_hierarchy

        sparse_cluster_indices = np.empty(len(self.dist_matrix), dtype=int)
        for i in range(0, hierarchy[0]):
             cluster = self.sparse_clusters[i]
             sparse_cluster_indices[cluster] = i

        self.sparse_cluster_indices = sparse_cluster_indices

        sparse_int_cluster_indices = {}
        if n_levels > 2:
            for level in range(1, n_levels-1):
                int_indices = np.empty(len(self.dist_matrix), dtype=int)
                for i in range(0, hierarchy[level]):
                    cluster = C_hierarchy[level][i]
                    int_indices[cluster] = i                    
                sparse_int_cluster_indices[level] = int_indices

            self.sparse_int_cluster_indices = sparse_int_cluster_indices



#   This method checks the presence of pathological arrangements of anchor points in the MDS (i.e. non-convex
#   and self-intersecting results) and improves the final MDS solution (free of pathologies)
    def convexity_check(self, clusters, prev_clusters, ind_anchor, dist_anchor, 
                        mds_anchor, embedding, max_perm=6, precision=None): 
        no_pathologies = 0
        final_vertices = []
        N_anchor = np.zeros((len(clusters)+1, ), dtype=int)
        if precision:
            precision = 'E' + str(precision)
        else:
            precision = 'QbB'
        for i in range(0, len(clusters)):
            if self.verbose:
                sys.stdout.write( '\rChecking convexity:%6.1f%%' % (float(i)*100./float(len(clusters)))  )
                sys.stdout.flush()
            N_anchor[i+1] = N_anchor[i] + len(ind_anchor[i])
            if len(prev_clusters[i]) <= 3:
                final_vertices.append(np.arange(0, len(prev_clusters[i]), 1))
                no_pathologies += 1
                continue
#           Check if there is a pathological quadrilateral (non-convex)
            hull = ConvexHull( mds_anchor[N_anchor[i]:N_anchor[i+1], :], qhull_options=precision )
            vertices = hull.vertices
            if len(vertices) == 3:
                final_vertices.append(vertices)
                no_pathologies += 0.5
            elif len(vertices) == 4:
#               Check if there is a pathological quadrilateral (self-intersecting)
                go_ahead = checkpermutation( ind_anchor[i][vertices], ind_anchor[i], verbose=False )
                if go_ahead:
                    final_vertices.append(vertices)
                    no_pathologies += 1
                else:
                    n_perm = 0       
                    temp_vertices = np.copy(vertices)
                    temp_mds = np.copy(mds_anchor)
                    while not go_ahead and (n_perm != max_perm):
#                       We need to permute the anchor coordinates for this cluster
                        perm_cluster = temp_mds[N_anchor[i]:N_anchor[i+1], :]
                        n_cycle = np.where(vertices == 0)[0][0]
                        perm_0 = np.roll(vertices, -n_cycle)
                        if (perm_0[1] == 2) & (n_perm <= 3): 
                            n_perm+=1
                            perm_cluster[[0,3]] = perm_cluster[[3,0]]
                        elif (perm_0[1] == 2): 
                            n_perm+=1
                            perm_cluster[[1,2]] = perm_cluster[[2,1]]
                        else:
                            n_perm+=1
                            perm_cluster[[0,1]] = perm_cluster[[1,0]]
                        init_embed = temp_mds
                        init_embed[N_anchor[i]:N_anchor[i+1], :] = perm_cluster
                        new_embed = embedding.fit(dist_anchor, init=init_embed)
                        temp_mds = new_embed.embedding_
                        hull = ConvexHull( temp_mds[N_anchor[i]:N_anchor[i+1], :], qhull_options=precision )
                        temp_vertices = hull.vertices
#                       Check if the convex hull is a triangle now
                        if len(temp_vertices) == 3:
                            go_ahead = True
                            temp_pathologies = 0.5
                        elif len(temp_vertices) == 4:
                            go_ahead = checkpermutation( ind_anchor[i][temp_vertices], 
                                                         ind_anchor[i], verbose=False )        
                            temp_pathologies = 1
                        else: 
#                           Improve this error message                                                          <-- check this
                            raise Warning("This is a pathological choice of anchor points, check cluster ", i) 
                    if n_perm == max_perm:
#                       We reached the maximum number of permutations
#                       Improve this choice (maybe triangle with maximum area in MDS local?)                    <-- comment
#                       BE CAREFUL with the MDS (do I need to remove the 4th vertex?)                           <-- comment
                        final_vertices.append(vertices[:3]) 
#                       Improve this print                                                                      <-- comment
                        if self.verbose:
                            print("\rSelf-intersecting quadrilateral on cluster %i, a linear transformation \
                                     will be used instead with anchor points " % i, final_vertices[i])
                    else:
#                       The current cluster is now non-pathological
#                       Check the effects of the new MDS on the convexity of the previous clusters
                        new_vertices = []
                        for j in range(0, i):
                            if len(ind_anchor[j]) <= 3:
                                temp_pathologies += 1
                                new_vertices.append(final_vertices[j])
                            else:
                                h = ConvexHull( temp_mds[N_anchor[j]:N_anchor[j+1], :], qhull_options=precision )
                                if len(h.vertices) == 3:
                                    temp_pathologies += 0.5
                                    new_vertices.append(h.vertices)
                                else:
                                    if checkpermutation( ind_anchor[j][h.vertices], ind_anchor[j], 
                                                         verbose=False ):
                                        temp_pathologies += 1
                                        new_vertices.append(h.vertices)
                                    else:
                                        new_vertices.append(h.vertices[:3])
#                                       Improve this print                                                       <-- comment
                                        if self.verbose:
                                            print("\rSelf-intersecting quadrilateral on cluster %i, a linear \
                                                     transf. will be used instead with anchor points " % j, 
                                                   ind_anchor[j][h.vertices[:3]])
                        if temp_pathologies > no_pathologies:
                            no_pathologies = temp_pathologies
                            mds_anchor = temp_mds
                            final_vertices = new_vertices + [temp_vertices]
                            self.MDS_stress = embedding.stress_
                        else:
#                           We keep the previous MDS
#                           Improve this choice (maybe triangle with maximum area in MDS local?)                 <-- comment
#                           BE CAREFUL with the MDS (do I need to remove the 4th vertex?)                        <-- comment
                            final_vertices.append(vertices[:3])             
            else:
#               We have a problem here (the code is assigning a wrong number of anchor points or its 
#               global MDS is a perfect line)                                                               
#               Improve this error message                                                                       <-- check this
                raise Warning("This is a highly pathological choice of anchor points, check cluster ", i) 
        if self.verbose:
            sys.stdout.write( '\rChecking convexity:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")

        self.final_n_anchor = N_anchor 
        self.order_anchor = final_vertices
        self.mds_anchor = mds_anchor


#   This method transforms each cluster from one 2D space (previous) to another (new)
#   The choice of transformation (linear or homography) depends on the anchor points given for each cluster
    def transform_2d(self, clusters, prev_clusters, ind_anchor, mds_clusters):
        N_anchor = self.final_n_anchor
        self.transformation = []
        for i in range(0, len(clusters)):
            if self.verbose:
                sys.stdout.write( '\rPerforming transformations:%6.1f%%' % (float(i)*100./float(len(clusters))) )
                sys.stdout.flush()
#           CASE 1: cluster with 1 anchor point (translation)
            if len(self.order_anchor[i]) == 1:
                self.sparse_coordinates[ind_anchor[i],:] = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :]
                self.transformation.append( [0] )
                continue
            indexes = ind_anchor[i][self.order_anchor[i]]
            X_prev = mds_clusters[indexes,:]
            X_new = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :][self.order_anchor[i]]
            diff_X_prev = X_prev - X_prev[1,:]
            diff_X_new = X_new - X_new[1,:]
#           CASE 2: cluster with 2 or 3 anchor points (linear transformation + regularization)
            if len(self.order_anchor[i]) in [2,3]:
                T = np.linalg.lstsq(diff_X_prev, diff_X_new, rcond=None )[0]
                self.transformation.append(T)
                if len(prev_clusters[i]) in [2,3]:
#                   Small clusters
                    self.sparse_coordinates[ind_anchor[i],:] = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :]
                else:
#                   Transform and translate each cluster to the origin of its transf. matrix T
                    product = np.dot(mds_clusters[prev_clusters[i], :] - X_prev[1,:], T)
                    self.sparse_coordinates[prev_clusters[i], :] = product + X_new[1,:]
#           CASE 3: cluster with 4 non-pathological anchor points (homography transf.)
            elif len(self.order_anchor[i]) == 4:
                axis = np.array([[1,0],[0,0],[0,1]])     
                T_prev = np.linalg.lstsq(diff_X_prev[:3,:], axis, rcond=None)[0]
                T_new = np.linalg.lstsq(diff_X_new[:3,:], axis, rcond=None)[0]
                T_new_inv = np.linalg.lstsq(axis, diff_X_new[:3,:], rcond=None)[0]
                a, b = np.dot(diff_X_prev[-1,:], T_prev)
                c, d = np.dot(diff_X_new[-1,:], T_new)
                s = a + b - 1
                t = c + d - 1
#               Decide if we should keep these warnings                                                         <-- comment 
                try:
                    assert (s > 0) & (t > 0)    
                except:
                    raise Warning('The anchor points of cluster %i form a non-convex quadrilateral' % i)
                F = np.array([[b*c*s,     0, b*(c*s - a*t)],
                              [    0, a*d*s, a*(d*s - b*t)],
                              [    0,     0,         a*b*t]])
                self.transformation.append( [X_prev[1,:], T_prev, F, T_new_inv, X_new[1,:]] )
                if len(prev_clusters[i]) == 4:
#                   Small clusters
                    self.sparse_coordinates[ind_anchor[i],:] = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :]
                else:
                    transf_prev = np.dot(mds_clusters[prev_clusters[i]]- X_prev[1,:], T_prev)
                    transf_prev_homog = np.concatenate((transf_prev, np.ones((len(transf_prev),1))), axis=1)
                    perspective_homog =  np.dot(transf_prev_homog, F)
                    perspective = perspective_homog/perspective_homog[:,-1][:,None] 
                    self.sparse_coordinates[prev_clusters[i]] = np.dot(perspective[:,:2], T_new_inv) + X_new[1,:]
            else:
                raise Warning("There must be something wrong, check the list of anchor points: ", ind_anchor)

        if self.verbose:
            sys.stdout.write( '\rPerforming transformations:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")            


#   This is a user friendly function that returns the clusters and medoids of the sparse set
    def get_sparse_coordinates(self, hierarchy):
        if not self.has_clmds:
            self.cluster_MDS(hierarchy = hierarchy)

        ext_coordinates = np.empty([self.n_env,3])
        ext_coordinates[0:self.n_env, 0:2] = self.sparse_coordinates
        ext_coordinates[0:self.n_env, 2] = self.sparse_cluster_indices

        return ext_coordinates


#   This is a user friendly function that returns the anchor points, the cluster or/and 
#   the transformations in each level of the hierarchy
    def extract_transf_info(self, info=None):
#       Extract specific information ("cluster", "anchor", "transf")
        if info: 
            n_levels = len(self.hierarchy)
            I = {}
            for level in range(1, n_levels):
                I[level] = {j: [] for j in range(0, self.hierarchy[level])}
                for i in range(0, self.hierarchy[0]):                        
                    cl = self.all_transformations[i][level]["cluster"]
                    if info == "cluster":
                        I[level][cl].append( i )
                    else:
                        temp = self.all_transformations[i][level][info]
                        if np.array_equal(temp, np.eye(2)):
                            I[level][cl].append(temp)
                            continue
                        already_there = False
                        for k in I[level][cl]:
                            if type(k) == type(temp) and len(k) == len(temp):
                                if isinstance(temp, np.ndarray) and np.allclose(temp, k):
                                    already_there = True
                                    break
                                elif isinstance(temp, list) and next( np.allclose(temp[j], k[j]) for j in range(0, 3) ):
                                    already_there = True  
                                    break   
                        if not already_there:
                            I[level][cl].append(temp)
            return I
#       Extract ALL the information in different arrays
        H = self.extract_transf_info(info="cluster") 
        A = self.extract_transf_info(info="anchor") 
        T = self.extract_transf_info(info="transf") 
        return H, A, T




#   This is a user friendly function that returns the clusters and medoids of the complete set
#   or of the specified indices
    def get_all_coordinates(self, hierarchy, sparse_info=False, precision=1e-8, reg_param=0.001, indices=None):
        """
        This function calls compute_estim_coordinates() to obtain an estimation for all the points
        in the dataset (default, indices=None) or for a given array of indices.
        If already computed, you can still use it to get the array of coordinates and clusters.
        """
        if hasattr(self, 'hierarchy'):
            if hierarchy != self.hierarchy:
                print("Warning: You provided a different hierarchy before, ", self.hierarchy)
                print("It will be used instead of the given one, please check before next time")  
                hierarchy = self.hierarchy

        if not self.has_clmds:
            self.cluster_MDS(hierarchy = hierarchy)

        if not self.has_estimation:
            self.compute_estim_coordinates(precision=precision, reg_param=reg_param, indices=indices)

        ext_coordinates = np.empty([self.all_env,3])
        ext_coordinates[0:self.all_env, 0:2] = self.all_coordinates
        ext_coordinates[0:self.all_env, 2] = self.all_cluster_indices

        if sparse_info:
            A = self.extract_transf_info(info="anchor")[1][0]
            labels = np.zeros(self.all_env)
            for i, ind in enumerate(self.sparse_list):
                ind_C = self.all_cluster_indices[ind]
                if i in self.sparse_medoids:
                    labels[ind] = 3
                elif i in A[ind_C]:
                    labels[ind] = 2
                else:
                    labels[ind] = 1
            ext_coordinates = np.concatenate((ext_coordinates, labels[:,None]), axis=1)

        return ext_coordinates


#   This method gives a "cheap" estimation of the MDS coordinates of points not included in the sparse set
#   If indices=None (default), the complete database is estimated. Otherwise, please provide an array or list of indices.
    def compute_estim_coordinates(self, precision=1e-8, reg_param=0.001, indices=None):
        hierarchy = self.hierarchy
        sparse_list = self.sparse_list
        all_env = self.all_env
#       Get the descriptors of all the atoms left out of the sparse set or the given ones
        if indices is None:
            indices = [i for i in range(0, all_env) if i not in sparse_list]
        if len(self.descriptor) != all_env:
            sparsify = self.sparsify
            all_descriptor = np.zeros( (all_env, np.shape(self.descriptor)[1]) )
            all_descriptor[sparse_list] = self.descriptor
            self.sparsify = indices
            self.build_descriptor()
            all_descriptor[indices] = self.descriptor
            self.descriptor = all_descriptor
#           Reassign the overwritten attributes
            self.sparsify = sparsify
            self.n_env = len(sparse_list)
            self.sparse_list = list(sparse_list) 
#       Classify them considering the clustering of the sparse set
        cluster_indices = - np.ones(all_env, dtype=int)
        cluster_indices[sparse_list] = self.sparse_cluster_indices
        n_env = len(indices)
        if self.verbose:
            print("")
        for i, ind in enumerate(indices):
            if self.verbose:
                sys.stdout.write('\rAssigning each point to cluster:%6.1f%%' % (float(i)*100./float(n_env)) )
                sys.stdout.flush()
            dist_med = np.zeros(hierarchy[0])
            for j in range(0, len(self.sparse_medoids)):
                med = sparse_list[self.sparse_medoids[j]]
                prod = np.dot(self.descriptor[ind], self.descriptor[med])**self.zeta 
                if prod <= 1. - precision:
                    dist_med[j] =  np.sqrt(1. - prod)
                else:
                    dist_med[j] = 0.
            cluster_indices[ind] = np.argmin(dist_med)           
        if self.verbose:
            sys.stdout.write('\rAssigning each point to cluster:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")

#       Compute the transformation from kernel space to 2D
        local_coordinates = np.zeros((all_env,2))       
        transf_coordinates = np.zeros((all_env,2))
        for i in range(0, hierarchy[0]):
            if self.verbose:
                print("")
#           distance matrix per cluster
            C = np.where(cluster_indices == i)[0] 
            C_sparse = self.sparse_clusters[i]
            dist_cluster = np.empty([len(C), len(C_sparse)])
            for j in range(0, len(C)):
                if self.verbose:
                    sys.stdout.write('\rEmbedding all data (cluster %i):%6.1f%%' % (i, float(j)*100./float(len(C))) )
                    sys.stdout.flush()
                if C[j] in sparse_list:
                    ind_sparse = np.where(sparse_list == C[j])[0]
                    assert ind_sparse in C_sparse
                    dist_cluster[j,:] = self.dist_matrix[ind_sparse, C_sparse]  
                    continue
                for k in range(0, len(C_sparse)):
                    ind_sparse = sparse_list[C_sparse[k]]
                    prod = np.dot(self.descriptor[C[j]], self.descriptor[ind_sparse])**self.zeta
                    if prod <= 1. - precision:
                        dist_cluster[j][k] = np.sqrt(1. - prod)
                    else:
                        dist_cluster[j][k] = 0.          

#           Transformation matrix T from distance space to 2D ( X·T = Y)
            dist_sparse = self.dist_matrix[np.ix_(C_sparse, C_sparse)]
            local_mds_sparse = self.local_sparse_coordinates[C_sparse,:]
            reg = np.ones(len(C_sparse))*reg_param        # regularization
            reg = np.diag(reg)
            T = np.linalg.lstsq(dist_sparse - reg, local_mds_sparse, rcond=None)[0]
            self.all_transformations[i]["dist"] = T
            local_coordinates[C] = np.dot(dist_cluster, T)

#           Transform the coordinates from local to global space 
            ref = self.all_transformations[i][1]["anchor"]
            if len(ref) > 1:
                ref = ref[1]
            local_mds_ref = self.local_sparse_coordinates[ref,:]
            global_mds_ref = self.sparse_coordinates[ref, :]
            transf_coordinates[C] = local_coordinates[C] - local_mds_ref
            for level in range(1, len(hierarchy)):
                T_2d = self.all_transformations[i][level]["transf"]
                if isinstance(T_2d, np.ndarray):
#                   Lineal transformation
                    transf_coordinates[C] = np.dot(transf_coordinates[C], T_2d)         
                elif T_2d == [0]:
#                   Sparse cluster with 1 point (translation)
                    continue
                else:
#                   Homography
                    ref_prev, T_prev, F, T_new_inv, ref_new = T_2d
                    if level > 1:
#                       We need to keep track of the reference anchor point in each step
                        transf_coordinates[C] = transf_coordinates[C] - ref_prev
                    X_prev = np.dot(transf_coordinates[C], T_prev)
                    X_prev_homog = np.concatenate((X_prev, np.ones((len(X_prev),1))), axis=1)
                    X_new_homog =  np.dot(X_prev_homog, F)
                    X_new = X_new_homog/X_new_homog[:,-1][:,None]
                    transf_coordinates[C] = np.dot(X_new[:,:2], T_new_inv)
                    if hierarchy[level] > 1:
#                       We need to keep track of the reference anchor point in each step
                        transf_coordinates[C] = transf_coordinates[C] + ref_new
            transf_coordinates[C] = transf_coordinates[C] + global_mds_ref
            if self.verbose:
                sys.stdout.write('\rEmbedding all data (cluster %i):%6.1f%%' % (i, 100.) )
                sys.stdout.flush()
                print("")
                                           
        self.has_estimation = True
        self.all_local_coordinates = local_coordinates

        self.all_cluster_indices = cluster_indices
        self.all_coordinates = transf_coordinates




#   This method writes an extended xyz file with the cl-MDS coordinates
    def write_xyz(self, filename=None):
        if filename == None:
            raise Exception("You must define a filename to write to disk")

        if self.atoms == None:
            raise Exception("You need to provide an input atoms file if you want to write the coordinates \
                             to an xyz file.")

        if not self.has_clmds:
            raise Exception("You haven't run a cl-MDS coordinate calculation yet!")

        from ase.io import write

        new_atoms = self.atoms.copy()
        n = 0
        for ats in new_atoms:
            natoms = len(ats)
            coords = np.empty([natoms, 2], dtype=float)
            cluster = np.empty(natoms, dtype=int)
            if not self.has_estimation:
                coords[:,:] = np.NaN
                cluster[:] = -1
            for j in range(0, natoms):
                if not self.has_estimation:
                    if n in self.sparse_list:
                        i = self.sparse_list.index(n)
                        coords[j] = self.sparse_coordinates[i]
                        cluster[j] = self.sparse_cluster_indices[i]
                else:
                    coords[j] = self.all_coordinates[n]
                    cluster[j] = self.all_cluster_indices[n]
                n += 1

            ats.new_array("clmds_coords", coords)
            ats.new_array("cluster_number", cluster)

        write(filename, new_atoms)

#   This method exports carved medoid environments to xyz files (ONLY for average_kernel=False)
    def medoids_to_xyz(self, dir=None, carve_radius=None, render=False, bond_cutoff=1.9, gnuplot=False):
        if dir == None:
            raise Exception("You must define a directory to write medoid's xyz files to")

        if self.atoms == None:
            raise Exception("You need to provide an input atoms file if you want to write the coordinates \
                             to an xyz file.")

        if not self.has_clmds:
            raise Exception("You haven't run a cl-MDS coordinate calculation yet!")

        if carve_radius == None:
            cutoff = self.cutoff
        else:
            cutoff = carve_radius

        from ase.io import write
        from ase import Atoms
        import os

        if not os.path.exists(dir):
            os.mkdir(dir)

        n = 0
        for ats in self.atoms:
            natoms = len(ats)
            for j in range(0, natoms):
                if n in self.sparse_list:
                    i = self.sparse_list.index(n)
                    if i in self.sparse_medoids:
                        pos = ats.get_positions()
                        cell = ats.get_cell()
                        new_pos = []
                        neighbors = []
                        site = Atoms()
                        for j2 in range(0, natoms):
                            d = ats.get_distance(j, j2, mic=True)
                            if d < cutoff:
                                neighbors.append(j2)
                                new_pos.append(pos[j2])
                                site += ats[j2]

                        site.set_cell(cell)
                        site.set_positions(new_pos)
                        site.set_pbc(True)
                        shift = np.array([cutoff, cutoff, cutoff]) - pos[j]
                        site.translate(shift); site.wrap()
                        site.set_cell([2.*cutoff, 2.*cutoff, 2.*cutoff])
                        site.set_pbc(False)
                        i2 = np.where(self.sparse_medoids == i)[0][0]
                        write(dir + "/medoid_%i.xyz" % i2, site)
                n += 1

        if render:
            try:
                from ovito.io import import_file
                from ovito.vis import Viewport
#                from ovito.modifiers import CreateBondsModifier
            except:
                raise Exception("You need Ovito (pip3 install ovito) to use the rendering capability")

            for i in range(0, len(self.sparse_medoids)):
                atoms = import_file(dir + "/medoid_%i.xyz" % i)
#               I didn't manage to get bonds to render                                                          <-- comment
#                modifier = CreateBondsModifier(cutoff = bond_cutoff)
#                modifier.vis.enabled = True
#                modifier.vis.width = 0.3
#                atoms.modifiers.append(modifier)
#                atoms.compute()
                atoms.source.data.cell.vis.render_cell = False
                atoms.add_to_scene()
                vp = Viewport()
                vp.type = Viewport.Type.Perspective
                vp.zoom_all()
                vp.render_image(filename=dir+"/medoid_%i.png" % i, size=(400,400), alpha=True)
                atoms.remove_from_scene()

        if gnuplot:
            if not self.has_estimation:
                ext_coords = self.get_sparse_coordinates(hierarchy=self.hierarchy)
            else:
                ext_coords = self.get_all_coordinates(hierarchy=self.hierarchy)
            f = open(dir + "/xy.dat", "w+")
            for i in ext_coords:
                print(i[0], i[1], i[2], file=f)

            f.close()
            f = open(dir + "/gnuplot.script", "w+")
            print("set term pngcairo size 640,640; set output 'clmds_map.png'", file=f)
            print("set size ratio -1", file=f)
            print("set xlabel 'MDS coordinate 1'", file=f)
            print("set ylabel 'MDS coordinate 2'", file=f)
            if render:
                print("plot 'xy.dat' u 1:2:3 lc var pt 7 not, \\", file=f)
                for i in range(0, len(self.sparse_medoids)):
                    x, y = self.sparse_coordinates[self.sparse_medoids[i]]
                    print("     'medoid_" + str(i) + ".png' binary filetype=png dx=0.0005" + \
                          "center=(" + str(x) + "," + str(y) + ") w rgbalpha not, \\", file=f)
            else:
                print("plot 'xy.dat' u 1:2:3 lc var not", file=f)

            f.close()

            try:
                os.system("cd " + dir + "; gnuplot gnuplot.script")
            except:
                print("I tried running gnuplot but it failed!")
#************************************************************************************************************





#************************************************************************************************************
#******************************************* Suporting functions ********************************************

# This method chooses the kmedoids clustering with minimum intra-cluster incoherence (relative or total)
def optim_kmedoids(D, n_clusters, incoherence="rel", n_iter=100,  tmax=100, 
                   init_Ms="random", n_iso=None, verbose=False):
    if verbose:
        print("")
    for t in range(0, n_iter):
        if verbose:
            sys.stdout.write('\rClustering data:%6.1f%%' % (float(t)*100./float(n_iter)) )
            sys.stdout.flush()
        temp_M, temp_C = kmedoids.kMedoids( D, n_clusters, tmax=tmax, 
                                            init_Ms=init_Ms, n_iso=n_iso )
#       Relative intra-cluster incoherence
        temp_I = 0.
        for i in range(0, n_clusters):
            if incoherence == "rel":
                temp_I += np.sum(D[temp_M[i]][temp_C[i]])/len(temp_C[i])
            elif incoherence == "tot":
                temp_I += np.sum(D[temp_M[i]][temp_C[i]])
            else:
                raise Exception("Wrong incoherence option, choose between: rel, tot")
#       Minimize this value
        if (t == 0) or (temp_I <= I):
            M, C, I = temp_M, temp_C, temp_I
    if verbose:
        sys.stdout.write('\rClustering data:%6.1f%%' % 100. )
        sys.stdout.flush()
        print("")

    return M, C, I


# Given a dataset, this method choose the N points corresponding to the N vertices of the polygon that fulfils  
# the selected criterion (area, number of points)
def anchor_points(N, points, method=None, n_random=None, criterion="area", precision=1.e-8):
    """
    3 available methods: 
        None (default) = use all possible combinations (without repetition) of the points given 
        "optimized" = consider all the combinations of the 30 furthest points in the dataset (or the 70% for small datasets)
        "random" = N random-chosen sequences, where n_random=N (less accurate)

    2 possible criteria:
        "area" (default) = choose the polygon with the largest area 
        "points" = choose the polygon including more data points
    """
#   Check if the number of samples given is enough to build at least 2 N-gons
    if len(points) <= N:
        return points
    s_opt = 0
    anchor_p = points[:N]
    n_comb = comb(len(points), N, exact=True)
    indexes = np.arange(0, len(points),1)
#   Generate vertices and their possible combinations
    precision_opt = 'E'+str(precision)
    h = ConvexHull(points, qhull_options=precision_opt)
    external_ind = h.vertices
    if len(external_ind) <= N:
        return points[external_ind]
    if method == "optimized":
        h = ConvexHull(points, qhull_options=precision_opt)
        external_ind = h.vertices
        while len(external_ind) < 0.7*len(points) and len(external_ind) < 30:
            temp = points[ ~np.isin(indexes, external_ind), :]
            h = ConvexHull(temp, qhull_options=precision_opt)
            temp_vert = [np.where(pt == points)[0][0] for pt in temp[h.vertices,:]]
            external_ind = np.concatenate((external_ind, temp_vert))
        vertices = itertools.combinations(np.sort(external_ind), N)
    elif method == "random":
        n_random = int(n_random)
        vertices = itertools.combinations(indexes, N)
        if n_random < n_comb:
            rand_mask = np.zeros(n_comb, dtype=int)
            rand_vertices = random.sample(range(0, n_comb), n_random)
            rand_mask[rand_vertices] = 1
            vertices = list(itertools.compress(vertices, rand_mask))
        elif not n_random:
            raise Exception("You need to provide a number of random combinations of vertices (n_random)")
    else:
        vertices = itertools.combinations(indexes, N)
#   Obtain the best polygon considering the chosen criterion
    for vert in vertices:
        temp_anchor = points[vert,:]
        if criterion == "area":
            try:
                h = ConvexHull(temp_anchor, qhull_options=precision_opt)
            except:
                continue
            s = h.volume
            temp_anchor = temp_anchor[h.vertices] 
        elif criterion == "points":
            try:
                h = ConvexHull(temp_anchor, qhull_options=precision_opt)
            except:
                continue
            temp_ind = indexes[ ~np.isin(indexes, vert) ]
            other_points = points[temp_ind,:]
            s = points_in_polygon(N, temp_anchor, other_points, qhull_opt=precision_opt)
        else:
            raise Exception("Choose a criterion included in the options: area, points")
        if s > s_opt:
            s_opt = s
            anchor_p = temp_anchor

    return anchor_p


# Computation of the number of points of a given set lying within a polygon with N vertices
def points_in_polygon(N, vertices, other_points, qhull_opt='QbB'):
    s=0
    if N != len(vertices):
        print("You need to provide %i vertices exactly" % N)
    for point in other_points:
        temp = np.concatenate((vertices, point[None,:]), axis=0)
        h = ConvexHull(temp, qhull_options=qhull_opt)
        if len(h.vertices) == N:
            if set(range(0,N)) <= set(h.vertices):  
                s += 1
    return s
    
    
# Find if a set of points is a cyclic permutation or a reflection (or both) of another set
# Should we keep the verbose?                                                                                   <-- comment
def checkpermutation(x, y, verbose=False):
    if len(x) != len(y):
        verdict = False
        if verbose:
            print("The arrays have different length, x can't be a permutation of y")
            print("x: ", x, ", y: ", y)
    else:
        cyclic_group = np.concatenate((y,y), axis=0)
        inv_cyclic_group = cyclic_group[::-1]
        i = np.where(cyclic_group == x[0])[0][0]
        i_inv = np.where(inv_cyclic_group == x[0])[0][0]
        if (cyclic_group[i:i+len(x)] == x).all():
            verdict = True
            if verbose:
                print(x, " is a cyclic permutation of ", y)
        elif (inv_cyclic_group[i_inv:i_inv+len(x)] == x).all():
            verdict = True
            if verbose:
                print(x, " is an inverted cyclic permutation of ", y)
        else:
            verdict = False
    return verdict

#************************************************************************************************************
