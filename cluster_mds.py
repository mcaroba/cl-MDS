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
from sklearn_mds import _mds
import kmedoids as km
from cur import cur
import random
import sys
from scipy import spatial
from fortran.anchor_selection import vertices_module as vmod


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
                 average_kernel=False, cutoff=None, do_species=None, verbose=True):
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
            self.n_env = len(dist_matrix)
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

#       Check if the user only wants specific species to be considered
        if do_species is None:
            self.do_species = True
        else:
            self.do_species = do_species

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
                                sparse_options, " or provide a list of indices")
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
    def build_descriptor(self, zeta=4):
        if not hasattr(self, 'descriptor_type'):
            raise Exception("You must define a descriptor, check the implemented options")

        descriptor = self.descriptor_type
        descriptor_string = self.descriptor_string
        if descriptor in ["quippy_soap", "quippy_soap_turbo"]:
            from ase.data import atomic_numbers
            from quippy.descriptors import Descriptor
            from quippy.convert import ase_to_quip
            self.zeta = zeta
            species_list = []
            for ats in self.atoms:
                for at in ats:
                    species_list.append(at.symbol)
            self.species_list = species_list
            if self.do_species is True:
                species = list(set(species_list)) 
            else:
                species = list(set(self.do_species))
#           This uses some default SOAP parameters
            if descriptor_string is None:
                species_string = ""
                for z in species:
                    species_string += " " + str(atomic_numbers[z])
                n_Z = len(species)
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
#                       Be careful with this allocation of cutoff per specie, it depends on the order of           <-- comment
#                       *species* which isn't preserved from one atoms file to another
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
                    for i, z in enumerate(species):
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
                    if n_Z < len(species):
                        raise Exception("Your database has a different amount of species than the \
                                         given on the descriptor string.")
#                   Take cutoff and average kernel from descriptor string
                    cutoff = self.get_info_string(quippy_string, label="cutoff", type_label=float)
                    average = self.get_info_string(quippy_string, label="average")     
                    self.cutoff = cutoff
                    self.average_kernel = average
                elif descriptor == "quippy_soap_turbo":
#                   Check string
                    if isinstance(quippy_string, str):   
                        Z = self.get_info_string(quippy_string, label="species_Z", type_label=int, 
                                                 is_array=True)
                        n_Z = len(Z) 
                        if len(species) == 1 and n_Z == 1:
                            quippy_string = {species[0]: quippy_string}   
                        else:
                            raise Exception("You need to give as many descriptor strings as n_species for \
                                             %s. Check it corresponds to the species in the database too." 
                                             % descriptor)                        
                    elif isinstance(quippy_string, dict):
                        Z = self.get_info_string(quippy_string[species[0]], label="species_Z", type_label=int, 
                                                 is_array=True)
                        n_Z = len(Z) 
                        if set(quippy_string.keys()) < set(species):
                            raise Exception("You need to give as many descriptor strings as n_species (at least) \
                                             for %s, using as dict. keys the appropriate species." % descriptor) 
                        elif n_Z < len(species):
                            raise Exception("Your database has a different amount of species than the \
                                             given on the descriptor string.")
                    else:
                        Z = self.get_info_string(quippy_string[0], label="species_Z", type_label=int, 
                                                 is_array=True)
                        n_Z = len(Z)  
                        if len(quippy_string) < len(species):
                            raise Exception("You need to give as many descriptor strings as n_species for ",
                                             descriptor) 
                        elif n_Z < len(species):
                            raise Exception("Your database has a lower amount of species than the given \
                                             on the descriptor string.")
                        else:
#                           Check this part (I assume the string always has ordered Z which is not true)           <-- comment
                            indices = [self.get_info_string(quippy_string[i], "central_index", type_label=int)
                                       - 1 for i in range(0, n_Z)]
                            quippy_string = {species[i]: quippy_string[indices[i]] for i in range(0, n_Z) 
                                             if atomic_numbers[species[i]] == Z[i]}
#                   Take cutoff from descriptor string
                    cutoff = []
                    for z in species:
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
            elif self.do_species is True:
#               Added to avoid possible crushes in other methods
                self.sparse_list = list(range(n_env))
            else:
                sparse_list = []
                for z in species:
                    sparse_list += list(np.where( np.array(species_list) == z )[0])
                self.sparse_list = sorted(sparse_list)

#           Descriptors
            if descriptor == "quippy_soap":
                d = Descriptor(quippy_string)
            elif descriptor == "quippy_soap_turbo":
                d = {z: Descriptor(quippy_string[z]) for z in species}
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
                        if species_list[n] not in species:
                            n += 1
                            continue
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
                    N = {z: 0 for z in species}
                    for i, z in enumerate(species):   
                        rcut_hard = max(cutoff[2*i:2*(i+1)])     
                        a = ase_to_quip(ats)
                        a.set_cutoff(rcut_hard)    
                        a.calc_connect()
                        q[z] = d[z].calc_descriptor(a) 
                    ind_ats = []
                    ats_Z = np.array(ats.get_chemical_symbols())
                    for z in species:
                        ind_ats += list(np.where(ats_Z == z)[0])
                    for i in sorted(ind_ats):
                        symb = ats[i].symbol
                        if not self.sparsify_per_cluster:
                            if isinstance(self.sparsify, (list,np.ndarray)) or self.sparsify == "random":
                                if n + i in sparse_list:       
                                    descriptor_list.append(q[symb][N[symb]])  
                            else:    
                                descriptor_list.append(q[symb][N[symb]])                                 
                        else:
                            descriptor_list.append(q[symb][N[symb]]) 
                        N[symb] += 1 
                    n += len(ats)
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



#   Cumbersome code to get a specific label from a descriptor string
    def get_info_string(self, quippy_string, label, type_label=str, is_array=False):
#       Check if additional descriptors are added                                                             <-- comment
        a = quippy_string.split()[0]
        if a != self.descriptor_type[7:]:
            raise Exception("The descriptor string doesn't correspond to the descriptor type, check this.")
        N = len(label)
        a = quippy_string.split('=')
        for i in range(0, len(a)):
            if label in a[i]:
                b = a[i+1]
                if is_array:
                    c = ( ( b.split('}')[0] ).split('{')[1] ).split()
                    param = np.array(c).astype(type_label)
                else:
                    param = type_label( b.split()[0] )
                break
        else:
            param = None

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
        dist_matrix, ind_dist, ind_dist_inv = remove_zero_entries(self.dist_matrix, return_indices=True)

        sparse_list = self.sparse_list
#       Compute the clustering with the initial sparse set (considering only unique entries)
        M, C = km.kMedoids( dist_matrix, n_clusters, incoherence="rel", n_inits=iter_med, 
                            tmax=tmax, init_Ms=init_medoids, n_iso=n_iso_med, verbosity=self.verbose)
#       Change this part (do not take into account the repeated entries to add new sparse entries)            <-- comment
        n_C = []
        C_incomplete = []
        C_indices = np.zeros(self.n_env, dtype=int)
        for i in range(0, n_clusters):
            all_Ci = np.concatenate(( [np.where(ind_dist == j)[0] for j in C[i]] ))
            n_C.append(len(all_Ci))
            C_indices[all_Ci] = i
            if n_C[i] < n_sparse:
                C_incomplete.append(i)
        if C_incomplete:
            non_sparse_list = [i for i in range(0, self.all_env) if i not in sparse_list]
            np.random.shuffle(non_sparse_list)
            for i in non_sparse_list:
#               Assign each non_sparse point to a cluster
                dist_med = np.zeros(n_clusters)
                for j in range(0, n_clusters):
                    med = sparse_list[np.unique(ind_dist)][M[j]]
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
            self.n_env = len(sparse_list)
            self.sparse_list = sorted(sparse_list)
            self.n_sparse = n_sparse
            self.build_dist_matrix()
            dist_matrix, ind_dist, ind_dist_inv = remove_zero_entries(self.dist_matrix, return_indices=True)
            new_indices = new_indices[ind_dist]
            M = np.array([np.where(new_indices == med)[0][0] for med in M])
#           Change this part                                                                              <-- comment
            C = {i: np.where(C_indices == i)[0] for i in range(0, n_clusters) }            

        else:
            print("The sparse set already fulfils your n_sparse request.")  

        return M, C



#   Method that prints a small analysis of intra-cluster incoherence and silhouette values for   
#   a given dataset, range of number of clusters and k-medoids parameters (no plot, only values)                                  
#   Improve how are shown the results                                                                     <-- comment 
    def clustering_analysis(self, n_cluster_min=2, n_cluster_max=10, specific_Ms="isolated",
                            specific_n_iso=None, I_min="tot", n_ranking=15):
        if not self.has_dist_matrix:
            self.build_dist_matrix()
        dist_matrix, ind_dist = remove_zero_entries(self.dist_matrix)

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

        C_ind = np.empty( len(dist_matrix), dtype=int )
        I_rel = 10*np.ones( (len(N_cl), len(param_iso)) )
        I_tot = np.ones( (len(N_cl), len(param_iso)) )*10**4
        average_sil = -10*np.ones( (len(N_cl), len(param_iso)) )
        for n in N_cl:
            ind = n - n_cluster_min
            for m, param in enumerate(param_iso):
                if param > n:
                    break
                M, C = km.kMedoids(dist_matrix, n, incoherence="rel", init_Ms=specific_Ms, 
                                   n_iso=param, n_inits=100)
                I_rel[ind,m] = km.cluster_incoherence(dist_matrix, M, C)
                I_tot[ind,m] = 0.
                for i in range(0, n):
                    C_ind[C[i]] = i
                    I_tot[ind,m] += np.sum(dist_matrix[M[i]][C[i]])
                average_sil[ind,m] = silhouette_score(dist_matrix, C_ind, metric="precomputed")
                sil_values = silhouette_samples(dist_matrix, C_ind, metric="precomputed")
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
                    n_init_mds_cluster=100, max_iter_cluster=300, n_jobs_cluster=1, verbose_cluster=0,
                    param_anchor=[70,80,90], n_init_mds_anchor=3500, max_iter_anchor=300, 
                    n_jobs_anchor=1, verbose_anchor=0, weight_cluster_mds=10, weight_anchor_mds=None,
                    eta=0., precision_qhull=1e-7):
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

        * Embedding (n_init_*, max_iter_*, n_jobs_*, verbose_*, weight_*_mds; with 
                     *=(initial) clusters, anchor (points))
        Check sklearn_mds and sklearn.manifold.MDS for additional information

        * Anchor points (param_anchor)
        Use param_anchor=[p1,p2,p3] tu customize the percentiles for the different cluster sizes: 
        1) 149 or less points, 2) from 150 to 1000 points, 3) else.
        Check anchor_points_ndim (from cluster_mds) for further information.

        * Kernel weight for the distance matrix (eta)

        * Transformations (precision_qhull)
        Check self.convexity_check (from cluster_mds) and scipy.spatial.ConvexHull for further 
        information.

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
            dist_matrix, ind_dist, ind_dist_inv = remove_zero_entries(self.dist_matrix, return_indices=True)
            ind_medoids, ind_clusters = km.kMedoids( dist_matrix, n_clusters, incoherence="rel",
                                                     n_inits=iter_med, tmax=tmax, init_Ms=init_medoids,
                                                     n_iso=n_iso_med, verbosity=self.verbose )
        else: 
            ind_medoids, ind_clusters = cluster_sparsification(n_clusters, init_medoids=init_medoids,
                                                               n_iso_med=n_iso_med, iter_med=iter_med,
                                                               tmax=tmax)
#           Maybe return the unique matrix directly from cluster_sparsification                              <-- comment
            dist_matrix, ind_dist, ind_dist_inv = remove_zero_entries(self.dist_matrix, return_indices=True)

        ind_medoids = np.array(ind_medoids)
        dist_clusters = [dist_matrix[np.ix_(ind_clusters[i], ind_clusters[i])] for i in range(0, n_clusters)]

#       Anchor points and MDS embedding of the initial clusters
        if len(param_anchor) != 3:
            param_anchor = [70, 80, 90]
            print("You didn't provide an array of 3 percentiles, so the default param_anchor" +
                  " will be used instead: ", param_anchor)
        precision = 'E' + str(precision_qhull)
        embedding_cl = _mds.MDS( n_components=2, dissimilarity="precomputed", n_init=n_init_mds_cluster,
                                 max_iter=max_iter_cluster, n_jobs=n_jobs_cluster, verbose=verbose_cluster )
        pot_indices = {}
        mds_clusters = np.zeros((len(dist_matrix), 2))
        mds_A = []
        ind_A = []
        for i in range(0, hierarchy[0]):
            if self.verbose:
                sys.stdout.write( '\rMDS embedding per initial cluster:%6.1f%%' 
                                  % (float(i)*100./float(hierarchy[0])) )
                sys.stdout.flush()
            L = len(ind_clusters[i])
            if L <= 4:
                pot_indices[i] = np.arange(0, L, 1)
                if L == 1:
                    mds_clusters[ind_clusters[i]] = np.zeros((1,2)) # avoid sklearn RuntimeWarning
                else:
                    mds_clusters[ind_clusters[i]] = embedding_cl.fit_transform(dist_clusters[i])
                mds_A.append( mds_clusters[ind_clusters[i],:] )
                ind_A.append( np.array(ind_clusters[i]) )
                continue
            if L - 1 < 70:
                M = np.where(ind_medoids[i] == ind_clusters[i])[0][0]
                pot_indices[i] = np.setdiff1d( np.arange(0, L, 1), [M] )
                param_method = 0
            else:
                if L - 1 < 150:
                    param_method = param_anchor[0]
                elif L - 1 < 1000:
                    param_method = param_anchor[1]
                else:
                    param_method = param_anchor[2]
                M = np.where(ind_medoids[i] == ind_clusters[i])[0][0]
                pot_indices[i] = select_further_points(dist_clusters[i], M, 90, n_max=500)
            ind_anchor = anchor_points_ndim(4, dist_clusters[i], method="percentile", 
                                            param_method=param_method, ref_point=M)
            ind_anchor = ind_clusters[i][ind_anchor]
            W_init = np.ones( dist_clusters[i].shape )
            if L > 4:
                ind_ref = list(ind_anchor) + [ind_medoids[i]]
                temp = [np.where(ind_clusters[i] == j)[0][0] for j in ind_ref]
                W_init[temp] = weight_cluster_mds
                W_init[:,temp] = weight_cluster_mds
                W_init[np.ix_(temp, temp)] = weight_cluster_mds**2
            mds_clusters[ind_clusters[i]] = embedding_cl.fit_transform(dist_clusters[i], weights=W_init)
            h = spatial.ConvexHull( mds_clusters[ind_anchor], qhull_options=precision )
            mds_A.append( mds_clusters[ind_anchor][h.vertices] )
            ind_A.append( ind_anchor[h.vertices] )
        if self.verbose:
            sys.stdout.write('\rMDS embedding per initial cluster:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")

        self.local_sparse_coordinates = mds_clusters[ind_dist_inv]


#       Hierarchy levels
        n_levels = len(hierarchy)
        M_prev = ind_medoids
        C_prev = ind_clusters

        embedding_h = _mds.MDS( n_components=2, dissimilarity="precomputed", n_init=n_init_mds_anchor,
                                    max_iter=max_iter_anchor, n_jobs=n_jobs_anchor, verbose=verbose_anchor )                       
        H = {}
        H[0] = {i: np.array([i]) for i in range(0, n_clusters)}
        T_hierarchy = {i:{} for i in range(0, n_clusters)}
        ind_anchor = {0: ind_A}
        A = np.concatenate((ind_A)).astype('int32')
        all_M = {0: ind_medoids}
        for level in range(1, n_levels-1):
            if self.verbose:
                print("")
                print( '\rHierarchy level %i (%i ---> %i clusters)' % (level-1, hierarchy[level-1], 
                        hierarchy[level]) )
#           Check the data reorganization needed for this new hierarchy level
            if hierarchy[level] > 1:
#               Assign the clusters of previous level to the current ones
                D = dist_matrix[np.ix_(M_prev, M_prev)]
                M, C = km.kMedoids( D, hierarchy[level], incoherence="tot", init_Ms="isolated",
                                       n_iso=hierarchy[level], n_inits=500, verbosity=self.verbose )
#               Obtain a dictionary with all the indices of each new cluster
                C_new = {i: np.concatenate( [C_prev[j] for j in C[i]] ) for i in C}
            elif hierarchy[level] == 1:
#               No clustering (consider all data points)
                D = dist_matrix[np.ix_(M_prev, M_prev)]
                M, C = km.kMedoids( D, 1, incoherence="tot", n_inits=1, verbosity=self.verbose )
                C_new = {0: np.arange(len(dist_matrix))}
            else:
                raise Exception("There is a wrong entry in the hierarchy parameter, it must have \
                                 non-zero integers only (e.g. hierarchy=[8,1])")
            H[level] = {i: np.concatenate( ([H[level-1][j] for j in C[i]]) ) for i in C}
            all_M[level] = M_prev[M]

#           Anchor points of current hierarchy level
            temp_A = []
            self.sparse_coordinates = np.zeros((len(dist_matrix), 2))
            for newcl in range(0, hierarchy[level]):
                if self.verbose:
                    print("")
                    print( '\rResult for new cluster %i' % newcl  )
#               Obtain the anchor points for the new level
                if len(C[newcl]) == 1:
                    ind_cl = H[level][newcl][0]
                    temp_A.append( ind_anchor[0][ind_cl] )
                else:
#                   Be careful, the (dis)order of C_prev[i] is important here
                    indices = []
                    l = 0
                    for i in H[level][newcl]:
                        indices.append( pot_indices[i] + l )
                        l += len(ind_clusters[i])
                    indices = np.concatenate( indices ).astype('int32')
                    D = dist_matrix[np.ix_(C_new[newcl],C_new[newcl])]
                    ind_A = anchor_points_ndim(4, D, method=indices)
                    ind_A = C_new[newcl][ind_A]
                    temp_A.append( ind_A ) 
            ind_anchor[level] = temp_A
            A = np.concatenate( (A, np.concatenate( (temp_A) )) ).astype('int32')
            if hierarchy[level] > 1:
#               Reassign the label "previous" to the new results
                M_prev = M_prev[M]
                C_prev = C_new

#       MDS of all anchor points and their transformations  
        self.sparse_coordinates = np.zeros((len(dist_matrix), 2))
        if len(np.unique(A)) == 1:
            self.sparse_coordinates[A[0],:] = np.zeros((1,2)) # avoid sklearn RuntimeWarning   
            self.order_anchor = [[0]]
            self.transformation = [[np.zeros(2)]]
            print('Trivial transformation (just 1 anchor point, check the dist. matrix)')
        else:
            dist_anchor = dist_matrix[np.ix_(A, A)]
#           MDS weight for intracluster distances
            W = weight_anchor_mds
            if W is not None:
                if isinstance(W, (int, float)):
                    w = [W]*n_levels
                elif isinstance(W, (list, np.ndarray)):
                    if len(W) == 1:
                        w = np.repeat(W, n_levels)
                    elif len(W) < n_levels:
                        w = np.concatenate( (W, np.ones(n_levels - len(W))) )
                elif isinstance(weight_anchor_mds, dict):
#                   Nested dict. such as { 0:{ cl0:w0, cl1:w1, ..., 'others':w }, ..., n_levels:{ 0: ... } }
#                   where cl0, cl1, ... are specific clusters with different weights w1, w2, ... from w
                    w = W
                W = np.ones(dist_anchor.shape)
                l=0
                for level in range(0, n_levels-1):
                    for n, a in enumerate(ind_anchor[level]):
                        I = np.arange(l, l+len(a), 1)
                        if isinstance(w, dict) and n in w[level].keys():
                            if not isinstance(w[level][n], (int, float)):
                                print('Cluster %i (level %i) weight is not a float/int, 1 will be used instead'
                                       % (n, level))
                                W[np.ix_(I,I)] = 1.
                            else:
                                W[np.ix_(I,I)] = w[level][n]
                        elif isinstance(w, dict):
                            W[np.ix_(I,I)] = w[level]['others']
                        else:
                            W[np.ix_(I,I)] = w[level]
                        l += len(a)
#               Additional weight using medoids kernel for intercluster distances
            if hasattr(self, 'descriptor_type') and eta != 0:
                weight_med = np.ones((len(A), len(A)))
                l=0
                indices = {}
                for level in range(0, n_levels-1):
                    eta_max = np.sum(eta[:level+1])
                    indices[level] = []
                    for a in ind_anchor[level]:
                        indices[level].append(np.arange(l, l+len(a), 1))
                        l+=len(a)
                    print('Level: ', level)
                    for i in range(0, len(ind_anchor[level])):
                        med_i = all_M[level][i]
                        for j in range(i+1, len(ind_anchor[level])):
                            med_j = all_M[level][j]
                            prod = np.dot(self.descriptor[med_i], self.descriptor[med_j])**self.zeta
                            w = prod**eta[level]
                            if w < 1.e-15:
                                w = 0.
                            weight_med[np.ix_(indices[level][i], indices[level][j])] = w
                            weight_med[np.ix_(indices[level][j], indices[level][i])] = w
                        for level_k in range(0, level):
                            for k in range(0, len(ind_anchor[level_k])):
                                if set(H[level_k][k]) <= set(H[level][i]):
                                    continue
                                med_k = all_M[level_k][k]
                                prod = np.dot(self.descriptor[med_i], self.descriptor[med_k])**self.zeta
                                w = prod**eta[level]  #( eta_max - np.sum(eta[:level_k]) )
                                if w < 1.e-15:
                                    w = 0.
                                weight_med[np.ix_(indices[level][i], indices[level_k][k])] = w
                                weight_med[np.ix_(indices[level_k][k], indices[level][i])] = w
                print('Medoids weight: ', weight_med.shape)
                dist_anchor = np.sqrt(1 + (dist_anchor**2 -1)*weight_med)

            mds_anchor = embedding_h.fit_transform(dist_anchor, weights=W)
            self.MDS_stress = embedding_h.stress_
#           Convexity check per cluster for their new MDS
            embedding_h.set_params(n_init=1)
            prev_clusters = [ind_clusters[i] for i in range(0, hierarchy[0])]
            self.convexity_check( n_clusters, ind_anchor[0], dist_anchor, mds_anchor, embedding_h,
                                  W_mds=W, precision=precision_qhull )
            embedding_h.set_params(n_init=n_init_mds_anchor)
#           Transformation from previous level to the new one
            self.transform_2d( range(0, n_clusters), prev_clusters, ind_anchor[0], mds_clusters )
            for level in range(0, n_levels-1):
                for i, a in enumerate(ind_anchor[level]):
                    if level == 0:
                        temp_anchor = a[self.order_anchor[i]]
                        ind_anchor[level][i] = ind_dist[temp_anchor]
                    else:
                        ind_anchor[level][i] = ind_dist[a]
            for i in ind_clusters:
                T_hierarchy[i].setdefault(0,{})["cluster"] = 0
                T_hierarchy[i].setdefault(0,{})["anchor"] = ind_anchor[0][i]
                T_hierarchy[i].setdefault(0,{})["transf"] = self.transformation[i]   

#       These indices refer to the original dist_matrix; we need to make sure that the information required   <--comment 
#       to retrieve the atomic structures from the original data base are consistent with the                 <-- comment
#       sparsification technique used
        self.has_clmds = True
        self.sparse_medoids = ind_dist[ind_medoids.astype(int)]
        self.sparse_coordinates = self.sparse_coordinates[ind_dist_inv]
        self.all_transformations = T_hierarchy
        self.ind_anchor = ind_anchor

        sparse_clusters = {}
        sparse_cluster_indices = np.empty(len(self.dist_matrix), dtype=int)
        for i in range(0, hierarchy[0]):
            temp = [np.where(ind_dist_inv == j)[0] for j in ind_clusters[i]]
            cluster = np.sort( np.concatenate((temp)) )
            sparse_clusters[i] = cluster
            sparse_cluster_indices[cluster] = i

        self.sparse_clusters = sparse_clusters
        self.sparse_cluster_indices = sparse_cluster_indices




#   This method checks the presence of pathological arrangements of anchor points in the MDS (i.e. non-convex
#   and self-intersecting results) and improves the final MDS solution (free of pathologies)
    def convexity_check(self, n_clusters, ind_anchor, dist_anchor, mds_anchor, embedding,
                        max_perm=6, W_mds=None, precision=None): 
        no_pathologies = 0
        final_vertices = []
        do_linear = []
        N_anchor = np.zeros((n_clusters+1, ), dtype=int)
        if precision:
            precision = 'E' + str(precision)
        else:
            precision = 'QbB'
        for i in range(0, n_clusters):
            if self.verbose:
                sys.stdout.write( '\rChecking convexity:%6.1f%%' % (float(i)*100./float(n_clusters))  )
                sys.stdout.flush()
            N_anchor[i+1] = N_anchor[i] + len(ind_anchor[i])
            if len(ind_anchor[i]) <= 3:
                final_vertices.append(np.arange(0, len(ind_anchor[i]), 1))
                no_pathologies += 1
                continue
#           Check if there is a pathological quadrilateral (non-convex)
            hull = spatial.ConvexHull( mds_anchor[N_anchor[i]:N_anchor[i+1], :], qhull_options=precision )
            vertices = hull.vertices
            if len(vertices) == 3:
#               We use the point excluded from the convex hull as the reference one (index 1 in vertices array)
#               This ensure a more accurate transformation later (regarding the resulting MDS)
                ref_point = np.setdiff1d(np.arange(0, 4, 1), vertices)
                final_vertices.append( np.concatenate(([vertices[0]], ref_point, vertices[1:])) )
                do_linear.append(i)
                no_pathologies += 0.5
            elif len(vertices) == 4:
#               Check if there is a pathological quadrilateral (self-intersecting)
                go_ahead = check_permutation( ind_anchor[i][vertices], ind_anchor[i], verbose=False )
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
                        temp_mds = embedding.fit_transform(dist_anchor, init=init_embed, weights=W_mds)
                        hull = spatial.ConvexHull( temp_mds[N_anchor[i]:N_anchor[i+1], :], qhull_options=precision )
                        temp_vertices = hull.vertices
#                       Check if the convex hull is a triangle now
                        if len(temp_vertices) == 3:
                            go_ahead = True
#                           We use the point excluded from the convex hull as the reference one 
                            ref_point = np.setdiff1d(np.arange(0, 4, 1), temp_vertices)
                            temp_vertices = np.concatenate(([temp_vertices[0]], ref_point, temp_vertices[1:]))
                            temp_pathology = 0.5
                        elif len(temp_vertices) == 4:
                            go_ahead = check_permutation( ind_anchor[i][temp_vertices], 
                                                         ind_anchor[i], verbose=False )        
                            temp_pathology = 1
                        else: 
#                           Improve this error message                                                          <-- check this
                            raise Warning("This is a pathological choice of anchor pts. on cluster ", i) 
                    if n_perm == max_perm:
#                       Decide if it is better to recompute the best 3 anchor points or                         <-- comment
#                       make a linear transformation with the current 4 anchor points                           <-- comment
                        final_vertices.append(vertices)
                        do_linear.append(i)
                    else:
#                       The current cluster is now non-pathological
#                       Check the effects of the new MDS on the convexity of the previous clusters
                        new_vertices = []
                        new_do_linear = []
                        new_no_pathologies = 0
                        for j in range(0, i+1):
                            if len(ind_anchor[j]) <= 3:
                                new_no_pathologies += 1
                                new_vertices.append(final_vertices[j])
                            else:
                                h = spatial.ConvexHull( temp_mds[N_anchor[j]:N_anchor[j+1], :], 
                                                        qhull_options=precision )
                                if len(h.vertices) == 3:
                                    new_no_pathologies += 0.5
#                                   We use the point excluded from the convex hull as the reference one 
                                    ref_point = np.setdiff1d(np.arange(0, 4, 1), h.vertices)
                                    new_vertices.append( np.concatenate(([h.vertices[0]], ref_point,
                                                          h.vertices[1:])) )
                                    new_do_linear.append(j)
                                else:
                                    if check_permutation( ind_anchor[j][h.vertices], ind_anchor[j], 
                                                         verbose=False ):
                                        new_no_pathologies += 1
                                        new_vertices.append(h.vertices)
                                    else:
                                        new_vertices.append(h.vertices)
                                        new_do_linear.append(j)
                        if new_no_pathologies > no_pathologies:
                            no_pathologies = new_no_pathologies
                            mds_anchor = temp_mds
                            final_vertices = new_vertices
                            do_linear = new_do_linear
                            self.MDS_stress = embedding.stress_
                        else:
#                           We keep the previous MDS
                            final_vertices.append(vertices)
                            do_linear.append(i) 
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
        self.do_linear_transf = do_linear


#   This method transforms each cluster from one 2D space (previous) to another (new)
#   The choice of transformation (linear or homography) depends on the anchor points given for each cluster
    def transform_2d(self, clusters, prev_clusters, ind_anchor, mds_clusters):
        N_anchor = self.final_n_anchor
        self.transformation = []
        print_label = []
        for i in range(0, len(clusters)):
            if self.verbose:
                sys.stdout.write( '\rPerforming transformations:%6.1f%%' % (float(i)*100./float(len(clusters))) )
                sys.stdout.flush()
#           CASE 1: cluster with 1 anchor point (translation)
            if len(self.order_anchor[i]) == 1:
                x_prev = mds_clusters[ind_anchor[i],:]
                x_new = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :]
                self.sparse_coordinates[ind_anchor[i],:] = x_new
                self.transformation.append( [[x_new - x_prev]] )
                print_label.append("translation")
                continue
            indices = ind_anchor[i][self.order_anchor[i]]
            X_prev = mds_clusters[indices,:]
            X_new = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :][self.order_anchor[i]]
            diff_X_prev = X_prev - X_prev[1,:]
            diff_X_new = X_new - X_new[1,:]
#           CASE 2: cluster with 2-3 anchor points or with a self-intersecting quadrilateral (linear transf.)
            if (len(self.order_anchor[i]) in [2,3]) or (i in self.do_linear_transf):
                T = np.linalg.lstsq(diff_X_prev, diff_X_new, rcond=None )[0]
                self.transformation.append( [X_prev[1,:], T, X_new[1,:]] )
                if len(prev_clusters[i]) in [2,3]:
#                   Small clusters
                    self.sparse_coordinates[ind_anchor[i],:] = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :]
                else:
#                   Transform and translate each cluster to the origin of its transf. matrix T
                    product = np.dot(mds_clusters[prev_clusters[i], :] - X_prev[1,:], T)
                    self.sparse_coordinates[prev_clusters[i], :] = product + X_new[1,:]
                if i in self.do_linear_transf:
                    print_label.append("linear (using 4 anchor points)")
                else:
                    print_label.append("linear")
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
                    raise Warning('The anchor points of cluster %i form a non-convex quadrilateral' % clusters[i])
#               Homography transformation and its residue
                F = np.array([[b*c*s,     0, b*(c*s - a*t)],
                              [    0, a*d*s, a*(d*s - b*t)],
                              [    0,     0,         a*b*t]])
                if len(prev_clusters[i]) == 4:
#                   Small clusters shortcut
                    self.sparse_coordinates[ind_anchor[i],:] = self.mds_anchor[N_anchor[i]:N_anchor[i+1], :]
                    result_homography = self.sparse_coordinates[ind_anchor[i],:]
                    print(i, "homography (small cluster)")
                else:
                    transf_prev = np.dot(mds_clusters[prev_clusters[i],:]- X_prev[1,:], T_prev)
                    transf_prev_homog = np.concatenate((transf_prev, np.ones((len(transf_prev),1))), axis=1)
                    perspective_homog =  np.dot(transf_prev_homog, F)
                    perspective = perspective_homog/perspective_homog[:,-1][:,None] 
                    result_homography = np.dot(perspective[:,:2], T_new_inv) + X_new[1,:]
                Rh = np.sum( (mds_clusters[prev_clusters[i],:] - result_homography)**2 )
#               Compute also the linear transf. for these points and compare their residues 
                T_lin = np.linalg.lstsq(diff_X_prev, diff_X_new, rcond=None )[0]
                result_linear = np.dot(mds_clusters[prev_clusters[i],:] - X_prev[1,:], T_lin) + X_new[1,:]
                Rl = np.sum( (mds_clusters[prev_clusters[i],:] - result_linear)**2 )
                if Rh < Rl:
                    self.sparse_coordinates[prev_clusters[i],:] = result_homography
                    self.transformation.append( [X_prev[1,:], T_prev, F, T_new_inv, X_new[1,:]] )
                    print_label.append("homography")
                else:
                    self.sparse_coordinates[prev_clusters[i],:] = result_linear
                    self.transformation.append( [X_prev[1,:], T_lin, X_new[1,:]] )
                    print_label.append("linear (homography was rejected)")
            else:
                raise Warning("There must be something wrong before cluster %i, check the list of anchor points: "
                               % clusters[i], ind_anchor)

        if self.verbose:
            sys.stdout.write( '\rPerforming transformations:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")
            print("--------------------------")
            print(" Cluster   Transformation")
            print("--------------------------")
            for i in range(0, len(clusters)):
                print(" %3i        %s" % (clusters[i], print_label[i]))         


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
#   or of the specified indices (CHECK THIS METHOD)                                                                 <-- comment
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
            self.sparse_list = sparse_list
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
            if len(C_sparse) in [1, 2] and len(C) > 2:
                print('NOTE: Cluster %i has only 1-2 sparse points. More sparse elements (3 minimum) are ' % i +
                      'needed to get a proper estimation for the other cluster points.')
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
            ind_sparse = np.array(sparse_list)[C_sparse]
            local_coordinates[ind_sparse] = local_mds_sparse
#           Transform the coordinates from local to global space
            transf_coordinates[C] = local_coordinates[C]                       
            transf_coordinates[C] = local_coordinates[C]
            T_2d = self.all_transformations[i][0]["transf"] # check if we change hierarchy computations                    <-- comment
            if len(T_2d) == 1:
#               Sparse cluster with 1 point (translation)
                transf_coordinates[C] = transf_coordinates[C] + T_2d[0]
                continue
            elif len(T_2d) == 3:
#               Linear transformation
                ref_prev, T_lin, ref_new = T_2d
#               We need to keep track of the reference anchor point in each step
                transf_coordinates[C] = np.dot(transf_coordinates[C] - ref_prev, T_lin) + ref_new    
            else:
#               Homography
                ref_prev, T_prev, F, T_new_inv, ref_new = T_2d
#               We need to keep track of the reference anchor point in each step
                transf_coordinates[C] = transf_coordinates[C] - ref_prev
                X_prev = np.dot(transf_coordinates[C], T_prev)
                X_prev_homog = np.concatenate((X_prev, np.ones((len(X_prev),1))), axis=1)
                X_new_homog =  np.dot(X_prev_homog, F)
                X_new = X_new_homog/X_new_homog[:,-1][:,None]
                transf_coordinates[C] = np.dot(X_new[:,:2], T_new_inv)
                transf_coordinates[C] = transf_coordinates[C] + ref_new
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

# This method find the repeated rows on a **square** matrix M (within a tolerance) 
def unique_rows_matrix(M, tol=1e-09, return_indices=False):
    n = len(M)
    I = np.arange(0, n, 1)
    ind_unique = np.unique( np.where((M + np.eye(n)) == 0.)[0] )
    if len(ind_unique) > 0: 
        ind_tol_u = [ind_unique[0]]
        M_tol_u = M[ind_unique][0][None,:]
        for i, row in zip(ind_unique, M[ind_unique]):
            for j, row_u in zip(ind_tol_u, M_tol_u):
                if np.allclose(row, row_u, atol=tol, rtol=0):
                    I[i] = j
                    break
            else:
                ind_tol_u = np.append(ind_tol_u, i)
                M_tol_u = np.concatenate((M_tol_u, row[None,:]), axis=0)
    M_unique = M[np.ix_(np.unique(I), np.unique(I))]
#   Return the matrix with unique entries and the unique row indices
    if not return_indices:
        return M_unique, np.unique(I)
#   Also return the indices to reconstruct the original matrix from the unique one
    I_inv = np.zeros(n, dtype=int)
    I_inv[np.unique(I)] = np.arange(0, len(M_unique), 1)
    for i in np.arange(0, len(M_unique), 1):
        temp = np.where(I == np.unique(I)[i])[0]
        I_inv[temp] = i
    return M_unique, np.unique(I), I_inv


# This method removes all the entries (except the first one) that form a block
# of zeros in a matrix M (even if they are different rows)
def remove_zero_entries(M, return_indices=False):
    n = len(M)
    I = np.arange(0, n, 1)
    indices = np.where((M + np.eye(n)) == 0.)
    ind_unique, counts = np.unique(indices[0], return_counts=True)
    if len(ind_unique) > 0:
        ind_rep = [indices[1][0]]
        s = 0
        for i, c in zip(ind_unique, counts):
             if i not in ind_rep:
                 ind_rep = ind_rep + list(indices[1][s:s + c])
                 I[indices[1][s:s + c]] = i
             s += c
    M_unique = M[np.ix_(np.unique(I), np.unique(I))]
#   Return the matrix with unique entries and the unique row indices
    if not return_indices:
        return M_unique, np.unique(I)
#   Also return the indices to reconstruct the original matrix from the unique one
    I_inv = np.zeros(n, dtype=int)
    I_inv[np.unique(I)] = np.arange(0, len(M_unique), 1)
    for i in np.arange(0, len(M_unique), 1):
        temp = np.where(I == np.unique(I)[i])[0]
        I_inv[temp] = i
    return M_unique, np.unique(I), I_inv


# This method chooses the kmedoids clustering with minimum intra-cluster incoherence (relative or total)
def optim_kmedoids(D, n_clusters, incoherence="rel", n_iter=100,  tmax=100, 
                   init_Ms="random", n_iso=None, verbose=False):
    from kmedoids_python import kmedoids
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


# This method gives te indices of those points whose distance to a reference point is bigger 
# than the chosen percentile
def select_further_points(dist_points, ref_point, percentile, n_min=30, n_max=None):
    indices = np.setdiff1d( np.arange(0, len(dist_points), 1), [ref_point] )
    I_complement_percent = []
    if percentile <= 50:
        return indices
    while len(I_complement_percent) < n_min and percentile > 50:
        dist_ref = dist_points[ref_point,:]
        dist_ref = np.delete(dist_ref, ref_point)
        complement_percent = (dist_ref > np.percentile(dist_ref, percentile))
        temp_ind = indices[complement_percent]
        if n_max is not None and len(temp_ind) > n_max:
            percentile += 1
            if percentile >= 100:
                t = n_max-len(I_complement_percent)
                if t <= 0:
                    rand_ind = I_complement_percent
                    np.random.shuffle(rand_ind)
                    I_complement_percent = np.sort(rand_ind[:n_limit])
                elif t == n_max:
                    np.random.shuffle(temp_ind)
                    I_complement_percent = np.sort(temp_ind[:n_limit])
                else:
                    rand_ind = np.setdiff1d(temp_ind, I_complement_percent)
                    np.random.shuffle(rand_ind)
                    I_complement_percent = np.concatenate((I_complement_percent, rand_ind[:t]))
                    I_complement_percent = np.sort(I_complement_percent)
                break
            else:
                continue
        I_complement_percent = temp_ind
        percentile -= 5
    return I_complement_percent


# Given the distance matrix of a dataset embedded in a n-dim space, this method returns the indices of 
# the points corresponding to the N vertices of the N-dim polytope that has max. volume
# NOTE: only N=4 is suported right now (a generalization of vmod.max_vol_vertices is needed otherwise)
def anchor_points_ndim(N, dist_points, method=None, param_method=None, ref_point=None, n_min=30,
                       n_max=None, indices_cl=None):
    indices = np.arange(0, len(dist_points), 1)
    if ref_point is not None:
        indices = np.setdiff1d(indices, [ref_point])
    if len(indices) <= N:
        return indices
#   Refine the list of indices using the following methods (if desired)
    if isinstance(method, (list, np.ndarray)):
        indices = np.sort(method)
    elif method == "random":
        if param_method is None:
            raise Exception("You need to provide a number of random combinations of vertices (param_method)")
        n_random = int(param_method)
        np.random.shuffle(indices)
        indices = indices[:n_random]
    elif method=="percentile":
        if param_method is None:
            raise Exception("You need to provide a percentile (param_method).")
        if ref_point is None:
            raise Exception("You need to provide a reference point (ref_point).")
        indices = select_further_points(dist_points, ref_point, param_method, n_min=n_min, n_max=n_max)
    elif method == "per cluster":
        if indices_cl is None:
            raise Exception("You need to provide the indices for each cluster, " +
                            "indices_cl = [ind_cl1, ..., ind_clN].")   
        if ref_point is None or len(indices_cl) != len(ref_point):
            raise Exception("You need to provide N reference points, ref_point = [p1,...,pN]," +
                            "where N = len(indices_cl).") 
        if param_method is None: 
            raise Exception("You need to provide a percentile (param_method).")
        elif isinstance(param_method, (int,float)):
            param_method = [param_method]*len(indices_cl)
        elif len(param_method) != len(indices_cl):
            raise Exception("Please, provide as many percentiles as number of clusters or choose one.")
        indices = []
        for i in range(0, len(indices_cl)):
            dist = dist_points[np.ix_(indices_cl[i], indices_cl[i])]
            I = select_further_points(dist, ref_point[i], param_method[i], n_min=n_min, n_max=n_max)
            indices.append(indices_cl[i][I])
        indices = np.sort( np.concatenate((indices)) )
#   Obtain the N indices defining the (N-1)-simplex with biggest volume
    ind_anchor = vmod.max_vol_vertices(len(indices), indices, dist_points, N)

    return ind_anchor


    
# Find if a set of points is a cyclic permutation or a reflection (or both) of another set
# Should we keep the verbose?                                                                                   <-- comment
def check_permutation(x, y, verbose=False):
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


# Computation of the number of points of a given set lying within a polygon with N vertices
def points_in_polygon(N, vertices, other_points, qhull_opt='QbB'):
    s=0
    if N != len(vertices):
        print("You need to provide %i vertices exactly" % N)
    for point in other_points:
        temp = np.concatenate((vertices, point[None,:]), axis=0)
        h = spatial.ConvexHull(temp, qhull_options=qhull_opt)
        if len(h.vertices) == N:
            if set(range(0,N)) <= set(h.vertices):  
                s += 1
    return s


#************************************************************************************************************
