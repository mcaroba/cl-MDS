#************************************************************************************************************
# This is the cluster-based MultiDimensional Scaling code for dimensionality reduction data analysis.       #
#                                                                                                           #
#                                                cl-MDS                                                     #
#                                                                                                           #
# This code has been written and is copyright (c) 2018-2022 of the following authors:                       #
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
#                                   https://github.com/mcaroba/cl-MDS                                       #
#                                                                                                           #
# Visit the repository for the latest version of this distribution.                                         #
#                                                                                                           #
#                                                                                                           #
# If you use cl-MDS for the compilation of academic/scientific/technical work, please cite, as appropriate: #
#                                                                                                           #
# P. Hernandez-Leon and M.A. Caro, XXX, YYY (2022)                                                          #
#                                                                                                           #
#************************************************************************************************************


# Import dependencies
import numpy as np
from sklearn_mds import _mds
import kmedoids as km
from cur import cur
import sys
from scipy import spatial
from src.anchor_selection import vertices_module as vmod


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
                 cutoff=None, do_species=None, average_kernel=False, verbose=True):
        """
        Arguments related to:

        - input (user must pass one of these): 
            *dist_matrix = 2D array, None(default)
            *atoms = str, None(default)
                Filename with the atomic structures (requires ase installation).

        - atomic descriptors (when passing atoms):
            *descriptor = str, None(default)
                Atomic descriptor from the implemented_descriptors list.
            *descriptor_string = str, None(default)
                String with the descriptor info.
            *cutoff = float, [float,float], None(default)
                Radius cutoff used in SOAP descriptors (first option for quippy_soap,
                second one for quippy_soap_turbo). Ignored when param. descriptor_string
                isn't None.
            *do_species = list of str, None(default)
                Selection of chemical species to be considered. 
                If None, all species in atoms are taken into account and the term
                "complete database" refers to the whole atoms file (self.n_env == self.all_env).
                Otherwise, it denotes the subset of atoms with the selected chemical species
                (self.n_env != self.all_env).
            *average_kernel = bool
                If True, the average kernel for each structure is computed instead of
                per atom (available only with quippy_soap).

        - sparsification:
            *sparsify = str, list/array, None(default)
                Sparsification method from sparse_options list.
                Alternatively, list/array with the indices of the desired sparse set
                within the whole atoms file (independently of do_species param.).
            *n_sparse = int, None(default)
                Number of elements in the sparse set (needed for sparsify=str, ignore
                otherwise).
            *sparsify_per_cluster = bool       --- in development (DO NOT USE)
            *max_n_sparse = int, None(default) --- in development (DO NOT USE)

        implemented_descriptors = ["quippy_soap","quippy_soap_turbo"]
        sparse_options = ["random", "cur"]
        """
#       This is the list of implemented atomic descriptors (it typically requires external
#       programs)
        implemented_descriptors = ["quippy_soap","quippy_soap_turbo"]
        sparse_options = ["random", "cur"]
        self.verbose = verbose
        self.sparsify = sparsify
        self.sparsify_per_cluster = sparsify_per_cluster
        self.max_n_sparse = max_n_sparse
        self.is_clustered = False
        self.has_clmds = False
        self.cutoff = cutoff
        self.do_species = do_species
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
                raise Exception("I couldn't find the atoms file or an ASE installation; \
                                 you need ASE to pass an atoms filename!")
#           Get chemical species and the number of environments
            species_list = []
            for ats in self.atoms:
                species_list += ats.get_chemical_symbols()
            self.species_list = species_list
            if self.average_kernel is True:
                n_env = len(self.atoms)
                self.all_env = n_env
            else:
                self.all_env = len(species_list)
            if do_species is None:
                n_env = len(species_list)
            else:
                self.do_species = set(do_species)
                try:
                    assert self.do_species <= set(species_list)
                except:
                    raise Exception("do_species param. has a chemical symbol that isn't in your \
                                     database, check it: ", self.do_species - set(species_list))
                do_species_list = []
                for z in self.do_species:
                    do_species_list += list(np.where(np.array(species_list) == z)[0])
                self.do_species_list = np.array(do_species_list)
                n_env = len(do_species_list)
            self.n_env = n_env
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
        sparse_list = np.arange(0, self.n_env, 1) # added to avoid possible crushes in other methods          <-- check this
        if do_species is not None:
            sparse_list = (self.do_species_list).copy()
        if sparsify is not None:
            if isinstance(sparsify, (list, np.ndarray)):
                sparsify = np.unique(sparsify)
                sparse_list = list(sparsify)
                n_sparse = len(sparse_list)
                if n_sparse > self.n_env:
                    raise Exception("The sparse set can't be larger than the complete dataset")
                if do_species is not None:
                    try:
                        assert set(sparsify) <= set(self.do_species_list)
                    except:
                        raise Exception("If you use do_species param., your sparse set can't have atoms \
                                         from other chemical species. Check the indices!")
            elif sparsify not in sparse_options:
                raise Exception("The sparsify option you chose is not available. Choose one of the following: ",
                                sparse_options, " or provide a list of indices")
            else:
                if n_sparse is None:
                    raise Exception("If you choose a sparsify option, you need to pass the n_sparse parameter")
                elif n_sparse > self.n_env:
                    raise Exception("The sparse set can't be larger than the complete dataset")
                if sparsify == "random":
                    np.random.shuffle(sparse_list)
                    sparse_list = sorted(sparse_list[:n_sparse])
#               Implement the "optimized sparse set" as a sparsify option                                           <--- comment
            if self.has_dist_matrix:
                if isinstance(sparsify, str) and sparsify == "cur":
                    sparse_list = np.unique(cur.cur_decomposition(self.dist_matrix, n_sparse)[-1])
                self.dist_matrix = dist_matrix[np.ix_(sparse_list, sparse_list)]
                self.all_dist_matrix = dist_matrix

        self.sparse_list = list(sparse_list)
        self.n_sparse = len(sparse_list)


#   This method takes care of adding a descriptor to the clMDS class:
    def build_descriptor(self, zeta=6, for_atoms_estim=False):
        if not hasattr(self, 'descriptor_type'):
            raise Exception("You must define a descriptor, check the implemented options")

#       Descriptor string
        descriptor = self.descriptor_type
        descriptor_string = self.descriptor_string
        if descriptor in ["quippy_soap", "quippy_soap_turbo"]:
            from ase.data import atomic_numbers
            from quippy.descriptors import Descriptor
            from quippy.convert import ase_to_quip
            self.zeta = zeta
            species = list(set(self.species_list)) 
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
                    if self.cutoff:
                        print('The information included in descriptor_string is used instead of the param. cutoff:')
                        print('   Chosen cutoff(s) =', cutoff)
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
#                       Preferred format
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
#                       Be careful with these formats (list, np.ndarray), the code assumes the strings are
#                       ordered regarding Z
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
                    if self.cutoff and for_atoms_estim is False:
                        print('The information included in descriptor_string is used instead of cutoff param.:')
                        print('   Chosen cutoff(s) =', cutoff)
                    self.cutoff = cutoff

#           Check sparsify hasn't change (required for coord. estimation)
            if for_atoms_estim is False:
                sparse_list = np.array(self.sparse_list)
            else:
                sparse_list = np.array(for_atoms_estim)

#           Descriptors
            if self.verbose:
                print("")
            if self.do_species is not None:
                species = self.do_species
            if descriptor == "quippy_soap":
                d = Descriptor(quippy_string)
            elif descriptor == "quippy_soap_turbo":
                d = {z: Descriptor(quippy_string[z]) for z in species}
            n = 0
            descriptor_list = []
            config_type_list = []
            for ats in self.atoms:
                if self.verbose:
                    sys.stdout.write('\rComputing descriptors:%6.1f%%' % (float(n)*100./float(self.all_env)) )
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
                    if isinstance(self.sparsify, (list, np.ndarray)) or self.sparsify == "random":
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
                        if self.species_list[n] not in species:
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
                    for i in range(0, len(ats)):
                        symb = ats[i].symbol
                        if symb not in species:
                            continue
                        if self.sparsify_per_cluster:
                            descriptor_list.append(q[symb][N[symb]]) 
                        elif isinstance(self.sparsify, str) and self.sparsify == "cur":
                            descriptor_list.append(q[symb][N[symb]]) 
                        else:
                            if n + i in sparse_list:       
                                descriptor_list.append(q[symb][N[symb]])  
                        N[symb] += 1 
                    n += len(ats)
            descriptor_list = np.array(descriptor_list)
            if for_atoms_estim is False:
                if isinstance(self.sparsify, str) and self.sparsify == "cur":
                    sparse_list = np.unique(cur.cur_decomposition(descriptor_list, self.n_sparse)[-1])
                    descriptor_list = descriptor_list[sparse_list]
                    self.sparse_list = list(self.do_species_list[sparse_list])
                    self.n_sparse = len(sparse_list)
            if self.verbose:
                sys.stdout.write('\rComputing descriptors:%6.1f%%' % 100. )
                sys.stdout.flush()
                print("")

            if not self.average_kernel:
                self.config_type_list = np.concatenate([c for c in config_type_list])
            else:
                self.config_type_list = np.array(config_type_list)
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
                    c = ( ( b.split('}')[0] ).split('{')[-1] ).split()
                    param = np.array(c).astype(type_label)
                else:
                    c = ( ( b.split('}')[0] ).split('{')[-1] ).split()
                    param = type_label(c[0])
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
            
            L = len(self.descriptor)
            dist_matrix = np.zeros([L, L])
            n = 0
            if self.verbose:
                print("")
            for i in range(0, L):
                if self.verbose:
                    sys.stdout.write('\rComputing dist_matrix:%6.1f%%' % (float(n)*100./float(L*(L+1)/2)) )
                    sys.stdout.flush()
#               Do this to remove numerical round-off problems
                dist_matrix[i,i] = 0.
                for j in range(i+1, L):
                    prod = np.dot(self.descriptor[i], self.descriptor[j])**self.zeta
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


#   This function ensures a minimum amount of points per cluster in the sparse set (DO NOT USE, IN DEVELOPMENT) <-- comment
    def cluster_sparsification(self, n_clusters, init_medoids="random", n_iso_med=None, 
                               iter_med=1000, tmax=100):
        if not self.sparsify_per_cluster:
            raise Exception("You haven't set sparsify_per_cluster=True. Please check if you want \
                             that kind of sparsification and set it right.")
        n_sparse = self.n_sparse
        self.n_sparse = self.n_sparse*n_clusters
        if not self.has_descriptor:
            self.build_descriptor()
        elif len(self.descriptor) < self.n_env:
            self.build_descriptor()
        self.build_dist_matrix()
        dist_matrix, ind_dist, ind_dist_inv = remove_zero_entries(self.dist_matrix, return_indices=True)

        sparse_list = self.sparse_list
#       Compute the clustering with the initial sparse set (considering only unique entries)
#       Remember to redefine init_medoids to unique entries indices (not implemented yet)                     <-- comment
        M, C = km.kMedoids( dist_matrix, n_clusters, incoherence="rel", n_inits=iter_med, 
                            tmax=tmax, init_Ms=init_medoids, n_iso=n_iso_med, verbosity=self.verbose)
#       Change this part (do not take into account the repeated entries to add new sparse entries)            <-- comment
        n_C = []
        C_incomplete = []
        C_indices = np.zeros(self.n_sparse, dtype=int)
        for i in range(0, n_clusters):
            all_Ci = np.concatenate(( [np.where(ind_dist == j)[0] for j in C[i]] ))
            n_C.append(len(all_Ci))
            C_indices[all_Ci] = i
            if n_C[i] < n_sparse:
                C_incomplete.append(i)
        if C_incomplete:
            non_sparse_list = [i for i in range(0, self.n_env) if i not in sparse_list]
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
            self.n_sparse = len(sparse_list)
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
            if self.verbose:
                print('\rClustering sparse points:')
            dist_matrix, ind_dist, ind_dist_inv = remove_zero_entries(self.dist_matrix, return_indices=True)
            if isinstance(init_medoids, (np.ndarray, list)):
#               The indices given by the user are regarding the FULL dist. matrix (including
#               zero/repeated entries), so they need to be redefine
                init_medoids = ind_dist_inv[init_medoids]
            ind_medoids, ind_clusters = km.kMedoids( dist_matrix, n_clusters, incoherence="rel",
                                                     n_inits=iter_med, tmax=tmax, init_Ms=init_medoids,
                                                     n_iso=n_iso_med, verbosity=self.verbose )
            if isinstance(init_medoids, (np.ndarray, list)) and iter_med == 0:
                assert set(ind_medoids) == set(init_medoids)
        else:
#           NOT READY YET, revise this part                                                                   <-- comment
            ind_medoids, ind_clusters = cluster_sparsification(n_clusters, init_medoids=init_medoids,
                                                               n_iso_med=n_iso_med, iter_med=iter_med,
                                                               tmax=tmax)
#           Maybe return the unique matrix directly from cluster_sparsification?                              <-- comment
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
            if L < 4:
                pot_indices[i] = np.arange(0, L, 1)
                if L == 1:
                    mds_clusters[ind_clusters[i]] = np.zeros((1,2)) # avoid sklearn RuntimeWarning
                else:
                    mds_clusters[ind_clusters[i]] = embedding_cl.fit_transform(dist_clusters[i])
                mds_A.append( mds_clusters[ind_clusters[i],:] )
                ind_A.append( ind_clusters[i] )
                continue
            elif L - 1 < 70:
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
        if self.verbose:
            print('\rMDS embedding of anchor points')
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
        sparse_cluster_indices = -np.ones(len(self.dist_matrix), dtype=int)
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
                sys.stdout.write( '\rChecking for pathological cases:%6.1f%%' % (float(i)*100./float(n_clusters))  )
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
            sys.stdout.write( '\rChecking for pathological cases:%6.1f%%' % 100. )
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
                print_label.append("affine (only translation)")
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
                    print_label.append("affine (overdetermined)")
                else:
                    print_label.append("affine")
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
                    print_label.append("affine (homography was rejected)")
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
    def get_sparse_coordinates(self, hierarchy, init_medoids="isolated", n_iso_med=1, eta=0.,
                               weight_anchor_mds=None):
        if not self.has_clmds:
            self.cluster_MDS(hierarchy = hierarchy, init_medoids = init_medoids,
                             n_iso_med = n_iso_med, eta = eta, weight_anchor_mds = weight_anchor_mds)

        ext_coordinates = np.empty([self.n_sparse, 3])
        ext_coordinates[0:self.n_sparse, 0:2] = self.sparse_coordinates
        ext_coordinates[0:self.n_sparse, 2] = self.sparse_cluster_indices

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



#   Given a list/array of indices, this method assigns each point to a cluster considering their
#   dissimilarity to the sparse medoids
    def assign_pts_to_cluster(self, indices=None):
        """
        This method returns a list of cluster indices for the points given in indices param. Each
        point is assigned to the cluster whose medoid has the lowest dissimilarity with that point.
        
        If indices=None (default), the complete distance matrix is used. Otherwise, please provide:
            (1) an array/list of indices (only available when sparsify is used),
            (2) a 2-dim. array with the distances of these points to the sparse medoids, that is,
                shape == (n. of points, n. of medoids).
        """
#       Classify all indices considering the sparse clustering
        if len(np.shape(indices)) == 2:
            dist_matrix = indices
        else:
            if self.sparsify is None:
                raise Exception("You need to use sparsify or provide a matrix with the distances to the \
                                 sparse medoids for these indices!")
            sparse_list = np.array(self.sparse_list)
            if indices is None:
                indices = np.setdiff1d(np.arange(0, self.n_env, 1), sparse_list)
            dist_matrix = self.all_dist_matrix[np.ix_(indices, sparse_list[self.sparse_medoids])]

        cluster_indices = np.argmin(dist_matrix, axis=1)
        if self.verbose:
            sys.stdout.write('\rAssigning each point to cluster:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")

        return cluster_indices


#   Given a list/array of indices, this method assigns each corresponding atom to a cluster
#   considering its dissimilarity to the sparse medoids
    def assign_atoms_to_cluster(self, indices=None, precision=1e-8, out_descriptor=False):
        """
        Classify new atomic data considering the sparse clustering. Each atom is assigned
        to the cluster whose medoid has the lowest dissimilarity with that atom.

        If indices=None, the complete database of length self.n_env is used. Otherwise,
        a list/array of indices is expected.
        """
#       Get the descriptors of all the atoms left out of the sparse set or the given ones
        sparse_list = self.sparse_list
        sparsify = self.sparsify
        if indices is None:
            if self.do_species is None:
                indices = np.setdiff1d(np.arange(0, self.all_env, 1), sparse_list)
            else:
                indices = np.setdiff1d(self.do_species_list, sparse_list)
        if len(self.descriptor) != self.n_env:
            Q_sparse = self.descriptor
            self.build_descriptor(for_atoms_estim=indices)
            Q = self.descriptor
#           Reassign the overwritten attributes
            self.descriptor = Q_sparse
        else:
            if self.do_species is None:
                Q_sparse = self.descriptor[sparse_list,:]
                Q = self.descriptor[indices,:]
            else:
                ind_sparse = [np.where(self.do_species_list == i)[0][0] for i in sparse_list]
                ind_new = [np.where(self.do_species_list == k)[0][0] for k in indices]
                Q_sparse = self.descriptor[ind_sparse,:]
                Q = self.descriptor[ind_new,:]
#       Classify them considering the clustering of the sparse set
        L = len(indices)
        cluster_indices = - np.ones(L, dtype=int)
        l = 0
        for i in range(0, L):
            if self.verbose:
                sys.stdout.write('\rAssigning each point to cluster:%6.1f%%' % (float(l)*100./float(L)))
                sys.stdout.flush()
            if indices[i] in sparse_list:
                temp = np.where(sparse_list == indices[i])[0][0]
                cluster_indices[i] = self.sparse_cluster_indices[temp]
                continue
            dist_med = np.zeros(self.hierarchy[0])
            for m, i_med in enumerate(self.sparse_medoids):
                prod = np.dot(Q[i], Q_sparse[i_med])**self.zeta 
                if prod <= 1. - precision:
                    dist_med[m] =  np.sqrt(1. - prod)
                else:
                    dist_med[m] = 0.
            cluster_indices[i] = np.argmin(dist_med)    
            l += 1       
        if self.verbose:
            sys.stdout.write('\rAssigning each point to cluster:%6.1f%%' % 100. )
            sys.stdout.flush()
            print("")

        if indices is None:
            I = np.empty(self.all_env, dtype=int)
            I[sparse_list] = self.sparse_cluster_indices
            I[indices] = cluster_indices
            if self.do_species is not None:
                I = I[self.do_species_list]
            cluster_indices = I

        if out_descriptor:
            return cluster_indices, Q

        return cluster_indices


#   This method transforms a distance matrix (N x N_sparse) to a 2-dim. Euclidean rep. using the
#   information from a previous sparse cl-MDS. The N points must correspond to the same cluster (i_cluster)
    def transform_dist_to_2d(self, i_cluster, dist_cluster):
        T = self.all_transformations[i_cluster]["dist"]
        local_coordinates = np.dot(dist_cluster, T)
        transf_coordinates = local_coordinates
        T_2d = self.all_transformations[i_cluster][0]["transf"] # check if we change hierarchy computations             <-- comment
        if len(T_2d) == 1:
#           Sparse cluster with 1 point (translation)
            transf_coordinates = transf_coordinates + T_2d[0]
        elif len(T_2d) == 3:
#           Linear transformation
            ref_prev, T_lin, ref_new = T_2d
#           We need to keep track of the reference anchor point in each step
            transf_coordinates = np.dot(transf_coordinates - ref_prev, T_lin) + ref_new    
        else:
#           Homography
            ref_prev, T_prev, F, T_new_inv, ref_new = T_2d
#           We need to keep track of the reference anchor point in each step
            transf_coordinates = transf_coordinates - ref_prev
            X_prev = np.dot(transf_coordinates, T_prev)
            X_prev_homog = np.concatenate((X_prev, [1]))
            X_new_homog =  np.dot(X_prev_homog, F)
            X_new = X_new_homog/X_new_homog[-1]
            transf_coordinates = np.dot(X_new[:2], T_new_inv)
            transf_coordinates = transf_coordinates + ref_new
        
        return local_coordinates, transf_coordinates 



#   This method gives a "cheap" estimation of the cl-MDS coordinates of points not included in the sparse set 
    def compute_pts_estim_coordinates(self, indices, reg_param=0.001):
        """
        Computes an estimation of the 2-dimensional coordinates of the points given by the
        indices param. or dist_info param., using the previously computed sparse transformations. 
        This method is specific for distance matrices.

        If indices=None (default), the complete distance matrix is used. Otherwise, please provide:
            (1) an array/list of indices (only available when sparsify is used),
            (2) a list where each member i is an array with its cluster index (first) and its distances
                to its cluster sparse points (use self.assign_to_cluster_dist to obtain cluster info).
        """
        sparse_list = np.array(self.sparse_list)
        L = len(indices)
#       Classify all indices considering the sparse clustering
        if isinstance(indices[0], (list, np.ndarray)):
            cluster_indices = np.array([int(indices[k][0]) for k in range(0, L)])
        else:
            cluster_indices = self.assign_pts_to_cluster(indices=indices) 

#       Compute the transformation from distance space to 2-dim. Euclidean space
        local_coordinates = np.zeros((L,2))
        transf_coordinates = np.zeros((L,2))
        self.warning_bad_estim = []
        l = 0
        for i in range(0, self.hierarchy[0]):
            if self.verbose:
                print("")
            C = np.where(cluster_indices == i)[0]
            C_sp = self.sparse_clusters[i]
            non_sparse = np.setdiff1d(indices[C], sparse_list[C_sp])
            if len(C_sp) in [1, 2] and len(non_sparse) > 0:
                self.warning_bad_estim.append(i)
#           Transf. matrix from high-dim. space to local 2-dim. space (with regularization)
            reg = np.diag( np.ones(len(C_sp))*reg_param )
            dist_sp = self.dist_matrix[np.ix_(C_sp, C_sp)] - reg
            T = np.linalg.lstsq(dist_sp, self.local_sparse_coordinates[C_sp,:], rcond=None)[0]
            self.all_transformations[i]["dist"] = T
            for j in C:
                if self.verbose:
                    sys.stdout.write('\rEstimating coordinates:%6.1f%%' % (float(l)*100./float(L)) )
                    sys.stdout.flush()
#               distance matrix of point j restricted to sparse cluster members
                if  isinstance(indices[0], (list, np.ndarray)):
                    dist_j = indices[j,1:][C_sp]
                else:
                    dist_j = self.all_dist_matrix[indices[j], sparse_list[C_sp]]
#               transforming coordinates using sparse transf. matrices
                local_coordinates[j], transf_coordinates[j] = self.transform_dist_to_2d(i, dist_j)
                l += 1
        if self.verbose:
            sys.stdout.write('\rEstimating coordinates:%6.1f%%' % (100.) )
            sys.stdout.flush()
            print("")
                                           
        self.has_estimation = True
        self.estim_local_coordinates = local_coordinates

        self.estim_cluster_indices = cluster_indices
        self.estim_coordinates = transf_coordinates


#   This method gives a "cheap" estimation of the cl-MDS coordinates of points not included in the sparse set 
    def compute_atoms_estim_coordinates(self, indices, precision=1e-8, reg_param=0.001):
        """
        Computes an estimation of the 2-dimensional coordinates of the points given by the
        indices param., using the previously computed sparse transformations. This method is
        specific for atoms files.

        Provide an array or list of indices via indices param.
        """
#       Classify all indices considering the sparse clustering
        indices = np.array(indices)
        if len(self.descriptor) != self.all_env:
            cluster_indices, Q = self.assign_atoms_to_cluster(indices=indices, precision=precision,
                                                              out_descriptor=True)
            Q_sparse = self.descriptor
        else:
            cluster_indices = self.assign_atoms_to_cluster(indices=indices, precision=precision)
            Q = self.descriptor[indices]
            Q_sparse = self.descriptor[self.sparse_list]

#       Compute the transformation from distance space to 2-dim. Euclidean space
        L = len(indices)
        local_coordinates = np.zeros((L, 2))
        transf_coordinates = np.zeros((L, 2))
        self.warning_bad_estim = []
        l = 0
        for i in range(0, self.hierarchy[0]):
            C = np.where(cluster_indices == i)[0] 
            C_sp = self.sparse_clusters[i]
            non_sparse = np.setdiff1d(indices[C], np.array(self.sparse_list)[C_sp])
            if len(C_sp) in [1, 2] and len(non_sparse) > 0:
                self.warning_bad_estim.append(i)
#           Transf. matrix from high-dim. space to local 2-dim. space (with regularization)
            reg = np.diag( np.ones(len(C_sp))*reg_param )
            dist_sp = self.dist_matrix[np.ix_(C_sp, C_sp)] - reg
            T = np.linalg.lstsq(dist_sp, self.local_sparse_coordinates[C_sp,:], rcond=None)[0]
            self.all_transformations[i]["dist"] = T
            for j in C:
                if self.verbose:
                    sys.stdout.write('\rEstimating coordinates:%6.1f%%' % (float(l)*100./float(L)) )
                    sys.stdout.flush()
#               distance matrix of atom j restricted to sparse cluster members
                dist_j = np.empty(len(C_sp))
                for n, j_sp in enumerate(C_sp):
                    prod = np.dot(Q[j], Q_sparse[j_sp])**self.zeta
                    if prod <= 1. - precision:
                        dist_j[n] = np.sqrt(1. - prod)
                    else:
                        dist_j[n] = 0. 
#               transforming coordinates using sparse transf. matrices
                local_coordinates[j], transf_coordinates[j] = self.transform_dist_to_2d(i, dist_j)
                l += 1
        if self.verbose:
            sys.stdout.write('\rEstimating coordinates:%6.1f%%' % (100.) )
            sys.stdout.flush()
            print("")
                                           
        self.has_estimation = True
        self.estim_local_coordinates = local_coordinates

        self.estim_cluster_indices = cluster_indices
        self.estim_coordinates = transf_coordinates


#   This is a user friendly function that returns the clusters and coordinates of the complete dataset
    def get_estim_coordinates(self, hierarchy=None, n_steps=10, precision=1e-8, reg_param=0.001):
        """
        This method give the clustering and embedding info. of the complete database. Those
        points not included in the sparse set are classified within the existing clustering and
        get estimated coordinates.

        * hierarchy = None (default), int, list
        Use None when the hierarchy attribute has already been used (usually for sparse
        calculations). Otherwise, you need to define a hierarchy of cluster levels by
        providing an integer or list (e.g., 8, [8,1] or [8,3,1]).

        * n_steps = positive int
        The n_steps param. establishes the size of the partition used to reduce memory consumption.
        That is, the complete dataset is divided in n_steps subsets (excluding the sparse set),
        each of them calling self.compute_estim_coordinates once.
        """
        if hasattr(self, 'hierarchy'):
            hierarchy = self.hierarchy
        elif hierarchy is None:
            raise Exception("You need to provide a hierarchy to proceed with the calculations")

        if not self.has_clmds:
            self.cluster_MDS(hierarchy=hierarchy)

        if self.atoms is None:
            ext_coordinates = np.empty([self.n_env,3])
        else:
            ext_coordinates = np.empty([self.all_env,3])
        ext_coordinates[self.sparse_list, 0:2] = self.sparse_coordinates
        ext_coordinates[self.sparse_list, 2] = self.sparse_cluster_indices

        L = self.n_env // n_steps
        for i in range(0, n_steps):
            if self.verbose:
                print("")
                print('\rEstimation for subset %i/%i' % (i+1, n_steps) )
            l_low = L*i
            l_high = L*(i+1)
            if i == n_steps - 1:
                l_high += self.n_env % n_steps
            I = np.arange(l_low, l_high, 1)
            if self.atoms is None:
                indices = np.setdiff1d(I, self.sparse_list)
                self.compute_pts_estim_coordinates(indices=indices)
            else:
                if self.do_species is None:
                    indices = np.setdiff1d(I, self.sparse_list)
                else:
                    indices = np.setdiff1d(self.do_species_list[I], self.sparse_list)
                self.compute_atoms_estim_coordinates(indices)
            ext_coordinates[indices, 0:2] = self.estim_coordinates
            ext_coordinates[indices, 2] = self.estim_cluster_indices
        if len(self.warning_bad_estim) > 0:
            print("")
            print('*** Warning: The following clusters have only 1-2 sparse points: ', self.warning_bad_estim)
            print('    A proper estimation of new cluster members requires 3 sparse points at least. ***')

        if self.do_species is not None:
            ext_coordinates = ext_coordinates[self.do_species_list,:]

        self.estim_cluster_indices = ext_coordinates[:,2].astype(int)
        self.estim_coordinates = ext_coordinates[:,:2]

        return ext_coordinates


#   This method saves to txt file the requested clMDS attributes
    def save_to_file(self, dir='./', debug=False):
        """
        Save to txt file the main cl-MDS attributes that have been computed, that is:
            self.sparse_list
            self.sparse_coordinates
            self.sparse_cluster_indices
            self.sparse_medoids
            self.estim_coordinates
            self.estim_cluster_indices
        
        If debug=True, a second file with all attributes related to transformations
        and anchor points is created too.
        """
        if not self.has_clmds:
            raise Exception("No information to save! Compute cl-MDS embedding first.")

        if not self.has_estimation:
            M = np.zeros(self.n_sparse)
            M[self.sparse_medoids] = 1
            with open(dir + 'clmds_results.dat', 'w+') as f:
                print('i_atoms X_clmds Y_clmds C M', file=f)
                for i in range(0, self.n_sparse):
                    if self.verbose:
                        sys.stdout.write('\rSaving results:%6.1f%%' % (float(i)*100./self.n_sparse) )
                        sys.stdout.flush()
                    print('%i %f %f %i %i' % (self.sparse_list[i], self.sparse_coordinates[i,0],
                          self.sparse_coordinates[i,1], self.sparse_cluster_indices[i], M[i]), file=f)
        else:
            M = np.zeros(self.n_env)
            S = np.zeros(self.n_env)
            if self.do_species:
                atoms_list = self.do_species_list
                sparse_list = np.empty(self.n_sparse, dtype=int)
                for i, s in enumerate(self.sparse_list):
                    sparse_list[i] = np.where(s == self.do_species_list)[0][0]
            else:
                atoms_list = np.arange(0, self.n_env, 1)
                sparse_list = np.array(self.sparse_list)
            M[sparse_list[self.sparse_medoids]] = 1
            S[sparse_list] = 1
            with open(dir + 'clmds_results.dat', 'w+') as f:
                print("i_atoms X_clmds Y_clmds C M sparse", file=f)
                for i in range(0, self.n_env):
                    if self.verbose:
                        sys.stdout.write('\rSaving results:%6.1f%%' % (float(i)*100./self.n_sparse))
                        sys.stdout.flush()
                    print('%i %f %f %i %i %i' % (atoms_list[i], self.estim_coordinates[i,0],
                          self.estim_coordinates[i,1], self.estim_cluster_indices[i], M[i], S[i]), file=f)
        if debug:
#           Save all transformations info
            with open(dir + 'clmds_debug_data.dat', 'w+') as g:
                print("all_env n_env n_sparse", file=g)
                print("%i %i %i" % (self.all_env, self.n_env, self.n_sparse), file=g)
                g.write(str(self.all_transformations))
            
        if self.verbose:
            sys.stdout.write('\rSaving results:%6.1f%%' % (100.) )
            sys.stdout.flush()
            print("")



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
            coords[:,:] = np.NaN
            cluster[:] = -1
            for j in range(0, natoms):
                if not self.has_estimation:
                    if n in self.sparse_list:
                        i = self.sparse_list.index(n)
                        coords[j] = self.sparse_coordinates[i]
                        cluster[j] = self.sparse_cluster_indices[i]
                else:
                    coords[j] = self.estim_coordinates[n]
                    cluster[j] = self.estim_cluster_indices[n]
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
            if descriptor == "quippy_soap_turbo":
                cutoff = np.max(self.cutoff)
            else:
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
                        pbc = ats.get_pbc()
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
                        site.set_pbc(pbc)
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
                from ovito.modifiers import CreateBondsModifier
            except:
                raise Exception("You need Ovito (pip3 install ovito) to use the rendering capability")

            for i in range(0, len(self.sparse_medoids)):
                atoms = import_file(dir + "/medoid_%i.xyz" % i)
                atoms.add_to_scene()
                vp = Viewport()
                vp.type = Viewport.Type.Perspective
                vp.zoom_all()
                atoms.source.data.cell.vis.render_cell = False
#               I didn't manage to get bonds to render                                                          <-- comment
                modifier = CreateBondsModifier(cutoff = bond_cutoff)
                modifier.vis.enabled = True
                modifier.vis.width = 0.3
                atoms.modifiers.append(modifier)
                atoms.compute()
                vp.render_image(filename=dir+"/medoid_%i.png" % i, size=(400,400), alpha=True)
                atoms.remove_from_scene()

        if gnuplot:
            if not self.has_estimation:
                ext_coords = self.get_sparse_coordinates(hierarchy=self.hierarchy)
            else:
                ext_coords = self.get_estim_coordinates(hierarchy=self.hierarchy)
            f = open(dir + "/xy.dat", "w+")
            for i in ext_coords:
                print(i[0], i[1], i[2], file=f)

            f.close()
            f = open(dir + "/gnuplot.script", "w+")
            print("set term pngcairo size 640,640; set output 'clmds_map.png'", file=f)
            print("set size ratio -1", file=f)
            print("set xlabel 'cl-MDS coordinate 1'", file=f)
            print("set ylabel 'cl-MDS coordinate 2'", file=f)
            if render:
                print("plot 'xy.dat' u 1:2:3 lc var pt 7 not, \\", file=f)
                for i in range(0, len(self.sparse_medoids)):
                    x, y = self.sparse_coordinates[self.sparse_medoids[i]]
                    print("     'medoid_" + str(i) + ".png' binary filetype=png dx=0.0001" + \
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
