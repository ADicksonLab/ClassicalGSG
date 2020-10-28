import os
import os.path as osp
import sys
import pickle as pkl

import numpy as np
from collections import defaultdict
from logp.molecular_models.GSGraph import GSGraph
from logp.molecular_models.utility import  adjacency_matrix
from logp.preprocessing.atomicparams import AtomicParams
from logp.molecular_models.anakinme import AnakinME

class PrepareDataset:
    def __init__(self, mol2_files_path,
                 str_files_path, logp_files_path):

        self.mol2_files_path = mol2_files_path
        self.str_files_path = str_files_path
        self.logp_files_path = logp_files_path


    def create(self,  wavelet_num_steps, sco_flags, molids,
               structure='3D', radial_cutoff=7.5, atype_categories_num=5):

        gsgraph = GSGraph(wavelet_num_steps)
        params = AtomicParams(atype_categories_num)
        cgenff_lj = params.CGENFFLJ
        CGENFF_ATOM_TYPES = cgenff_lj.keys()

        dataset = defaultdict(list)

        if not osp.exists(self.mol2_files_path):
            print(f'{self.mol2_files_path} does not exists')
            return None



        if not osp.exists(self.str_files_path):
            print(f'{self.str_files_path} does not exists')
            return None

        if not osp.exists(self.logp_files_path):
            print(f'{self.logp_files_path} does not exists')
            return None


        failed_number = 0
        #read the molecules in the mol2 files
        for mol_id in molids:
            mol2_file_name = f'{mol_id}.mol2'
            print(mol_id)

            mol2_file = osp.join(self.mol2_files_path, mol2_file_name)

            cgenffsrt_file = osp.join(self.str_files_path, f'{mol_id}.str')
            logp_file = osp.join(self.logp_files_path, f'{mol_id}.exp')

            all_paths = [mol2_file, cgenffsrt_file, logp_file]

            if all(osp.isfile(mol_path) for mol_path in all_paths):

                cgenff_molecule = params.cgenff_str(cgenffsrt_file)
                mol_atom_types = [atom.atom_type for atom in cgenff_molecule]

                atom_names, encodings, coords = params.molecule_props(mol2_file)

                #check_types = [atom_type in CGENFF_ATOM_TYPES for atom_type in mol_atom_types]
                # if len(mol_atom_types)==0:
                #     check_types = False
                if (coords.shape[0] > 0 and coords.shape[0] == len(cgenff_molecule)):
                    logp = params.logp(logp_file)

                    #cgenff featured signals
                    cgenff_signals = params.atoms_signal(cgenff_molecule, cgenff_lj, encodings)

                    #Determine the features of the molecules using Feng,
                    #Geometric Scattering Graph representation

                    if structure == '3D':
                        adj_mat = adjacency_matrix(coords, radial_cutoff)
                    elif structure == '2D':
                        adj_mat = params.connec_mat(mol2_file)


                    wavelets = gsgraph.wavelets(adj_mat)

                    cgenff_features = gsgraph.molecule_features(adj_mat, cgenff_signals, sco_flags)

                        #make a data set using the params
                    dataset['molid'].append(mol_id)
                    dataset['atom_names'].append(atom_names)
                    dataset['cgenff_features'].append(cgenff_features)
                    dataset['logp'].append(logp)
                else:
                    failed_number += 1


        return dataset
