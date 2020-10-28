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
    def __init__(self, mol2_files_path, gaffmol2_files_path, logp_files_path):

        self.mol2_files_path = mol2_files_path
        self.gaffmol2_files_path = gaffmol2_files_path
        self.logp_files_path = logp_files_path


    def create(self, wavelet_num_steps, features, dataset_save_path,
               structure='3D', radial_cutoff=7.5, atype_categories_num=5):

        gsgraph = GSGraph(wavelet_num_steps)
        params = AtomicParams(atype_categories_num)
        gaff_lj = params.GAFFLJ

        dataset = defaultdict(list)

        mol2_files = [f for f in os.listdir(self.mol2_files_path)
                      if f.endswith(".mol2")]

        if not osp.exists(self.mol2_files_path):
            print(f'{self.mol2_files_path} does not exists')
            return None


        if not osp.exists(self.gaffmol2_files_path):
            print(f'{self.gaffmol2_files_path} does not exists')
            return None


        if not osp.exists(self.logp_files_path):
            print(f'{self.logp_files_path} does not exists')
            return None


        failed_number = 0
        #read the molecules in the mol2 files
        for idx, mol2_file_name in enumerate(mol2_files):

            mol_id, _ = osp.splitext(mol2_file_name)

            mol2_file = osp.join(self.mol2_files_path, mol2_file_name)

            gaffmol2_file = osp.join(self.gaffmol2_files_path, mol2_file_name)

            logp_file = osp.join(self.logp_files_path, f'{mol_id}.exp')

            all_paths = [mol2_file, gaffmol2_file, logp_file]

            if all(osp.isfile(mol_path) for mol_path in all_paths):

                gaff_molecule = params.gaff_mol2(gaffmol2_file)


                atom_names, encodings, coords = params.molecule_props(mol2_file)

                if coords.shape[0] == len(gaff_molecule):

                    logp = params.logp(logp_file)

                    #gaff featured signals
                    #TODO use each molecule atom type
                    gaff_signals = params.atoms_signal(gaff_molecule, gaff_lj, encodings)


                    if structure == '3D':
                        adj_mat = adjacency_matrix(coords, radial_cutoff)
                    elif structure == '2D':
                        adj_mat = params.connec_mat(mol2_file)

                    wavelets = gsgraph.wavelets(adj_mat)


                    gaff_features = gsgraph.molecule_features(adj_mat, gaff_signals, features)

                    #make a data set using the params
                    dataset['molid'].append(mol_id)
                    dataset['atom_names'].append(atom_names)
                    dataset['coords'].append(coords)
                    dataset['adjacency'].append(adj_mat)
                    dataset['wavelets'].append(wavelets)

                    dataset['gaff_signals'].append(gaff_signals)

                    dataset['gaff_features'].append(gaff_features)

                    dataset['logp'].append(logp)
                    #print(f'Successfuly processed Molecule {mol_id}')
                else:
                    failed_number += 1
                    #print(f'Failed to processe Molecule {mol_id}')


        print(f'{failed_number} molecules faild to be processed')

        if osp.exists(dataset_save_path):
            print("The dataset already exists, Deleted files")
            os.remove(dataset_save_path)

        with open(dataset_save_path, 'wb') as pklf:
            pkl.dump(dataset, pklf)
            print(f"dataset {dataset_save_path}")
