import os
import os.path as osp
import sys

import numpy as np
import pickle as pkl

import openbabel, pybel
from biopandas.mol2 import PandasMol2

class ProcessSDF:

    def __init__(self, sdf_file, ff_optimization_steps=2000):
        self.sdf_file = sdf_file
        self.ff_optimization_steps = ff_optimization_steps

    def smiles2pdb(self, save_path):

        if not osp.exists(save_path):
            os.makedirs(save_path)

        failed_number = 0
        database = pybel.readfile('sdf', self.sdf_file)
        #read the molecules in the sdf files
        for sd_record in database:

            mol_id = sd_record.data['MOLECULEID']
            molecule = pybel.readstring("smi", sd_record.data['SMILES'])

            if (molecule):
                #add hydrogen
                molecule.OBMol.AddHydrogens()

                #minimize the energy
                molecule.make3D(forcefield="gaff",
                                steps=self.ff_optimization_steps)
                molecule.localopt(forcefield="gaff",
                                  steps=self.ff_optimization_steps)

                pdb_file_path = osp.join(save_path, f'{mol_id}.pdb')

                molecule.write('pdb', pdb_file_path)

                print(f'Successfuly  converted molecule {mol_id}')

            else:

                failed_number += 1
                print(f'Openbabel failed to convert molecule {mol_id}')


        print(f'{failed_number} failed to convert')


    def sdfto_pdb_mol2(self, arguments):


        indxs, sdf_file_name, base_path = arguments[0], arguments[1], arguments[2]

        database = pybel.readfile('sdf', sdf_file_name)
        if not osp.exists(base_path):
            os.makedirs(base_path)

        pdb_save_path = osp.join(base_path, 'pdbs')
        mol2_save_path = osp.join(base_path, 'mol2')
        logp_save_path = osp.join(base_path, 'logp_values')

        save_paths = [pdb_save_path, mol2_save_path, logp_save_path]

        for spath in save_paths:
            if not osp.exists(spath):
                os.makedirs(spath)
        failed_number = 0
        #read the molecules in the sdf files
        for idx, molecule in enumerate(database):
            mol_id = f'NCI_{indxs[idx]+1}'
            if (molecule):
                #add hydrogen
                molecule.OBMol.AddHydrogens()

                #minimize the energy
                molecule.make3D(forcefield="gaff",
                                steps=self.ff_optimization_steps)
                molecule.localopt(forcefield="gaff",
                                  steps=self.ff_optimization_steps)

                pdb_file_path = osp.join(pdb_save_path, f'{mol_id}.pdb')
                mol2_file_path = osp.join(mol2_save_path, f'{mol_id}.mol2')
                logp_file_path = osp.join(logp_save_path, f'{mol_id}.exp')
                #write them in the files
                molecule.write('pdb', pdb_file_path)
                molecule.write('mol2', mol2_file_path)

                with open(logp_file_path, 'w') as txtf:
                    txtf.write(molecule.data['LogP'])

                # print(f'Successfuly converted molecule')
            else:

                failed_number += 1
                # print(f'Openbabel failed to convert molecule {mol_id}')


        print(f'{failed_number} failed to convert')

    def smiles2mol2(self, save_path):

        failed_number = 0

        if not osp.exists(save_path):
            os.makedirs(save_path)

        database = pybel.readfile('sdf', self.sdf_file)
        #read the molecules in the sdf files
        for sd_record in database:
            mol_id = sd_record.data['MOLECULEID']
            print(f'Processing Molecule {mol_id}')

            molecule = pybel.readstring("smi", sd_record.data['SMILES'])

            if (molecule):
                #add hydrogen
                molecule.OBMol.AddHydrogens()

                #minimize the energy
                molecule.make3D(forcefield="gaff",
                                steps=self.ff_optimization_steps)
                molecule.localopt(forcefield="gaff",
                                  steps=self.ff_optimization_steps)

                mol2_file_path = osp.join(save_path, f'{mol_id}.mol2')

                molecule.write('mol2', mol2_file_path)
                print(f'Successfuly  converted molecule {mol_id}')

            else:
                failed_number += 1
                print(f'Openbabel failed to convert molecule {mol_id}')

        print(f'{failed_number} failed to convert')

    def save_logp(self, save_path):

        if not osp.exists(save_path):
            os.makedirs(save_path)

        failed_number = 0
        database = pybel.readfile('sdf', self.sdf_file)
        #read the molecules in the sdf files
        for sd_record in database:
            mol_id = sd_record.data['MOLECULEID']
            print(f'Processing Molecule {mol_id}')

            molecule = pybel.readstring("smi", sd_record.data['SMILES'])

            if (molecule):

                logp = sd_record.data['logPow {measured}']

                log_file_path = osp.join(save_path, f'{mol_id}.exp')

                with open(log_file_path, 'w') as txtf:
                    txtf.write(logp)

            else:
                failed_number += 1
                print(f'Openbabel failed to raed molecule {mol_id}')


        print(f'{failed_number} failed to read')
