import os.path as osp
import sys

import numpy as np
import pickle as pkl

import openbabel, pybel
from biopandas.mol2 import PandasMol2

STEPS = 2000

DATA_SETS = ['../SD_Databases/huuskonen/huuskonen_logp_training_.sdf',
             '../SD_Databases/huuskonen/huuskonen_logp_test_.sdf']

DATA_SET_NAMES = ['../make_databases/logp&coords/huuskonen/huuskonen_logp_training.pkl',
                  '../make_databases/logp&coords/huuskonen/huuskonen_logp_test.pkl']

databae_path = '../database/huuskonen_train'

mol2_file_path = ['../database/huuskonen_train/mol2/',
                  '../database/huuskonen_test/mol2/']

def get_coords(ac_mol2_file):

    pmol = PandasMol2().read_mol2(ac_mol2_file)
    coords = []
    molecule = []
    for atom in pmol.df.itertuples():
        coords.append([atom.x, atom.y, atom.z])

    return np.array(coords)

if __name__=="__main__":


    for idx, sdf_dataset in enumerate(DATA_SETS):
        logp_dataset = dict()
        database = pybel.readfile('sdf', sdf_dataset)
        #read the molecules in the sdf files
        for sd_record in database:

            mol_id = sd_record.data['MOLECULEID']
            file_path = mol2_file_path[idx] + mol_id+'.mol2'
            molecule_coords = get_coords(file_path)


            #molecule.data.keys() gives all the properties
            molecule = pybel.readstring("smi", sd_record.data['SMILES'])
            #add hydrogen
            molecule.OBMol.AddHydrogens()
            #minimize the energy
            molecule.make3D(forcefield="gaff", steps=STEPS)
            molecule.localopt(forcefield="gaff", steps=STEPS)

            #get the coordinates
            molecule_coords = []
            for atom in molecule.atoms:
                molecule_coords.append(atom.coords)

            # #save in the data set
            logp_dataset[mol_id] = {'logp':float(sd_record.data['logPow {measured}']),
                                    'coords':np.array(molecule_coords)}
            molecule.write("pdb", f"{databae_path}/pdbs/{sd_record.data['MOLECULEID']}.pdb")
            molecule.write("mol2", f"{databae_path}/pdbs/{sd_record.data['MOLECULEID']}.mol2")


        #save in the pickle file

        outfile = DATA_SET_NAMES[idx]
        if osp.exists(outfile):
            print("The dataset already exists")
        else:
            with open(outfile, 'wb') as pklf:
                pkl.dump(logp_dataset, pklf)
