import os
import os.path as osp
import numpy as np
import warnings
import csv
from openbabel import pybel
from tqdm import tqdm
from rdkit import Chem


# from OpenChem
def read_smiles_property_file(path, cols_to_read,
                              delimiter=',', keep_header=False):
    reader = csv.reader(open(path, 'r'), delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]

    return data[0], data[1]


def canonize_smile(smile, canonize=True):
    new_smile = None
    try:
        new_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile,
                                                        sanitize=canonize))

    except:

        warnings.warn(smile + ' can not be canonized:'
                      'nvalid SMILES string!')

    return new_smile


def save_smile(smiles, molecule_ids, save_path):

    if not osp.exists(save_path):
        os.mkdir(save_path)

    for sm_idx in tqdm(range(len(smiles))):
        smile = smiles[sm_idx]
        canonized_smile = canonize_smile(smile)
        if canonized_smile:
            smile_file_path = osp.join(save_path,
                                       f'{molecule_ids[sm_idx]}.smi')
            with open(smile_file_path, 'w') as wfile:
                wfile.write(smile)


def save_logP(logp_values, mol_names, save_path):

    if not osp.exists(save_path):
        os.mkdir(save_path)

    for idx, logp in enumerate(logp_values):
        logp_file_path = osp.join(save_path, f'{mol_names[idx]}.exp')

        with open(logp_file_path, 'w') as wfile:
            wfile.write(logp)


def save_to_pdb(molecules, mol_names, save_path):

    if not osp.exists(save_path):
        os.mkdir(save_path)

    for mol_idx, molecule in enumerate(molecules):
        pdb_path = osp.join(save_path, f'{mol_names[mol_idx]}.pdb')
        molecule.write('pdb', pdb_path)


def save_to_mol2(molecules, mol_names, save_path):

    if not osp.exists(save_path):
        os.mkdir(save_path)

    for mol_idx, molecule in enumerate(molecules):
        mol2_path = osp.join(save_path, f'{mol_names[mol_idx]}.mol2')
        molecule.write('mol2', mol2_path)


def convert_smiles(smiles, opt_steps=2000):

    # if not osp.exists(save_path):
    #     os.makedirs(save_path)

    clean_idxs = []

    molecules = []

    failed_number = 0

    for sm_idx in tqdm(range(len(smiles))):
        smile = smiles[sm_idx]
        canonized_smile = canonize_smile(smile)

        if canonized_smile:

            molecule = pybel.readstring("smi", smile)

            if (molecule):
                # add hydrogen
                molecule.OBMol.AddHydrogens()

                # minimize the energy
                molecule.make3D(forcefield="gaff",
                                steps=opt_steps)
                molecule.localopt(forcefield="gaff",
                                  steps=opt_steps)
                molecules.append(molecule)
                clean_idxs.append(sm_idx)
            else:

                failed_number += 1

    print(f'{failed_number} failed to convert')

    return molecules


def make_3D_mols(molecules, opt_steps=2000):

    # if not osp.exists(save_path):
    #     os.makedirs(save_path)

    failed_number = 0
    molecules_3d = []

    for mol_idx in tqdm(range(len(molecules))):

        molecule = molecules[mol_idx]

        if (molecule):
            # add hydrogen
            molecule.OBMol.AddHydrogens()

            # minimize the energy
            molecule.make3D(forcefield="gaff",
                            steps=opt_steps)
            molecule.localopt(forcefield="gaff",
                              steps=opt_steps)
            molecules_3d.append(molecule)
        else:

            failed_number += 1

    print(f'{failed_number} failed to convert')

    return molecules_3d
