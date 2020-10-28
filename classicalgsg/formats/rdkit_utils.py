import os
import os.path  as osp
import numpy as np
import warnings
import csv
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

#from OpenChem
def read_smiles_property_file(path, cols_to_read, delimiter=',', keep_header=False):
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
        new_smile = Chem.MolToSmiles(Chem.MolFromSmiles(smile, sanitize=canonize))

    except:

        warnings.warn(smile + ' can not be canonized:'
                      'nvalid SMILES string!')


    return new_smile


def save_to_pdb(molecules, mol_names, save_path):

    if not osp.exists(save_path):
        os.mkdir(save_path)

    for mol_idx, molecule in enumerate(molecules):
        pdb_path = osp.join(save_path, f'{mol_names[mol_idx]}.pdb')
        Chem.MolToPDBFile(molecule, pdb_path)


def save_logP(logp_values, mol_names, save_path):

    if not osp.exists(save_path):
        os.mkdir(save_path)

    for idx, logp in enumerate(logp_values):
        logp_file_path = osp.join(save_path, f'{mol_names[idx]}.exp')

        with open(logp_file_path, 'w') as wfile:
            wfile.write(logp)

def save_smile(smiles, molecule_ids, save_path):

    if not osp.exists(save_path):
        os.mkdir(save_path)


    for sm_idx in tqdm(range(len(smiles))):
        smile = smiles[sm_idx]
        canonized_smile = canonize_smile(smile)
        if canonized_smile:
            smile_file_path = osp.join(save_path, f'{molecule_ids[sm_idx]}.smi')
            with open(smile_file_path, 'w') as wfile:
                wfile.write(smile)

def convert_smiles(smiles, opt_steps=3000):

    clean_idxs = []

    molecules = []

    for sm_idx in tqdm(range(len(smiles))):
        smile = smiles[sm_idx]
        canonized_smile = canonize_smile(smile)

        if canonized_smile:

            # generating 3D coordinates
            m = Chem.MolFromSmiles(smile)
            mHs = Chem.AddHs(m)
            embedError = AllChem.EmbedMolecule(mHs, useRandomCoords=True)

            if embedError == 0 :

                UffoptError = AllChem.UFFOptimizeMolecule(mHs, opt_steps)

                if UffoptError != 0 :

                    warnings.warn(f'UFF optimization failed, trying MMFF optimization for {smile}')

                    MMFFoptError = AllChem.MMFFOptimizeMolecule(mHs, opt_steps)

                    if MMFFoptError != 0 :

                        warnings.warn(f'MMFF optimizaiton has also failed on: {smile}')

                    else:
                        molecules.append(mHs)
                        clean_idxs.append(sm_idx)

                else:

                    molecules.append(mHs)
                    clean_idxs.append(sm_idx)

            else:

                warnings.warn(f'Embedding Failed for:{smile}')



    return molecules
