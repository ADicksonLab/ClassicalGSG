import os.path as osp
import numpy as np
from openbabel import openbabel, pybel


def mol2_parser(mol2_file_name):
    sections = {}
    cursor = None
    with open(mol2_file_name) as mol2file:
        for line in mol2file:
            if "@<TRIPOS>" in line:
                cursor = line.split("@<TRIPOS>")[1].strip().lower()
                sections[cursor] = []
                continue
            elif line.startswith("#") or line == "\n":
                continue
            sections[cursor].append(line.strip())

    return sections


def connectivy_matrix(mol2_file):
    sections = mol2_parser(mol2_file)
    n_atoms = len(sections['atom'])
    connect_mat = np.zeros((n_atoms, n_atoms))
    for line in sections['bond']:
        (_, atom1_idx, atom2_idx, _) = line.split()
        connect_mat[int(atom1_idx) - 1, int(atom2_idx) - 1] = 1
        connect_mat[int(atom2_idx) - 1, int(atom1_idx) - 1] = 1

    return connect_mat


def coordinates(mol2_file):

    molecule = next(pybel.readfile("mol2", mol2_file))

    coords = []

    # data = defaultdict(list)
    for atom in molecule.atoms:
        coords.append(atom.coords)

    return np.array(coords)


def read_logp(logp_file):

    if osp.exists(logp_file):
        with open(logp_file, 'r') as rfile:
            return float(rfile.read().strip())

    else:
        print(f'There is no file names {logp_file}')
        return None


def one_hot_encode(num_cats, cat_idx):
    return np.eye(num_cats)[cat_idx]


def smi_to_2D(smiles):
    mol = pybel.readstring('smi', smiles)
    mol.OBMol.AddHydrogens()
    mol.make2D()
    num_atoms = mol.OBMol.NumAtoms()
    connect_mat = np.zeros((num_atoms, num_atoms))

    for bond in openbabel.OBMolBondIter(mol.OBMol):
        bpoint = bond.GetBeginAtomIdx() - 1
        epoint = bond.GetEndAtomIdx() - 1
        connect_mat[bpoint, epoint] = 1
        connect_mat[epoint, bpoint] = 1

    return connect_mat


def smi_to_3D(smiles):
    OPT_STEP = 2000
    mol = pybel.readstring('smi', smiles)
    mol.OBMol.AddHydrogens()

    # make 3D
    mol.make3D(forcefield="gaff", steps=OPT_STEP)
    mol.localopt(forcefield="gaff", steps=OPT_STEP)

    coords = []

    for atom in mol.atoms:
        coords.append(atom.coords)

    return np.array(coords)
