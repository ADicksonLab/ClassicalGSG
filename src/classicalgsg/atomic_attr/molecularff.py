import os.path as osp
import numpy as np

from collections import namedtuple, defaultdict

from openbabel import openbabel, pybel
from parmed.charmm import CharmmParameterSet

from classicalgsg.atomic_attr.utils import (mol2_parser,
                                            one_hot_encode)

Atom = namedtuple('Atom', 'element, atom_type, charge, hyb')
LJParam = namedtuple('LJParam',
                     'atom_type, atype_encodings, radius, epsilon')


class MolecularFF(object):

    PARAM_FILE_PATH = osp.join(osp.dirname(__file__),
                               'forcefields_params')
    CGENFF_PARAM_FILE = osp.join(PARAM_FILE_PATH,
                                 'par_all36_cgenff.prm')
    GAFFLJ_FILE = osp.join(PARAM_FILE_PATH, 'gaff2_lj.dat')

    # Read Lj paramter files of cgenff and gaff
    def __init__(self, AC_type='AC36'):

        self.ATOM_TYPE_CATEGORIES = AC_type

        self.CGENFF_PARAMS = CharmmParameterSet(
            osp.join(self.PARAM_FILE_PATH,
                     self.CGENFF_PARAM_FILE))
        # 159
        self.NUM_CGENFF_ATOM_TYPES = len(self.CGENFF_PARAMS.atom_types)

        self.NUM_GAFF_ATOM_TYPES = 95

        self.cgenff_AC36 = self.AC36()
        self.gaff_AC31 = self.AC31()

    # make a dictionary of cgenff lj parameters for each atom type
    def cgenff_lj(self):

        ljparams = {}
        atom_idx = 0
        for atom_type, params in self.CGENFF_PARAMS.atom_types.items():

            # generate one hot encoding for all atom types in the cgenff
            one_hot_encoding = one_hot_encode(self.NUM_CGENFF_ATOM_TYPES,
                                              atom_idx)

            ljparams[atom_type] = LJParam(atom_type=atom_type,
                                          atype_encodings=one_hot_encoding,
                                          radius=params.rmin,
                                          epsilon=params.epsilon)

            atom_idx += 1

        return ljparams

    # reads the gaff lj parameter files
    def gaff_lj(self):

        f = open(self.GAFFLJ_FILE, 'r')
        ljparams = {}
        for atom_idx, line in enumerate(f):
            line = line.strip()
            words = line.split()
            atom_type = words[0]
            radius = float(words[1])
            epsilon = float(words[2])
            one_hot_encoding = one_hot_encode(self.NUM_CGENFF_ATOM_TYPES,
                                              atom_idx)
            ljparams[atom_type] = LJParam(atom_type=atom_type,
                                          atype_encodings=one_hot_encoding,
                                          radius=radius,
                                          epsilon=epsilon)
        return ljparams

    def molecule(self, mol2_file):

        molecule = next(pybel.readfile("mol2", mol2_file))

        mol_attr = defaultdict(list)

        for atom in molecule.atoms:
            element = openbabel.GetSymbol(atom.atomicnum)
            mol_attr['element'].append(element)
            mol_attr['hyb'].append(atom.hyb)

        return mol_attr

    def cgenff_molecule(self, mol2_file, str_file):
        """FIXME! briefly describe function

        :param str_file:
        :param mol2_file:
        :returns:
        :rtype:

        """

        str_file = open(str_file, 'r')
        molecule = self.molecule(mol2_file)
        cgenffmolecule = []
        idx = 0
        for line in str_file:
            line = line.strip()
            if line[:4].upper() == 'ATOM':
                words = line.split()
                atom_type = words[2].upper()
                charge = float(words[3])
                if atom_type not in self.CGENFFLJ.keys():
                    return []
                if atom_type != 'LPH':
                    atom = Atom(atom_type=atom_type,
                                element=molecule['element'][idx],
                                hyb=molecule['hyb'][idx],
                                charge=charge)

                    cgenffmolecule.append(atom)
                    idx += 1

        return cgenffmolecule

    def gaff_molecule(self, mol2_file, gaffmol2_file):
        sections = mol2_parser(gaffmol2_file)
        molecule = self.molecule(mol2_file)

        gaffmolecule = []
        for idx, line in enumerate(sections['atom']):
            words = line.split()
            gaff_atom_type = words[6]
            charge = float(words[8])
            gaffmolecule.append(Atom(element=molecule['element'][idx],
                                     atom_type=gaff_atom_type,
                                     hyb=molecule['hyb'][idx],
                                     charge=charge))

        return gaffmolecule

    @property
    def CGENFFLJ(self):
        return self.cgenff_lj()

    @property
    def GAFFLJ(self):
        return self.gaff_lj()

    def AC36(self):
        atom_encodings = {}
        num_cats = 36
        cgenff_36_path = osp.join(self.PARAM_FILE_PATH, 'cgenff_AC36.dat')
        cgenff_36 = open(cgenff_36_path, 'r')
        for line in cgenff_36:
            line = line.strip()
            words = line.split()

            atom_encodings.update({words[0]:
                                   one_hot_encode(num_cats, int(words[2]))})

        return atom_encodings

    def AC31(self):
        atom_encodings = {}
        num_cats = 31
        gaff_31_path = osp.join(self.PARAM_FILE_PATH, 'gaff_AC31.dat')
        gaff_31 = open(gaff_31_path, 'r')
        for line in gaff_31:
            line = line.strip()
            words = line.split()
            atom_encodings.update({words[0]:
                                   one_hot_encode(num_cats, int(words[2]))})

        return atom_encodings

    def AC5(self, element, hyb_value):
        """FIXME! briefly describe function

        :param element:
        :param hyb_value:
        :returns:
        :rtype:

        """

        element = element.strip()

        if element == 'H':
            category = 0
        elif element == 'O' or element == 'N':
            category = 1
        elif element == 'C' and hyb_value < 3:
            category = 2
        elif element == 'C' and hyb_value == 3:
            category = 3
        else:
            category = 4

        return np.eye(self.ATOM_TYPE_CATEGORIES)[np.array(category)]

    def atomic_attributes(self, molecule, forcefield='CGenFF'):

        if forcefield == 'CGenFF':
            ff_ljparams = self.CGENFFLJ

        elif forcefield == 'GAFF':
            ff_ljparams = self.GAFFLJ

        mol_signals = []
        for atom_idx, atom in enumerate(molecule):
            atom_params = ff_ljparams[atom.atom_type]

            if self.ATOM_TYPE_CATEGORIES == 'AC1':

                atom_signal = np.array([atom.charge,
                                        atom_params.radius,
                                        atom_params.epsilon])

            elif self.ATOM_TYPE_CATEGORIES == 'AC5':
                atype_encodings = self.AC5(atom.element, atom.hyb)
                atom_signal = np.concatenate((atype_encodings,
                                              np.array([atom.charge,
                                                        atom_params.radius,
                                                        atom_params.epsilon]
                                              )))

            elif self.ATOM_TYPE_CATEGORIES == 'AC36':
                atom_signal = np.concatenate((self.cgenff_AC36[atom.atom_type],
                                              np.array([atom.charge,
                                                        atom_params.radius,
                                                        atom_params.epsilon])))

            elif self.ATOM_TYPE_CATEGORIES == 'AC31':
                atom_signal = np.concatenate((self.gaff_AC31[atom.atom_type],
                                              np.array([atom.charge,
                                                        atom_params.radius,
                                                        atom_params.epsilon]
                                              )))

            elif self.ATOM_TYPE_CATEGORIES == 'ACall':
                atom_signal = np.concatenate((atom_params.atype_encodings,
                                              np.array([atom.charge,
                                                        atom_params.radius,
                                                        atom_params.epsilon]
                                              )))

            mol_signals.append(atom_signal)

        return np.array(mol_signals)
