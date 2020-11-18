import sys
import os.path as osp
import numpy as np
import pkgutil

from collections import namedtuple, defaultdict


from openbabel import openbabel, pybel
from parmed.charmm import CharmmParameterSet

from classicalgsg.atomic_attr.utils import (mol2_parser,
                                            one_hot_encode)

Atom = namedtuple('Atom', 'element, atom_type, charge, hyb')
FFParam = namedtuple('FFParam',
                     'atom_type, atype_encodings, radius, epsilon')


class MolecularFF(object):

    FFPARAMS_DIRNAME = 'forcefields_params'

    def __init__(self, AC_type='AC36'):

        self.ATOM_TYPE_CATEGORIES = AC_type
        self.gaff_params = self.get_gaff_params()
        self.cgenff_params = self.get_cgenff_params()
        self.cgenff_AC36 = self.AC36()
        self.gaff_AC31 = self.AC31()

    # make a dictionary of cgenff lj parameters for each atom type
    def get_cgenff_params(self):

        path = osp.dirname(sys.modules[__name__].__file__)
        cgenff_paramfile_path = osp.join(path,
                                         self.FFPARAMS_DIRNAME,
                                         'par_all36_cgenff.prm')
        cgenff_params = CharmmParameterSet(cgenff_paramfile_path)
        atype_nums = len(cgenff_params.atom_types)

        ffparams = {}
        atom_idx = 0
        for atom_type, params in cgenff_params.atom_types.items():

            # generate one hot encoding for all atom types in the cgenff
            one_hot_encoding = one_hot_encode(atype_nums,
                                              atom_idx)

            ffparams[atom_type] = FFParam(atom_type=atom_type,
                                          atype_encodings=one_hot_encoding,
                                          radius=params.rmin,
                                          epsilon=params.epsilon)

            atom_idx += 1

        return ffparams

    # reads the gaff lj parameter files
    def get_gaff_params(self):

        gaff_param_file = pkgutil.get_data(__name__,
                                           osp.join(self.FFPARAMS_DIRNAME,
                                                    'gaff2.dat'))
        gaff_param_text = gaff_param_file.decode()

        Flag = False
        lj_section = []
        for line in gaff_param_text.splitlines():
            line = line.strip()
            if Flag and line.startswith('END'):
                break

            if Flag and len(line) > 0:
                lj_section.append(line)

            if line.startswith('MOD4'):
                Flag = True

        atype_nums = len(lj_section)
        params = {}
        for atom_idx, line in enumerate(lj_section):
            line = line.strip()
            words = line.split()
            atom_type = words[0]
            radius = float(words[1])
            epsilon = float(words[2])
            one_hot_encoding = one_hot_encode(atype_nums,
                                              atom_idx)
            params[atom_type] = FFParam(atom_type=atom_type,
                                        atype_encodings=one_hot_encoding,
                                        radius=radius,
                                        epsilon=epsilon)
        return params

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
                if atom_type not in self.cgenff_params.keys():
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


    def AC36(self):
        cgenff36_param_file = pkgutil.get_data(__name__,
                                               osp.join(self.FFPARAMS_DIRNAME,
                                                        'cgenff_AC36.dat'))
        cgenff36_param_text = cgenff36_param_file.decode()

        atom_encodings = {}
        num_cats = 36
        for line in cgenff36_param_text.splitlines():
            line = line.strip()
            words = line.split()

            atom_encodings.update({words[0]:
                                   one_hot_encode(num_cats, int(words[2]))})

        return atom_encodings

    def AC31(self):

        gaff31_param_file = pkgutil.get_data(__name__,
                                             osp.join(self.FFPARAMS_DIRNAME,
                                                      'gaff_AC31.dat'))
        gaff31_param_text = gaff31_param_file.decode()

        atom_encodings = {}
        num_cats = 31
        for line in gaff31_param_text.splitlines():
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
            ff_params = self.cgenff_params

        elif forcefield == 'GAFF':
            ff_params = self.gaff_params

        mol_signals = []
        for atom_idx, atom in enumerate(molecule):
            atom_params = ff_params[atom.atom_type]

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
