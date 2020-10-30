import os
import os.path as osp

import numpy as np
from collections import defaultdict
from classicalgsg.molreps_models.utils import adjacency_matrix
from classicalgsg.atomic_attr.classicalmd import ClassicalMD
from classicalgsg.atomic_attr.utils import (coordinates,
                                            connectivy_matrix)


AC_TYPES = ['AC1', 'AC5', 'AC31', 'AC36', 'ACall']

class ClassicalGSG(object):

    def __init__(self, **kwargs):
        pass

    def features(self, **kwargs):
        pass

class GAFFGSG(ClassicalGSG):
    def __init__(self, gsg, structure='3D', AC_type='AC1',
                 radial_cutoff=7.5):
        """FIXME! briefly describe function

        :param mrmodel:
        :param structure:
        :param AC_type:
        :param radial_cutoff:
        :returns:
        :rtype:

        """

        self.gsg = gsg
        self.structure = structure
        self.AC_type = AC_type
        self.radial_cutoff = radial_cutoff


    def features(self, mol2_file_path, gaffmol2_file_path):

        assert not osp.exists(mol2_file_path), \
            "mol2 file does not exists"


        assert not osp.exists(gaffmol2_file_path), \
            "gaffmol2 file does not exists"


        classicalmd = ClassicalMD(self.AC_type)

        molecule = classicalmd.gaff_molecule(mol2_file_path, gaffmol2_file_path)


        atomic_attributes = classicalmd.atomic_attributes(molecule, forcefield='GAFF')


        if self.structure == '3D':
            coords = coordinates(mol2_file_path)
            adj_matrix = adjacency_matrix(coords, self.radial_cutoff)

        elif self.structure == '2D':
            adj_matrix = connectivy_matrix(mol2_file_path)

        wavelets = self.gsg.wavelets(adj_matrix)


        return self.gsg.molecular_features(adj_matrix, atomic_attributes)



class CGenFFGSG(ClassicalGSG):
    def __init__(self, gsg, structure='3D', AC_type='AC1',
                 radial_cutoff=7.5):
        """FIXME! briefly describe function

        :param mrmodel:
        :param structure:
        :param AC_type:
        :param radial_cutoff:
        :returns:
        :rtype:

        """

        self.gsg = gsg
        self.structure = structure
        self.AC_type = AC_type
        self.radial_cutoff = radial_cutoff


    def features(self, mol2_file_path, str_file_path):

        # assert osp.exists(mol2_file_path), \
        #     "mol2 file does not exists"


        # assert  osp.exists(str_file_path), \
        #     "str file does not exists"


        classicalmd = ClassicalMD(self.AC_type)

        molecule = classicalmd.cgenff_molecule(mol2_file_path, str_file_path)


        atomic_attributes = classicalmd.atomic_attributes(molecule,
                                                          forcefield='CGenFF')


        if self.structure == '3D':
            coords = coordinates(mol2_file_path)
            adj_matrix = adjacency_matrix(coords, self.radial_cutoff)

        elif self.structure == '2D':
            adj_matrix = connectivy_matrix(mol2_file_path)



        return self.gsg.features(adj_matrix, atomic_attributes)
