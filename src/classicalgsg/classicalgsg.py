import os.path as osp

from classicalgsg.molreps_models.utils import adjacency_matrix
from classicalgsg.atomic_attr.molecularff import MolecularFF
from classicalgsg.atomic_attr.utils import (coordinates,
                                            connectivy_matrix, smi_to_2D,
                                            smi_to_3D)


AC_TYPES = ['AC1', 'AC5', 'AC31', 'AC36', 'ACall']


class ClassicalGSG(object):

    def __init__(self, **kwargs):
        pass

    def features(self, **kwargs):
        pass


class GAFF2GSG(ClassicalGSG):
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

        assert osp.exists(mol2_file_path), \
            "mol2 file does not exists"

        assert osp.exists(gaffmol2_file_path), \
            "gaffmol2 file does not exists"

        molecularff = MolecularFF(self.AC_type)

        molecule = molecularff.gaff_molecule(mol2_file_path,
                                             gaffmol2_file_path)

        atomic_attributes = molecularff.atomic_attributes(molecule,
                                                          forcefield='GAFF')

        if self.structure == '3D':
            coords = coordinates(mol2_file_path)
            adj_matrix = adjacency_matrix(coords, self.radial_cutoff)

        elif self.structure == '2D':
            adj_matrix = connectivy_matrix(mol2_file_path)

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

        assert osp.exists(mol2_file_path), \
             'mol2 file does not exists'

        assert osp.exists(str_file_path), \
            'str file does not exists'

        molecularff = MolecularFF(self.AC_type)

        molecule = molecularff.cgenff_molecule(mol2_file_path, str_file_path)

        atomic_attributes = molecularff.atomic_attributes(molecule,
                                                          forcefield='CGenFF')
        if self.structure == '3D':
            coords = coordinates(mol2_file_path)
            adj_matrix = adjacency_matrix(coords, self.radial_cutoff)

        elif self.structure == '2D':
            adj_matrix = connectivy_matrix(mol2_file_path)

        return self.gsg.features(adj_matrix, atomic_attributes)


class OBFFGSG(ClassicalGSG):

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

    def features(self, smiles, forcefield):

        molecularff = MolecularFF(self.AC_type)

        molecule = molecularff.openbabel_molecule(smiles,
                                                  forcefield=forcefield)
        if molecule is None:
            return None

        atomic_attr = molecularff.atomic_attributes(molecule,
                                                    forcefield=forcefield)
        if self.structure == '3D':
            coords = smi_to_3D(smiles)
            adj_matrix = adjacency_matrix(coords, self.radial_cutoff)

        elif self.structure == '2D':
            adj_matrix = smi_to_2D(smiles)

        return self.gsg.features(adj_matrix, atomic_attr)
