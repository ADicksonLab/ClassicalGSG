import numpy as np


class OpenChem:

    def __init__(self):
        """FIXME! briefly describe function

        :returns:
        :rtype:

        """

        self.onehot_nums = {'valence': 6, 'charge': 6, 'hybridization': 8,
                            'aromatic': 2, 'atom_element': 11}

    def get_atomic_attributes(self, atom):
        attr_dict = {}
        atomic_num = atom.GetAtomicNum()
        atomic_mapping = {5: 0, 7: 1, 6: 2, 8: 3, 9: 4,
                          15: 5, 16: 6, 17: 7, 35: 8, 53: 9}
        if atomic_num in atomic_mapping.keys():
            attr_dict['atom_element'] = atomic_mapping[atomic_num]
        else:
            attr_dict['atom_element'] = 10

        attr_dict['valence'] = atom.GetTotalValence()
        attr_dict['charge'] = atom.GetFormalCharge()
        attr_dict['hybridization'] = atom.GetHybridization().real
        attr_dict['aromatic'] = int(atom.GetIsAromatic())
        return attr_dict

    def connectivity_matrix(self, rdmol):
        num_nodes = rdmol.GetNumAtoms()
        adj_matrix = np.eye(num_nodes)

        for _, bond in enumerate(rdmol.GetBonds()):
            begin_atom_idx = bond.GetBeginAtomIdx()
            end_atom_idx = bond.GetEndAtomIdx()

            adj_matrix[begin_atom_idx,
                       end_atom_idx] = 1.0
            adj_matrix[end_atom_idx,
                       begin_atom_idx] = 1.0
        return adj_matrix

    def get_molecule_attribute(self, rdmol):

        signals = []

        for atom in rdmol.GetAtoms():
            atom_attributes = self.get_atomic_attributes(atom)

            atom_signal = []
            for key, value in atom_attributes.items():
                atom_signal.extend(list(np.eye(self.onehot_nums[key])[value]))

            signals.append(np.array(atom_signal))

        return signals
