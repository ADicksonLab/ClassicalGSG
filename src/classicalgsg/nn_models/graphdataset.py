import numpy as np
from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, node_attributes, adj_matrices, labels):
        super(GraphDataset, self).__init__()

        num_nodes = []
        for adj_matrix in adj_matrices:
            num_nodes.append(adj_matrix.shape[0])

        self.max_size = max(num_nodes)

        # Padd data
        self.adj_matrices = self.pad_adj_matrices(adj_matrices)
        self.node_attributes = self.pad_node_attributes(node_attributes)

        self.target = [np.array([label]) for label in labels]
        self.num_features = self.node_attributes[0].shape[1]

    def pad_adj_matrices(self, adj_matrices):

        padded_adj_matrices = []
        for adj_matrix in adj_matrices:
            num_nodes = adj_matrix.shape[0]
            padded_adj_matrix = np.zeros((self.max_size, self.max_size))
            padded_adj_matrix[:num_nodes, :num_nodes] = adj_matrix
            padded_adj_matrices.append(padded_adj_matrix)

        return padded_adj_matrices

    def pad_node_attributes(self, node_attributes):

        padded_node_attributes = []
        for attributes in node_attributes:
            padded_attributes = np.zeros((self.max_size, attributes.shape[1]))
            padded_attributes[:attributes.shape[0],
                              :attributes.shape[1]] = attributes
            padded_node_attributes.append(padded_attributes)

        return padded_node_attributes

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        sample = {'adj_matrix': self.adj_matrices[index].astype('float32'),
                  'node_feature_matrix':
                      self.node_attributes[index].astype('float32'),
                  'labels': self.target[index].astype('float32')}
        return sample
