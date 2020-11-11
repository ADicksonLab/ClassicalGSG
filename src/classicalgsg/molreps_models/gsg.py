import numpy as np
import numpy.linalg as la
import scipy.stats as stats


class GSG(object):

    def __init__(self, wavelet_scale=4,
                 sm_operateros=(True, True, True)):

        self.wavelet_scale = wavelet_scale
        self.sm_operateros = sm_operateros

    def lazy_random_walk(self, adjacency_mat):

        # calcuate degree matrix
        degree_mat = np.sum(adjacency_mat, axis=0)

        # calcuate A/D
        adj_degree = np.divide(adjacency_mat, degree_mat)

        # sets NAN vlaues to zero
        adj_degree = np.nan_to_num(adj_degree)

        identity = np.identity(adj_degree.shape[0])

        return 1/2 * (identity + adj_degree)

    # calcuate the graph wavelets based on the paper
    def graph_wavelet(self, probability_mat):

        # 2^j
        steps = []
        for step in range(self.wavelet_scale):

            steps.append(2 ** step)

        wavelets = []
        for i, j in enumerate(steps):
            wavelet = la.matrix_power(probability_mat, j) \
                - la.matrix_power(probability_mat, 2*j)

            wavelets.append(wavelet)

        return np.array(wavelets)

    def zero_order_feature(self, signal):
        # zero order feature calcuated using signal of the graph.
        features = []

        features.append(np.mean(signal, axis=0))
        features.append(np.var(signal, axis=0))
        features.append(stats.skew(signal, axis=0, bias=False))
        features.append(stats.kurtosis(signal, axis=0, bias=False))

        return np.array(features).reshape(-1, 1)

    def first_order_feature(self, wavelets, signal):

        wavelet_signal = np.abs(np.matmul(wavelets, signal))
        features = []
        features.append(np.mean(wavelet_signal, axis=1))
        features.append(np.var(wavelet_signal, axis=1))
        features.append(stats.skew(wavelet_signal, axis=1, bias=False))
        features.append(stats.kurtosis(wavelet_signal, axis=1, bias=False))

        return np.array(features).reshape(-1, 1)

    def second_order_feature(self, wavelets, signal):

        wavelet_signal = np.abs(np.matmul(wavelets, signal))

        coefficents = []
        for i in range(1, len(wavelets)):
            coefficents.append(np.einsum('ij,ajt ->ait', wavelets[i],
                                         wavelet_signal[0:i]))

        coefficents = np.abs(np.concatenate(coefficents, axis=0))

        features = []

        features.append(np.mean(coefficents, axis=1))
        features.append(np.var(coefficents, axis=1))
        features.append(stats.skew(coefficents, axis=1, bias=False))
        features.append(stats.kurtosis(coefficents, axis=1, bias=False))
        return np.array(features).reshape(-1, 1)

    def wavelets(self, adj_mat):

        probability_mat = self.lazy_random_walk(adj_mat)
        return self.graph_wavelet(probability_mat)

    def features(self, adj_mat, signal):

        probability_mat = self.lazy_random_walk(adj_mat)

        wavelets = self.graph_wavelet(probability_mat)

        gsg_features = []

        if self.sm_operateros[0]:
            gsg_features.append(self.zero_order_feature(signal))

        if self.sm_operateros[1]:
            gsg_features.append(self.first_order_feature(wavelets, signal))

        if self.sm_operateros[2]:
            gsg_features.append(self.second_order_feature(wavelets, signal))

        return np.concatenate(gsg_features, axis=0)
