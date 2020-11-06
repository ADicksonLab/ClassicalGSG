import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.numpy.linalg as la
import autograd.scipy.stats as stats
from .utils import (distance_matrix, adjacency_matrix,
                                       angle_records, fc)
from .stats import skew, kurtosis, matrix_power
#is in Angstruma

class GSG(object):

    def __init__(self, wavelet_step_num=4,
                 sm_operateros=(True, True, True)):

        self.wavelet_step_num = wavelet_step_num
        self.sm_operateros = sm_operateros

    def lazy_random_walk(self, adjacency_mat):

        # calcuate degree matrix
        degree_mat = np.sum(adjacency_mat, axis=0)

        # calcuate A/D
        adj_degree = np.divide(adjacency_mat, degree_mat)

        # sets NAN vlaues to zero
        #adj_degree = np.nan_to_num(adj_degree)

        identity = np.identity(adj_degree.shape[0])

        return 1/2 * (identity + adj_degree)

    #calcuate the graph wavelets based on the paper
    def graph_wavelet(self, probability_mat):

        # 2^j
        steps = []
        for step in range(self.wavelet_step_num):

            steps.append(2 ** step)

        wavelets = []
        for i, j in enumerate(steps):
            wavelet = matrix_power(probability_mat, j) \
                - matrix_power(probability_mat, 2*j)

            wavelets.append(wavelet)


        return np.array(wavelets)

    def zero_order_feature(self, signal):
        #zero order feature calcuated using signal of the graph.
        features = []

        features.append(np.mean(signal, axis=0))
        features.append(np.var(signal, axis=0))
        features.append(skew(signal, bias=False, axis=0))
        features.append(kurtosis(signal, bias=False, axis=0))

        return np.array(features).reshape(-1, 1)

    def first_order_feature(self, wavelets, signal):

        wavelet_signal = np.abs(np.matmul(wavelets, signal))
        features = []
        features.append(np.mean(wavelet_signal, axis=1))
        features.append(np.var(wavelet_signal, axis=1))
        features.append(skew(wavelet_signal, bias=False, axis=1))
        features.append(kurtosis(wavelet_signal, bias=False, axis=1))

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
        features.append(skew(coefficents, bias=False, axis=1))
        features.append(kurtosis(coefficents, bias=False, axis=1))
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