import numpy as np
import scipy.stats as st
from tabulate import tabulate
from sklearn import metrics


class EvalMetrics:
    def __init__(self, prediction, experimental):
        self.prediction = prediction
        self.experimental = experimental

    @property
    def RMSE(self):
        return self.MSE ** (1/2)

    @property
    def MSE(self):
        return metrics.mean_squared_error(self.prediction, self.experimental)

    @property
    def MUE(self):
        return metrics.mean_absolute_error(self.prediction, self.experimental)

    @property
    def PCC(self):
        x = self.prediction
        y = self.experimental
        _, _, r_value, _, _ = st.linregress(x, y)
        return r_value**2

    @property
    def ErrorRange(self):

        # 0.5 < acceptable
        # 0.5< <1 disputable
        # >1 unacceptable

        categories = {'<0.5': 0, '<1': 0, '>1': 0}

        diffs = np.abs(self.prediction - self.experimental)

        for diff_value in diffs:
            if diff_value < 0.5:
                count = categories['<0.5'] + 1
                categories.update({'<0.5': count})

            elif diff_value >= 0.5 and diff_value < 1.0:
                count = categories['<1'] + 1
                categories.update({'<1': count})

            elif diff_value >= 1:
                count = categories['>1'] + 1
                categories.update({'>1': count})

        n_items = diffs.shape[0]

        # make percentages
        for key, value in categories.items():
            categories.update({key: np.round(value*100/n_items, 3)})

        return categories

    def evaluate(self, metrics):

        results = {}

        for metric in metrics:
            if metric == 'MSE':
                results.update({metric: self.MSE})

            elif metric == 'MUE':
                results.update({metric: self.MUE})

            elif metric == 'RMSE':
                results.update({metric: self.RMSE})

            elif metric == 'PCC':
                results.update({metric: self.PCC})
            elif metric == 'ErrorRange':
                results.update(self.ErrorRange.items())

        return results


class BBBEvalMetrics:
    def __init__(self, prediction, experimental):
        self.prediction = prediction
        self.experimental = experimental
        self.init()

    def init(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0

        for idx in range(self.experimental.shape[0]):

            if self.prediction[idx] == self.experimental[idx] == 1:
                self.TP += 1

            if self.prediction[idx] == self.experimental[idx] == 0:
                self.TN += 1

            if self.prediction[idx] == 1 and self.experimental[idx] == 0:
                self.FP += 1

            if self.prediction[idx] == 0 and self.experimental[idx] == 1:
                self.FN += 1

    @property
    def Accuracy(self):
        return metrics.accuracy_score(self.prediction, self.experimental)

    @property
    def Sensitivity(self):
        return self.TP/(self.TP + self.FN)

    @property
    def Specificity(self):
        return self.TN/(self.TN+self.FP)

    @property
    def AUC(self):

        fpr, tpr, thresholds = metrics.roc_curve(self.prediction,
                                                 self.experimental)
        return metrics.auc(fpr, tpr)

    def evaluate(self, metrics):

        results = {}

        for metric in metrics:
            if metric == 'AUC':
                results.update({metric: self.AUC})

            elif metric == 'Accuracy':
                results.update({metric: self.Accuracy})

            elif metric == 'Sensitivity':
                results.update({metric: self.Sensitivity})

            elif metric == 'Specificity':
                results.update({metric: self.Specificity})

        return results


def print_results(results, headers):

    table_headers = [[item for item in results.keys()]]

    data = [value for value in results.values()]
    if headers:
        print(tabulate(data, headers=table_headers))
    else:
        print(tabulate(data))
