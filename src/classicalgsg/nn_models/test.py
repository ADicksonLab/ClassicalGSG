import numpy as np

from classicalgsg.nn_models.evalmetrics import EvalMetrics


class Test:

    def __init__(self, device):
        self.device = device

    def test(self, model, test_set):

        test_x, test_y = test_set

        predictions = np.squeeze(model.predict(test_x.astype(np.float32)))

        experimental = np.squeeze(test_y.astype(np.float32))

        return predictions, experimental

    def evaluate(self, predictions, experimental):
        """FIXME! briefly describe function

        :param predictions:
        :param experimental:
        :returns:
        :rtype:

        """

        metrics = EvalMetrics(predictions, experimental)
        return metrics.evaluate(['PCC', 'RMSE', 'MUE', 'ErrorRange'])
