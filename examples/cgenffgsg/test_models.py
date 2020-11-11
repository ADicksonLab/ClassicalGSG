import os
import os.path as osp

from itertools import product

import torch
from sklearn.preprocessing import StandardScaler

from classicalgsg.nn_models.dataloader import DataLoader
from classicalgsg.nn_models.test import Test
from classicalgsg.nn_models.reporter import TestReporter
from classicalgsg.molreps_models.utils import scop_to_str, scop_to_boolean

TRAIN_DATASET = 'ChEMBL'
TRAINED_MODELS_PATH = f'models/{TRAIN_DATASET}'
FORCEFIELD = 'CGenFF'
AC_TYPE = 'AC36'


GSG_PARAMS = {'wavelet_scale': [4],
              'scattering_operators': ['(z,f,s)']}
DEVICE = 'cpu'


def run_tests(results_save_path):

    params = product(GSG_PARAMS['wavelet_scale'],
                     GSG_PARAMS['scattering_operators'])

    reporter = TestReporter(TRAIN_DATASET, FORCEFIELD, AC_TYPE)

    results_str = []
    results_df = []

    for wavelet_scale, scattering_operators in params:

        print(f"Start testing model {wavelet_scale} {scattering_operators}")

        # load the models
        model_save_path = osp.join(TRAINED_MODELS_PATH,
                                   f'model_{wavelet_scale}_'
                                   f'{scop_to_str(scattering_operators)}.pkl')

        trained_model = torch.load(model_save_path)

        dataset_path = osp.join(f'data_{wavelet_scale}_'
                                f'{scop_to_str(scattering_operators)}')

        data_loader = DataLoader(TRAIN_DATASET, dataset_path)

        ChEMBL_data = data_loader.load_data()

        test_sets = {'ChEMBL': ChEMBL_data['ChEMBL_test']}

        x_train = ChEMBL_data['ChEMBL_training'][0]
        scaler = StandardScaler()
        scaler.fit(x_train)

        for key in test_sets.keys():
            test_sets[key] = (scaler.transform(test_sets[key][0]),
                              test_sets[key][1])

        tester = Test(DEVICE)

        results = {}
        for test_set_name, test_set in test_sets.items():
            predictions, experimental = tester.test(trained_model, test_set)

            results.update({test_set_name:
                            tester.evaluate(predictions, experimental)})

        scop_boolean = scop_to_boolean(scattering_operators)
        test_result_df, test_str = reporter.result(wavelet_scale,
                                                   scop_boolean,
                                                   results)
        results_str.append(test_str)

        test_result_df.insert(0,
                              'wavelet_scale',
                              [wavelet_scale for _ in range(len(test_sets))]
                              )

        test_result_df.insert(1,
                              'Scattetring Operators',
                              [scattering_operators for _
                               in range(len(test_sets))]
                              )

        results_df.append(test_result_df)

        reporter.save_txt(results_str,
                          f'{results_save_path}/results.org')

        reporter.save_pickle(results_df,
                             f'{results_save_path}/results.pkl')


if __name__ == '__main__':

    results_save_path = 'results'
    if not osp.exists(results_save_path):
        os.makedirs(results_save_path)

    run_tests(results_save_path)
