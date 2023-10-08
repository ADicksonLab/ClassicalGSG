import os
import os.path as osp
import numpy as np
from joblib import dump

import pandas as pd
import pickle as pkl
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from skorch import NeuralNetRegressor
from torch.optim.lr_scheduler import StepLR
from skorch.callbacks.lr_scheduler import LRScheduler

from classicalgsg.nn_models.dataloader import DataLoader
from classicalgsg.nn_models.models import GSGNN
from classicalgsg.molreps_models.utils import scop_to_str

# Running training guowei dataset of different features
# The Gaff forcefield without atom types is going to be used

DATASET = 'OpenChem'

MODELS_SAVE_PATH = f'models/{DATASET}'

# Example GSG parameters
# GSG_PARAMS = {'wavelet_scale': [4, 5],
#               'scattering_operators': ['(z,f,s)', '(z,f)', '(z,s)', '(f,s)']}

GSG_PARAMS = {'wavelet_scale': [4],
              'scattering_operators': ['(z,f,s)']}


def report(results, n_top):
    num_tests = results['rank_test_score'].shape[0]
    for i in range(min(n_top, num_tests)):
        candidate = np.flatnonzero(results['rank_test_score'] == i+1)[0]
        print(f'Model with rank: {i:0}')
        print('Mean validation score: '
              f'{results["mean_test_score"][candidate]:.3f}'
              f' (std: {results["std_test_score"][candidate]:.3f})')
        print(f'Parameters: {results["params"][candidate]}')


def split_dataset(wavelet_step, scattering_operators, ratio=0.2):
    dataset_save_path = osp.join(f'data_{wavelet_step}_'
                                 f'{scop_to_str(scattering_operators)}',
                                 f'{DATASET}')

    save_file_path = osp.join(dataset_save_path, f'{DATASET}.pkl')
    # Split the data
    with open(save_file_path, 'rb') as rfile:
        data = pkl.load(rfile)

    data = pd.DataFrame.from_dict(data)
    train_data, test_data = train_test_split(data,
                                             test_size=ratio,
                                             random_state=21)
    train_data = train_data.to_dict('list')
    test_data = test_data.to_dict('list')

    train_data_path = osp.join(dataset_save_path, f'{DATASET}_training.pkl')
    test_data_path = osp.join(dataset_save_path, f'{DATASET}_test.pkl')

    with open(train_data_path, 'wb') as wfile:
        pkl.dump(train_data, wfile)

    with open(test_data_path, 'wb') as wfile:
        pkl.dump(test_data, wfile)


def create_model(wavelet_scale, scattering_operators):

    print('Start training model '
          f'{wavelet_scale} {scattering_operators}')

    split_dataset(wavelet_scale, scattering_operators)

    model_save_path = osp.join(MODELS_SAVE_PATH,
                               f'model_{wavelet_scale}_'
                               f'{scop_to_str(scattering_operators)}.pkl')

    if not osp.exists(model_save_path):
        data_loader = DataLoader(DATASET,
                                 f'data_{wavelet_scale}_'
                                 f'{scop_to_str(scattering_operators)}')

        data = data_loader.load_data()

        x_train, y_train, _ = data[f'{DATASET}_training']
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # normalize data to 0 mean and unit std
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)

        # save the scaler for testing
        scaler_save_path = osp.join(MODELS_SAVE_PATH,
                               f'std_scaler_{wavelet_scale}_'
                               f'{scop_to_str(scattering_operators)}.sav')
        dump(scaler, scaler_save_path, compress=True)

        n_in = x_train.shape[1]
        lr_policy = LRScheduler(StepLR, step_size=15, gamma=0.5)

        net = NeuralNetRegressor(
            GSGNN,
            module__n_in=n_in,
            criterion=torch.nn.MSELoss,
            max_epochs=400,
            optimizer=torch.optim.Adam,
            optimizer__lr=.005,
            callbacks=[lr_policy],
            device='cpu',
            batch_size=256,
            #verbose=0,
        )

        params = {
                'module__n_h': [300, 400, 500],
                'module__dropout': [0.2, 0.4],
                'module__n_layers': [1, 2, 3, 4],
        }

        gs = GridSearchCV(net,
                          params,
                          refit=True,
                          cv=5,
                          scoring='r2',
                          n_jobs=-1)
  
        gs.fit(x_train, y_train)

        # save the trained model
        print(f"Save the model in {model_save_path}")
        torch.save(gs.best_estimator_, model_save_path)
        report(gs.cv_results_, 10)


if __name__ == '__main__':

    if not osp.exists(MODELS_SAVE_PATH):
        os.makedirs(MODELS_SAVE_PATH)

    for wavelet_scale in GSG_PARAMS['wavelet_scale']:
        for scattering_operators in GSG_PARAMS['scattering_operators']:
            create_model(wavelet_scale, scattering_operators)
