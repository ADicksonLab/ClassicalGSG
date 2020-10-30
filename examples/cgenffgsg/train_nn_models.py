import os
import sys
import os.path as osp
import numpy as np

import pandas as pd
import pickle as pkl


import torch

from classicalgsg.nn_models.dataloader import DataLoader
from classicalgsg.nn_models.models import GSGNN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import skorch
from skorch import NeuralNetRegressor

from sklearn.model_selection import GridSearchCV

from torch.optim.lr_scheduler import StepLR
from skorch.callbacks.lr_scheduler import LRScheduler



# Running training guowei dataset of different features
# The Gaff forcefield without atom types is going to be used

DATASET = 'chEMBL21'

OUTPUT_PATH = f'models/{DATASET}'
FORCEFIELD = 'CGenFF'

GSG_PARAMS = {'wavelet_step_num': [4, 5],
          'scattering_operators': ['(z,f,s)','(z,f)', '(z,s)', '(f,s)']}



def report(results, n_top=5):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print(f'Model with rank: {i:0}')
                print(f'Mean validation score: {results["mean_test_score"][candidate]:.3f}'\
                '(std: {results["std_test_score"][candidate]:.3f})')
                print(f'Parameters: {results["params"][candidate]:0}')

def split_dataset(wavelet_step, scattering_operators, ratio=0.2):
    dataset_save_path = osp.join(f'data_{wavelet_step}_{scattering_operators}',
                                 f'{DATASET}')

    save_file_path = osp.join(dataset_save_path, f'{DATASET}.pkl')
    #Split the data
    with open(save_file_path, 'rb') as rfile:
        data = pkl.load(rfile)

    data = pd.DataFrame.from_dict(data)
    train_data, test_data = train_test_split(data, test_size=ratio, random_state=21)
    train_data = train_data.to_dict('list')
    test_data = test_data.to_dict('list')

    train_data_path = osp.join(dataset_save_path, f'{DATASET}_training.pkl')
    test_data_path = osp.join(dataset_save_path, f'{DATASET}_test.pkl')

    with open(train_data_path, 'wb') as wfile:
        pkl.dump(train_data, wfile)

    with open(test_data_path, 'wb') as wfile:
        pkl.dump(test_data, wfile)


def create_model(wavelet_step_num, scattering_operators):


    print(f"Start training model {wavelet_step_num} {scattering_operators}")

    split_dataset(wavelet_step_num, scattering_operators)

    data_loader = DataLoader(DATASET,
                             f'data_{wavelet_step_num}_{scattering_operators}')


    model_save_path = osp.join(OUTPUT_PATH,
                               f'model_{FORCEFIELD}_{wavelet_step_num}'\
                               '{scattering_operators}.pkl')

    if not osp.exists(model_save_path):

        data = data_loader.load_data()


        x_train, y_train, _ = data[f'{DATASET}_training']

        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # normalize data to 0 mean and unit std
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)

        n_in = x_train.shape[1]
        # normalize data to 0 mean and unit std
        lr_policy = LRScheduler(StepLR, step_size=15, gamma=0.5)
        net = NeuralNetRegressor(
            GSGNN,
            criterion=torch.nn.MSELoss,
            max_epochs=400,
            optimizer=torch.optim.Adam,
            optimizer__lr=.005,
            callbacks=[lr_policy],
            device='cpu',
            batch_size=256,
            verbose=0,
        )

        #train_split:None or callable (default=skorch.dataset.CVSplit(5)) in params

        params = {
                'module__n_h': [300, 400, 500],
                'module__dropout' : [0.2, 0.4],
                'module__n_layers': [1, 2, 3, 4],
                'module__n_in': [n_in]
        }

        #scoring parameters
        #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

        gs = GridSearchCV(net, params, refit=True, cv=3,
                                 scoring='r2', n_jobs=21)

        gs.fit(x_train, y_train)


        report(gs.cv_results_, 10)

        #save the trained model
        print(f"Save the model in {model_save_path}")
        torch.save(gs.best_estimator_, model_save_path)

if __name__=='__main__':


    if not osp.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for wavelet_step_num in GSG_PARAMS['wavelet_step_num']:
        for scattering_operators in GSG_PARAMS['scattering_operators']:
            create_model(wavelet_step_num, scattering_operators)
