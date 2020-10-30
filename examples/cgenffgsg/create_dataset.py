import  os
import os.path as osp

import numpy as np
import pickle as pkl
from itertools import product

import multiprocessing as mp

from classicalgsg.molreps_models.gsg import GSG
from classicalgsg.classicalgsg import CGenFFGSG
from classicalgsg.nn_models.datasetbuilder import DatasetBuilder
from classicalgsg.molreps_models.utils import sco_to_boolean

PARAMS = {'wavelet_step_num': [4, 5],
          'scattering_operators': ['(z,f,s)', '(z,f)', '(z,s)', '(f,s)']}

DATASETS = [('NonStar', 'test'), ('chEMBL21', '')]

DATASETS_PATH = '/mnt/home/nazanin/projects/logp_sims/datasets'


def conv_to_str(feature_flag):

    out_str = ''

    for flag in feature_flag:
        out_str += str(int(flag))

    return out_str



def create_dataset(dataset_name, dataset_type, wavelet_step_num, scattering_operators):

    print(f'Started {dataset_name}, {wavelet_step_num}, {scattering_operators}')


    dataset_files_path = osp.join(DATASETS_PATH, dataset_name,
                                  dataset_type)



    mol2_files_path = osp.join(dataset_files_path, 'mol2')
    cgenffsrt_files_path = osp.join(dataset_files_path, 'str')
    logp_files_path = osp.join(dataset_files_path, 'logp_values')


    dataset_save_path = osp.join(f'data_{wavelet_step_num}_{scattering_operators}',
                                 f'{dataset_name}')

    if not osp.exists(dataset_save_path):
        os.makedirs(dataset_save_path)


    if dataset_type != '':
        save_file_path = osp.join(dataset_save_path,
                                  f'{dataset_name}_{dataset_type}.pkl')
    else:
        save_file_path = osp.join(dataset_save_path,
                                  f'{dataset_name}.pkl')


    gsg = GSG(wavelet_step_num,
              sco_to_boolean(scattering_operators))

    cgenffgsg = CGenFFGSG(gsg, structure='2D', AC_type='AC36')



    dataset_builder = DatasetBuilder(cgenffgsg, save_file_path)
    dataset_builder.create(mol2_files_path, cgenffsrt_files_path,
                            logp_files_path)


    print(f'Finished {dataset_name}, { wavelet_step_num}, {scattering_operators}')

if __name__=='__main__':


    GSG_params = list(product(PARAMS['wavelet_step_num'],
                         PARAMS['scattering_operators']))

    arguments = []

    for dataset in DATASETS:
        for gsg_param in GSG_params:
            arguments.append(dataset + gsg_param)



    with mp.Pool(processes=10) as pool:
        pool.starmap(create_dataset, arguments)
