import os
import os.path as osp
from collections import defaultdict
import pickle as pkl
from itertools import product
import multiprocessing as mp

from openbabel import pybel

from classicalgsg.molreps_models.gsg import GSG
from classicalgsg.classicalgsg import OBFFGSG
from classicalgsg.molreps_models.utils import scop_to_boolean, scop_to_str


GSG_PARAMS = {'wavelet_scale': [4, 5, 6, 7, 8],
              'scattering_operators': ['(z,f,s)', '(z,f)', '(z,s)', '(f,s)']}

DATASETS = [('ChEMBL21_V2.sdf', 'ChEMBL', '')]

DATASETS_PATH = '../training_sdf_sets'

FORCEFIELD = 'MMFF94'


def create_dataset(sdf_file, dataset_name, dataset_type,
                   wavelet_scale, scattering_operators):

    print(f'Started {dataset_name}',
          f'{wavelet_scale}',
          f'{scattering_operators}')

    dataset_save_path = osp.join(f'data_{wavelet_scale}_'
                                 f'{scop_to_str(scattering_operators)}',
                                 f'{dataset_name}')

    if not osp.exists(dataset_save_path):
        os.makedirs(dataset_save_path)

    if dataset_type != '':
        save_file_path = osp.join(dataset_save_path,
                                  f'{dataset_name}_{dataset_type}.pkl')
    else:
        save_file_path = osp.join(dataset_save_path,
                                  f'{dataset_name}.pkl')

    gsg = GSG(wavelet_scale,
              scop_to_boolean(scattering_operators))

    uffgsg = OBFFGSG(gsg, structure='2D', AC_type='ACall')

    molecules = pybel.readfile('sdf',
                               osp.join(DATASETS_PATH, sdf_file))
    dataset = defaultdict(list)

    for mol in molecules:
        smiles = mol.data['smiles']
        features = uffgsg.features(smiles, FORCEFIELD)
        if features is not None:
            dataset['molid'].append(mol.data['molid'])
            dataset['logp'].append(float(mol.data['logP']))
            dataset['features'].append(features)

    with open(save_file_path, 'wb') as wfile:
        pkl.dump(dataset, wfile)
        print(f"dataset {save_file_path}")

    print(f'Finished {dataset_name},'
          f'{ wavelet_scale},'
          f'{scattering_operators}')


if __name__ == '__main__':

    params = list(product(GSG_PARAMS['wavelet_scale'],
                          GSG_PARAMS['scattering_operators']))

    arguments = []

    for dataset in DATASETS:
        for gsg_param in params:
            arguments.append(dataset + gsg_param)

    with mp.Pool(processes=20) as pool:
        pool.starmap(create_dataset, arguments)
