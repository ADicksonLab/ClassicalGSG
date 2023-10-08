import os
import sys
import os.path as osp

from itertools import product
import multiprocessing as mp

from classicalgsg.molreps_models.gsg import GSG
from classicalgsg.classicalgsg import CGenFFGSG
from classicalgsg.nn_models.datasetbuilder import DatasetBuilder
from classicalgsg.molreps_models.utils import scop_to_boolean, scop_to_str

# Examples params
# GSG_PARAMS = {'wavelet_scale': [4, 5],
#               'scattering_operators': ['(z,f,s)', '(z,f)', '(z,s)', '(f,s)']}

GSG_PARAMS = {'wavelet_scale': [4],
              'scattering_operators': ['(z,f,s)']}
DATASETS = [('OpenChem', ''), ('Star', 'test'), ('NonStar', 'test'),
            ('Huuskonen', 'test')]

if __name__ == '__main__':

    if sys.argv[1] == '-h' or sys.argv[1] == '--h':
        print('python extract_features.py path_to_the_dataset')
        exit()

    else:
        FILES_PATH = sys.argv[1]

def create_dataset(dataset_name, dataset_type,
                   wavelet_scale, scattering_operators):

    print(f'Started {dataset_name}',
          f'{wavelet_scale}',
          f'{scattering_operators}')

    dataset_files_path = osp.join(FILES_PATH,
                                  dataset_name,
                                  dataset_type)

    mol2_files_path = osp.join(dataset_files_path, 'mol2')
    cgenffsrt_files_path = osp.join(dataset_files_path, 'str')
    logp_files_path = osp.join(dataset_files_path, 'logp_values')

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

    cgenffgsg = CGenFFGSG(gsg, structure='2D', AC_type='AC36')

    dataset_builder = DatasetBuilder(cgenffgsg, save_file_path)
    dataset_builder.create(mol2_files_path,
                           cgenffsrt_files_path,
                           logp_files_path)

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

    with mp.Pool(processes=5) as pool:
        pool.starmap(create_dataset, arguments)
