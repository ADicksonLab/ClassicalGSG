import os
import os.path as osp
from collections import defaultdict
import pickle as pkl

from classicalgsg.atomic_attr.utils import read_logp
from classicalgsg.classicalgsg import CGenFFGSG, GAFF2GSG


class DatasetBuilder(object):

    def __init__(self, classicalgsg_method, dataset_save_path):

        self.classicalgsg_method = classicalgsg_method
        self.dataset_save_path = dataset_save_path

    def create(self, mol2_files_path, param_files_path,
               logp_files_path, molids=None):

        if not osp.exists(mol2_files_path):
            print(f'{mol2_files_path} does not exists')
            return None

        if not osp.exists(param_files_path):
            print(f'{param_files_path} does not exists')
            return None

        if not osp.exists(logp_files_path):
            print(f'{logp_files_path} does not exists')
            return None

        mol2_files = [f for f in os.listdir(mol2_files_path)
                      if f.endswith(".mol2")]

        failed_number = 0
        # read the molecules in the mol2 files

        if isinstance(self.classicalgsg_method, CGenFFGSG):
            param_extension = 'str'

        elif isinstance(self.classicalgsg_method, GAFF2GSG):
            param_extension = 'mol2'

        dataset = defaultdict(list)

        if molids is None:
            molids = [osp.splitext(mol2_file_name)[0]
                      for mol2_file_name in mol2_files]

        for idx, molid in enumerate(molids):

            mol2_file = osp.join(mol2_files_path, f'{molid}.mol2')

            param_file = osp.join(param_files_path,
                                  f'{molid}.{param_extension}')

            logp_file = osp.join(logp_files_path, f'{molid}.exp')

            all_paths = [mol2_file, param_file, logp_file]

            if all(osp.isfile(mol_path) for mol_path in all_paths):

                logp = read_logp(logp_file)

                features = self.classicalgsg_method.features(mol2_file,
                                                             param_file)
                dataset['molid'].append(molid)
                dataset['logp'].append(logp)
                dataset['features'].append(features)

            else:
                failed_number += 1

        print(f'{failed_number} molecules faild to be processed')

        with open(self.dataset_save_path, 'wb') as wfile:
            pkl.dump(dataset, wfile)
            print(f"dataset {self.dataset_save_path}")
