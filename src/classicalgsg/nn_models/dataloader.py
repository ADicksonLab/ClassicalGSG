import os.path as osp

import numpy as np
import pickle as pkl

from collections import defaultdict
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


#datasets's fields name ['molid', 'atom_names', 'coords',
#'adjacency', 'wavelets', 'gaff_signals', 'gaff_signals_notype',
#'cgenff_signals', 'cgenff_signals_notype', 'gaff_features',
#'gaff_features_notype', 'cgenff_features', 'cgenff_features_notype',
#'logp']


#opens a dataset and divides into given partioning
class DataLoader:
    def __init__(self, dataset_name, dataset_path=None):
        self.dataset_name = dataset_name

        if dataset_path:
            self.base_dir = dataset_path

        else:

            self.base_dir = osp.join(osp.dirname(__file__),
                                     'datasets')


        self.dataset_file_names = self._get_dataset_file_names()

    #atomtype is boolean
    def _get_data(self, dataset_name):
        """
        Loads the dataset  and normalize
        """
        with open(dataset_name, 'rb') as pklf:
             dataset = pkl.load(pklf)


        molids = dataset['molid']
        x = np.array(dataset['features'])
        y = np.array(dataset['logp'])
        x = x.reshape((x.shape[0], -1))
        y = y.reshape(y.shape[0], -1)

        return x, y, molids


    def _load_dataset(self, dataset_path):

        with open(dataset_path, 'rb') as pklf:
            dataset_data = pkl.load(pklf)

        return dataset_data


    def _load_fields_data(self, field_names):

        data = defaultdict(dict)
        for dataset_file_name in self.dataset_file_names:

            dataset_path = osp.join(self.base_dir, self.dataset_name,
                                dataset_file_name)
            data_type_name , _ = osp.splitext(dataset_file_name)


            dataset = self._load_dataset(dataset_path)

            for field_name in field_names:
                data[data_type_name][field_name] = dataset[field_name]

        return data


    def fields_data(self, field_names):

        if self._check_exists():



            return self._load_fields_data(field_names)
        else:
            exit()

    def _load_data(self):
        """FIXME! briefly describe function

        :param ff:
        :param atom_type:
        :returns:
        :rtype:

        """

        data = {}
        for dataset_file_name in self.dataset_file_names:

            dataset_path = osp.join(self.base_dir, self.dataset_name,
                                dataset_file_name)
            data_type_name , _ = osp.splitext(dataset_file_name)

            x, y, molids = self._get_data(dataset_path)

            x = x.astype(np.float32)
            y_train = y.astype(np.float32)


            data.update({data_type_name: (x, y, molids)})

        return data


    def _check_exists(self):

        if len(self.dataset_file_names) < 1:
            return False

        for dataset_file_name in self.dataset_file_names:
            dataset_path = osp.join(self.base_dir, self.dataset_name,
                                dataset_file_name)
            if not osp.exists(dataset_path):
                print(f'{dataset_path} does not exists')
                return False

        return True

    def _get_dataset_file_names(self):

        dataset_file_names = []
        if self.dataset_name == 'huuskonen':
            dataset_file_names.append('huuskonen_training.pkl')
            dataset_file_names.append(f'huuskonen_test.pkl')

        if self.dataset_name == 'guowei':
            dataset_file_names.append('guowei_training.pkl')

        if self.dataset_name == 'FDA':
            dataset_file_names.append('FDA_test.pkl')

        if self.dataset_name == 'Star':
            dataset_file_names.append('Star_test.pkl')

        if self.dataset_name == 'NonStar':
            dataset_file_names.append('NonStar_test.pkl')

        if self.dataset_name == 'chEMBL21':
            dataset_file_names.append('chEMBL21_training.pkl')
            dataset_file_names.append(f'chEMBL21_test.pkl')

        if self.dataset_name == 'SAMPL6':
            dataset_file_names.append(f'SAMPL6_test.pkl')

        if self.dataset_name == 'SAMPL7':
            dataset_file_names.append(f'SAMPL7_test.pkl')


        return dataset_file_names


    def load_data(self):
        """Loads a dataset and normalize it

        :param ff:
        :param atom_type:
        :param train_percent:
        :param validation_percent:
        :param test_percent:
        :param outfile_name:
        :returns:
        :rtype:

        """

        if self._check_exists():
            return self._load_data()

        else:
            exit()