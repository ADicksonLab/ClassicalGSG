import pickle as pkl
import pandas as pd
from tabulate import tabulate

class TestReporter:

#     REPORT_HEADER_TEMPLATE=\
# """* Deep Neural model return results:
# - Training dataset name: {dataset_name}
# - Feature's force field: {forcefield}
# - Atom type: {atom_type}
# ** Training hyperparameters
# - Batch size: {batch_size}
# - Number of hidden layers: {n_layers}
# - Hidden layer's size: {h_size}
# - Dropout: {dropout}
# - Number of training steps: {num_steps}
# """
    REPORT_HEADER_TEMPLATE=\
                        """* Deep Neural model return results:
- Training dataset name: {dataset_name}
- Feature's force field: {forcefield}
- Atom type: {atom_type}
"""

    TEST_RESULTS_TEMPLATE=\
"""* Test Results:
** Numner of Wavelet matrix steps: {wavelet_steps}
** Scattetring moments:
- Zero order: {zero_order}
- First order: {first_order}
- Second order: {second_order}
** results:

{test_results}
"""

    def __init__(self, dataset_name, forcefield, atom_type):
        self.header_str = self.header_string(dataset_name, forcefield,
                                             atom_type)

    def header_string(self, dataset_name, forcefield, atom_type):

        header_str = self.REPORT_HEADER_TEMPLATE.format(
            dataset_name=dataset_name,
            forcefield=forcefield,
            atom_type=atom_type,
            )

        return header_str

    def result(self, wavelet_steps, features_flags, results):

        results_dic = {}

        for dataset_name, result in results.items():
            results_dic.update({dataset_name: result.values()})

        df = pd.DataFrame.from_dict(results,  orient='index')

        results_str = tabulate(df, headers='keys', tablefmt='fancy_grid')

        test_result_str = self.TEST_RESULTS_TEMPLATE.format(
            wavelet_steps=wavelet_steps,
            zero_order=features_flags[0],
            first_order=features_flags[1],
            second_order=features_flags[2],
            test_results=results_str)

        return df, test_result_str

    def save_txt(self, results_str, save_path):

        report_string = self.header_str
        mode = 'w'

        for result_str in results_str:
            report_string += result_str

        with open(save_path, mode=mode) as reporter_file:
            reporter_file.write(report_string)

    def save_pickle(self, results, save_file_name):

        df = pd.concat(results)
        df.insert(0, 'Dataset', df.index)
        df_index = pd.Series([i for i in range(df.shape[0])])

        df = df.set_index(df_index)

        with open(save_file_name, 'wb') as pklf:
            pkl.dump(df, pklf)
