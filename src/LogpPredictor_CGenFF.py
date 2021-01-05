import sys
import os.path as osp
import numpy as np
from joblib import load

import torch
from classicalgsg.molreps_models.gsg import GSG
from classicalgsg.classicalgsg import CGenFFGSG
from classicalgsg.molreps_models.utils import scop_to_boolean


PRETRAINED_MODEL_PATH = 'classicalgsg/pretrained_models'
PRETRAINED_MODEL = 'model_4_zfs_CGenFF.pt'
SCALER = 'std_scaler_CGenFF.sav'

if __name__ == '__main__':

    if sys.argv[1] == '-h' or sys.argv[1] == '--h':
        print('LogpPredictor molecule.mol2 molecule.str')
        exit()

    else:
        mol2_file_path = sys.argv[1]
        str_file_path = sys.argv[2]

    wavelet_step_num = 4

    scattering_operators = scop_to_boolean('(z,f,s)')

    gsg = GSG(wavelet_step_num, scattering_operators)

    cgenffgsg = CGenFFGSG(gsg, structure='2D', AC_type='AC36')

    x = cgenffgsg.features(mol2_file_path, str_file_path)
    x = x.reshape((-1, x.shape[0]))

    scaler_file_path = osp.join(osp.dirname(sys.modules[__name__].__file__),
                                PRETRAINED_MODEL_PATH,
                                SCALER)

    model_file_path = osp.join(osp.dirname(sys.modules[__name__].__file__),
                               PRETRAINED_MODEL_PATH,
                               PRETRAINED_MODEL)

    scaler = load(scaler_file_path)
    model = torch.load(model_file_path)
    x = scaler.transform(x)

    predicted_logP = np.squeeze(model.predict(x.astype(np.float32)))

    print(f'Predicted logP value is: {predicted_logP:.2f}')
