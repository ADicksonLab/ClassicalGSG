import sys
import os
import os.path as osp
import numpy as np
from joblib import dump, load

import torch
from classicalgsg.molreps_models.gsg import GSG
from classicalgsg.classicalgsg import SCATTERING_MOMENT_OPERATORS, CGenFFGSG
from classicalgsg.molreps_models.utils import sco_to_boolean

BASE_DIR = osp.join(osp.dirname(__file__),
                                     'trained_models')
TRAINED_MODEL_PATH = f'{BASE_DIR}/model_openchem_CGENFF_Y_4_111.pt'
SandardScaler_PATH = f'{BASE_DIR}/std_scaler.sav'


def load_SandardScaler(sc_path):
    return load(sc_path)

def load_model(model_path):
    return torch.load(model_path)

if __name__=='__main__':

    if sys.argv[1]=='-h' or sys.argv[1]=='--h':
        print('LogpPredictor molecule.mol2 molecule.str')
        exit()

    else:
        mol2_file_path = sys.argv[1]
        str_file_path = sys.argv[2]

    wavelet_step_num = 4

    scattering_operators = sco_to_boolean('(z,f,s)')

    gsg = GSG(wavelet_step_num, scattering_operators)

    cgenffgsg = CGenFFGSG(gsg, structure='2D', AC_type='AC36')

    x = cgenffgsg.features(mol2_file_path, str_file_path)
    x = x.reshape((-1, x.shape[0]))

    model = load_model(TRAINED_MODEL_PATH)
    scaler = load_SandardScaler(SandardScaler_PATH)
    x = scaler.transform(x)

    print(np.squeeze(model.predict(x.astype(np.float32))))
