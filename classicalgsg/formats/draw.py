import os
import os.path as osp

import numpy  as np
from rdkit import Chem
from rdkit.Chem import Draw

DATA_SET = '../mol_files/Guo_Wei/first_test/minimized'



def drawfrompdb():

    pdbfiles_path = osp.join(DATA_SET, 'pdbs')
    pdb_files = [f for f in os.listdir(pdbfiles_path) if f.endswith(".pdb")]

    molecules = []
    legends = []

    image_grid_size = 50
    image_perrow = 10
    image_size = (300, 300)
    for filename in pdb_files:

        pdb_path =  osp.join(pdbfiles_path, filename)
        molecules.append(Chem.rdmolfiles.MolFromPDBFile(pdb_path))
        legends.append(filename[:-4])





    strides = list(np.arange(0, len(molecules), image_grid_size))

    if strides[-1] < len(molecules):
        strides.append(len(molecules))

    for idx in range(len(strides)-1):

        group_img = Draw.MolsToGridImage(molecules[strides[idx]:strides[idx+1]], molsPerRow=image_perrow,
                                 subImgSize=image_size,
                                 legends=legends[strides[idx]:strides[idx+1]])


        imgfiles_path = osp.join(DATA_SET, 'images')

        img_file_path = osp.join(imgfiles_path, f'group{idx}.png')
        group_img.save(img_file_path)

def drawfrommol2():
    mol2_files_path = osp.join(DATA_SET, 'nonconsistent')
    mol2_files = [f for f in os.listdir(mol2_files_path) if f.endswith(".mol2")]

    molecules = []
    legends = []

    image_grid_size = 50
    image_perrow = 10
    image_size = (300, 300)
    for filename in mol2_files:

        mol2_path =  osp.join(mol2_files_path, filename)
        molecules.append(Chem.MolFromMol2File(mol2_path))
        legends.append(filename[:-5])





    strides = list(np.arange(0, len(molecules), image_grid_size))

    if strides[-1] < len(molecules):
        strides.append(len(molecules))

    for idx in range(len(strides)-1):

        group_img = Draw.MolsToGridImage(molecules[strides[idx]:strides[idx+1]], molsPerRow=image_perrow,
                                 subImgSize=image_size,
                                 legends=legends[strides[idx]:strides[idx+1]])


        imgfiles_path = osp.join(DATA_SET, 'images')

        img_file_path = osp.join(imgfiles_path, f'group{idx}.png')
        group_img.save(img_file_path)

if __name__=='__main__':

    drawfrommol2()
