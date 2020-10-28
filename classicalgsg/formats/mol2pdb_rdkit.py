import os
import os.path as osp

import numpy  as np
from rdkit import Chem
from rdkit.Chem import AllChem


DATA_SET = '../mol_files/Star&NonStar'

outpath = '../mol_files/Star&NonStar/mol2'
if __name__=='__main__':
    mol2_files_path = osp.join(DATA_SET, 'mol2')
    mol2_files = [f for f in os.listdir(mol2_files_path) if f.endswith(".mol2")]

    molecules = []


    count = 0
    failed_mol2 = open('failed_mol2.txt', 'w')
    for filename in mol2_files:


        molid = filename[:-5]
        mol2_path =  osp.join(mol2_files_path, filename)

        print(f'processing molecule {molid}')
        #Create the molecule and
        molecule = Chem.MolFromMol2File(mol2_path)

        if molecule:
            AllChem.Compute2DCoords(molecule)

            #Add
            AllChem.EmbedMolecule(molecule)

            pdb_path = osp.join(outpath, f'{molid}.pdb')
            Chem.MolToPDBFile(molecule, pdb_path)
        else:
            count +=1

            failed_mol2.write(molid)


    print(f'failed to create 3D coords for {count} number of molecules!')
    failed_mol2.close()
