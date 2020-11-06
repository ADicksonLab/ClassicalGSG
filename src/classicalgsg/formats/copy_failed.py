import os
import os.path as osp

dataset_path='../mol_files/Star_NonStar/guowei/NonStar'

def read_streamfile(str_filename):
    f = open(str_filename, 'r')
    molecule = []
    for i, line in enumerate(f):
        line = line.strip()
        if line[:4].upper() == 'ATOM':
            words = line.split()
            name = words[1].upper()
            atom_type = words[2].upper()
            charge = float(words[3])
            if type!='LPH':
                molecule.append(name)
    return molecule


def gaff_failed():
    mol2_files_path = osp.join(dataset_path, 'mol2')
    gaff_files_path = osp.join(dataset_path, 'gaffmol2')

    in_files_path = osp.join(dataset_path, 'mol2')
    out_files_path = osp.join(dataset_path, 'gaff_failed')



    mol2_files = [f for f in os.listdir(mol2_files_path)
                  if f.endswith(".mol2")]

    gaff_files = [ f for f in os.listdir(gaff_files_path)
                                               if f.endswith(".mol2")]

    diff = set(mol2_files) - set(gaff_files)
    count = len(diff)

    os.system(f'mkdir -p {out_files_path}')

    for molfile_name in diff:
        file_name = molfile_name[:-5] +'.mol2'
        print(f'Copying {file_name}')
        source = osp.join(in_files_path, file_name)
        dist = osp.join(out_files_path, file_name)
        os.system(f"cp {source} {dist} ")

    print(f'GAff failed for {count} number of molecules')

def cgenff_failed():
    strfiles_path = osp.join(dataset_path, 'str')

    out_files_path = osp.join(dataset_path, 'cgenff_failed')


    os.system(f'mkdir -p {out_files_path}')
    str_files = [f for f in os.listdir(strfiles_path) if f.endswith(".str")]

    failed_files = []
    count = 0
    for str_filename in str_files:
        mol_id = str_filename[:-4]
        molecule = read_streamfile(os.path.join(strfiles_path,
                                                str_filename))

        if len(molecule)==0:
            count +=1
            failed_files.append(str_filename)
            source = osp.join(strfiles_path, str_filename)
            dist = osp.join(out_files_path, str_filename)
            os.system(f"cp {source} {dist} ")

            print(f'Cgenff failed for {str_filename}')


    print(f'Cgenff failed for {count} number of molecules')
if __name__ =='__main__':
    gaff_failed()
    cgenff_failed()
