# Compare the hash of all files needed for the training
import pandas as pd
import hashlib
import h5py

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path,"rb") as f:
        # Leer y actualizar el hash en bloques de 4K
        for byte_block in iter(lambda: f.read(4096),b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

def generate_hash_from_h5(path):
    hash_obj = hashlib.sha256()
    with h5py.File(path, 'r') as f:
        # obtain the dataset
        dset = f['data_frame']['block0_values']
        # obtain hash from the dataset
        hash_obj.update(dset[()].tobytes())

    return hash_obj.hexdigest()

patients_list = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb12', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17', 'chb18', 'chb19', 'chb20', 'chb21', 'chb22', 'chb23', 'chb24']

# dictionary with the patients and the hash of the files
hash_dict = pd.DataFrame()
# load the hash file if exists
old_dict = pd.read_csv('patients_hash.csv', index_col=0)

for patient in patients_list:
    error = 0
    print(f'Patient: {patient} ...', end=' ')
    file1 = f'CHBMIT/{patient}/signal_{patient}.h5'
    file2 = f'CHBMIT/{patient}/{patient}_input_params.csv'
    file3 = f'CHBMIT/{patient}/{patient}_seizure_table.csv'
    file4 = f'CHBMIT/{patient}/seizures0_{patient}.csv'
    file5 = f'CHBMIT/{patient}/{patient}-summary.txt'

    hash1 = generate_hash_from_h5(file1)
    # print(f'{file1}: {hash1}')
    hash_dict.loc[file1, 'hash'] = hash1
    error += 1 if old_dict.loc[file1, 'hash'] != hash1 else 0
    hash2 = calculate_sha256(file2)
    # print(f'{file2}: {hash2}')
    hash_dict.loc[file2, 'hash'] = hash2
    error += 1 if old_dict.loc[file2, 'hash'] != hash2 else 0
    hash3 = calculate_sha256(file3)
    # print(f'{file3}: {hash3}')
    hash_dict.loc[file3, 'hash'] = hash3
    error += 1 if old_dict.loc[file3, 'hash'] != hash3 else 0
    hash4 = calculate_sha256(file4)
    # print(f'{file4}: {hash4}')
    hash_dict.loc[file4, 'hash'] = hash4
    error += 1 if old_dict.loc[file4, 'hash'] != hash4 else 0
    hash5 = calculate_sha256(file5)
    # print(f'{file5}: {hash5}')
    hash_dict.loc[file5, 'hash'] = hash5
    error += 1 if old_dict.loc[file5, 'hash'] != hash5 else 0

    if error > 0:
        print(f'{error} errors')
    else:
        print('OK')

# save the dataframe to a csv file to check the patients signals
hash_dict.to_csv('new_patients_hash.csv', header=True, index=True)