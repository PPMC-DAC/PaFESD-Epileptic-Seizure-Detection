"""
Description:
    We have a list of desired channels:
    desired_ch = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 
    'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']
    And we read a file from a dataset like this way:
            for myrecord in patientmatches:
                record = read_edf(myrecord, pn_dir=f'chbmit/chb12', header_only=False, verbose=False, rdedfann_flag=False)
                id_desired_ch = [record.sig_name.index(ch) for ch in desired_ch]
    Where patientmatches is a list with all the recordings. record is the collected record from the dataset. id_desired_ch the ids of the desired channels in the record. And eeg_inter the matrix with only the desired channels from the record.
    The problem is that the signals do not have the same labels in all the recordings for patient 12. For instance, 'FP1-F7' is not in all the recordings. Instead, we have 'FP1-CS2' and 'F7-CS2', or 'FP1' and 'F7'. In this case, to get the desired signal 'FP1-F7' we need to substract 'FP1-CS2' - 'F7-CS2', or 'FP1' - 'F7'.

Notes:
    _description_
"""
import re
import os
import pandas as pd
import numpy as np

from datetime import datetime
from wfdb.io.convert import read_edf


def to_seconds(a,b):
    ha = datetime.strptime(a,'%H:%M:%S')
    if re.search('24:\d+:\d{2}', b): # when it ends with 24:00:00
        b = b.replace("24:", "00:", 1)
    if re.search('25:\d+:\d{2}', b): # when it ends with 24:00:00
        b = b.replace("25:", "01:", 1)
    if re.search('26:\d+:\d{2}', b): # when it ends with 24:00:00
        b = b.replace("26:", "02:", 1)
    if re.search('27:\d+:\d{2}', b): # when it ends with 24:00:00
        b = b.replace("27:", "03:", 1)
        print(b)
        print(datetime.strptime(b,'%H:%M:%S'))
    hb = datetime.strptime(b,'%H:%M:%S')
    if hb < ha: # Eg: 23h y 00h
        return hb.second + hb.minute*60 + (hb.hour+24)*3600 - ha.second - ha.minute*60 - ha.hour*3600
    return hb.second + hb.minute*60 + hb.hour*3600 - ha.second - ha.minute*60 - ha.hour*3600

header = ['file', 'file_start', 'seizure_start', 'seizure_end', 'total_start', 'total_end', 'duration']

patient = 'chb12'

df_table = pd.DataFrame(columns=header)
with open(f'CHBMIT/chb12/chb12-summary.txt', 'r') as fp:
    # The pattern indicates that the pattern before ? could appear or not {0,1} times
    # The "[abc]*" patttern is used for chb17 case
    patientmatches = re.findall(f'chb12[abc]*_\d+\+?.edf', fp.read())
    print('Number of recordings:', len(patientmatches))
            
    fp.seek(0)

    matches = re.findall('\d+:\d+:\d{2}', fp.read())
    duration = [to_seconds(a,b) for a,b in list(zip(matches,matches[1:]))[::2]]
    
    if patient == 'chb17':
        id17 = patientmatches.index('chb17b_69.edf')
        duration[id17] = 1440
    
    assert len(patientmatches) == len(duration)
    
    print('duration:', [(a,b) for a,b in zip(patientmatches, duration)])
    durationsum = np.append(0, np.cumsum(duration))
    file_duration = durationsum[-1]
    print('durationsum:', durationsum)
    print('durationsum:', [(a,b) for a,b in zip(patientmatches, durationsum)])
    print(len(durationsum), len(patientmatches))

    # Go to the beginning of the file
    fp.seek(0)

    count = 0
    nseizures = 0
    # Strips the newline character
    seizure_list = []
    row_values = {'file':'chbX_X','file_start':0,'seizure_start':0,'seizure_end':0,'total_start':0,'total_end':0,'duration':0}

    # Read the file line by line
    Lines = fp.readlines()
    for line in Lines:
        count += 1
        if 'File Name' in line:
            item = re.search(f'chb12[abc]*_\d+\+?.edf', line).group()
            row_values['file'] = item
        elif 'Number of Seizures in File:' in line:
            nseizures = int(re.search('\d', line).group())
        elif nseizures > 0:
            if 'Start Time:' in line:
                index = patientmatches.index(item)
                start = int(re.search('\d{2,5}', line).group())
                seizure_list.append(durationsum[index]+start)
                row_values['file_start'] = durationsum[index]
                row_values['seizure_start'] = start
                row_values['total_start'] = durationsum[index]+start
            if 'End Time:' in line:
                end = int(re.search('\d{2,5}', line).group())
                nseizures -= 1
                seizure_list.append(durationsum[index]+end)
                row_values['seizure_end'] = end
                row_values['total_end'] = durationsum[index]+end
                row_values['duration'] = end-start
                df_table.loc[len(df_table)] = row_values
    print(seizure_list)
    print(seizure_list[::2])
    print(list(df_table['duration'].values))
    seizure_file_duration = np.sum(df_table['duration'])
    row_values = {'file':'Resume','file_start':-1,'seizure_start':-1,
                'seizure_end':-1,'total_start':0,'total_end':durationsum[-1],
                'duration':seizure_file_duration}
    df_table.loc[len(df_table)] = row_values
    print(df_table)

    dfs = pd.DataFrame()
    dfs.insert(0, 'seizure_list', seizure_list)
    dfs.to_csv(f'CHBMIT/chb12/chb12_input_params.csv', header=True, index=False)
    df_table.to_csv(f'CHBMIT/chb12/chb12_seizure_table.csv', header=True, index=False)

    n_records = len(patientmatches)
    index_i = 0
    index_f = 0

    record = read_edf(patientmatches[0], pn_dir=f'chbmit/chb12', header_only=False, verbose=False, rdedfann_flag=False)

    desired_ch = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 
                'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']

    n_sig = len(desired_ch)

    sig_len = record.sig_len

    sig_list = desired_ch

    # El orden del array es muy importante porque si hacemos resize, dependiendo de en que dirección crezca el array,
    # se nos van a descolocar los valores o no. Eg: order='C' (default): si aumentamos el número de filas no hay problema,
    # en cambio si decidimos aumentar las columnas, se descolocan todos los valores. Como estamos guardando las
    # señales por columnas porque es mejor para el almacenamiento, tenemos que utilizar order='C'.
    total = np.zeros((sig_len*n_records, n_sig), order='C')+np.nan # suposicion inicial de tamaño
    print('Initial suposition of size:', np.shape(total))

    id_desired_ch = [record.sig_name.index(ch) for ch in sig_list]
    eeg_inter = record.p_signal[:,id_desired_ch] * np.asarray(record.adc_gain)[id_desired_ch]
    print(f'Record {patientmatches[0]}:', np.shape(eeg_inter))

    # comparo un INT con FLOAT; si ponemos array_equal falla
    assert(np.allclose(np.asarray(record.init_value)[id_desired_ch], eeg_inter[0,:])) 

    index_f += sig_len

    print(f'Record {patientmatches[0]} from {index_i} to {index_f}')
    total[index_i:index_f,:] = eeg_inter

    index_i = index_f

    for myrecord in patientmatches[1:]:

        print(f'Record {myrecord}:', end=' ')
        # Download the record
        record = read_edf(myrecord, pn_dir=f'chbmit/chb12', header_only=False, verbose=False, rdedfann_flag=False)

        sig_name = record.sig_name
        eeg_inter = {}
        if '01' in sig_name:
            # print(sig_name.index('01'))
            record.sig_name[sig_name.index('01')] = 'O1'
        for ch in desired_ch:
            if ch in sig_name:
                # print(ch, end=' ')
                eeg_inter[ch] = record.p_signal[:,sig_name.index(ch)] * record.adc_gain[sig_name.index(ch)]
            else:
                ch1, ch2 = ch.split('-')
                ch1_found = False
                ch2_found = False
                for name in sig_name:
                    if name.startswith(f'{ch1}'):
                        ch1_found = True
                        ch1_index = sig_name.index(name)
                    if name.startswith(f'{ch2}'):
                        ch2_found = True
                        ch2_index = sig_name.index(name)
                assert ch1_found and ch2_found, f'Channels {ch1} and {ch2} not found'
                # print(f'{ch1}-{ch2}', end=' ')
                eeg_inter[f'{ch1}-{ch2}'] = (record.p_signal[:,ch1_index] - record.p_signal[:,ch2_index]) * record.adc_gain[ch1_index]
        
        index_f += record.sig_len

        print(f'({index_f - index_i}, {len(eeg_inter)})')
        print(f'Record {myrecord} from {index_i} to {index_f}')

        if sig_len*n_records < index_f: # si la suposicion inicial de tamaño falla
            total.resize((index_f, n_sig), refcheck=False)

        for i,ch in enumerate(desired_ch):
            total[index_i:index_f,i] = eeg_inter[ch]

        # total[index_i:index_f,:] = eeg_inter
        index_i = index_f

    # Si he reservado más espacio de la cuenta, ajusto
    total.resize((index_f, n_sig), refcheck=False)

    df = pd.DataFrame(data=total, columns=sig_list)

    # Por si hay alguna etiqueta de columna repetida, que la hay: T8-P8
    s = df.columns.to_series()
    df.columns = s + s.groupby(s).cumcount().astype(str).replace({'0':''})

    assert(len(df)//256 == file_duration)
    # df.to_hdf(f'CHBMIT/chb12/signal_chb12.h5', key='data_frame', format='table', mode='w')
    # df.to_hdf(f'CHBMIT/chb12/signal_chb12.h5', key='data_frame', format='fixed', complevel=9, complib='zlib', mode='w')
    df.to_hdf(f'CHBMIT/chb12/signal_chb12.h5', key='data_frame', format='fixed', mode='w')

    channels = list(df.keys())

    sfreq = 256
    matches = [x*sfreq for x in seizure_list]
    wide = 0
    dfs = pd.DataFrame()
    for ii,ch in enumerate(channels):
        signal = df[ch].values
        seizures = []
        for a,b in list(zip(matches,matches[1:]))[::2]:
            #print(a,b)
            #print(np.shape(signal_chb06[a-10*256:b+10*256]))
            #seizures = np.append(seizures, signal_chb06[a:b])
            seizures = np.append(seizures, signal[a-wide*sfreq:b+wide*sfreq])
        #print(np.shape(signal_chb06[1724*256-10*256:1738*256+10*256]))
        #print(np.shape(seizures))
        dfs.insert(ii, ch, seizures)
    assert(len(dfs)//256 == seizure_file_duration)
    dfs.to_csv(f'CHBMIT/chb12/seizures{wide}_chb12.csv', header=True, index=False)

# clean
os.system(f'rm chb12*.edf')