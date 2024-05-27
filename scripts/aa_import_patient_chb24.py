import re
import os
import pandas as pd
import numpy as np

from datetime import datetime
import wfdb
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

sfreq = 256
header = ['file', 'file_start', 'seizure_start', 'seizure_end', 'total_start', 'total_end', 'duration']
df_table = pd.DataFrame(columns=header)

allrecords = wfdb.get_record_list('chbmit/1.0.0')
patient = 'chb24'

record_list = []
for record in allrecords:
    item = re.search(f'{patient}[abc]*_\d+\+?.edf', record)
    if item:
        record_list.append(item.group())
print(record_list)

with open(f'CHBMIT/{patient}/{patient}-summary.txt', 'r') as fp:
    # The pattern indicates that the pattern before ? could appear or not {0,1} times
    # The "[abc]*" patttern is used for chb17 case
    patientmatches = re.findall(f'{patient}[abc]*_\d+\+?.edf', fp.read())
    #print('patientmatches:', patientmatches)
    print('Number of recordings:', len(patientmatches))

    print(patientmatches)

    duration = []
    # The problem with chb24 is that the file start and end times are not in the summary file, we need to read the edf files to get the duration, which implies downloading the files. I already have them downloaded, so I will use them to get the duration and save it in a txt file
    with open(f'CHBMIT/{patient}/{patient}_seizures_duration.txt', 'r') as fd:
        Lines = fd.readlines()
        for line in Lines:
            mylist = line.strip().split()
            duration.append((int(mylist[1]) - int(mylist[0]))//sfreq)
    print(duration)
        
    fp.seek(0)
    print(len(duration))
    durationsum = np.append(0, np.cumsum(duration))
    file_duration = durationsum[-1]
    Lines = fp.readlines()
  
    count = 0
    nseizures = 0
    # Strips the newline character
    seizure_list = []
    row_values = {'file':'chbX_X','file_start':0,'seizure_start':0,'seizure_end':0,'total_start':0,'total_end':0,'duration':0}
    for line in Lines:
        count += 1
        # get the name of the file from the summary file
        if 'File Name' in line:
            item = re.search(f'{patient}[abc]*_\d+\+?.edf', line).group()
            row_values['file'] = item
            # index = patientmatches.index(item)
            index = record_list.index(item)
        # get the number of seizures in the file
        elif 'Number of Seizures in File:' in line:
            nseizures = int(re.search('\d', line).group())
        elif nseizures > 0:
            if 'Start Time:' in line:
                # index = patientmatches.index(item)
                start = int(re.search('\d{2,5}', line).group())
                #print(durationsum[index], start, end=' ')
                seizure_list.append(durationsum[index]+start)
                row_values['file_start'] = durationsum[index]
                row_values['seizure_start'] = start
                row_values['total_start'] = durationsum[index]+start
            if 'End Time:' in line:
                end = int(re.search('\d{2,5}', line).group())
                #print(end, durationsum[index]+start, durationsum[index]+end)
                nseizures -= 1
                seizure_list.append(durationsum[index]+end)
                row_values['seizure_end'] = end
                row_values['total_end'] = durationsum[index]+end
                row_values['duration'] = end-start
                df_table.loc[len(df_table)] = row_values
    print(seizure_list)
    # print(seizure_list[::2])
    print(list(df_table['duration'].values))
    seizure_file_duration = np.sum(df_table['duration'])
    row_values = {'file':'Resume','file_start':-1,'seizure_start':-1,
                  'seizure_end':-1,'total_start':0,'total_end':durationsum[-1],
                  'duration':seizure_file_duration}
    df_table.loc[len(df_table)] = row_values
    print(df_table)

    dfs = pd.DataFrame()
    dfs.insert(0, 'seizure_list', seizure_list)
    dfs.to_csv(f'CHBMIT/{patient}/{patient}_input_params.csv', header=True, index=False)
    df_table.to_csv(f'CHBMIT/{patient}/{patient}_seizure_table.csv', header=True, index=False)

dir_l = 'chbmit/chb24'
index_i = 0
index_f = 0

n_records = len(record_list)

record = read_edf(record_list[0], pn_dir=dir_l, header_only=False, verbose=False, rdedfann_flag=False)

n_sig = record.n_sig

sig_len = record.sig_len

sig_list = record.sig_name

# El orden del array es muy importante porque si hacemos resize, dependiendo de en que dirección crezca el array,
# se nos van a descolocar los valores o no. Eg: order='C' (default): si aumentamos el número de filas no hay problema,
# en cambio si decidimos aumentar las columnas, se descolocan todos los valores. Como estamos guardando las
# señales por columnas porque es mejor para el almacenamiento, tenemos que utilizar order='C'.
total = np.zeros((sig_len*n_records, n_sig), order='C')+np.nan # suposicion inicial de tamaño
print(np.shape(total))

eeg_inter = record.p_signal * record.adc_gain
print(np.shape(eeg_inter))

if not(np.allclose(record.init_value, eeg_inter[0,:])): # comparo un INT con FLOAT; si ponemos array_equal falla
    raise ValueError("Sorry, the signals don't match")

index_f += sig_len

print(index_i, index_f)
total[index_i:index_f,:] = eeg_inter

index_i = index_f

for mysignal in record_list[1:]:

    print(mysignal)
    record = read_edf(mysignal, pn_dir=dir_l, header_only=False, verbose=False, rdedfann_flag=False)
    eeg_inter = record.p_signal * record.adc_gain
    #print(record.init_value == eeg_inter[:,0].astype(int))
    if not(np.allclose(record.init_value, eeg_inter[0,:])): # comparo un INT con FLOAT; si ponemos array_equal falla
        raise ValueError("Sorry, the signals don't match")
    index_f += record.sig_len
    print(np.shape(eeg_inter))
    print(index_i, index_f)
    if sig_len*n_records < index_f: # si la suposicion inicial de tamaño falla
        total.resize((index_f, n_sig), refcheck=False)
    total[index_i:index_f,:] = eeg_inter
    index_i = index_f
# Si he reservado más espacio de la cuenta, ajusto
total.resize((index_f, n_sig), refcheck=False)

data = pd.DataFrame(data=total, columns=sig_list)

# Por si hay alguna etiqueta de columna repetida, que la hay: T8-P8
s = data.columns.to_series()
data.columns = s + s.groupby(s).cumcount().astype(str).replace({'0':''})

# data.to_hdf('CHBMIT/chb24/signal_chb24.h5', key='data_frame', format='table', mode='w')
# data.to_hdf('CHBMIT/chb24/signal_chb24.h5', key='data_frame', format='fixed', complevel=9, complib='zlib', mode='w')
data.to_hdf('CHBMIT/chb24/signal_chb24.h5', key='data_frame', format='fixed', mode='w')

channels = data.columns.values

print(channels)

matches = [x*sfreq for x in seizure_list]
wide = 0
df = pd.DataFrame()
for ii,ch in enumerate(channels):
    signal_chb24 = data[ch].values
    seizures = []
    for a,b in list(zip(matches,matches[1:]))[::2]:
        seizures = np.append(seizures, signal_chb24[a-wide*sfreq:b+wide*sfreq])
    df.insert(ii, ch, seizures)
df.to_csv(f'chb24/seizures{wide}_chb24.csv', header=True, index=False)

# clean
os.system(f'rm chb24*.edf')