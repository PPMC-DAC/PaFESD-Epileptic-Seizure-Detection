import re
import os
import pandas as pd
from pandas import HDFStore
import numpy as np
import time

from datetime import datetime
from wfdb.io.convert import read_edf

class SafeHDF5Store(HDFStore):
    """Implement safe HDFStore by obtaining file lock. Multiple writes will queue if lock is not obtained."""
    def __init__(self, *args, **kwargs):
        """Initialize and obtain file lock."""
        interval = kwargs.pop('probe_interval', 1)
        self._lock = "%s.lock" % args[0]
        while True:
            try:
                self._flock = os.open(self._lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except IOError:
                time.sleep(interval)
        HDFStore.__init__(self, *args, **kwargs)

    def __exit__(self, *args, **kwargs):
        """Exit and remove file lock."""
        HDFStore.__exit__(self, *args, **kwargs)
        os.close(self._flock)
        os.remove(self._lock)
    
def write_hdf(f, key, df, complib):
    """
    Append pandas dataframe to hdf5.

    Args:
    f       -- File path
    key     -- Store key
    df      -- Pandas dataframe
    complib -- Compress lib 
    """

    with SafeHDF5Store(f, complevel=9, complib=complib) as store:
        df.to_hdf(store, key, format='fixed')
    
    return

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

patient_list = ['chb01','chb02','chb03','chb04','chb05', 'chb06', 'chb07', 'chb08', 'chb09', 'chb10', 'chb11', 'chb13', 'chb14', 
                'chb15', 'chb16', 'chb17', 'chb18', 'chb19', 'chb20', 'chb21', 'chb22', 'chb23']
for patient in patient_list:
    df_table = pd.DataFrame(columns=header)
    with open(f'CHBMIT/{patient}/{patient}-summary.txt', 'r') as fp:
        # The pattern indicates that the pattern before ? could appear or not {0,1} times
        # The "[abc]*" patttern is used for chb17 case
        patientmatches = re.findall(f'{patient}[abc]*_\d+\+?.edf', fp.read())
        #print('patientmatches:', patientmatches)
        print('Number of recordings:', len(patientmatches))
        
        #nextmatches = [int(item.replace(f'{patient}_', '').replace('.edf', '')) for item in patientmatches]
        #nextmatches = [item.replace(f'{patient}_', '').replace('.edf', '') for item in patientmatches]
        #nextmatches = [item.replace(re.search(f'{patient}[abc]*_', item).group(), '').replace('.edf', '') for item in patientmatches]
        #print('nextmatches:', nextmatches)
        #print('Number of recordings:', len(nextmatches))
        
        fp.seek(0)
    
        matches = re.findall('\d+:\d+:\d{2}', fp.read())
        duration = [to_seconds(a,b) for a,b in list(zip(matches,matches[1:]))[::2]]
        
        if patient == 'chb17':
            id17 = patientmatches.index('chb17b_69.edf')
            duration[id17] = 1440
        
        assert len(patientmatches) == len(duration)
        
        print('duration:', [(a,b) for a,b in zip(patientmatches, duration)])
        #display(list(enumerate(np.cumsum(duration))))
        durationsum = np.append(0, np.cumsum(duration))
        file_duration = durationsum[-1]
        print('durationsum:', durationsum)
        #print('Number of recordings:', len(duration))
        print('durationsum:', [(a,b) for a,b in zip(patientmatches, durationsum)])
        print(len(durationsum), len(patientmatches))


        fp.seek(0)
        #Con los números podemos obtener los índices y buscar en el vector de duracion acumulada en que segundo
        Lines = fp.readlines()
    
        count = 0
        nseizures = 0
        # Strips the newline character
        seizure_list = []
        row_values = {'file':'chbX_X','file_start':0,'seizure_start':0,'seizure_end':0,'total_start':0,'total_end':0,'duration':0}
        for line in Lines:
            count += 1
            if 'File Name' in line:
                item = re.search(f'{patient}[abc]*_\d+\+?.edf', line).group()
                #record = int(item.replace(f'{patient}_', '').replace('.edf', ''))
                #print(re.search(f'{patient}[abc]*_', line).group())
                #record = item.replace(f'{patient}_', '').replace('.edf', '')
                #ppattern = re.search(f'{patient}[abc]*_', line).group()
                #record = item.replace(ppattern, '').replace('.edf', '')
                #row_values['file'] = f'{ppattern}{record}'
                row_values['file'] = item
                #print(item, end=' ')
            elif 'Number of Seizures in File:' in line:
                nseizures = int(re.search('\d', line).group())
                #if nseizures > 0:
                    #print(nseizures, end=' ')
                #else:
                    #print(nseizures)
            elif nseizures > 0:
                if 'Start Time:' in line:
                    index = patientmatches.index(item)
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
        dfs.to_csv(f'CHBMIT/{patient}/{patient}_input_params.csv', header=True, index=False)
        df_table.to_csv(f'CHBMIT/{patient}/{patient}_seizure_table.csv', header=True, index=False)

        # dir_l = f'chbmit/{patient}'
        n_records = len(patientmatches)
        index_i = 0
        index_f = 0

        print(patientmatches[0])
        #record = wfdb.edf2mit(f'{patient}_{nextmatches[0]}.edf', pn_dir=f'chbmit/{patient}', header_only=False, verbose=False, rdedfann_flag=False)
        #record = read_edf(f'{patient}_{nextmatches[0]}.edf', pn_dir=f'chbmit/{patient}', header_only=False, verbose=False, rdedfann_flag=False)
        record = read_edf(patientmatches[0], pn_dir=f'chbmit/{patient}', header_only=False, verbose=False, rdedfann_flag=False)

        desired_ch = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 
                    'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']

        n_sig = len(desired_ch)

        sig_len = record.sig_len

        sig_list = desired_ch

        # Check all the channels
        #for ch in desired_ch:
        #    assert(ch in sig_list)

        # El orden del array es muy importante porque si hacemos resize, dependiendo de en que dirección crezca el array,
        # se nos van a descolocar los valores o no. Eg: order='C' (default): si aumentamos el número de filas no hay problema,
        # en cambio si decidimos aumentar las columnas, se descolocan todos los valores. Como estamos guardando las
        # señales por columnas porque es mejor para el almacenamiento, tenemos que utilizar order='C'.
        total = np.zeros((sig_len*n_records, n_sig), order='C')+np.nan # suposicion inicial de tamaño
        print(np.shape(total))

        id_desired_ch = [record.sig_name.index(ch) for ch in desired_ch]
        eeg_inter = record.p_signal[:,id_desired_ch] * np.asarray(record.adc_gain)[id_desired_ch]
        print(np.shape(eeg_inter))

        # comparo un INT con FLOAT; si ponemos array_equal falla
        assert(np.allclose(np.asarray(record.init_value)[id_desired_ch], eeg_inter[0,:])) 

        index_f += sig_len

        print(index_i, index_f)
        total[index_i:index_f,:] = eeg_inter

        index_i = index_f

        # for i in range(2,n_records+1):
        #for i in nextmatches:
        for mysignal in patientmatches[1:]:
            #if mysignal == patientmatches[0]: # already saved
                #continue
            
            #mysignal = f'{patient}_{i}.edf'

            print(mysignal)
            #record = wfdb.edf2mit(mysignal, pn_dir=f'chbmit/{patient}', header_only=False, verbose=False, rdedfann_flag=False)
            record = read_edf(mysignal, pn_dir=f'chbmit/{patient}', header_only=False, verbose=False, rdedfann_flag=False)

            id_desired_ch = [record.sig_name.index(ch) for ch in desired_ch]
            # there is a problem with this recording
            # only the first 1440 seconds are ok
            if mysignal == 'chb17b_69.edf':
                eeg_inter = record.p_signal[:1440*256,id_desired_ch] * np.asarray(record.adc_gain)[id_desired_ch]
                index_f += 1440*256
            else:
                eeg_inter = record.p_signal[:,id_desired_ch] * np.asarray(record.adc_gain)[id_desired_ch]
                index_f += record.sig_len
            
            #print(record.init_value == eeg_inter[:,0].astype(int))
            # comparo un INT con FLOAT; si ponemos array_equal falla
            assert(np.allclose(np.asarray(record.init_value)[id_desired_ch], eeg_inter[0,:])) 
            
            #index_f += record.sig_len
            print(np.shape(eeg_inter))
            print(index_i, index_f)
            if sig_len*n_records < index_f: # si la suposicion inicial de tamaño falla
                total.resize((index_f, n_sig), refcheck=False)
            total[index_i:index_f,:] = eeg_inter
            index_i = index_f
        # Si he reservado más espacio de la cuenta, ajusto
        total.resize((index_f, n_sig), refcheck=False)

        df = pd.DataFrame(data=total, columns=sig_list)

        # Por si hay alguna etiqueta de columna repetida, que la hay: T8-P8
        s = df.columns.to_series()
        df.columns = s + s.groupby(s).cumcount().astype(str).replace({'0':''})

        assert(len(df)//256 == file_duration)
        # df.to_hdf(f'CHBMIT/{patient}/signal_{patient}.h5', key='data_frame', format='table', mode='w')
        # df.to_hdf(f'CHBMIT/{patient}/signal_{patient}.h5', key='data_frame', format='fixed', complevel=9, complib='zlib', mode='w')
        df.to_hdf(f'CHBMIT/{patient}/signal_{patient}.h5', key='data_frame', format='fixed', mode='w')

        channels = list(df.keys())
        #matches = [2670,2841,13656,13846,20988,21111,27617,27777,56083,56347]

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
        dfs.to_csv(f'CHBMIT/{patient}/seizures{wide}_{patient}.csv', header=True, index=False)

    # clean
    os.system(f'rm {patient}*.edf')