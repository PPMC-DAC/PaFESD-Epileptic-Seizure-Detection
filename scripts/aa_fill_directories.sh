#!/bin/bash
# This script creates a directory for each patient and downloads the summary file for each patient
for patient in 'chb01' 'chb02' 'chb03' 'chb04' 'chb05' 'chb06' 'chb07' 'chb08' 'chb09' 'chb10' 'chb11' 'chb12' 'chb13' 'chb14' 'chb15' 'chb16' 'chb17' 'chb18' 'chb19' 'chb20' 'chb21' 'chb22' 'chb23' 'chb24'
do
    mkdir $patient
    wget "https://physionet.org/files/chbmit/1.0.0/$patient/$patient-summary.txt" -P "CHBMIT/$patient"
done
