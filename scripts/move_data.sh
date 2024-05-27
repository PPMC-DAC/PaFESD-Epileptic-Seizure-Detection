#!/bin/bash

src_dir=$FSCRATCH/LAMP_project
dst_dir=$FSCRATCH/seizure-algorithm/CHBMIT

for patient in 'chb01' 'chb02' 'chb03' 'chb04' 'chb05' 'chb06' 'chb07' 'chb08' 'chb09' 'chb10' 'chb11' 'chb12' 'chb13' 'chb14' 'chb15' 'chb16' 'chb17' 'chb18' 'chb19' 'chb20' 'chb21' 'chb22' 'chb23' 'chb24'
do
    echo "Moving $patient"
    cp $src_dir/$patient/$patient\_* $dst_dir/$patient/
    cp $src_dir/$patient/seizures* $dst_dir/$patient/
    ln -s $src_dir/$patient/signal* $dst_dir/$patient/
done