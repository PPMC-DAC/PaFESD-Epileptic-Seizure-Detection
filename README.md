# PaFES: Patterns augmented by Features Epileptic Seizure Detection

The detection of epileptic seizures can have a significant impact on the patients' quality of life and on their caregivers. In this paper we propose a method for detecting such seizures from electroencephalogram (EEG) data named PaFESD -Patterns augmented by Features Epileptic Seizure Detection-. The main novelty of our proposal consists in a detection model that combines EEG signal features with pattern matching. After cleaning the signal and removing artifacts (as eye blinking or muscle movement noise), time-domain and frequency-domain features are extracted to filter out non-seizure regions of the EEG. Jointly, pattern matching based on Dynamic Time Warping (DTW) distance is also leveraged to identify the most discriminative patterns of the seizures, even under scarce training data. The proposed model is evaluated on all patients in the CHB-MIT database, and the results show that it is able to detect seizures with an average F1 score of 98.9. Furthermore, our approach achieves a F1 score of 100% (no false alarms or missed true seizures) for 20 of the patients (out of 24). Additionally, we automatically detect the most seizure/non-seizure discriminative EEG channel so that a wearable with only two electrodes would suffice to warn patients of seizures.

## Requirements

The code was developed using Python 3.9.13 and the required packages are listed in the `requirements.txt` file. To install them, run the following command:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This will create a virtual environment and install the required packages. 

## Dataset

The dataset used in this work is the CHB-MIT Scalp EEG Database. It is available at https://physionet.org/content/chbmit/1.0.0/. In order to use the dataset, we facilitate several scripts to download and preprocess the data to be used in the algorithm. The scripts are available in the `script` folder.

... details about the scripts ...

In order to check the correct download of the dataset, you can compare the SHA256 hash of the downloaded files with the ones available in the `patients_hash.csv`.

## Usage

The main script to run the algorithm is `find_pattern.py`. We provide a script to run the algorithm for one patient and one channel. The script is available in the file `script/launch_execution.sh`. The script receives the following arguments:

```bash
python -u -W ignore find_pattern.py -P chb06 -S F3-C3 -w 16 -s 256 -it 1000 -z 4 -b 3 -est 'scm' -f True
```
- **-P chb06**: Specifies the patient number as `chb06`.
- **-S F3-C3**: Sets the channel label to `F3-C3`.
- **-w 16**: Defines the warping window size as `16` samples.
- **-s 256**: Sets the stride size to `256` samples.
- **-it 1000**: Specifies the number of iterations for the optimizer as `1000`.
- **-z 4**: Indicates that the algorithm will be trained with `4` seizures.
- **-b 3**: Sets the maximum number of patterns to use in a batch as `3`.
- **-est 'scm'**: Uses `scm` (Sample Covariance Matrix) as the covariance estimator to compute the distance.
- **-f True**: Activates the bandpass filter.

The rest of the arguments are set to their default values.