import numpy as np
import pandas as pd
import sys
import time

import matplotlib.pyplot as plt

import pickle

# garbage collector
import gc

import mne
from mne.filter import filter_data
from mne import make_fixed_length_epochs

from pyriemann.estimation import Covariances
from pyriemann.utils.covariance import normalize
from pyriemann.clustering import PotatoField

# command-line
import argparse

import multiprocessing as mp

from mylibrary.common.utils import call_psd_tri
from mylibrary.common.utils import filter_bandpass
from mylibrary.common.utils import call_cov_est
from mylibrary.common.utils import call_dnc_artifact
from mylibrary.common.utils import call_max_dist
from mylibrary.common.utils import call_energy
from mylibrary.common.utils import get_cdf_categories
from mylibrary.common.utils import find_train_limits
from mylibrary.common.utils import search_bound_values
from mylibrary.common.utils import check_bound_values
from mylibrary.common.utils import closest_point_to_cluster_center
from mylibrary.common.utils import get_contained_evaluations
from mylibrary.common.utils import load_matrix_classes
from mylibrary.common.utils import dtime
from mylibrary.common.utils import def_bounds_categories

from mylibrary.common.classes import DTW_Matrix

# global variables
import mylibrary.common.config as config

from mylibrary.dtw_functions.find_seizure import check_dnc
from mylibrary.dtw_functions.find_seizure import query_mask
from mylibrary.dtw_functions.find_seizure import call_dnc_aware_full

from mylibrary.optimizer.functions import f_evaluation as f_evaluate
from mylibrary.optimizer.functions import f_optimization as f_optimize
from mylibrary.optimizer.functions import f_validation as f_validate

from lipo import GlobalOptimizer

from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

import EEGLIB

from os.path import exists, getsize

# def exploration_func(rp1_min, rp1_max, rp2_min, rp2_max, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards):
#     return f_exploration_rate(rp1_min, rp1_max, rp2_min, rp2_max, rp3_max, rd_min, rd_max, re_min, re_max, exclusion, lookbackwards, xiter, yiter, d_p1, d_p2, d_p3, d_d, d_e, q_p1, q_p2, q_p3, q_d, q_e, qmask, idx_query, idx_seizure_list, DoNotCompute, ex_dnc_time)

def evaluation_wrapper(x, args):
    return (x, *f_evaluate(**x, **args))

def validation_wrapper(x, lids, th, args):
    return (x, f_validate(**x, lids=lids, th=th, **args))

# define a tolerance
TOL = 0.1

# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////
# ███╗░░░███╗░█████╗░██╗███╗░░██╗
# ████╗░████║██╔══██╗██║████╗░██║
# ██╔████╔██║███████║██║██╔██╗██║
# ██║╚██╔╝██║██╔══██║██║██║╚████║
# ██║░╚═╝░██║██║░░██║██║██║░╚███║
# ╚═╝░░░░░╚═╝╚═╝░░╚═╝╚═╝╚═╝░░╚══╝
# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////
# ///////////////////////////////////////////////////////////////////

"""
Description:
    The main differences between this version and the previous one are:
        - The use of the EEGLIB library to compute the matrix
        - Avoid to explore or evaluate since all the dtws are already computed

Notes:
    _description_
"""


def main():

    # first, parse the input arguments

    # there is 12 possible arguments, but only 2 are required: patient (P) and signal (S)

    # from the point of view of the algorithm, the important parameters are the template (t), stride (s), warping window (w), filter (f), lookback (lb), number of iterations (it) and the threshold (th)

    parser = argparse.ArgumentParser(description='Computes the PSD using a slide window. The result is a vector whose size depends on the chosen STRIDE and QUERY sizes.')

    parser.add_argument('-P','--patient', required=True, help='Patient number. Eg: chb24')
    parser.add_argument('-S','--signal', required=True, help='Signal label. Eg: F4-C4')
    parser.add_argument('-t','--template', type=int, default=5*256, help='Template window size in samples.')
    parser.add_argument('-s','--stride', type=int, default=256, help='Stride size in samples.')
    parser.add_argument('-w','--warp', type=int, default=16, help='Warping window size in samples.')
    parser.add_argument('-n','--nprocs', type=int, default=mp.cpu_count(), help='Number of processes.')
    parser.add_argument('-c','--chunks', type=int, default=mp.cpu_count(), help='Divides the signal in N chunks for parallel processing.')
    parser.add_argument('-f','--filter', type=bool, default=False, help='Active bandpass filter.')

    parser.add_argument('-lb','--lookback', type=int, default=1, help='Number of consecutive seconds to check before compute')
    parser.add_argument('-il','--input_logs', type=str, default='empty', help='Log file with evaluations. Format: [(dict, pattern_id, DR, dist_threshold)]')

    parser.add_argument('-it','--num_iterations', type=int, default=100, help='Number of runs for each optimizer iteration.')

    parser.add_argument('-th','--threshold', type=float, default=.05, help='DTW distance threshold. Eg: .05 means 5% more than the worst distance.')

    parser.add_argument('-est','--estimator', type=str, default='scm', help='Covariance estimator used to compute the distance. Eg: scm, oas, lwf, sch')

    parser.add_argument('-z', '--seizures', type=int, default=4, help='Number of seizures to train the algorithm.')

    parser.add_argument('-b', '--batch', type=int, default=3, help='Maximum number of patterns to use as batch.')

    parser.add_argument('-ts', '--training_size', type=float, default=30., help='Percentage of the signal to use as training.')

    parser.add_argument('-cdf', '--cdf_limit', type=float, default=.5, help='This is the probability limit to design the bounds of the categories. Eg: .25 means that the LB are in [0, .25] and the UB in [.75, 1].')

    args = parser.parse_args()
    print(sys.argv[0])
    print(args)

    samplerate = 256

    int_seed = 321
    # rs = RandomState(MT19937(SeedSequence(int(time.time()))))
    rs = RandomState(MT19937(SeedSequence(int_seed)))
    # The seed for numpy.random is set. This way, we can reproduce the results
    # np.random.seed(int_seed)

    patient = args.patient
    signal = args.signal
    mw = args.template
    stride = args.stride
    nprocs = args.nprocs
    nchunks = args.chunks
    filtered = args.filter
    maxwarp = args.warp

    # This variable saved the number of epochs per second. We use this factor to convert seconds to epochs (or indices)
    factor = samplerate//stride
    lookbackwards = 1 + args.lookback*factor # actual position + N epochs
    # We use this to exclude X seconds from the beginning and end of the seizures. This way we avoid that the limits of the seizures are included in the evaluation of the Discrimination Ratio
    exclusion = 10*factor # 10 seconds

    input_logs = args.input_logs
    # This is the number of iterations for each loop of the optimizer
    num_iterations = args.num_iterations

    # threshold percentage
    thp = args.threshold

    # covariance estimator
    cov_estimator = args.estimator

    # We choose a desired number of seizures to train
    until_seizure = args.seizures

    # Number of patterns to use as batch
    batch_size = args.batch

    # Training size minimum limit
    training_size_limit = args.training_size

    # CDF limit
    cdf_limit = args.cdf_limit
    assert cdf_limit > 0. and cdf_limit < 1., 'The CDF limit must be in (0,1)'

    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////

    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////
    # ///////////////////////////////////////////////////////////////////

    start_time = time.time()
    # This is the start of the algorithm, the preprocessing phase
    # First we need to load the annotations previously extracted from the summary file
    dfs = pd.read_csv(f'CHBMIT/{patient}/{patient}_input_params.csv')
    seizure_list = dfs['seizure_list'].values

    # as we have start-stop pairs in the seizure_list vector, we need to divide by 2 to get the number of seizures
    nseizures = len(seizure_list)//2

    # In this case the signal has a lower number of seizures, so we choose n-1 seizures to train
    if nseizures < (until_seizure + 1):
        print('The signal doesn\'t have enough seizures')
        until_seizure = nseizures - 1
    
    print(f'We choose {until_seizure} seizures to train')
    # at least 1 seizure
    assert until_seizure > 0, 'The number of seizures must be greater than 0'

    # The next step is to load the entire time-series. This was previously extracted from the .edf files and conveniently concatenated in a single .h5 file, with a format of "table", as it is faster to read than a csv file
    dfi = pd.read_hdf(f'CHBMIT/{patient}/signal_{patient}.h5', key='data_frame')

    # max_length in samples. All the signals have the same length
    max_length = dfi.count()[0]
    print('MAX_LENGTH', max_length)

    # max_length in seconds
    max_length = max_length//samplerate

    # This is the list of desired channels. There is 18 channels in total
    channels = ['FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 
                'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 'FZ-CZ', 'CZ-PZ']

    # check that all the channels are in dfi
    assert all([ch in dfi.columns for ch in channels]), 'Some channels are missing in the dataframe'

    # We choose the desired channel from the ts
    ts = dfi[signal].values.reshape(1,-1)

    # Load the query. This contains the seizures concatenated
    dfq = pd.read_csv(f'CHBMIT/{patient}/seizures0_{patient}.csv')
    # dfq = metadata_to_seizures(dfi, channels, seizure_list)

    # We choose the desired channel from the query
    seizures = dfq[signal].values.reshape(1,-1)

    # This is the name of the file where should be saved the results of a previous run
    logs_file = f'CHBMIT/{patient}/opt_logs_{signal}_W{mw}_S{stride}_w{maxwarp}_L{lookbackwards}'
    # This is the name of the file where the DTW distances computed for each seizure should be saved
    matrix_name = f'CHBMIT/{patient}/dtw_matrix_{patient}_{signal}_W{mw}_S{stride}_w{maxwarp}'
    matrix_name += '_filtered.hdf' if filtered else '.hdf'

    # Convert seconds to indices of the epochs
    idx_seizure = [x*factor for x in seizure_list]
    # Add the first and last index of the list
    aux_list = [0] + idx_seizure + [-1]
    # This list contains the start and end of each region non-seizure/seizure
    idx_seizure_list_all = list(zip(aux_list,aux_list[1:]))
    print('idx_seizure_list:', idx_seizure_list_all)

    # Filter the signal and the query if that is the case
    if filtered:
        logs_file += '_filtered'
        lowcut = .5
        highcut = 70.
        print('Applying filter...', end = ' ')
        data_all = filter_data(ts, samplerate, lowcut, highcut, verbose=False)[0]
        query_all = filter_data(seizures, samplerate, lowcut, highcut, verbose=False)[0]
        print('Done.')
    else:
        # logs_file += '.pkl'
        data_all = ts[0]
        query_all = seizures[0]

    pos_eval = []
    pos_eval_x = []
    
    # If a log file wasn't provided, we save the results in a default file
    if input_logs == 'empty':
        input_logs = logs_file + '_categories.pkl'
    else:
        # If the file exists, we load the results from a previous run
        if exists(input_logs) and (getsize(input_logs) > 0):
            with open(input_logs, 'rb') as f:
                print('Loading logs:')
                init_lb = pickle.load(f)
                init_ub = pickle.load(f)
                pos_eval = pickle.load(f)
                print('  Number of evaluations loaded:', len(pos_eval))
                # pos_eval_x = [x for x,_,val,_ in pos_eval if val > 1.0]
                pos_eval_x = [x for x,_,val,_ in pos_eval if val > TOL]
                print('  Number of GOOD evaluations loaded:', len(pos_eval_x))
                # if we have already evaluated 5k configurations: limit the number of iterations
                num_iterations = 1000 if len(pos_eval) > 5000 else num_iterations
            
    print('Data shape: ', data_all.shape)
    print('Query shape: ', query_all.shape)

    # This is the number of epochs that we can analyze from the signal
    # xiter = (len(data_all)-mw+1)//stride
    xiter = (len(data_all)-mw)//stride + 1

    # This is the stride for the epochs of the query. I thought that it could be useful to have a different stride for the query, for example to have a more refined search for the patterns. But in the end I didn't use it, so it is the same as the stride of the signal.
    qstride = stride
    # Number of epochs that we can analyze from the query
    # yiter = (len(query_all)-mw+1)//qstride
    yiter = (len(query_all)-mw)//qstride + 1
    # This is the same as the "factor" variable but for the query, it is used to convert the seconds into indices of the epochs of the query
    qfactor = samplerate//qstride

    # This list is used to divide the chunks of the signal that each process will analyze in parallel
    chunks = np.linspace(0, xiter, num=nchunks+1, endpoint=True, dtype=int)
    # Indices list. Each element of the list is a tuple with the start and end of the chunk
    dlist = list(zip(chunks,chunks[1:]))

    # Same as chunks but for the query
    qchunks = np.linspace(0, yiter, num=nchunks+1, endpoint=True, dtype=int)
    # Indices list
    qlist = list(zip(qchunks,qchunks[1:]))

    # This list contains the cumulative sum of the duration of the seizures. We use this vector to, once we have decided the beginning and the end of the training in the signal, be able to locate the seizures we will use to train in the query
    duration_query_list = [0]
    for a,b in list(zip(seizure_list,seizure_list[1:]))[::2]:
        duration_query_list.append(b-a + duration_query_list[-1])
    print('duration_query_list', duration_query_list)
    # Check that duration_query doesn't exceed yiter. This way we can be sure that any epoch of the query will overflow the duration of the query
    if yiter < (duration_query_list[-1]*qfactor):
        print('We need to correct the duration_query_list')
        duration_query_list[-1] = yiter//qfactor

    # This mask avoids to compute patterns that overlap two seizures or exceed the length of the query
    qmask_all = np.asarray(query_mask(duration_query_list, mw, qstride, qfactor, yiter))

    # This is the start of the artifact rejection process
    # Obtain the labels of all the channels
    ch_names = list(dfq.keys())
    # Create the info object for the mne library
    info = mne.create_info(ch_names, sfreq=samplerate, ch_types='eeg')
    # Create the raw objects of the ts for the mne library. This contains the data and the info
    rawi = mne.io.RawArray(dfi.to_numpy().transpose(), info) 
    # Same for the query
    rawq = mne.io.RawArray(dfq.to_numpy().transpose(), info) 

    del(dfi)
    del(dfq)
    gc.collect()

    # Pick the desired channels. This discard the rest of the channels
    rawq.pick_channels(channels, ordered=True)
    rawi.pick_channels(channels, ordered=True)

    # RPF definition
    # Threshold on z-score of distance to reject artifacts. It is the number of standard deviations from the mean of distances to the centroid.
    z_th = 5
    # Threshold on probability to being clean, in (0, 1), combining probabilities of potatoes using Fisher’s method.
    p_th = 0.05

    if patient == 'chb18':
        p_th = 0.025
    # RPF configuration. It is a dictionary with the name of the potato as key and the parameters of the potato as value. Each potato has a name, a list of channels, a low and a high frequency, and a normalization method for the covariance matrix that could be 'corr', 'trace' or 'determinant'. We naively use the configuration of the paper. This coarse configuration is not optimal. I think that this is a good feature of the code that should be commented in the paper. TODO: Comment this in the paper
    rpf_config = {
        'RPF eye_blinks': {  # for eye-blinks
            'channels': ['FP1-F7', 'FP1-F3', 'FP2-F4', 'FP2-F8'],
            'low_freq': 1.,
            'high_freq': 20.},
        'RPF occipital': {  # for high-frequency artifacts in occipital area
            'channels': ['P7-O1', 'P3-O1', 'P4-O2', 'P8-O2'],
            'low_freq': 25.,
            'high_freq': 45.,
            'cov_normalization': 'trace'},  # trace-norm to be insensitive to power
        'RPF global_lf': {  # for low-frequency artifacts in all channels
            'channels': None,
            'low_freq': 0.5,
            'high_freq': 3.}
    }
    n_potatoes=len(rpf_config)
    # Create the RPF object
    rpf = PotatoField(metric='riemann', z_threshold=z_th, p_threshold=p_th,
                    n_potatoes=n_potatoes)

    # Epoch duration in seconds
    duration = mw/samplerate
    # Stride duration in seconds
    interval = stride/samplerate
    print(duration, interval)

    # EEG processing for RPF
    rpf_covs = []
    # First, process the epochs of the query
    for p in rpf_config.values():  # loop on potatoes
        # Use the bandpass filter of the mne library to filter the query
        rpf_sig = filter_bandpass(rawq, p.get('low_freq'), p.get('high_freq'),
                                channels=p.get('channels'))
        # This function creates an object with the epochs of the query.
        rpf_epochs = make_fixed_length_epochs(
            rpf_sig, duration=duration, overlap=duration - interval, verbose=False)
        # Compute the covariance matrices of the epochs
        covs_ = Covariances(estimator=cov_estimator).transform(rpf_epochs.get_data())
        # Normalize the covariance matrices if necessary
        if p.get('cov_normalization'):
            covs_ = normalize(covs_, p.get('cov_normalization'))
        rpf_covs.append(covs_)

    print('Train RPF...', end=' ')
    # RPF training
    # Notice that we are using the information of the query to train the RPF. This is not optimal either. We could provide the possibility for a specific segment to be chosen by the user, being the query segment the default behavior.
    # TODO: Comment this in the paper
    train_covs = yiter      # nb of matrices for training
    train_set = range(train_covs)
    # Fit the potato field from covariance matrices.
    rpf.fit([c[train_set] for c in rpf_covs])
    print('Done!')

    del(rawq)
    del(rpf_covs)
    del(rpf_sig)
    del(rpf_epochs)
    del(covs_)
    gc.collect()

    # pool = mp.get_context('forkserver').Pool(nprocs)
    pool = mp.Pool(nprocs)

    # When we have hundreds of epochs, as in the case of the query, we don't need parallelization. But when we have hundreds of thousands of epochs, as in the case of the signal, we do need parallelization.
    print('Parallel cov calculation...')
    # EEG processing for RPF
    rpf_covs = []
    for p in rpf_config.values():  # loop on potatoes
        # Filter the entire time-series
        rpf_sig = filter_bandpass(rawi, p.get('low_freq'), p.get('high_freq'),
                                channels=p.get('channels'))
        # Compute in parallel the covariance matrices of the epochs       
        cov_aux = pool.starmap(call_cov_est, [(rpf_sig._data[:,start*stride:stop*stride+mw-1], mw, stride, stop-start, cov_estimator) for start,stop in dlist])
        # cov_aux is a list of lists of covariance matrices. We need to flatten it to a list of covariance matrices
        covs_ = np.asarray([element for sublist in cov_aux for element in sublist])
        print(covs_.shape)
        # Normalize the covariance matrices if necessary
        if p.get('cov_normalization'):
            covs_ = normalize(covs_, p.get('cov_normalization'))
        rpf_covs.append(covs_)

    del(rawi)
    del(rpf_sig)
    del(cov_aux)
    del(covs_)
    gc.collect()

    print(f'Done!')

    start_artifact = time.time()

    print('Parallel artifact rejection...', end=' ', flush=True)
    # This reject the epochs with artifacts in parallel; decide which epochs to compute
    predictions = pool.starmap(call_dnc_artifact, [([rpf_covs[i][start:stop] for i in range(n_potatoes)], mw, stride, stop-start, rpf) for start,stop in dlist])
    # predictions is a list of lists of booleans. We need to flatten it to a list of booleans
    DoNotCompute_all = np.asarray([element for sublist in predictions for element in sublist])

    """
    I will explain how DoNotCompute is created and used in our algorithm:
        - DoNotCompute is a list of booleans, where each element corresponds to an epoch. A value of 'True' indicates that the epoch is artifacted or non-seizure, while a value of 'False' indicates that the epoch is clean or potentially seizure.
        - First, we apply the RPF method to detect the artifacted or non-seizure epochs, and we store the result in DoNotCompute.
        - Second, we precompute the metrics for all the epochs, obtain the minimum LB and maximum UB for each category, and apply the thresholds to prune the non-seizure epochs.
        - Third, within the optimization function, we use the selected category configuration to further prune the non-seizure epochs based on DoNotCompute.
    """

    print(f'Done! {time.time() - start_artifact} s')

    del(predictions)
    del(rpf_covs)
    del(rpf)
    gc.collect()

    # Check if the at least one epoch of the each seizure survived the artifact rejection
    compute_DTW = check_dnc(DoNotCompute_all, idx_seizure_list_all, exclusion)

    print(f'DoNotCompute: {np.count_nonzero(DoNotCompute_all)} out of {xiter}: Pruning = {(np.count_nonzero(DoNotCompute_all)/xiter)*100 :.2f}%')

    # This is a BIG "if". Here is where the preprocess step ends and the optimization step begins.
    if 0 < compute_DTW:

        print('Compute PSD, Energy and Amplitude once...', end=' ', flush=True)
        # Instead of perform 3 different algorithms, as calc_delta_power perform the complete welch algorithm, we can obtain the 3 psd only with 3 simps calls. This way, we compute the psd only once for all the epochs of the signal.
        d_aux = pool.starmap(call_psd_tri, [(data_all[start*stride:stop*stride+mw-1], mw, stride, stop-start) for start,stop in dlist])
        d_psd = np.asarray([element for sublist in d_aux for element in sublist])
        d_p1_all = d_psd[:,0]
        d_p2_all = d_psd[:,1]
        d_p3_all = d_psd[:,2]
        # The same for the energy and the max_dist
        d_aux = pool.starmap(call_energy, [(data_all[start*stride:stop*stride+mw-1], mw, stride, stop-start) for start,stop in dlist])
        d_e_all = np.asarray([element for sublist in d_aux for element in sublist])
        d_aux = pool.starmap(call_max_dist, [(data_all[start*stride:stop*stride+mw-1], mw, stride, stop-start) for start,stop in dlist])
        d_d_all = np.asarray([element for sublist in d_aux for element in sublist])

        # The same for the query
        q_aux = pool.starmap(call_psd_tri, [(query_all[start*qstride:stop*qstride+mw-1], mw, qstride, stop-start) for start,stop in qlist])
        q_psd = np.asarray([element for sublist in q_aux for element in sublist])
        q_p1_all = q_psd[:,0]
        q_p2_all = q_psd[:,1]
        q_p3_all = q_psd[:,2]
        # And the energy and the max_dist of the query
        q_aux = pool.starmap(call_energy, [(query_all[start*qstride:stop*qstride+mw-1], mw, qstride, stop-start) for start,stop in qlist])
        q_e_all = np.asarray([element for sublist in q_aux for element in sublist])
        q_aux = pool.starmap(call_max_dist, [(query_all[start*qstride:stop*qstride+mw-1], mw, qstride, stop-start) for start,stop in qlist])
        q_d_all = np.asarray([element for sublist in q_aux for element in sublist])

        pool.close()

        print('Done!')

        del(d_aux)
        del(q_aux)
        gc.collect()

        # Fix the number of categories for each measure. TODO: This is another parameter to optimize
        bins = 200
        # min_range is the minimum distance between the LB and the UB of the categories. We have to use this to avoid errors in the optimizer that needs LB < UB. TODO: This is another parameter to optimize; if this occurs we could remove the category from the optimization
        min_range = int(bins*.02) #2%
        assert(min_range > 1)

        # We need to compute the limits untile we get a good compute_DTW
        get_limits = True

        from_t = None
        end_t = None
        halo = None

        while get_limits:

            # Get the categories for the query and the data. The cdf of each measure is used to divide the categories and select the min and max categories for each measure        
            query_categories = dict()
            query_cdfs = dict()

            # We choose the limits of the training signal
            from_t,end_t,halo = find_train_limits(seizure_list, until_seizure, max_length, exclusion=10, min_id=from_t, max_id=end_t, halo=halo)

            assert from_t >= 0, 'Its not possible to obtain good limits'
            # training size in percentage of the signal
            training_size = (end_t-from_t)/max_length*100
            print(f'\tNew training limits: {from_t} - {end_t} ({training_size :.2f}%)')

            # We can select the end of the number of samples in reference to the number of templates
            # This list contains the end of each seizure
            end_seizure_list = seizure_list[1:][::2]
            # this mask points to those seizures that have been selected
            seizure_mask = (end_seizure_list >= from_t) & (end_seizure_list <= end_t)
            # Now the list contains only the selected seizures
            end_seizure_list = end_seizure_list[seizure_mask]
            # Just to make sure that the number of seizures is correct
            assert len(end_seizure_list) == until_seizure, 'There is a problem with the seizure selection'

            # This is a cropped list with tuples that contains the (start,end) of each seizure in the query. This list is cropped because we only want to use the seizures that are inside the training query
            idx_query_crop = np.asarray(list(zip(duration_query_list, duration_query_list[1:])))[seizure_mask]
            # Save the start of the training query
            from_tq = idx_query_crop[0][0]
            # The same way we choose the instant when the training query starts, we choose the index where the training query starts. This is necessary because when we find a pattern, its index will be relative to the training query not to the whole query. We need to pass this index to the function that select the patterns: function 1
            from_idq = from_tq*qfactor

            # Compute the limits for the epochs in the signal; we will use this to define the matrix
            dini = from_t * factor
            dend = end_t * factor
            # We need to define the categories using only the data in the training set
            qini = from_tq * qfactor # The query starts at from_tq seconds
            until_tq = idx_query_crop[-1][1] # The query ends at until_tq seconds
            qend = until_tq * qfactor
            # Crop the measures vectors to the query
            q_p1 = q_p1_all[qini:qend]
            q_p2 = q_p2_all[qini:qend]
            q_p3 = q_p3_all[qini:qend]
            q_d = q_d_all[qini:qend]
            q_e = q_e_all[qini:qend]
            # q_p1 = q_p1_all
            # q_p2 = q_p2_all
            # q_p3 = q_p3_all
            # q_d = q_d_all
            # q_e = q_e_all
            # categories for PSD1
            cdfs,categories = get_cdf_categories(q_p1, bins)
            query_categories['rp1_min'] = categories
            query_categories['rp1_max'] = categories
            query_cdfs['rp1_min'] = cdfs
            query_cdfs['rp1_max'] = cdfs
            # categories for PSD2
            cdfs,categories = get_cdf_categories(q_p2, bins)
            query_categories['rp2_min'] = categories
            query_categories['rp2_max'] = categories
            query_cdfs['rp2_min'] = cdfs
            query_cdfs['rp2_max'] = cdfs
            # categories for PSD3
            cdfs,categories = get_cdf_categories(q_p3, bins)
            query_categories['rp3_min'] = categories
            query_categories['rp3_max'] = categories
            query_cdfs['rp3_min'] = cdfs
            query_cdfs['rp3_max'] = cdfs
            # categories for energy
            cdfs,categories = get_cdf_categories(q_e, bins)
            query_categories['re_min'] = categories
            query_categories['re_max'] = categories
            query_cdfs['re_min'] = cdfs
            query_cdfs['re_max'] = cdfs
            # categories for max_dist
            cdfs,categories = get_cdf_categories(q_d, bins)
            query_categories['rd_min'] = categories
            query_categories['rd_max'] = categories
            query_cdfs['rd_min'] = cdfs
            query_cdfs['rd_max'] = cdfs

            # vcat_limit = {item:[np.min(query_categories[item]),np.max(query_categories[item])] for item in query_categories}
            # vcat_limit = {item:[0, len(query_categories[item])-1] for item in query_categories}

            # Add lookbackwards to categories. A lookbackwards=0 is wrong with this implementation
            query_categories['lookbackwards'] = [1,2,3,4,5]

            # We can add the maxwarp to the optimization. TODO: This is another parameter to optimize; initially we can use the same maxwarp
            # query_categories['maxwarp'] = [8,12,16,24]

            # This is to avoid to compute a lot of times the limits of the categories
            min_max_values = {}
            for key, value in query_categories.items():
                if key.endswith('_min'):
                    min_max_values[key] = min(value)
                elif key.endswith('_max'):
                    min_max_values[key] = max(value)
                elif key == 'lookbackwards':
                    min_max_values[key] = min(value)

            # Compute DNC for the categories limit for all the data not only the training set. This does not affect the optimization. It is only to avoid to compute the DNC of the validation epochs later
            DoNotCompute_all = np.asarray(call_dnc_aware_full( DoNotCompute_all, xiter, d_p1_all, d_p2_all, d_p3_all, d_d_all, d_e_all, 
                                                        min_max_values['rp1_min'], 
                                                        min_max_values['rp1_max'], 
                                                        min_max_values['rp2_min'],
                                                        min_max_values['rp2_max'],
                                                        min_max_values['rp3_min'], 
                                                        min_max_values['rp3_max'], 
                                                        min_max_values['rd_min'], 
                                                        min_max_values['rd_max'], 
                                                        min_max_values['re_min'], 
                                                        min_max_values['re_max'], 
                                                        min_max_values['lookbackwards']))

            # Check if at least one epoch of the each seizure survived the artifact rejection
            compute_DTW = check_dnc(DoNotCompute_all, idx_seizure_list_all, exclusion)
        
            if compute_DTW:
                # training size in percentage of the signal
                training_size = (end_t-from_t)/max_length*100
                if training_size < training_size_limit: # not yet
                    print('WARNING: the training set is not representative of the signal. Try to increase the training set')
                else:
                    print('GOOD training limits:', from_t, end_t)
                    print(f'Training with {training_size :.2f}% of the signal')
                    # End the search
                    get_limits = False
            else:
                print('BAD limits!')

            # This assertion cloud fail. This means that the training set is not representative of all the data. We have several options: 1) increase the training set, 2) loop this step until the assertion is true, 3) abandon the optimization
            # TODO: loop this step until the assertion is true
            # assert compute_DTW != 0, 'This should not happen! Compute the categories limits should not affect to compute_DTW'

        print(f'Minimum and maximum values of categories: {min_max_values}')

        # Compute DNCQ only for assertion; use only the training set
        dncq = (q_p1 < min_max_values['rp1_min']) | \
                (min_max_values['rp1_max'] < q_p1) | \
                (q_p2 < min_max_values['rp2_min']) | \
                (min_max_values['rp2_max'] < q_p2) | \
                (q_p3 < min_max_values['rp3_min']) | \
                (min_max_values['rp3_max'] < q_p3) | \
                (q_d < min_max_values['rd_min']) | \
                (min_max_values['rd_max'] < q_d) | \
                (q_e < min_max_values['re_min']) | \
                (min_max_values['re_max'] < q_e)
        # check that using min_max_values does not affect to the query
        assert len(dncq) == np.count_nonzero(~dncq), f'This should not happen! The query should not be affected by the categories limits. {len(dncq)} != {np.count_nonzero(~dncq)}'

        print(f'DoNotCompute Second Check: {np.count_nonzero(DoNotCompute_all)} out of {xiter}: Pruning = {(np.count_nonzero(DoNotCompute_all)/xiter)*100 :.2f}%')
        print(f'Compute DTW: {compute_DTW}')

        gc.collect()

        # Main matrix to store the results; We do not reserve the memory until the evaluation starts
        tDTW = DTW_Matrix(qini, qend, dini, dend)
        mDTW = DTW_Matrix(0, yiter, 0, xiter)
        # global variables to be used in the functions
        config.tDTW = tDTW
        config.mDTW = mDTW

        mtime = time.time()

        cdata = data_all.astype(dtype=np.float32)
        cquery = query_all.astype(dtype=np.float32)

        all_distances = EEGLIB.GetDistMtx(cdata, len(cdata), cquery, len(cquery), mw, stride, maxwarp, True)

        distMtx = np.reshape(all_distances, (xiter,yiter)).transpose()

        print(f'Elapsed time distMtx: {time.time()-mtime :.2f}')

        mDTW.data = distMtx

        # These are the timers to track the execution
        ex_time = []
        ex_rel = []
        ex_pruning = [[],[]] # dncq_sum & dnc_sum lists
        ex_dnc_time = []
        ex_opt_time = []
        ex_comp = []
        ex_epoch = [.005] # starting time
        # We use theses lists to handle the cropped data and query for training
        # data = []
        # query = []
        # The same for the measures, theses are the handlers for the query
        q_p1 = []
        q_p2 = []
        q_p3 = []
        q_d = []
        q_e = []
        # And the data
        d_p1 = []
        d_p2 = []
        d_p3 = []
        d_d = []
        d_e = []
        # Handler for the mask of the query
        qmask = []
        # Handler for the DNC of the signal
        DoNotCompute = []
        # Handler for the indices
        idx_seizure_list = []
        idx_query = []
                
        pre_eval = []
        init_lb = []
        init_ub = []

        # Not flexible at the start
        flexible_bound_threshold = -1.0

        # Number max of iterations of each optimization round. The optimization perform another round if it did not converge in the previous round, i.e., the number of DR>1.0 is less than a fixed number: by default 20 patterns
        max_loop_iterations = 2

        # Limit the number of the iterations of the optimization.
        num_iterations_opt = num_iterations

        # Number of random samples per X
        num_random_samples = 5000

        # Pure random search probability [0,1]. Search the next X randomly withouth using the previous pairs of X and Y. This is useful to scape from local maxima.
        random_search_probability = 0.0

        # Fix a seizure from where to start the training. start_from_seizure = 0 is the default behavior, i.e., start to explore from the first seizure.
        start_from_seizure = until_seizure - 1
        assert (0 <= start_from_seizure) and (start_from_seizure < until_seizure), f'This should not happen! start_from_seizure = {start_from_seizure} is not valid. It should be between 0 and {until_seizure-1}'

        # When the optimization with a seizure is already perform, we have a number of evaluations, good and bad evaluations. The bad evaluations give us information too, because we can discard it for the next round. This evaluations are provided to the optimizer before the next round. This time limit this process of testing later evaluations. TODO: This timer will disappear when we improve the execution time.
        evaluation_limit = 12 * 3600 

        # This is the time limit for the whole optimization process. TODO: This timer will disappear when we improve the execution time.
        search_limit = 16 * 3600 

        # If we hit max_loop_iterations or search_limit before the optimization finds a fixed number of good evaluations (20 as default), we stop the optimization setting this flag to False.
        converged = True

        # This flag does not control anything yet.
        validation = True

        print(f'Boiler-plate ends at {dtime(time.time()-start_time)}')

        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////


        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////

        # We do not loop in this case
        ii = until_seizure - 1

        print("""
            ███╗░░██╗███████╗░██╗░░░░░░░██╗  ░██████╗███████╗██╗███████╗██╗░░░██╗██████╗░███████╗
            ████╗░██║██╔════╝░██║░░██╗░░██║  ██╔════╝██╔════╝██║╚════██║██║░░░██║██╔══██╗██╔════╝
            ██╔██╗██║█████╗░░░╚██╗████╗██╔╝  ╚█████╗░█████╗░░██║░░███╔═╝██║░░░██║██████╔╝█████╗░░
            ██║╚████║██╔══╝░░░░████╔═████║░  ░╚═══██╗██╔══╝░░██║██╔══╝░░██║░░░██║██╔══██╗██╔══╝░░
            ██║░╚███║███████╗░░╚██╔╝░╚██╔╝░  ██████╔╝███████╗██║███████╗╚██████╔╝██║░░██║███████╗
            ╚═╝░░╚══╝╚══════╝░░░╚═╝░░░╚═╝░░  ╚═════╝░╚══════╝╚═╝╚══════╝░╚═════╝░╚═╝░░╚═╝╚══════╝
        """)

        print('Computed seizure:', ii)

        # if it is the last training, we use the already computed until_t
        until_t = end_t

        # we need to recompute idx_seizure_list for each seizure
        print(f'signal {ii} from {from_t} to {until_t}')
        # select the seizure limits
        idx_seizure_crop = seizure_list[(seizure_list >= from_t) & (seizure_list <= until_t)]
        # obtain the non-seizure/seizure indices inside cropped data
        idx_seizure = ((idx_seizure_crop - from_t)*factor).tolist()
        # append limits to idx_seizure
        aux_list = [0] + idx_seizure + [-1]
        # "non-seizure + Seizure" list
        idx_seizure_list = list(zip(aux_list,aux_list[1:]))
        print('idx_seizure_list:', idx_seizure_list)

        assert len(idx_seizure_list) == (ii+1)*2+1, f'idx_seizure_list has wrong length {len(idx_seizure_list)}, it should be {(ii+1)*2+1}. This could be due to a wrong HALO value. Check that the HALO value is correct within the find_train_limits function.'

        # obtain seizure indices inside cropped data
        idx_query = [((x-from_tq)*qfactor, (y-from_tq)*qfactor) for x,y in idx_query_crop]
        # print('idx_query:', idx_query)

        # once we have the limits for seizures, generate a list of colors, where each pattern from a different seizure has a different color. Eg: [0,0,0,1,1,1,2,2,2,3,3,3] for 4 seizures with 3 patterns each
        seizure_colors = np.repeat(np.arange(len(idx_query)), [y-x for x,y in idx_query])

        # Select the chunk query for this training
        qini = from_tq * qfactor
        qend = until_tq * qfactor
        yiter = qend - qini
        q_p1 = q_p1_all[qini:qend]
        q_p2 = q_p2_all[qini:qend]
        q_p3 = q_p3_all[qini:qend]
        q_d = q_d_all[qini:qend]
        q_e = q_e_all[qini:qend]
        # Mask
        qmask = qmask_all[qini:qend]
        # Select the chunk data for this training
        dini = from_t * factor
        dend = until_t * factor
        xiter = dend - dini
        d_p1 = d_p1_all[dini:dend]
        d_p2 = d_p2_all[dini:dend]
        d_p3 = d_p3_all[dini:dend]
        d_d = d_d_all[dini:dend]
        d_e = d_e_all[dini:dend]
        # DNC with artifact rejection
        DoNotCompute = DoNotCompute_all[dini:dend]
            
        # Search
        num_iterations = num_iterations_opt

        # Read the main matrix if it exists
        load_matrix_classes(matrix_name, tDTW, mDTW)

        repeat = True
        # Number of re-trainings
        n_opt = 0

        # all flexible at start; it needs flexible_bounds_threshold to be greater than 0 to be flexible, but it is decided later
        flexible_bounds = {key:[True,True] for key in query_categories}

        print('Looking for logs:')
        # if we have saved logs, we load them
        if pos_eval:

            # obtain bounds from latest exploration; use only evaluations with DR > 1.0
            init_lb,init_ub = search_bound_values(pos_eval_x, query_categories)
            # check if the bounds are valid or fix them
            check_bound_values(init_lb, init_ub, query_categories)

            # Flexible
            flexible_bound_threshold= .05
            # If we make epsilon = 0.0, the optimizer will search until the floating point precision. This scenario is not good for the optimizer if we have normal bounds, because the "local-minimum" problem will be more likely to happen. TODO: What happen now we have discrete categories? Is this still a problem or not?
            # epsilon = .01

            # sort the evaluations
            # its better this way because it perform the re-computations in order, based on the magnitude of the DR achieved (from the best to the worst)
            pos_eval = sorted(pos_eval, key=lambda x: x[2], reverse=True)
            # List of DRs ordered from the best to the worst
            # print([x[2] for x in pos_eval])
            # join all the executions; keep only the configurations; TOL to avoid problems with floating point precision
            pre_eval_x = [x for x,_,val,_ in pos_eval if val > TOL]
            # the last M are the relations == 0
            pre_eval = pos_eval[len(pre_eval_x):]

            print('\n\nINIT LB:', init_lb)
            print('INIT UB:', init_ub)
            print('Using categories:', query_categories.keys())
            print('INIT flexible_bounds:', flexible_bounds)
            print('INIT flexible_bound_threshold:', flexible_bound_threshold)
            print('Bins:', bins, 'Min Range:', min_range)

            # function to evaluate
            # evaluation_func = lambda rp1_min, rp1_max, rp2_min, rp2_max, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards: \
            #                 f_evaluate(rp1_min=rp1_min, rp1_max=rp1_max, rp2_min=rp2_min, rp2_max=rp2_max, rp3_max=rp3_max, rd_min=rd_min, rd_max=rd_max, re_min=re_min, re_max=re_max, lookbackwards=lookbackwards, exclusion=exclusion, xiter=xiter, yiter=yiter, d_p1=d_p1, d_p2=d_p2, d_p3=d_p3, d_d=d_d, d_e=d_e, q_p1=q_p1, q_p2=q_p2, q_p3=q_p3, q_d=q_d, q_e=q_e, qmask=qmask, idx_query=idx_query, idx_seizure_list=idx_seizure_list, DoNotCompute=DoNotCompute, from_idq=from_idq, tDTW=tDTW.data)

            print(f'\t{len(pos_eval)} logs; {len(pre_eval)} evaluations == 0.0;')
            # We recompute uploading the evaluations with DR == 0.0
            max_evaluations = len(pre_eval_x)

            eval_t = time.time()

            evargs = {
                'exclusion': exclusion,
                'xiter': xiter,'yiter': yiter,
                'd_p1': d_p1,'d_p2': d_p2,'d_p3': d_p3,'d_d': d_d,'d_e': d_e,
                'q_p1': q_p1,'q_p2': q_p2,'q_p3': q_p3,'q_d': q_d,'q_e': q_e,
                'qmask': qmask,
                'idx_query': idx_query,
                'idx_seizure_list': idx_seizure_list,
                'DoNotCompute': DoNotCompute,
                'from_idq': from_idq,
                'batch_size': batch_size,
                'colors': seizure_colors,
            }
            
            # Parallelize the search function
            with mp.Pool(processes=nprocs) as pool:
                pre_eval += pool.starmap(evaluation_wrapper, [(x, evargs) for x in pre_eval_x])
            evaluated = len(pre_eval_x)

            print(f'\tEvaluated TOTAL: {evaluated}/{max_evaluations} configurations; Finished in {dtime(time.time()-eval_t)}; {dtime(time.time()-start_time)} elapsed')
                                
        # in the case we do not have saved logs
        else:
            print('Default logs')
            # Initialize the lower and upper bounds using the CDFs
            init_lb, init_ub = def_bounds_categories(query_cdfs, lb_limit=cdf_limit, ub_limit=1.-cdf_limit)

            # Check if the bounds are valid or fix them
            check_bound_values(init_lb, init_ub, query_categories, min_range=min_range)
            # If we make epsilon = 0.0, the optimizer will search until the floating point precision. This scenario is not good for the optimizer if we have normal bounds, because the "local-minimum" problem will be more likely to happen. TODO: What happen now we have discrete categories? Is this still a problem or not?
            # epsilon = .01

        # vector of weights for the objective function: DR and dmax
        weights = np.array([1., 50.])

        # Change the function to optimize once we evaluate all the explored data
        search_func = lambda rp1_min, rp1_max, rp2_min, rp2_max, rp3_min, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards: \
                f_optimize(rp1_min=rp1_min, rp1_max=rp1_max, rp2_min=rp2_min, rp2_max=rp2_max, rp3_min=rp3_min, rp3_max=rp3_max, rd_min=rd_min, rd_max=rd_max, re_min=re_min, re_max=re_max, exclusion=exclusion, lookbackwards=lookbackwards, xiter=xiter, yiter=yiter, d_p1=d_p1, d_p2=d_p2, d_p3=d_p3, d_d=d_d, d_e=d_e, q_p1=q_p1, q_p2=q_p2, q_p3=q_p3, q_d=q_d, q_e=q_e, qmask=qmask, idx_query=idx_query, idx_seizure_list=idx_seizure_list, DoNotCompute=DoNotCompute, from_idq=from_idq, batch_size=batch_size, colors=seizure_colors)

        loaded_lb = init_lb.copy()
        loaded_ub = init_ub.copy()
        # start the optimization
        while repeat:

            gc.collect() # garbage collector

            print('\n\nUsing LB:', loaded_lb)
            print('Using UB:', loaded_ub)
            print('Using categories:', query_categories.keys())
            print('flexible_bounds:', flexible_bounds)
            print('flexible_bound_threshold:', flexible_bound_threshold)
            print('Bins:', bins, 'Min Range:', min_range, flush=True)

            print(f'\n\tOptimization loop {n_opt} (seizure {ii}) starts at {dtime(time.time()-start_time)}')

            opt_t = time.time()
            # optimizer handler; here is where the later evaluations are added
            search = GlobalOptimizer(
                search_func,
                lower_bounds=loaded_lb,
                upper_bounds=loaded_ub,
                categories=query_categories,
                log_args=[],
                evaluations=pre_eval,
                maximize=True,
                flexible_bounds=flexible_bounds,
                flexible_bound_threshold=flexible_bound_threshold,
                random_state=rs.randint(1, 10000),
                random_search_probability=random_search_probability,
                num_random_samples=num_random_samples,
            )

            # run the optimizer
            search.run(num_function_calls=num_iterations)

            opt_t = time.time() - opt_t
            print(f'\t{num_iterations} iterations; Finished in {dtime(opt_t)}; {dtime(time.time()-start_time)} elapsed')
            # print(f'opt_t: {opt_t} s')
            ex_opt_time.append(opt_t)

            # number of optimizer trainings
            n_opt += 1

            # make the bounds flexible
            flexible_bound_threshold = 0.05

            # print('Obtain evaluations:')
            pos_eval = search.evaluations
            # keep only the configurations with a DR greater than 1.0 (good evaluations)
            # pos_eval_x = [dicc for dicc,_,val,_ in pos_eval if val > 1.0]
            # as we use normalized DR, we can use TOL to filter the good evaluations
            pos_eval_x = [dicc for dicc,_,val,_ in pos_eval if val > TOL]

            # this is the best configuration found by the optimizer
            print('Search optimum:', search.optimum)
            search_elapsed = int(time.time()-start_time)
            # print(f'Search elapsed: {timedelta(seconds=search_elapsed)} elapsed')

            # Limit by number of trainings and time
            if (n_opt == max_loop_iterations) | (search_limit < search_elapsed):
            # elif (n_opt == max_loop_iterations):
                repeat = False
                print('LIMIT REACHED')

                # minimum number of good results to accept the search. TODO: This is another possible parameter to adjust. In this case we are good with 1 good evaluation. Notice that when the easier is the problem of finding the best configuration, the greater can be the number of good evaluations, and viceversa.
                if len(pos_eval_x) < 1:
                    converged = False
                    print('The OPTIMIZER doesn\'t converged!')
                    if ex_time:
                        print('DTW ex. mean time:', np.mean(ex_time))
                        print('EX total time:', np.sum(ex_time))
                        print('EX per epoch:', np.mean(ex_epoch))
                        print('Optimizer mean time:', np.mean(ex_opt_time))
                        print('Optimizer total time:', np.sum(ex_opt_time))
                        print('DNC mean time:', np.mean(ex_dnc_time))
                        print('DNC total time:', np.sum(ex_dnc_time))

            # If we do not hit the limit, but we have enough good evaluations
            elif (20 < len(pos_eval_x)):
                # repeat = False
                print(f'OK evaluations: {len(pos_eval_x)}; New bounds!')
                # Instead of end the execution, keep searching with more accuracy. TODO: We can also reduce the epsilon value to make the search more accurate.
                loaded_lb,loaded_ub = search_bound_values(pos_eval_x, query_categories)
                # Check bounds
                check_bound_values(loaded_lb, loaded_ub, query_categories)
                # TODO: Revise this; Let's focus on the evaluations that we already have and not in the new ones.
                # epsilon = 0.0

            # If we don't have enough good logs -> repeat
            else:
                print(f'\n\n\nREPEAT! Opt iterations:{n_opt}; Total eval:{len(pos_eval)}; Good eval:{len(pos_eval_x)}\n\n\n')
                # TODO: We can also reduce the epsilon value to make the search more accurate.
                # epsilon = epsilon * 0.5
                
            pre_eval = pos_eval

        # At this point the training process has finished. Getting one of these errors means that when repeating the search the optimizer doesn't know that we have already used that combination of arguments.
        n_err = 0
        print('Final evaluation check...', end=' ')
        for i,item in enumerate(pos_eval):
            for j in range(i+1, len(pos_eval)):
                if item == pos_eval[j]:
                    n_err += 1
                    print('\t Error: Repeated evaluation')
        if n_err == 0:
            print('OK')
        else:
            # converged = False
            print('\n')

        # If the training converged
        if converged:

            # Obtain bounds
            int_lb,int_ub = search_bound_values(pos_eval_x, query_categories)
            # check if the bounds are valid or fix them
            check_bound_values(init_lb, init_ub, query_categories)

            print('Integer LB:', int_lb)
            print('Integer UB:', int_ub)
                    
            # If I save all de evaluations could have a problem with the limits LB and UB
            # We dont need to be concern if we discard the values < 1.0 at the start of the loop, as we did
            with open(logs_file + '_categories.pkl', 'wb') as f:
                pickle.dump(int_lb, f)
                pickle.dump(int_ub, f)
                pickle.dump(pos_eval, f)

            # print('Sorted evaluations:', end=' ')
            pos_eval = sorted(pos_eval, key=lambda x: x[2], reverse=True)
            # print(pos_eval[:20])

            print('Logs saved:', logs_file + '_categories.pkl')
            print('Number of evaluations:', len(pos_eval))
            print('Good evaluations:', len(pos_eval_x))

        print(f'\n///////// TRAINING elapsed time: {time.time()-start_time :.2f}', flush=True)

        if converged & validation:

            # print(f'\n\n\tChecking evaluations:')
            # optlogs = logs_file + '_gold_opt.pkl'
            # # check the evaluations from the optimization process
            # check_evaluations(optlogs, pos_eval, ex_time)
            
            print("""
                ██╗░░░██╗░█████╗░██╗░░░░░██╗██████╗░░█████╗░████████╗██╗░█████╗░███╗░░██╗
                ██║░░░██║██╔══██╗██║░░░░░██║██╔══██╗██╔══██╗╚══██╔══╝██║██╔══██╗████╗░██║
                ╚██╗░██╔╝███████║██║░░░░░██║██║░░██║███████║░░░██║░░░██║██║░░██║██╔██╗██║
                ░╚████╔╝░██╔══██║██║░░░░░██║██║░░██║██╔══██║░░░██║░░░██║██║░░██║██║╚████║
                ░░╚██╔╝░░██║░░██║███████╗██║██████╔╝██║░░██║░░░██║░░░██║╚█████╔╝██║░╚███║
                ░░░╚═╝░░░╚═╝░░╚═╝╚══════╝╚═╝╚═════╝░╚═╝░░╚═╝░░░╚═╝░░░╚═╝░╚════╝░╚═╝░░╚══╝
            """)

            # For the validation we need to use the whole dataset
            idx_seizure_list = idx_seizure_list_all

            # yiter = (len(query_all)-mw+1)//qstride
            yiter = (len(query_all)-mw)//qstride + 1
            q_p1 = q_p1_all
            q_p2 = q_p2_all
            q_p3 = q_p3_all
            q_d = q_d_all
            q_e = q_e_all
            
            qmask = qmask_all

            # xiter = (len(data_all)-mw+1)//stride
            xiter = (len(data_all)-mw)//stride + 1
            d_p1 = d_p1_all
            d_p2 = d_p2_all
            d_p3 = d_p3_all
            d_d = d_d_all
            d_e = d_e_all

            DoNotCompute = DoNotCompute_all

            # vector with thresholds percentages to test different thresholds
            fth = np.arange(1.0, 2.0, 0.05, dtype=float)

            del(tDTW)

            # We want to make idy a list of indices, so we can return several patterns
            # val_func = lambda rp1_min, rp1_max, rp2_min, rp2_max, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards, lids, th: \
            #                 f_validate(rp1_min=rp1_min, rp1_max=rp1_max, rp2_min=rp2_min, rp2_max=rp2_max, rp3_max=rp3_max, rd_min=rd_min, rd_max=rd_max, re_min=re_min, re_max=re_max, lookbackwards=lookbackwards, lids=lids, th=th, fth=fth, exclusion=exclusion, xiter=xiter, d_p1=d_p1, d_p2=d_p2, d_p3=d_p3, d_d=d_d, d_e=d_e, idx_seizure_list=idx_seizure_list, DoNotCompute=DoNotCompute, mDTW=mDTW.data)

            # Initially, try all the good evaluations
            # final_eval =[(x, val_func(**x, lids=lids, th=th)) for x,lids,val,th in pos_eval if val > 1.0]

            # final_eval_x = [(x, lids, th) for x,lids,val,th in pos_eval if val > 1.0]
            final_eval_x = [(x, lids, dr, th) for x,lids,dr,th in pos_eval if dr > TOL]

            # Generate an stadistics of the evaluations
            print('Evaluations statistics:')
            evals_dr = np.asarray([dr for _,_,dr,_ in final_eval_x])
            evals_dmax = np.asarray([dmax for _,_,_,dmax in final_eval_x])
            print(f'\tDR: mean:{np.mean(evals_dr):.2f}; std:{np.std(evals_dr):.2f}; max:{np.max(evals_dr):.2f}; min:{np.min(evals_dr):.2f}')
            print(f'\tDmax: mean:{np.mean(evals_dmax):.2f}; std:{np.std(evals_dmax):.2f}; max:{np.max(evals_dmax):.2f}; min:{np.min(evals_dmax):.2f}')
            # Generate a vector of boolean where dmax is close to 0.0
            evals_dmax_bool = np.array(evals_dmax) < 0.1
            # Print the evaluations with dmax close to 0.0
            print(f'\tEvaluations with dmax close to 0.0: {evals_dmax_bool.sum()}')
            for i in np.where(evals_dmax_bool)[0]:
                print(f'\t\t{final_eval_x[i][1:]}')

            val_t = time.time()

            vargs = {
                'fth': fth,
                'exclusion': exclusion,
                'xiter': xiter,
                'd_p1': d_p1,'d_p2': d_p2,'d_p3': d_p3,'d_d': d_d,'d_e': d_e,
                'idx_seizure_list': idx_seizure_list,
                'DoNotCompute': DoNotCompute,
                # 'mDTW': mDTW.data
            }

            # Parallelize the search function
            with mp.Pool(processes=nprocs) as pool:
                final_eval = pool.starmap(validation_wrapper, [(x, lids, th, vargs) for x,lids,_,th in final_eval_x])
            max_validations = len(final_eval_x)

            print(f'\tValidated TOTAL: {len(final_eval)}/{max_validations} configurations; Finished in {dtime(time.time()-val_t)}; {dtime(time.time()-start_time)} elapsed')

            # vallogs = logs_file + '_gold_val.pkl'
            # check_validations(vallogs, final_eval)

            # save the list of ids
            # pattern_list = [lids for _,lids,val,_ in pos_eval if val > 1.0]
            pattern_list = [lids for _,lids,val,_ in pos_eval if val > TOL]
            # Check the results of the evaluation
            count_good = np.zeros(len(fth), dtype=int)
            # the output of the function is a list of tuples: (tp, tn, fp, fn) for each threshold
            btp = 0 # best true positives
            bfp = np.inf # best false positives
            bp = 0 # best position
            bev = 0 # best evaluation
            n_patterns = 0 # number of patterns

            # to save the good validations
            prev_id = -1
            good_val = []

            for i,res in enumerate(final_eval):
                position = 0
                # the wrapper function returns: res = list((x, vres))
                for tp,tn,fp,fn in res[1]:
                    if (tp == nseizures) & (fp == 0):
                        count_good[position] += 1
                        # Save the good validations; we only save one validation per configuration
                        if prev_id < i:
                            good_val.append(final_eval_x[i])
                            prev_id = i
                            # assert that both dictionaries are the same
                            assert final_eval_x[i][0] == res[0], 'The dictionaries are not the same'
                    # If there is not a perfect match, we search for the best one
                    if tp > btp:
                        # if there is a better tp, we allow more false positives
                        btp = tp
                        bfp = fp
                        bp = position
                        bev = i
                        # print(f'New best: {final_eval_x[bev][1:]}')
                    elif tp == btp:
                        # if there is a tie in tp, we choose the one with less false positives
                        if fp < bfp:
                            bfp = fp
                            bp = position
                            bev = i
                            # print(f'New best: {final_eval_x[bev][1:]}')
                    position += 1

            assert len(pattern_list[bev]) == len(final_eval_x[bev][1]), 'The number of patterns is not the same as the number of ids'

            if good_val:
                # Generate an stadistics of the Validations
                print('Validations statistics:')
                print(f'\tNumber of good validations: {len(good_val)}')
                vals_dr = np.asarray([dr for _,_,dr,_ in good_val])
                vals_dmax = np.asarray([dmax for _,_,_,dmax in good_val])
                print(f'\tDR: mean:{np.mean(vals_dr):.2f}; std:{np.std(vals_dr):.2f}; max:{np.max(vals_dr):.2f}; min:{np.min(vals_dr):.2f}')
                print(f'\tDmax: mean:{np.mean(vals_dmax):.2f}; std:{np.std(vals_dmax):.2f}; max:{np.max(vals_dmax):.2f}; min:{np.min(vals_dmax):.2f}')
                # Print the validation with min dmax
                print(f'\tValidation with min dmax: {good_val[np.argmin(vals_dmax)]}')
                # Print the validation with max dr
                print(f'\tValidation with max dr: {good_val[np.argmax(vals_dr)]}')
            else:
                # Print the best validation
                print(f'Best validation: {final_eval_x[bev]}')


            print(f'Best ALL POS: {bev}, Patterns: {len(pattern_list[bev])}, TP: {btp}, FP: {bfp}, Th: {fth[bp]:.2f}%')

            print('Number of validations:', len(final_eval))
            print(f'Training size: {training_size :.2f}%')
            print('Number of seizures:', nseizures)
            for i in range(len(fth)):
                print(f'THRESHOLD {fth[i]:.2f}')
                print(f'Good FINAL validations: {count_good[i]}')
                print(f'SUCCESS RATE: {(count_good[i]/len(final_eval))*100 :.2f}%')

            gc.collect()

            n_clusters = 1

            # In the case that the number of good evaluations is less than the number of clusters we want to apply, we cannot apply the clustering techniques
            if len(final_eval) < n_clusters:
                print('Not enough good evaluations to apply clustering')
            else:
                # Let's apply k-medoids to the evaluations that performed well
                good_logs = np.asarray([x for x in pos_eval if x[2] > TOL], dtype=object)

                # We need to convert the dictionary to a list
                X = np.asarray([[x[key] for key in x.keys() if key != 'lookbackwards'] for x,_,_,_ in good_logs])

                #TODO: we need to find a way to get the best k; maybe we can use the elbow method or the silhouette method; meanwhile, we will use k=3
                kmedoids = KMedoids(n_clusters=n_clusters, random_state=0).fit(X)
                
                #using the indices of the medoids, we can get the parameters of the medoids
                medoids_args = good_logs[kmedoids.medoid_indices_]
                print(medoids_args)
                # evaluate the medoids
                # medoid_eval =[(x, val_func(**x, lids=lids, th=th)) for x,lids,_,th in medoids_args]
                medoid_eval =[validation_wrapper(x, lids, th, vargs) for x,lids,_,th in medoids_args]

                if n_clusters > 1:
                    # in this case we have to count how many evaluations are good for each threshold
                    # we can do this by counting how many evaluations have tp == nseizures and fp == 0
                    count_good = np.zeros(len(fth), dtype=int)
                    # same as using final_eval
                    for _,vres in medoid_eval:
                        position = 0
                        for tp,tn,fp,fn in vres:
                            if (tp == nseizures) & (fp == 0):
                                count_good[position] += 1
                            position += 1

                    print('Number of KMEDOIDS evaluations:', len(medoid_eval))
                    for i in range(len(fth)):
                        print(f'KMEDOIDS Th: {fth[i]:.2f}')
                        print(f'Good KMEDOIDS evaluations: {count_good[i]}')
                        print(f'KMEDOIDS RATE: {(count_good[i]/len(medoid_eval))*100 :.2f}%')

                else:
                    n_patterns = len(medoids_args[0][1])
                    # in this case it displays the number of true positives and false positives, instead of counting how many evaluations are good
                    _,vres = medoid_eval[0]
                    position = 0
                    btp = 0 # best tp
                    bfp = np.inf # best fp
                    bp = 0 # best position
                    for tp,tn,fp,fn in vres:
                        print(f'KMEDOIDS POS: {position}, TP: {tp}, FP: {fp}')
                        if btp == nseizures:
                            position += 1
                            continue
                        if tp > btp:
                            # if there is a better tp, we allow more false positives
                            btp = tp
                            bfp = fp
                            bp = position
                        elif tp == btp:
                            # if there is a tie in tp, we choose the one with less false positives
                            if fp < bfp:
                                bfp = fp
                                bp = position
                        position += 1
                    print(f'Best KMEDOIDS POS: {bp}, Patterns: {n_patterns}, TP: {btp}, FP: {bfp}, Th: {fth[bp]:.2f}%')

                DR_values = np.asarray([x[2] for x in good_logs])

                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X, sample_weight=DR_values)

                # actual_ids = index_representative_points(kmeans, X)
                # actual_ids = index_representative_points2(kmeans, X)
                actual_ids = closest_point_to_cluster_center(kmeans, X)

                kmeans_args = good_logs[actual_ids]

                # kmeans_eval = [(x, val_func(**x, lids=lids, th=th)) for x,lids,_,th in kmeans_args]
                kmeans_eval =[validation_wrapper(x, lids, th, vargs) for x,lids,_,th in kmeans_args]

                if n_clusters > 1:
                    # in this case we have to count how many evaluations are good for each threshold
                    # we can do this by counting how many evaluations have tp == nseizures and fp == 0
                    count_good = np.zeros(len(fth), dtype=int)
                    # same as using final_eval
                    for _,vres in kmeans_eval:
                        position = 0
                        for tp,tn,fp,fn in vres:
                            if (tp == nseizures) & (fp == 0):
                                count_good[position] += 1
                            position += 1

                    print('Number of KMEANS evaluations:', len(kmeans_eval))
                    for i in range(len(fth)):
                        print(f'KMEANS Th: {fth[i]:.2f}')
                        print(f'Good KMEANS evaluations: {count_good[i]}')
                        print(f'KMEANS RATE: {(count_good[i]/len(kmeans_eval))*100 :.2f}%')

                else:
                    n_patterns = len(kmeans_args[0][1])
                    # in this case it displays the number of true positives and false positives, instead of counting how many evaluations are good
                    _,vres = kmeans_eval[0]
                    position = 0
                    btp = 0 # best tp
                    bfp = np.inf # best fp
                    bp = 0 # best position
                    for tp,tn,fp,fn in vres:
                        print(f'KMEANS POS: {position}, TP: {tp}, FP: {fp}')
                        if btp == nseizures:
                            position += 1
                            continue
                        if tp > btp:
                            # if there is a better tp, we allow more false positives
                            btp = tp
                            bfp = fp
                            bp = position
                        elif tp == btp:
                            # if there is a tie in tp, we choose the one with less false positives
                            if fp < bfp:
                                bfp = fp
                                bp = position
                        position += 1
                    print(f'Best KMEANS POS: {bp}, Patterns: {n_patterns}, TP: {btp}, FP: {bfp}, Th: {fth[bp]:.2f}%')

                # another solution could be to select those evaluations that contain the greater number of evaluations inside their own limits
                # e.g. if we have an evaluation with rp1_min = 10 and rp1_max = 20, we can count how many evaluations have rp1_min >= 10 and rp1_max <= 20.
                # we can do this for each couple, _min/_max, of parameters. In the case we don't have _min/_max, we can use the most permissive value.

                contained_args = get_contained_evaluations(good_logs, n_clusters)

                # contained_evals = [(x, val_func(**x, lids=lids, th=th)) for x,lids,_,th in contained_args]
                contained_evals = [validation_wrapper(x, lids, th, vargs) for x,lids,_,th in contained_args]

                if n_clusters > 1:
                    # in this case we have to count how many evaluations are good for each threshold
                    # we can do this by counting how many evaluations have tp == nseizures and fp == 0
                    count_good = np.zeros(len(fth), dtype=int)
                    # same as using final_eval
                    for _,vres in contained_evals:
                        position = 0
                        for tp,tn,fp,fn in vres:
                            if (tp == nseizures) & (fp == 0):
                                count_good[position] += 1
                            position += 1

                    print('Number of CONTAINED evaluations:', len(contained_evals))
                    for i in range(len(fth)):
                        print(f'CONTAINED Th: {fth[i]:.2f}')
                        print(f'Good CONTAINED evaluations: {count_good[i]}')
                        print(f'CONTAINED RATE: {(count_good[i]/len(contained_evals))*100 :.2f}%')

                else:
                    n_patterns = len(contained_args[0][1])
                    # in this case it displays the number of true positives and false positives, instead of counting how many evaluations are good
                    _,vres = contained_evals[0]
                    position = 0
                    btp = 0 # best tp
                    bfp = np.inf # best fp
                    bp = 0 # best position
                    for tp,tn,fp,fn in vres:
                        print(f'CONTAINED POS: {position}, TP: {tp}, FP: {fp}')
                        # if btp == nseizures:
                        #     position += 1
                        #     continue
                        if tp > btp:
                            # if there is a better tp, we allow more false positives
                            btp = tp
                            bfp = fp
                            bp = position
                        elif tp == btp:
                            # if there is a tie in tp, we choose the one with less false positives
                            if fp < bfp:
                                bfp = fp
                                bp = position
                        position += 1
                    print(f'Best CONTAINED POS: {bp}, Patterns: {n_patterns}, TP: {btp}, FP: {bfp}, Th: {fth[bp]:.2f}%')

    else:
        pool.close()
        print('DO NOT COMPUTE [ARTIFACTS]')

    print(f'\n///////// OPTIMIZATION elapsed time: {time.time()-start_time :.2f} seconds', flush=True)

    print("""\n\n
        ███████╗███╗░░██╗██████╗░  ░█████╗░███████╗  ░██████╗░█████╗░██████╗░██╗██████╗░████████╗
        ██╔════╝████╗░██║██╔══██╗  ██╔══██╗██╔════╝  ██╔════╝██╔══██╗██╔══██╗██║██╔══██╗╚══██╔══╝
        █████╗░░██╔██╗██║██║░░██║  ██║░░██║█████╗░░  ╚█████╗░██║░░╚═╝██████╔╝██║██████╔╝░░░██║░░░
        ██╔══╝░░██║╚████║██║░░██║  ██║░░██║██╔══╝░░  ░╚═══██╗██║░░██╗██╔══██╗██║██╔═══╝░░░░██║░░░
        ███████╗██║░╚███║██████╔╝  ╚█████╔╝██║░░░░░  ██████╔╝╚█████╔╝██║░░██║██║██║░░░░░░░░██║░░░
        ╚══════╝╚═╝░░╚══╝╚═════╝░  ░╚════╝░╚═╝░░░░░  ╚═════╝░░╚════╝░╚═╝░░╚═╝╚═╝╚═╝░░░░░░░░╚═╝░░░
    \n\n""")

if __name__ == '__main__':
    main()