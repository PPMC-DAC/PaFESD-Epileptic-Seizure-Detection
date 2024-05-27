import os
import time

from scipy.integrate import simps
from scipy.signal import welch

from pyriemann.estimation import Covariances

from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances_chunked

from scipy.spatial.distance import euclidean

from itertools import zip_longest

import numpy as np
import pandas as pd
from pandas import HDFStore
import pickle

from itertools import combinations

from datetime import timedelta

# This functions are used to sort the list of tuples; the difference between them is the index of the tuple that is used to sort the list
def keyfunc(e):
    return e[1]
def keyfunc2(e):
    return e[2]
def keyfunc3(e):
    return e[3]

def calc_delta_power(x, low, high):    
    # nperseg=512 because I want to choose 2.5Hz
    freqs, Pxx = welch(x, 256., nperseg=512)
    idx_delta = np.logical_and(freqs >= low, freqs < high)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    # Compute the absolute power by approximating the area under the curve
    return simps(Pxx[idx_delta], dx=freq_res)

def calc_delta_tri(x):    
    # nperseg=512 because I want to choose 2.5Hz
    freqs, Pxx = welch(x, 256., nperseg=512)
    idx_delta1 = np.logical_and(freqs >= 2.5, freqs < 12.)
    idx_delta2 = np.logical_and(freqs >= 12., freqs < 18.)
    idx_delta3 = np.logical_and(freqs >= 18., freqs < 35.)
    # Frequency resolution
    freq_res = freqs[1] - freqs[0]  # = 1 / 4 = 0.25
    # Compute the absolute power by approximating the area under the curve
    return [simps(Pxx[idx_delta1], dx=freq_res),
            simps(Pxx[idx_delta2], dx=freq_res),
            simps(Pxx[idx_delta3], dx=freq_res)]

def call_psd(signal, mw, stride, operations, low, high):
    pP = [np.nan]*operations

    for idx in range(operations):
        pP[idx] = calc_delta_power(signal[idx*stride:idx*stride+mw], low, high)
    return pP

def call_psd_tri(signal, mw, stride, operations):
    pP = [[np.nan]*3]*operations

    for idx in range(operations):
        pP[idx] = calc_delta_tri(signal[idx*stride:idx*stride+mw])
    return pP

def filter_bandpass(signal, low_freq, high_freq, channels=None, method="iir"):
    """Filter signal on specific channels and in a specific frequency band"""
    sig = signal.copy()
    if channels is not None:
        sig.pick_channels(channels)
    sig.filter(l_freq=low_freq, h_freq=high_freq, method=method, verbose=False)
    return sig

def call_cov(signal, mw, stride, operations):
    nchannels = len(signal)
    vcov = np.zeros((operations,nchannels,nchannels))

    for idx in range(operations):
        vcov[idx] = Covariances(estimator='scm').transform(signal[np.newaxis,:,idx*stride:idx*stride+mw])
    return vcov

def call_cov_est(signal, mw, stride, operations, estimator='scm'):
    nchannels = len(signal)
    vcov = np.zeros((operations,nchannels,nchannels))

    for idx in range(operations):
        vcov[idx] = Covariances(estimator=estimator).transform(signal[np.newaxis,:,idx*stride:idx*stride+mw])
    return vcov

def call_dnc_artifact(rpf_covs, mw, stride, operations, rpf):
    """
    Description:
        Applies the Riemannian Potato Field (RPF) to reject epochs with artifacts.

    Args:
        rpf_covs (_type_): Covariance matrices for each potato.
        mw (_type_): Epoch length in samples.
        stride (_type_): Stride length in samples.
        operations (_type_): Number of epochs to process.
        rpf (_type_): The RPF object.

    Returns:
        List[booleans]: List of booleans indicating whether the epoch should be computed or not.

    Notes:
        _description_
    """
    # Initialize the dnc to True because the "predict" function returns True if the epoch is not an artifact
    ldnc = [True]*operations
    for idx in range(operations):
        # Predict if the epoch is an artifact
        if (rpf.predict([c[np.newaxis, idx] for c in rpf_covs])[0]):
            # If predict returns True, then the epoch is not an artifact
            ldnc[idx] = False
            # This additional step updates the RPF with the new information, as we have a new epoch that is not an artifact. TODO: I'm not sure if this is necessary
            # rpf.partial_fit([c[np.newaxis, idx] for c in rpf_covs], alpha=1 / (train_covs + idx))
    return ldnc

def call_max_dist(signal, mw, stride, operations):
    vD = [np.nan]*operations

    for idx in range(operations):
        window = signal[idx*stride:idx*stride+mw]
        mu = window - np.mean(window)
        vD[idx] = max(mu) - min(mu)
    return vD

def call_energy(signal, mw, stride, operations):
    vE = [np.nan]*operations

    for idx in range(operations):
        window = signal[idx*stride:idx*stride+mw]
        mu = window - np.mean(window)
        vE[idx] = np.sum(mu**2)/mw
    return vE

def get_cdf_categories(data, bins=200):
    """
    Description:
        This function compute the distribution function of the data

    Args:
        data (List[float]): List with the data
        bins (int, optional): Number of bins. Defaults to 200.

    Returns:
        cdf: Distribution function
        categories: List with the categories

    Notes:
        _description_
    """
    H,X1 = np.histogram( data, bins = bins, density = True )
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H)*dx
    return F1,X1.tolist()

def find_rand_epochs(dnc, xiter, idx_list, factor=1):

    """
    Description
    -----------
    This functions is used when we successfully prune all the non-seizure epochs
    in that case, we need to randomly pick 4 epochs from the non-seizure data
    because we need to find a possible pattern. Anyways, we continue returning 
    a DR equal to infinity in the search function.

    Parameters
    -----------
    dnc : array of DoNotCompute epochs
    xiter : number of epochs
    idx_list : list of tuples with the limits of the non-seizure/seizure epochs
    factor : how many epochs to pick, related to the number of survived epochs. E.g. if we have 4 epochs, we pick 4*factor epochs

    Returns
    -------

    Notes
    -----

    Examples
    --------
    """
    # odd indices are the limits of the non-seizure epochs
    compute = True
    # how many epochs survived
    survivors = xiter - np.sum(dnc)
    # in total, how many epochs to pick
    nsamples = survivors*factor
    # number of seizures
    nseizures = (len(idx_list)-1)//2
    # nseizures+1 is the number of non-seizures. 
    # k is the number of epochs to pick from each non-seizure
    # if the number of epochs is not divisible by the number of non-seizures, it is not a problem
    k = nsamples//(nseizures+1)
    for i, (a, b) in enumerate(idx_list):
        if i%2 == 0: # non-seizure
            # if the greater limit is negative, it means that is the limit of the last non-seizure epoch, the limit of dnc
            if b < 0:
                b = xiter
            # pick k epochs from the non-seizure randomly
            llist = np.random.randint(a,b,k)
            for ii in llist:
                dnc[ii] = False
            # next iteration is even
        else: # seizure
            continue


def func_min_dr(*patterns):
    """
    Description:
        Obtain the Discrimination Ratio (DR) of a set of patterns

    Returns:
        float: The Discrimination Ratio
        float: The threshold

    Notes:
        _description_
    """
    # the list of distances is the last element of each pattern
    patterns = [np.array(pattern[-1]) for pattern in patterns]
    # obtain the list of minimum distances of non-seizure epochs
    vec_min = np.minimum.reduce(patterns)
    # obtain the best distance of non-seizure epochs
    min_non_seizure = vec_min[::2].min()
    # obtain the worst distance of seizure epochs
    max_seizure = vec_min[1::2].max()
    # compute the DR
    DR = min_non_seizure / max_seizure
    return DR, max_seizure


def func_min_dr2(patterns, compute_patterns):
    # the list of distances is the last element of each pattern
    ldist = [np.array(pattern[-1]) for pattern in patterns]
    # obtain the list of minimum distances of non-seizure epochs
    vec_min = np.minimum.reduce(ldist)
    # obtain the best distance of non-seizure epochs
    min_non_seizure = vec_min[::2].min()
    assert not np.isnan(min_non_seizure), 'min_non_seizure is nan'
    # obtain the worst distance of seizure epochs
    dmax = vec_min[1::2].max()
    # compute DR
    DR = min_non_seizure/dmax
    # let's check if DR is greater than individual DR
    if DR > max([pattern[1] for pattern in patterns]):
        # if it is, this patterns pass the test
        compute_patterns[[pattern[0] for pattern in patterns]] = True
    # return DR and dmax
    return DR, dmax


def check_bound_values(lb, ub, categories, min_range=1):
    for key in lb:
        if (ub[key] - lb[key]) < min_range:
            #TODO a problem could be lb = vcat_limit[key][1] in _min categories
            # but it's a extremely rare possibility; same way for _max
            if '_min' in key:
                print(f'Modify {key} UB')
                ub[key] = min(ub[key] + min_range, len(categories[key])-1)
            elif '_max' in key:
                print(f'Modify {key} LB')
                lb[key] = max(lb[key] - min_range, 0)
            else:
                print(f'Modify {key}')
                ub[key] = min(ub[key] + min_range, len(categories[key])-1)
                lb[key] = max(lb[key] - min_range, 0)
    return

def search_bound_values(pos_eval_x, categories):
    lb = pos_eval_x[0].copy()
    ub = pos_eval_x[0].copy()

    # search bound values
    for dicc in pos_eval_x:
        for key in dicc:
            if dicc[key] < lb[key]:
                lb[key] = dicc[key]
            if ub[key] < dicc[key]:
                ub[key] = dicc[key]

    int_lb = dict()
    int_ub = dict()

    # translate the bound values to indices
    for key in lb:
        int_lb[key] = categories[key].index(lb[key])
        int_ub[key] = categories[key].index(ub[key])

    return int_lb,int_ub

def find_train_limits(seizure_list, until_seizure, max_length, exclusion=10, min_id=None, max_id=None, halo=None):
    """
    Description:
        Function to find the limits of the training set

    Args:
        seizure_list (_type_): List of the seizure times in seconds
        until_seizure (_type_): Number of seizures to use
        max_length (_type_): Number of seconds of the entire time-series
        exclusion (int): Number of seconds to exclude before and after a seizure

    Returns:
        min_id: first index of the training set
        max_id: last index of the training set
        halo: minimum separation between the limits of the training set and the limits of the seizures

    Notes:
        _description_
    """
    # list of the seizures (start, end) tuples
    mylist = list(zip(seizure_list,seizure_list[1:]))[::2]

    if halo is None:
        # default halo; We use this to avoid that the training set ends or start in the limits of a seizure
        halo = 300 # seconds; 5 minutes
        # search the min halo
        for ii in range(len(mylist)-1):
            # I only take into account the exclusion time of the next seizure instead of 2*exclusion
            halo = min(halo, mylist[ii+1][0] - mylist[ii][1] - exclusion)
        assert halo > 60, 'The separation between seizures must be > 60s'

    displacement = 600 # seconds; 10 minutes

    length_limit = int(max_length*0.75)
    # number of samples to increase the window size in each iteration: 5% of the total length
    factor = max_length//20
    # at least 25% of the ts to train, because this length will be initially the window center
    length = max_length//8 if min_id is None else (max_id - min_id)//2
    # obtain the center
    center = length if min_id is None else (min_id + max_id)//2 + displacement
    # initialize the limits
    min_id,max_id = (0,0) if min_id is None else (min_id,max_id)

    sfound = False

    # if we find a proper pair of limits or we reach the maximum length, we stop
    while (not sfound) & (length < length_limit):
        # move the window until we find a pair of limits
        while (not sfound) & (max_id < max_length):
            min_id = max(0, center - length - halo)
            max_id = min(max_length, center + length + halo)
            if np.count_nonzero((seizure_list >= min_id) & (seizure_list <= max_id)) == (until_seizure*2):
                sfound = True
            # Displace the center and try again
            center += displacement
        # if we don't find a pair of limits, we enlarge the window
        if not sfound:
            length += factor
            center = length 
            min_id,max_id = (0,0)
    
    if sfound:
        # if we find a pair of limits, we return them and the halo
        return min_id,max_id,halo
    # if we don't find a pair of limits, we return -1 as an error
    return -1,-1,halo

def index_representative_points(km, X):
    ret = []
    for k in range(km.n_clusters):
        mask = (km.labels_ == k).nonzero()[0]
        s = []
        for _ in pairwise_distances_chunked(X=X[mask]):
            s.append(np.square(_).sum(axis=1))
        ret.append(mask[np.argmin(np.concatenate(s))])
    return np.array(ret)

def index_representative_points2(km, X):
    ret = []
    for k in range(km.n_clusters):
        mask = (km.labels_ == k).nonzero()[0]
        centroid = np.mean(X[mask], axis=0)
        i0 = mask[pairwise_distances_argmin(centroid[None, :], X[mask])[0]]
        ret.append(i0)
    return np.array(ret)

def closest_point_to_cluster_center(kmeans, X):
    # Loop over all clusters and find index of closest point to the cluster center and append to closest_pt_idx list.
    closest_pt_idx = []
    for k in range(kmeans.n_clusters):
        # get all points assigned to each cluster:
        # cluster_pts = X[kmeans.labels_ == k]
        # get all indices of points assigned to this cluster:
        cluster_pts_indices = np.where(kmeans.labels_ == k)[0]

        cluster_cen = kmeans.cluster_centers_[k]
        min_idx = np.argmin([euclidean(X[idx], cluster_cen) for idx in cluster_pts_indices])
        
        # # Testing:    
        # print('closest point to cluster center: ', cluster_pts[min_idx])
        # print('closest index of point to cluster center: ', cluster_pts_indices[min_idx])
        # print('  ', X[cluster_pts_indices[min_idx]])
        closest_pt_idx.append(cluster_pts_indices[min_idx])

    return closest_pt_idx

def get_contained_evaluations(good_evals, n=3):
    good_evals_x = [x for x,_,_,_ in good_evals]
    weights = np.zeros(len(good_evals_x), dtype=int)
    for i in range(len(good_evals_x)):
        x = good_evals_x[i]
        for j in range(len(good_evals_x)):
            if i == j:
                # trivial case
                continue
            y = good_evals_x[j]
            if x['rp1_min'] <= y['rp1_min'] and x['rp1_max'] >= y['rp1_max'] and \
                x['rp2_min'] <= y['rp2_min'] and x['rp2_max'] >= y['rp2_max'] and \
                x['rp3_max'] >= y['rp3_max'] and \
                x['rd_min'] <= y['rd_min'] and x['rd_max'] >= y['rd_max'] and \
                x['re_min'] <= y['re_min'] and x['re_max'] >= y['re_max']:
                # +1 if y is contained in x
                weights[i] += 1

    print(f'weights: {weights}')

    sorted_ids = np.argsort(weights)
    print(f'sorted_ids: {sorted_ids[-n:]}')
    print(f'evals: {weights[sorted_ids[-n:]]}')
    return good_evals[sorted_ids[-n:]]

def intercalate_lists(list1, list2):
    result = []
    for a, b in zip_longest(list1, list2):
        if a is not None:
            result.append(a)
        if b is not None:
            result.append(b)
    return result

def split_list(lst):
    list1 = lst[::2]
    list2 = lst[1::2]
    return list1, list2

def load_matrix_classes(matrix_name, tDTW, mDTW):

    if mDTW.is_empty():
        # Read the main matrix if it exists
        if os.path.exists(matrix_name):
            dfm = pd.read_hdf(matrix_name, key='df')
            mDTW.data = dfm.values
            print(f'Loaded mDTW: {np.count_nonzero(~np.isnan(mDTW.data))} elements; Shape: {mDTW.data.shape} ')
            # Check if mDTW has enough elements
            assert mDTW.data.shape[0] == mDTW.qend, f"Error: mDTW has not enough rows: {mDTW.data.shape[0]} < {mDTW.qend}"
            assert mDTW.data.shape[1] == mDTW.dend, f"Error: mDTW has not enough columns: {mDTW.data.shape[1]} < {mDTW.dend}"
            assert mDTW.data.shape[0] >= tDTW.qend, f"Error: mDTW has not enough rows: {mDTW.data.shape[0]} < {tDTW.qend}"
            assert mDTW.data.shape[1] >= tDTW.dend, f"Error: mDTW has not enough columns: {mDTW.data.shape[1]} < {tDTW.dend}"
            # Load tDTW matrix
            tDTW.data = mDTW.data[tDTW.qini:tDTW.qend, tDTW.dini:tDTW.dend]
            print(f'Loaded tDTW: {np.count_nonzero(~np.isnan(tDTW.data))} elements; Shape: {tDTW.data.shape}')
        else:
            print(f"File {matrix_name} does not exist. Creating matrix for training...")
            # Reserve the memory for the main matrix. This is only the main matrix for training.
            tDTW.create_empty_matrix()
            print(f'tDTW shape: {tDTW.data.shape}')
    else:
        # Select the tDTW matrix for this training
        tDTW.data = mDTW.data[tDTW.qini:tDTW.qend, tDTW.dini:tDTW.dend]
        print(f'Loaded tDTW: {np.count_nonzero(~np.isnan(tDTW.data))} elements; Shape: {tDTW.data.shape}')

    return

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

def save_matrix(matrix_name, mDTW):
    # create a dataframe from the matrix
    df = pd.DataFrame(mDTW)
    # it is extremely rare that a process saves the matrix meanwhile we are in the middle of the evaluation, but it can happen
    if os.path.exists(matrix_name):
        # If the file exists, we need to load it and compare what matrices have more values different than None
        df_old = pd.read_hdf(matrix_name, 'df')
        # Combine the matrices
        df = pd.DataFrame(np.where(~np.isnan(df), df, df_old))
        # Delete the old matrix
        del(df_old)

    # Save the matrix safely; it is extremely rare that two processes try to save the matrix at the same time, but it can happen
    write_hdf(matrix_name, 'df', df, 'zlib')

    return

def check_evaluations(newlogs, evals, ex_time):
    # newlogs = logs_file + '_gold_eval.pkl'
    if os.path.exists(newlogs) and (os.path.getsize(newlogs) > 0):
        with open(newlogs, 'rb') as f:
            print('\tLoading logs')
            gold_eval = pickle.load(f)
            gold_time = pickle.load(f)

        gold_eval = sorted(gold_eval, key=lambda x: x[2], reverse=True)
        evals = sorted(evals, key=lambda x: x[2], reverse=True)

        for i in range(len(gold_eval)):
            # Compare dictionary values
            assert gold_eval[i][0] == evals[i][0], f'Error: gold_eval[{i}][0] and evals[{i}][0] are not equal'
            # Compare pattern indices
            assert np.all(gold_eval[i][1] == evals[i][1]), f'Error: gold_eval[{i}][1] and evals[{i}][1] are not equal'
            # Compare DR values
            assert np.allclose(gold_eval[i][2], evals[i][2]), f'Error: gold_eval[{i}][2] and evals[{i}][2] are not equal'
            # Compare dmax values
            assert np.allclose(gold_eval[i][3], evals[i][3]), f'Error: gold_eval[{i}][3] and evals[{i}][3] are not equal'

        print('\tPASS: All the evaluations are equal!')

        if ex_time:
            # Compare the time
            print(f'\tNew is {np.sum(gold_time)/np.sum(ex_time) :.2f} times faster than the old one')
            # Compare execution by execution
            counter = 0
            for i in range(len(gold_time)):
                if gold_time[i] > ex_time[i]:
                    counter += 1
            print(f'\tNew is faster in {counter}/{len(gold_time)} executions; {counter/len(gold_time)*100 :.2f}%\n')


    else:
        print('\tSaving evaluation logs\n\n')
        # save evals in a pickle file as a gold standard
        evals = sorted(evals, key=lambda x: x[2], reverse=True)
        with open(newlogs, 'wb') as f:
            pickle.dump(evals, f)
            pickle.dump(ex_time, f)


def check_validations(newlogs, vals):

    if os.path.exists(newlogs) and (os.path.getsize(newlogs) > 0):
        with open(newlogs, 'rb') as f:
            print('Loading logs')
            gold_val = pickle.load(f)

        for i in range(len(gold_val)):
            # Compare dictionary values
            assert gold_val[i][0] == vals[i][0], f'Error: gold_val[{i}][0] and vals[{i}][0] are not equal'
            # Compare the validation values tp, fp, tn, fn
            assert gold_val[i][1] == vals[i][1], f'Error: gold_val[{i}][1] and vals[{i}][1] are not equal'

        print('\tPASS:All the validations are equal!\n\n')

    else:
        print('\tSaving validation logs\n\n')
        # save vals in a pickle file as a gold standard
        with open(newlogs, 'wb') as f:
            pickle.dump(vals, f)


def search_colors_best(relation, compute_relation, batch_size, min_value_DR, colors, bsf):
    # Find best combination of patterns using func_min
    best_patterns = None
    max_dr = bsf
    max_th = float('inf')
    
    # use as many patterns as seizures-1 in the training
    for n_patterns in range(2, batch_size+1):
        # search in relation how many patterns has a best_DR_case > max_dr
        for idy, dr, ith, vd in relation[compute_relation]:
            # check if the best_DR_case > min_value_DR
            if np.nanmin(vd[::2])/np.min(vd[1::2]) < max_dr:
                compute_relation[idy] = False
        # if the desired DR was found or there are no enough patterns to find it, return
        if (max_dr > min_value_DR) or (compute_relation.sum() <= n_patterns) or len(set(colors[compute_relation])) < n_patterns:
            return max_dr, max_th, best_patterns
        # track the patterns that pass the test
        crc = np.zeros(len(compute_relation), dtype=bool)
        # find the best combination of patterns; instead of combine the relation, combine the indices
        for idxs in combinations(np.where(compute_relation)[0], n_patterns):
            patterns = [relation[idx] for idx in idxs]
            # check that all patterns are from different colors
            if len(set(colors[[pattern[0] for pattern in patterns]])) != n_patterns:
                continue
            # maximize this function
            # dr,ith = func_min_dr(patterns)
            dr,ith = func_min_dr2(patterns, crc)
            if dr > max_dr:
                max_dr = dr
                max_th = ith
                best_patterns = patterns
        # keep only good for next iteration
        compute_relation[~crc] = False

    return max_dr, max_th, best_patterns


def dtime(fseconds):
    """
    Description:
        Return a timedelta object from a float number of seconds.

    Args:
        fseconds (float): # seconds

    """
    return timedelta(seconds=int(fseconds))


def def_bounds_categories(cdfs, lb_limit=0.5, ub_limit=0.5):
    init_lb = {}
    init_ub = {}

    init_lb = {
        'rp1_min':0,
        'rp1_max':next(x for x, val in enumerate(cdfs['rp1_max']) if val > ub_limit),
        'rp2_min':0,
        'rp2_max':next(x for x, val in enumerate(cdfs['rp2_max']) if val > ub_limit),
        'rp3_min':0,
        'rp3_max':next(x for x, val in enumerate(cdfs['rp3_max']) if val > ub_limit),
        'rd_min':0,
        'rd_max':next(x for x, val in enumerate(cdfs['rd_max']) if val > ub_limit),
        're_min':0,
        're_max':next(x for x, val in enumerate(cdfs['re_max']) if val > ub_limit)
    }

    init_ub = {
        'rp1_min':next(x for x, val in enumerate(cdfs['rp1_min']) if val > lb_limit),
        'rp1_max':len(cdfs['rp1_max'])-1,
        'rp2_min':next(x for x, val in enumerate(cdfs['rp2_min']) if val > lb_limit),
        'rp2_max':len(cdfs['rp2_max'])-1,
        'rp3_min':next(x for x, val in enumerate(cdfs['rp2_min']) if val > lb_limit),
        'rp3_max':len(cdfs['rp3_max'])-1,
        'rd_min':next(x for x, val in enumerate(cdfs['rd_min']) if val > lb_limit),
        'rd_max':len(cdfs['rd_max'])-1,
        're_min':next(x for x, val in enumerate(cdfs['re_min']) if val > lb_limit),
        're_max':len(cdfs['re_max'])-1
    }

    return init_lb, init_ub


