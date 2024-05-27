import numpy as np
from dtw import dtw

from mylibrary.common.utils import intercalate_lists

def call_find_seizure_dnc(signal, template, mw, stride, operations, ww, dnc):
    # pDTW = np.zeros(stop-start)+np.nan # Initialized to NAN partial DTW
    pDTW = [np.nan]*operations
    # print(len(pDTW), f'signal: {len(signal)}, stop: {len(template)}, dnc:{len(dnc)}')
    for idx in range(operations):
        if dnc[idx]:
            continue
        # data = signal[idx*stride:idx*stride+mw]
        # print(calc_delta_power(data, 2.5, 12.))
        pDTW[idx] = dtw(signal[idx*stride:idx*stride+mw], template,
                                dist_method='euclidean',
                                window_type='sakoechiba',
                                window_args={'window_size':ww},
                                keep_internals=False, 
                                distance_only=True).normalizedDistance
        # pDTW[idx] = mts.mass2(signal[idx*stride:idx*stride+mw], template)
    return pDTW

def call_find_seizure_dnc_query(signal, query, mw, stride, qstride, xiter, ww, dnc, dncq, operations):
    # pDTW = np.zeros(stop-start)+np.nan # Initialized to NAN partial DTW
    lDTW = np.zeros((operations,xiter))+np.nan
    # print(pDTW.size, f'start: {start}, stop: {stop}')
    for idy in range(operations):
        if dncq[idy]:
            continue
        template = query[idy*qstride:idy*qstride+mw]
        for idx in range(xiter):
            if dnc[idx]:
                continue
            # data = signal[idx*stride:idx*stride+mw]
            # print(calc_delta_power(data, 2.5, 12.))
            lDTW[idy,idx] = dtw(signal[idx*stride:idx*stride+mw], template,
                                    dist_method='euclidean',
                                    window_type='sakoechiba',
                                    window_args={'window_size':ww},
                                    keep_internals=False, 
                                    distance_only=True).normalizedDistance
            # pDTW[idx] = mts.mass2(signal[idx*stride:idx*stride+mw], template)
    return lDTW


def call_dnc_aware_full(dnc, operations, d_p1, d_p2, d_p3, d_d, d_e, psd_th, p1_max, psd2_th, p2_max, psd3_th, p3_max, d_min, d_th, e_min, e_th, lb):
    """
    Description:
        Computes de epoch pruning using all the measures and the lookbackwards, taking into account the previous artifact rejection.

    Returns:
        _type_: _description_

    Notes:
        _description_
    """
    # All True by default
    ldnc = [True]*operations
    # This vector keep the track of the last lb epochs that are good. With this I can compute the lookbackwards
    vtruth = [False]*lb
    # The first lb epochs are slightly different
    for idx in range(lb):
        if dnc[idx]:
            continue

        if (d_d[idx] < d_min) | (d_th < d_d[idx]):
            continue

        if (d_e[idx] < e_min) | (e_th < d_e[idx]):
            continue

        if (d_p1[idx] < psd_th) | (p1_max < d_p1[idx]):
            continue

        if (d_p2[idx] < psd2_th) | (p2_max < d_p2[idx]):
            continue

        if (d_p3[idx] < psd3_th) | (p3_max < d_p3[idx]):
            continue

        # If I am here, the epoch is good
        vtruth[idx] = True
        # I allow the computation only if the previous ones are good
        if np.all(vtruth[:idx+1]):
            ldnc[idx] = False
    # The rest of the epochs are computed in the same way
    for idx in range(lb, operations):
        # Next iteration; Discard the first element (the oldest)
        vtruth = vtruth[1:] + [False]
        if dnc[idx]:
            continue

        if (d_d[idx] < d_min) | (d_th < d_d[idx]):
            continue

        if (d_e[idx] < e_min) | (e_th < d_e[idx]):
            continue

        if (d_p1[idx] < psd_th) | (p1_max < d_p1[idx]):
            continue

        if (d_p2[idx] < psd2_th) | (p2_max < d_p2[idx]):
            continue

        if (d_p3[idx] < psd3_th) | (p3_max < d_p3[idx]):
            continue
        # If I am here, the epoch is good
        vtruth[-1] = True
        # I allow the computation only if the previous ones are good
        if np.all(vtruth):
            ldnc[idx] = False
    # Return the local dnc
    return ldnc


def find_dtw_distances(distances, idx_seizure_list, exclusion, thp=.05):  
    """
    Description
    -----------
    This functions is used to find the relation between the minimum distance in non-seizures and the maximum distance in seizures. This is used to find the threshold to use in the DTW algorithm.
    The difference with the previous function is that here we return the threshold to use, in addition to the ratio.

    Parameters
    -----------
    distances: numpy array with the distances between the pattern and the signal
    idx_seizure_list: list of tuples with the start and end of the non-seizures/seizures
    exclusion: (int) number of samples to exclude from the beginning and end of the non-seizures regions; it have the opposite effect in the seizures regions; it is used to avoid problems with the extremes
    thp: (float) percentage of the maximum distance in seizures to use as threshold. Default: 0.05 (5%)

    Returns
    -------
    Discrimination ratio, threshold to use and the vector of min distances

    Notes
    -----

    Examples
    --------
    """
    
    # Two different lists: first for minimus in non-seizures regions
    min_no_seizure_list = []
    # second for minimus in seizures regions
    min_seizure_list = []

    for i, (a, b) in enumerate(idx_seizure_list):
        if i % 2 == 0: # List of minimus in non-seizures regions
            nextmin = np.nanmin(distances[a+exclusion:b-exclusion])
            # if there is no value, we put it to infinity so that it is not taken into account because we want the minimum in this case
            nextmin = float('inf') if np.isnan(nextmin) else nextmin
            min_no_seizure_list.append(nextmin)
        else: # List of minimus in seizures regions
            # As we checked before, with check_dnc_query, that at least one of the values is not nan, we can use it without problems
            nextmin = np.nanmin(distances[a-exclusion:b+exclusion])
            min_seizure_list.append(nextmin)    
  
    # Instead of return the ratio with respect to each of the non-seizures, we'll return the ratio with respect to the seizures in the worst-case scenario, which is when we have the smallest distance from the non-seizures
    no_seizure_min = np.nanmin(min_no_seizure_list) # best non-seizure
    seizure_max = np.max(min_seizure_list) # worst seizure
    # We'll use a threshold greater than the maximum distance in seizures
    # th = seizure_max * (1.0 + thp)
    # Just for comparison, we'll use a threshold equal to the maximum distance in seizures. This way we can choose different thresholds in the evaluation step and see how it affects the results
    th = seizure_max
    # saturate the DR to 100 if the maximum distance in seizures is 0 or no_seizure_min is infinite
    dr = min(no_seizure_min/seizure_max if 0.0 < seizure_max else float('inf'), 100.)

    return dr, th, intercalate_lists(min_no_seizure_list, min_seizure_list)


def find_validation_params(distances, xiter, th, idx_seizure_list, exclusion):  
    """
    Description
    -----------
    This functions is used to find the validation parameters for the DTW algorithm.

    Parameters
    -----------
    distances: numpy array with the distances between the pattern and the signal
    xiter: number of epochs to use in the validation
    th: threshold to use in the DTW algorithm to decide if the epoch is a seizure or not
    idx_seizure_list: list of tuples with the start and end of the non-seizures/seizures
    exclusion: number of samples to exclude from the beginning and end of the non-seizures regions; it have the opposite effect in the seizures regions; it is used to avoid problems with the extremes

    Returns
    -------
    True positive, false positive, true negative and false negative

    Notes
    -----

    Examples
    --------
    """
    # this mask identifies the epochs that meet the conditions to be considered as a seizure
    mask = distances <= th

    tp = 0 # true positive
    tn = 0 # true negative
    fp = 0 # false positive
    fn = 0 # false negative

    # each chunk_size we'll check if a false positive has been detected
    chunk_size = 100

    for i, (a, b) in enumerate(idx_seizure_list):
        if i % 2 == 0: # non-seizures regions
            # as this is a non-seizure region, we need to check chunks of the signal to see if there is any seizure
            start = a+exclusion
            # b could be < 0, indicating the end of the vector, so we need to check it
            end = b-exclusion if b > 0 else xiter
            # this list will contain the number of false positives
            remainder = (end - start) % chunk_size
            cropped_mask = mask[start : end - remainder]
            fp_list = np.any(cropped_mask.reshape(-1, chunk_size), axis=1)
            # if there is any detection, we'll count it as a false positive
            aux = np.count_nonzero(fp_list)
            fp += aux
            # the rest of chunks will be true negatives
            tn += len(fp_list) - aux
        else: # seizures regions
            # as this is a seizure, we need to check if the mask is true in this region for any of the values
            if np.any(mask[a-exclusion:b+exclusion]):
                tp += 1
            else:
                fn += 1
    
    return tp, tn, fp, fn


def check_dnc(dnc, idx_seizure_list, exclusion):
    """
    Description:
        Check if the dnc vector of the signal allows at least one epoch of each seizure to be computed

    Args:
        dnc (List[booleans]): Do Not Compute vector
        idx_seizure_list (_type_): List contains the start and end of each region non-seizure/seizure
        exclusion (_type_): Number of epochs to exclude in order to avoid problems with the limits of the seizures

    Returns:
        int: _description_

    Notes:
        _description_
    """
    # The first region is always a non-seizure
    no_seizure_list = []
    for i, (a, b) in enumerate(idx_seizure_list):
        if i%2 == 0:
            # The final end never overflows because I terminate it with negative index which in Python is relative to the end of the vector.
            no_seizure_list.append(np.all(dnc[a+exclusion:b-exclusion]))
        else:
            # Check if any has survived
            if np.all(dnc[a-exclusion:b+exclusion]):
                # If all True, then there is no epoch that can be computed. Error!
                return 0
    # If there are no survivors in no-seizures, the Discrimination Ratio will always be INF
    if np.all(no_seizure_list):
        # print('\tBINGO! REL == INF')
        return -1
    # If everything is ok, return 1
    return 1    

def query_mask(duration_query_list, mw, qstride, qfactor, yiter):
    """
    Description:
        This function avoid to compute epochs that overlaps samples of different seizures, i.e. if the seizures are concatenated in the query, we don't want to compute the epochs that overlaps different seizures.
    Args:
        duration_query_list (List[int]): List of seizures limits.
        mw (int): Length of the epochs.
        qstride (int): Stride of the query.
        qfactor (int): Factor that relates indices with seconds.
        yiter (int): Number of epochs in the query.
    Returns:
        mask (List[boolean]): epochs to compute.
    Notes:
        _description_
    """
    aux_list = [x*qfactor for x in duration_query_list]
    idx_seizure = list(zip(aux_list, aux_list[1:]))
    # Initialize the mask
    mask = [False]*yiter
    # Obtain the limits of the seizures
    for a,b in idx_seizure:
        assert(b <= yiter)
        # This is the limit in samples. From this sample, the epochs should be not computed because they overlap with the next seizure
        limit = b*qstride
        # For each epoch, check if it overlaps with the next seizure
        for ii in range(a, b):
            if ii*qstride+mw <= limit:
                continue
            # If it overlaps, mark it as True (i.e. do not compute)
            mask[ii] = True
    # print(yiter, np.sum(mask))
    return mask

def check_dnc_query(dnc, idx_query):  
    """
    Description:
        Check if the dnc vector of the query allows at least one epoch of each seizure to be computed.

    Args:
        dnc (List[booleans]): Do Not Compute vector
        idx_query (List[tuple(int,int)]): List contains the start and end of each seizure

    Returns:
        _type_: _description_

    Notes:
        _description_
    """
    # Get start and end of each seizure
    for x,y in idx_query:
        # Check if any has survived
        if np.all(dnc[x:y]):
            # If all True, then there is no epoch that can be computed.
            return 0
    return 1

