import numpy as np
# import time
import gc
from itertools import combinations

from mylibrary.common.utils import find_rand_epochs
from mylibrary.common.utils import func_min_dr
from mylibrary.common.utils import search_colors_best

from mylibrary.dtw_functions.find_seizure import call_dnc_aware_full
from mylibrary.dtw_functions.find_seizure import find_dtw_distances
from mylibrary.dtw_functions.find_seizure import find_validation_params
from mylibrary.dtw_functions.find_seizure import check_dnc
from mylibrary.dtw_functions.find_seizure import check_dnc_query

import mylibrary.common.config as config

# All the arguments are keyword arguments and required
def f_exploration(*, rp1_min, rp1_max, rp2_min, rp2_max, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards, exclusion, xiter, yiter, d_p1, d_p2, d_p3, d_d, d_e, q_p1, q_p2, q_p3, q_d, q_e, qmask, idx_query, idx_seizure_list, DoNotCompute, ex_dnc_time):

    # start = time.time()

    # Compute the DNC for the query. Be careful!!!! if we put a "liter" different from "yiter": the dimensions would not match those of qmask
    DoNotComputeQ = (q_p1 < rp1_min) | (rp1_max < q_p1) | (q_p2 < rp2_min) | (rp2_max < q_p2) | (rp3_max < q_p3) | (q_d < rd_min) | (rd_max < q_d) | (q_e < re_min) | (re_max < q_e) | qmask
    # Check if at least one pattern survives in each seizure
    compute_DTW = check_dnc_query(DoNotComputeQ, idx_query)
    # A value of DR=0 means that this configuration is not valid
    dr = 0.0

    if compute_DTW:
        # Compute the DNC in parallel for the signal
        DoNotCompute = np.asarray(call_dnc_aware_full( DoNotCompute, xiter, d_p1, d_p2, d_p3, d_d, d_e, rp1_min, rp1_max, rp2_min, rp2_max, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards))

        # I need to check everything again because when applying lookbackwards, it may stop computing seizure zones because of the lookbackwards parameter. If we do not use lookbackwards, this is not necessary because the seizures regions of the DNC were already checked with check_dnc_query above.
        compute_DTW = check_dnc(DoNotCompute, idx_seizure_list, exclusion)

        if compute_DTW == 0:
            # print('DO NOT COMPUTE [DATA]')
            dr = 0.0

        elif compute_DTW > 0:
            # print('PASS', end='\n\n\n')
            # Mark it as good; rel > 1.0; TODO: we can use a different value to guide the optimization process. Could this value cause problems with the optimizer? This is a interesting parameter, because now that we use the pruning rate, we have a method to compare good normal configurations (i.e. compute_DTW > 0 and a high pruning rate) with bad "infinite" configurations (i.e. compute_DTW < 0 and a low pruning rate).
            dr = 1.5
            # Instead of returning this fixed value, let's ponderate it with the pruning rate
            dr *= np.count_nonzero(DoNotCompute)/xiter

        else:
            # print('INFINITE', end='\n\n\n')
            # At this point, the only possibility is that rel = Inf. Instead of returning Inf, we return a DR value that is double that of a normal good evaluation.
            dr = 2.0
            # Instead of returning this fixed value, let's ponderate it with the pruning rate
            dr *= np.count_nonzero(DoNotCompute)/xiter
        
    # else:
    #     print('DO NOT COMPUTE [QUERY]')

    # ex_dnc_time.append(time.time() - start)

    # return idx + Discrimination Ratio + Threshold; we need to return these values because later we will use the function that computes the DTWs, and the optimizer only accepts one return value, in this case a tuple
    return [-1],dr,0.0


# All the arguments are keyword arguments and required
def f_evaluation(*, rp1_min, rp1_max, rp2_min, rp2_max, rp3_min, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards, exclusion, xiter, yiter, d_p1, d_p2, d_p3, d_d, d_e, q_p1, q_p2, q_p3, q_d, q_e, qmask, idx_query, idx_seizure_list, DoNotCompute, from_idq, batch_size=3, colors):

    #  Collect all de memory from last iteration
    gc.collect()

    tDTW = config.tDTW.data

    bsf = 0.0
    th = float('inf')
    # we need to use a list for ids because we allow to select more than one pattern
    bsf_id = [-1]

    # TODO: this is another parameter to be optimized; the value of bsf that we want to reach
    # TODO: in the case of a patient with a lot of seizures, we may want to use a greater value than 1.0 to avoid problems with future seizures. 
    # stop when we find a pattern with DR > 1.1
    min_value_DR = 1.2

    # start = time.time()

    # TODO: Be careful!!!! if we put a "liter" different from "yiter": the dimensions would not match those of qmask
    DoNotComputeQ = (q_p1 < rp1_min) | (rp1_max < q_p1) | (q_p2 < rp2_min) | (rp2_max < q_p2) | (q_p3 < rp3_min) | (rp3_max < q_p3) | (q_d < rd_min) | (rd_max < q_d) | (q_e < re_min) | (re_max < q_e) | qmask

    # assert check_dnc_query(DoNotComputeQ, idx_query) != 0, "If we are evalauting this configuration, the compute_DTW should be different from 0"
    # this is a possibility since we can use a saved configuration that does not have the same categories as the current one
    if not check_dnc_query(DoNotComputeQ, idx_query):
        bsf = 0.0        
        return bsf_id,bsf,th

    DoNotCompute = np.asarray(call_dnc_aware_full( DoNotCompute, xiter, d_p1, d_p2, d_p3, d_d, d_e, rp1_min, rp1_max, rp2_min, rp2_max, rp3_min, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards))

    # I need to check everything again because when applying lookbackwards, it may stop computing seizure zones; it might be a good idea to apply it to call_dnc_query as well
    compute_DTW = check_dnc(DoNotCompute, idx_seizure_list, exclusion)

    # this is a possibility since we can use a saved configuration that does not have the same categories as the current one
    if compute_DTW == 0:
        # print('DO NOT COMPUTE [DATA]')
        bsf = 0.0        
        return bsf_id,bsf,th
            
    if compute_DTW < 0:

        # At this point the only possibility is DR = Inf. In that case we need to find at least one pattern
        find_rand_epochs(DoNotCompute, xiter, idx_seizure_list, factor=1)
    
    lDTW = np.full((yiter, xiter), np.nan)
    # Create a boolean mask for non-NaN values in tDTW; True -> No computation needed
    mask = np.logical_or.outer(DoNotComputeQ, DoNotCompute)
    # Read the values from tDTW
    lDTW[~mask] = tDTW[~mask]

    relation = np.full(yiter, None, dtype=object)


    # now we need to find the best pattern
    for idy in np.where(~DoNotComputeQ)[0]:

        # find the best DR
        dr,ith,vd = find_dtw_distances(lDTW[idy], idx_seizure_list, exclusion)
        # save idx, DR, th, and DR vector
        # relation.append([idy, dr, ith, vd])
        relation[idy] = [idy, dr, ith, vd]
        # is it the best so far?
        if bsf < dr:
            bsf = dr
            th = ith
            bsf_id[0] = idy
            # print('  New:', relation[-1])
        
    # at this point the index of the best match is bsf_id and it is relative to the training query
    # we need to find the index of the best match in the original query
    bsf_id[0] += from_idq

    if bsf < min_value_DR:
        
        # when we use function1, idx_query is the list of indexes of all the seizures used in the training
        # batch_size = min(len(idx_query)-1, batch_size)
        assert batch_size < len(idx_query), 'Batch size is too large'

        # Find best combination of patterns using func_min        
        max_dr, max_th, best_patterns = search_colors_best(relation, ~DoNotComputeQ, batch_size, min_value_DR, colors, bsf)

        # if we found a combination of patterns with DR > bsf
        if max_dr > bsf:
            # set the best so far to the best combination of patterns
            bsf = max_dr
            th = max_th
            # get the selected patterns ids
            sids = [pattern[0] for pattern in best_patterns]
            # be sure to add the offset to the ids; these are the returned ids
            bsf_id = [x+from_idq for x in sids]
    
    if compute_DTW < 0:
        # We no longer return infinity. This way we guide the optimization process better. The "score", which in our case is the DR, worths more because the configuration already discarded all the non-seizure epochs. This is the reason why we return bsf*2.0 instead of bsf.
        bsf *= 2.0

    return bsf_id,bsf,th

# this function is used when we have all the distances computed in tDTW
# All the arguments are keyword arguments and required
def f_optimization(*, rp1_min, rp1_max, rp2_min, rp2_max, rp3_min, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards, exclusion, xiter, yiter, d_p1, d_p2, d_p3, d_d, d_e, q_p1, q_p2, q_p3, q_d, q_e, qmask, idx_query, idx_seizure_list, DoNotCompute, from_idq, batch_size=3, colors):

    #  Collect all de memory from last iteration
    gc.collect()

    tDTW = config.tDTW.data

    bsf = 0.0
    th = float('inf')
    # we need to use a list for ids because we allow to select more than one pattern
    bsf_id = [-1]

    # TODO: this is another parameter to be optimized; the value of bsf that we want to reach
    # TODO: in the case of a patient with a lot of seizures, we may want to use a greater value than 1.0 to avoid problems with future seizures. 
    # stop when we find a pattern with DR > 1.1
    min_value_DR = 1.2

    # TODO: Be careful!!!! if we put a "liter" different from "yiter": the dimensions would not match those of qmask
    DoNotComputeQ = (q_p1 < rp1_min) | (rp1_max < q_p1) | (q_p2 < rp2_min) | (rp2_max < q_p2) | (q_p3 < rp3_min) | (rp3_max < q_p3) | (q_d < rd_min) | (rd_max < q_d) | (q_e < re_min) | (re_max < q_e) | qmask

    # if the we discard all the epochs in any of the seizures, we do not proceed with this configuration
    if not check_dnc_query(DoNotComputeQ, idx_query):
        bsf = 0.0        
        return bsf_id,bsf,th

    DoNotCompute = np.asarray(call_dnc_aware_full( DoNotCompute, xiter, d_p1, d_p2, d_p3, d_d, d_e, rp1_min, rp1_max, rp2_min, rp2_max, rp3_min, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards))

    # I need to check everything again because when applying lookbackwards, it may stop computing seizure zones; it might be a good idea to apply it to call_dnc_query as well
    compute_DTW = check_dnc(DoNotCompute, idx_seizure_list, exclusion)
    # compute_DTW can be: 0, if there are no points in the seizures, -1, if there are no points in non-seizures, and 1 if everything went well
            
    if compute_DTW == 0:
        # print('DO NOT COMPUTE [DATA]')
        bsf = 0.0        
        return bsf_id,bsf,th

    if compute_DTW < 0:

        # At this point the only possibility is DR = Inf. In that case we need to find at least one pattern
        find_rand_epochs(DoNotCompute, xiter, idx_seizure_list, factor=1)
    
    lDTW = np.full((yiter, xiter), np.nan)
    # Create a boolean mask for non-NaN values in tDTW; True -> No computation needed
    mask = np.logical_or.outer(DoNotComputeQ, DoNotCompute)
    # Read the values from tDTW
    lDTW[~mask] = tDTW[~mask]

    relation = np.full(yiter, None, dtype=object)

    # now we need to find the best pattern
    for idy in np.where(~DoNotComputeQ)[0]:

        # find the best DR
        dr,ith,vd = find_dtw_distances(lDTW[idy], idx_seizure_list, exclusion)
        # save idx, DR, th, and DR vector
        relation[idy] = [idy, dr, ith, vd]
        # is it the best so far?
        if bsf < dr:
            bsf = dr
            th = ith
            bsf_id[0] = idy
            # print('  New:', relation[-1])
        
    # at this point the index of the best match is bsf_id and it is relative to the training query
    # we need to find the index of the best match in the original query
    bsf_id[0] += from_idq


    if bsf < min_value_DR:
        # min_value_DR = bsf if bsf > min_value_DR else min_value_DR
        
        # when we use function1, idx_query is the list of indexes of all the seizures used in the training
        assert batch_size < len(idx_query), 'Batch size is too large'

        # Find best combination of patterns using func_min        
        max_dr, max_th, best_patterns = search_colors_best(relation, ~DoNotComputeQ, batch_size, min_value_DR, colors, bsf)

        # if we found a combination of patterns with DR > bsf
        if max_dr > bsf:
            # set the best so far to the best combination of patterns
            bsf = max_dr
            th = max_th
            # get the selected patterns ids
            sids = [pattern[0] for pattern in best_patterns]
            # be sure to add the offset to the ids; these are the returned ids
            bsf_id = [x+from_idq for x in sids]
    
    if compute_DTW < 0:
        # We no longer return infinity. This way we guide the optimization process better. The "score", which in our case is the DR, worths more because the configuration already discarded all the non-seizure epochs. This is the reason why we return bsf*2.0 instead of bsf.
        bsf *= 2.0

    return bsf_id,bsf,th


# This is the function for the evaluation process
# The difference with the previous one is that this one collapses the selected rows of the patterns in a single row
# All the arguments are keyword arguments and required
def f_validation(*, rp1_min, rp1_max, rp2_min, rp2_max, rp3_min, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards, lids, th, fth, exclusion, xiter, d_p1, d_p2, d_p3, d_d, d_e, idx_seizure_list, DoNotCompute):

    #  Collect all de memory from last iteration
    gc.collect()

    mDTW = config.mDTW.data

    # compute the DoNotCompute vector
    DoNotCompute = np.asarray(call_dnc_aware_full( DoNotCompute, xiter, d_p1, d_p2, d_p3, d_d, d_e, rp1_min, rp1_max, rp2_min, rp2_max, rp3_min, rp3_max, rd_min, rd_max, re_min, re_max, lookbackwards))
    # dnc_sum = np.count_nonzero(DoNotCompute)
    # dleft = xiter - dnc_sum

    # reserve the memory for the DTWs; the value is not important
    lDTW = np.full((len(lids),xiter), np.nan)

    lDTW[:, ~DoNotCompute] = mDTW[lids, :][:, ~DoNotCompute]

    # once we have the DTWs we need to collapse them
    cDTW = np.min(lDTW, axis=0)

    # We test different thresholds
    vth = fth * th
    vres = [find_validation_params(cDTW, xiter, ith, idx_seizure_list, exclusion) for ith in vth]

    # return [(tp, tn, fp, fn)]
    return vres


