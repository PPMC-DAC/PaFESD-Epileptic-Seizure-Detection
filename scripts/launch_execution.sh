#!/usr/bin/env bash

# ('-P','--patient', required=True, help='Patient number. Eg: chb24')
# ('-S','--signal', required=True, help='Signal label. Eg: F4-C4')
# ('-t','--template', type=int, default=5*256, help='Template window size in samples.')
# ('-s','--stride', type=int, default=256, help='Stride size in samples.')
# ('-w','--warp', type=int, default=16, help='Warping window size in samples.')
# ('-n','--nprocs', type=int, default=mp.cpu_count(), help='Number of processes.')
# ('-c','--chunks', type=int, default=mp.cpu_count(), help='Divides the signal in N chunks for parallel processing.')
# ('-f','--filter', type=bool, default=False, help='Active bandpass filter.')
# ('-lb','--lookback', type=int, default=1, help='Number of consecutive seconds to check before compute')
# ('-il','--input_logs', type=str, default='empty', help='Log file with evaluations. Format: [(dict, pattern_id, DR, dist_threshold)]')
# ('-it','--num_iterations', type=int, default=100, help='Number of runs for each optimizer iteration.')
# ('-th','--threshold', type=float, default=.05, help='DTW distance threshold. Eg: .05 means 5% more than the worst distance.')
# ('-est','--estimator', type=str, default='scm', help='Covariance estimator used to compute the distance. Eg: scm, oas, lwf, sch')
# ('-z', '--seizures', type=int, default=4, help='Number of seizures to train the algorithm.')
# ('-b', '--batch', type=int, default=3, help='Maximum number of patterns to use as batch.')
# ('-ts', '--training_size', type=float, default=30., help='Percentage of the signal to use as training.')
# ('-cdf', '--cdf_limit', type=float, default=.5, help='This is the probability limit to design the bounds of the categories. Eg: .25 means that the LB are in [0, .25] and the UB in [.75, 1].')

python -u -W ignore find_pattern.py -P chb06 -S F3-C3 -w 16 -s 256 -it 1000 -n 64 -z 4 -b 3 -est 'scm' -f True
