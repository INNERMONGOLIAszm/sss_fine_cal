"""
erm_shielding_v1.py

@author: wronk

Calculate shielding factors for ERM data and save results
"""
import os
import os.path as op
import numpy as np
from numpy.linalg import norm
import pickle as pkl
from time import strftime
from datetime import datetime as dt
import pprint

import mne
from mne.preprocessing import maxwell_filter
from mne.io.pick import pick_types
from mne.parallel import parallel_func

from shielding_func import comp_shielding, erm_error_checks, get_power

#######################################
# Define global vars
#######################################
file_head_dir = op.join(os.environ['CODE_ROOT'], 'sss_fine_cal_files')
fineCal_1d_fname = op.join(file_head_dir, 'sss_cal_3053.dat')
fineCal_3d_fname = op.join(file_head_dir, 'sss_cal_3053_141027_close_3d.dat')

# Parameters in dict to easily save later
params = dict(erm_len=180.,  # Length (in seconds) of raw data to process
              st_duration=None,  # Length (in seconds) or None (for no tSSS)
              coord_frame='meg',
              cal_keys=['None', '1D', '3D'],
              cal_fnames=[None, fineCal_1d_fname, fineCal_3d_fname],
              ctc_fname=op.join(file_head_dir, 'ct_sparse.fif'),
              regularize=None,  # 'in' or None
              verbose=False)

data_dir = op.join(file_head_dir, 'ERM_data')
erm_filenames = os.listdir(data_dir)
erm_filenames.sort()
save_dir = op.join(file_head_dir, 'cache')

pkl_data = True
pp = pprint.PrettyPrinter(indent=2)

debug = False
if debug:
    erm_filenames = erm_filenames[0:10]  # Subselect a few files
    params['erm_len'] = 30.
    pkl_data = False


#############################################
# SSS function to be used in parallel methods
#############################################
def _maxwell_fun(f_name):
    '''Aux function to run SSS and store shielding calculations'''

    # Load data, crop, and update magnetometer coil type info
    raw = mne.io.Raw(op.join(data_dir, f_name), verbose=params['verbose'])
    erm_error_checks(raw)
    print ('Time of recording: ' +
           dt.fromtimestamp(raw.info['meas_date'][0]).strftime('%Y-%m-%d %H:%M:%S')
           + '\n')

    raw.crop(tmax=params['erm_len'], copy=False)
    raw.fix_mag_coil_types()

    shielding_dict = dict(f_name=f_name)

    # Store power to get relative change
    shielding_dict['raw_norm'] = get_power(raw)

    # Do SSS processing using different calibration files
    for cal_key, cal_fname in zip(params['cal_keys'], params['cal_fnames']):
        raw_sss = maxwell_filter(raw, calibration=cal_fname,
                                 cross_talk=params['ctc_fname'],
                                 st_duration=params['st_duration'],
                                 coord_frame=params['coord_frame'],
                                 regularize=params['regularize'],
                                 verbose=params['verbose'])
        shielding_dict[cal_key] = comp_shielding(raw, raw_sss)

    print 'Spatial filtering mean/max:'
    for cal_key in params['cal_keys']:
        # Print mean and max shielding factor
        print (cal_key + '\tMean: {mean} \tMax: {max}'.format(
            mean=np.mean(shielding_dict[cal_key]),
            max=np.max(shielding_dict[cal_key])))
    return shielding_dict

####################################################
# Read in ERM files, apply SSS and compute shielding
####################################################
# List storing a dict of results for each .fif file
sf_list = [None] * len(erm_filenames)

# Print relevant processing info
print '\nMaxwell filter fine calibrations choosen:'
pp.pprint(params['cal_keys'])
if '1D' in params['cal_keys']:
    print '1D calibration file: ' + fineCal_1d_fname
if '3D' in params['cal_keys']:
    print '3D calibration file: ' + fineCal_3d_fname
print ('\nProcessing {num} files (started @ '.format(num=len(erm_filenames)) +
       strftime('%D %H:%M:%S') + '):')
pp.pprint(erm_filenames)

# Parallelized functions
parallel, my_maxwell, n_jobs = parallel_func(_maxwell_fun, n_jobs=-1,
                                             verbose=False)
sf_list = parallel(my_maxwell(f_name) for f_name in erm_filenames)

####################################################
# Save data
####################################################
if pkl_data:
    print '\nSaving data',
    with open(op.join(save_dir, 'erm_shielding_factors.pkl'), 'wb') as pkl_file:
        pkl.dump(sf_list, pkl_file)
    with open(op.join(save_dir, 'shielding_factor_params.pkl'), 'wb') as pkl_file:
        params['finish_time'] = strftime('%D %H:%M:%S')
        pkl.dump(params, pkl_file)
    print ' ... Done'

print 'Finished @ ' + strftime('%D %H:%M:%S')
