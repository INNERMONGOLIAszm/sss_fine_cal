"""
coilNoise_shielding_v1.py

@author: wronk

Calculate shielding factors when coil was on introducing noise into the MEG
data and save results
"""
import os
import os.path as op
import numpy as np
import pickle as pkl
from time import strftime
import pprint

import mne
from mne.preprocessing import maxwell_filter
from mne.io.pick import pick_types

from shielding_func import comp_shielding, coilNoise_error_checks

#######################################
# Define global vars
#######################################
file_head_dir = op.join(os.environ['CODE_ROOT'], 'sss_fine_cal_files')
fineCal_1d_fname = op.join(file_head_dir, 'sss_cal_3053.dat')
fineCal_3d_fname = op.join(file_head_dir, 'sss_cal_3053_141027_close_3d.dat')

# Parameters in dict to easily save later
params = dict(n_secs=120.,  # Length (in seconds) of raw data to process
              st_duration=None,  # Length (in seconds) or None (for no tSSS)
              coord_frame='meg',
              cal_keys=['None', '1D', '3D'],
              coil_freqs=[1, 10, 20, 50, 100, 200, 500, 1000],
              cal_fnames=[None, fineCal_1d_fname, fineCal_3d_fname],
              ctc_fname=op.join(file_head_dir, 'ct_sparse.fif'),
              regularize=None,  # 'in' or None
              verbose=False)

date = '160302'
file_beg = date + '_empty_room_rotating_coil_'
file_end = '_hz_raw.fif'

data_dir = op.join(file_head_dir, 'coil_data', date)
#coil_filenames = os.listdir(data_dir)
save_dir = op.join(file_head_dir, 'cache')

debug = False
pkl_data = True
pp = pprint.PrettyPrinter(indent=2)

if debug:
    params['coil_freqs'] = params['coil_freqs'][0:3]  # Subselect a few files
    params['n_secs'] = 30.
    pkl_data = False

####################################################
# Read in ERM files, apply SSS and compute shielding
####################################################
sf_list = []  # List storing a [1 x n_times] vector for each .fif file

# Print relevant processing info
print '\nMaxwell filter fine calibrations choosen:'
pp.pprint(params['cal_keys'])
if '1D' in params['cal_keys']:
    print '1D calibration file: ' + fineCal_1d_fname
if '3D' in params['cal_keys']:
    print '3D calibration file: ' + fineCal_3d_fname
print ('\nProcessing {num} frequency files (started @ '.format(
    num=len(params['coil_freqs'])) + strftime('%D %H:%M:%S') + '):')
pp.pprint(params['coil_freqs'])


for fi, freq in enumerate(params['coil_freqs']):
    print '\n=================================================='
    f_name = file_beg + str(freq) + file_end
    print ('Processing {num} of {tot}: \n'.format(
        num=fi + 1, tot=len(params['coil_freqs'])) + f_name + ' \n')

    # Load data, crop, and update magnetometer coil type info
    raw = mne.io.Raw(op.join(data_dir, f_name), verbose=params['verbose'],
                     allow_maxshield=False)

    # Error checks, crop data, and fix coil types
    coilNoise_error_checks(raw)
    raw.crop(tmax=params['n_secs'], copy=False)
    raw.fix_mag_coil_types()

    shielding_dict = dict(f_name=f_name, freq=freq)

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

    sf_list.append(shielding_dict)

####################################################
# Save data
####################################################
if pkl_data:
    print '\nSaving data',
    with open(op.join(save_dir, date + '_coilNoise_shielding_factors.pkl'), 'wb') as pkl_file:
        pkl.dump(sf_list, pkl_file)
    with open(op.join(save_dir, date + '_coilNoise_shielding_factor_params.pkl'), 'wb') as pkl_file:
        params['finish_time'] = strftime('%D %H:%M:%S')
        pkl.dump(params, pkl_file)
    print ' ... Done'

print 'Finished @ ' + strftime('%D %H:%M:%S')
