"""
plot_normal_diff_v1.py

@author: wronk

Plots differences in the angles for each sensor
"""
import os
import os.path as op
import numpy as np
import itertools as it

import mne
from mne.preprocessing._fine_cal import read_fine_calibration as read_cal
from mne.io.pick import pick_types
from shielding_func import angle_between

# Requires:
#    fine cal (.dat) files in '...sss_fine_cal_files/cal_files/'


def extract_znorm(fname, mag_picks):
    """Helper to get z-normal vector from fine cal files"""
    fine_cal = read_cal(fname)
    locs = np.array([fine_cal['locs'][ii] for ii in mag_picks])

    return locs[:, 9:12]

#######################################
# Define global vars
#######################################
file_head_dir = op.join(os.environ['CODE_ROOT'], 'sss_fine_cal_files')

data_dir = op.join(file_head_dir, 'cal_files')
save_dir = op.join(file_head_dir, 'figures', 'normalComparison')

std_freq = 10
coil_freqs = [1, 10, 20, 50, 100, 200, 500, 1000]
coil_freqs_str = [str(cf) for cf in coil_freqs]
std_freq_i = coil_freqs.index(std_freq)
cal_dims = [1, 3]  # dimension of calibration files to test (1 and/or 3)

###########################
# Load raw for channel info
###########################
# Construct fname
date = '160302'
file_beg = date + '_empty_room_rotating_coil_'
file_end = '_hz_raw.fif'
fname_raw = op.join(file_head_dir, 'coil_data', '160302',
                    file_beg + str(std_freq) + '_hz_raw.fif')

# Load file and get mag inds
raw = mne.io.Raw(fname_raw)
mag_picks = pick_types(raw.info, meg='mag')

loc_arr = np.zeros((len(coil_freqs), len(mag_picks), 3))

###########################################
# Loop over all combinations of frequencies
###########################################
# Fill array with normal vectors for both sets of frequencies
for ci, freq in enumerate(coil_freqs):
    freq_fname = op.join(data_dir, 'sss_' + str(freq) + 'Hz_1D_cal.dat')
    loc_arr[ci, :, :] = extract_znorm(freq_fname, mag_picks)

########################################################
# Compare all vector normals
########################################################
diff = {}
for f1, f2 in it.combinations(range(len(coil_freqs)), 2):
    angle_diffs = np.zeros(len(mag_picks))
    for ii in range(len(mag_picks)):
        angle_diffs[ii] = angle_between(loc_arr[f1, ii, :], loc_arr[f2, ii, :],
                                        metric='degree')

    diff[coil_freqs_str[f1] + ' ' + coil_freqs_str[f2]] = angle_diffs

# With data gathered, compute mean angle difference
diff_mean = np.zeros((len(coil_freqs), len(coil_freqs)))
for f1, f2 in it.combinations(range(len(coil_freqs)), 2):
    diff_mean[f1, f2] = np.mean(diff[coil_freqs_str[f1] + ' ' +
                                     coil_freqs_str[f2]])

###########################
# Plot each set of points
###########################
from matplotlib import pyplot as plt

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.close('all')  # Close all previous plots
plt.ion()

label_fontsize = 14
tickLabel_fontsize = 12
ang_max = 0.25

# Create figure, and plot image
fig1, ax1 = plt.subplots(figsize=(6, 5))
im = ax1.imshow(diff_mean, interpolation='none', cmap='viridis', vmax=ang_max)
fig1.colorbar(im)

plt.xticks(range(len(coil_freqs)), coil_freqs_str)
plt.yticks(range(len(coil_freqs)), coil_freqs_str)

ax1.xaxis.tick_top()  # Set axis ticks on top of plot
ax1.set_ylabel('Frequency')
ax1.set_title(('Mean z-norm angle difference between freqs\n(using arccos of'
               ' dot product; clipped at %0.2f deg)' % ang_max), y=1.125)
fig1.tight_layout()

fname_save = op.join(save_dir, 'meanAngleDifs.png')
fig1.savefig(fname_save)
