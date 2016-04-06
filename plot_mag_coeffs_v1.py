"""
plot_mag_coeffs_v1.py

@author: wronk

Scatter plots for magnetometer coeffs across different freqencies
"""
import os
import os.path as op
import numpy as np

import mne
from mne.preprocessing._fine_cal import read_fine_calibration as read_cal
from mne.io.pick import pick_types

# Requires:
#    empty room data with rotating coil in sss_fine_cal_files/coil_data/(date)

#######################################
# Define global vars
#######################################
file_head_dir = op.join(os.environ['CODE_ROOT'], 'sss_fine_cal_files')

date = '160302'
file_beg = date + '_empty_room_rotating_coil_'
file_end = '_hz_raw.fif'

data_dir = op.join(file_head_dir, 'cal_files')
save_dir = op.join(file_head_dir, 'figures', 'magCoeff_freqComparison')

std_freq = 10
coil_freqs = [1, 10, 20, 50, 100, 200, 500, 1000]
std_freq_i = coil_freqs.index(std_freq)
cal_dims = [1, 3]  # dimension of calibration files to test (1 and/or 3)

###########################
# Load raw for channel info
###########################
# Construct fname
fname_raw = op.join(file_head_dir, 'coil_data', date,
                    file_beg + str(std_freq) + '_hz_raw.fif')

# Load file and get mag inds
raw = mne.io.Raw(fname_raw)
mag_picks = pick_types(raw.info, meg='mag')

########################################################
# Get mag coeffs for all other frequencies and cal types
########################################################
cal_arr = np.zeros((len(coil_freqs), len(mag_picks)))

# Loop over frequencies
for fi, freq in enumerate(coil_freqs):
    # Loop over calibration types
    # Only need 1D cals because magnetometer coeffs are same for 1D and 3D
    fname_fine_calc = op.join(data_dir, 'sss_' + str(freq) + 'Hz_1D_cal.dat')
    fine_cal = read_cal(fname_fine_calc)
    imb_cals = np.array([fine_cal['imb_cals'][ii] for ii in mag_picks])

    # Store mag coeffs in array
    cal_arr[fi, :] = imb_cals.squeeze()

###########################
# Plot each set of points
###########################
from matplotlib import pyplot as plt
from scipy import stats

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.close('all')  # Close all previous plots
plt.ion()

label_fontsize = 14
tickLabel_fontsize = 12
figsize = (12, 6)
nrows, ncols = 2, 4

fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                         figsize=figsize)
for flat_i, arr_i in enumerate(np.ndindex(nrows, ncols)):
    axes[arr_i].scatter(cal_arr[std_freq_i, :],
                        cal_arr[flat_i, :],
                        edgecolor='none', alpha=0.5)
    params = stats.linregress(cal_arr[std_freq_i, :], cal_arr[flat_i, :])
    axes[arr_i].annotate(r'$R^2:$ %0.2f' % (params[2] ** 2), xy=(0.65, 0.1), xycoords='axes fraction',
                         fontsize=tickLabel_fontsize)

    #axes[arr_i].set_aspect('equal')
    axes[arr_i].set_xlim([0.95, 1.05])
    axes[arr_i].set_ylim([0.95, 1.05])
    #axes[arr_i].locator_params(axis='both', nbins=8)
    axes[arr_i].set_title('Freq: ' + str(coil_freqs[flat_i]) + ' Hz',
                          fontsize=label_fontsize)

fig.suptitle('1D Fine Calibration', fontsize=label_fontsize)
fig.tight_layout(pad=2.5, h_pad=.75, w_pad=.75)

fig.savefig(op.join(save_dir, 'magCoeffComparison.png'))
