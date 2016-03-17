"""
plot_erm_shielding_v1.py

@author: wronk

Plot saved shielding factors
"""
import os
import os.path as op
import numpy as np
import pickle as pkl

#######################################
# Define global vars and load data
code_root = os.environ['CODE_ROOT']
date = '160302'
saved_coilNoise_fname = op.join(code_root, 'sss_fine_cal_files', 'cache',
                                date + '_coilNoise_shielding_factors.pkl')
saved_params_fname = op.join(code_root, 'sss_fine_cal_files', 'cache',
                             date + '_coilNoise_shielding_factor_params.pkl')

with open(saved_coilNoise_fname, 'rb') as pkl_file:
    sf_list = pkl.load(pkl_file)
with open(saved_params_fname, 'rb') as pkl_file:
    params = pkl.load(pkl_file)

save_dir = op.join(code_root, 'sss_fine_cal_files', 'figures')
save_plots = True
#######################################
# Helper functions
#######################################


def get_sf_arr(sf_data, key):
    """Helper to get all data from shielding factor for one freq"""
    sf_keyed_list = [freq_dict[key] for freq_dict in sf_data]
    #sf_keyed_list_filter = [freq_filt for freq_filt in sf_keyed_list
    #                        if freq_filt.shape[0] == 120001]

    #return np.array(sf_keyed_list_filter)
    return np.array(sf_keyed_list)

#######################################
# Plot
#######################################
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
plot_prefix = 'coilNoise_'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.close('all')  # Close all previous plots
plt.ion()

label_fontsize = 14
tickLabel_fontsize = 12
fig_width_1col = 4

colors2 = ['#4477AA', '#CC6677']
colors3_hex = ['#4477AA', '#DDCC77', '#CC6677']
colors3 = [hex2color(hex_col) for hex_col in colors3_hex]
colors4 = ['#4477AA', '#117733', '#DDCC77', '#CC6677']

print 'Loading data from: ' + params['finish_time']

mean_sf_list = []
max_sf_list = []
freqs = params['coil_freqs']

for ri, reg_key in enumerate(params['cal_keys']):
    sf_arr = get_sf_arr(sf_list, reg_key)
    mean_sf_list.append(np.mean(sf_arr, axis=1))
    max_sf_list.append(np.max(sf_arr, axis=1))

############################
# Plot mean/max shielding factor

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(3 * fig_width_1col, 4))

for ri, cal_key in enumerate(params['cal_keys']):
    axs[0].plot(freqs, mean_sf_list[ri], color=colors3_hex[ri],
                label='Fine cal: ' + cal_key, lw=2)
    axs[1].plot(freqs, max_sf_list[ri], color=colors3_hex[ri], lw=2)


ax_ylims = axs[0].get_ylim()
#axs[0].set_ylim(ax_ylims[0], ax_ylims[1] + 0.1 * (ax_ylims[1] - ax_ylims[0]))
axs[0].set_ylim(0, 120)
axs[0].legend(loc='best', fontsize=label_fontsize - 4)
axs[0].set_ylabel('Mean Shielding Factor', fontsize=label_fontsize)
axs[1].set_ylabel('Max Shielding Factor', fontsize=label_fontsize)

for ax in axs:
    ax.set_xlabel('Coil Noise Frequency', fontsize=label_fontsize)
    ax.set_xticks(range(len(freqs)))
    ax.set_xticklabels(freqs)
    #ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=8)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.grid(True)
    ax.set_xscale('log')

plt.tight_layout()

if save_plots:
    fig.savefig(op.join(save_dir, date + '_' + plot_prefix +
                        'shielding_mean_max.png'), dpi=150)
    #fig.savefig(op.join(save_dir) + 'shielding_mean_max.pdf', dpi=300)


#############################################
# Plot histogram of mean/max shielding factor
bins_mean = np.linspace(0, 120, 20)
bins_max = np.linspace(0, 600, 20)

fig2, axs2 = plt.subplots(nrows=1, ncols=2, figsize=(2 * fig_width_1col, 4))

for ri, cal_key in enumerate(params['cal_keys']):
    axs2[0].hist(mean_sf_list[ri], bins_mean, histtype='step', fill=True,
                 fc=colors3[ri], ec='k', label='Fine cal: ' + cal_key,
                 alpha=0.5)
    axs2[1].hist(max_sf_list[ri], bins_max, histtype='step', fill=True,
                 fc=colors3[ri], ec='k', alpha=0.5)

ax_ylims = axs2[0].get_ylim()
axs2[0].set_ylim(0, ax_ylims[1] + 2)
ax_ylims = axs2[1].get_ylim()
axs2[1].set_ylim(0, ax_ylims[1] + 2)

axs2[0].legend(loc='best', fontsize=label_fontsize - 4)
axs2[0].set_xlabel('Mean Shielding Factor', fontsize=label_fontsize)
axs2[1].set_xlabel('Max Shielding Factor', fontsize=label_fontsize)

for ax in axs2:
    ax.set_ylabel('Count (# of freqs)')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

plt.tight_layout()

if save_plots:
    fig2.savefig(op.join(save_dir, date + '_' + plot_prefix +
                         'shielding_mean_max_hist.png'), dpi=150)

#############################################
# Plot violin plot of mean/max shielding factor

fig3, axs3 = plt.subplots(nrows=1, ncols=2, figsize=(2 * fig_width_1col, 6))
axs3[0].violinplot(mean_sf_list, showmeans=True, showmedians=False)
axs3[1].violinplot(max_sf_list, showmeans=True, showmedians=False)

fig3.suptitle('Shielding across {num} frequencies'.format(num=len(freqs)))
axs3[0].set_ylabel('Mean Shielding Factor', fontsize=label_fontsize)
axs3[1].set_ylabel('Max Shielding Factor', fontsize=label_fontsize)

axs3[0].set_ylim([0, 120])
axs3[1].set_ylim([0, 600])

for ax in axs3:
    ax.set_xlabel('Fine calibration type', fontsize=label_fontsize)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([y + 1 for y in range(len(mean_sf_list))])
    ax.set_xticklabels(params['cal_keys'])

    ax.yaxis.grid(True)

plt.tight_layout(pad=2.5)

if save_plots:
    fig3.savefig(op.join(save_dir, date + '_' + plot_prefix +
                         'shielding_mean_max_violin.png'), dpi=150)

plt.show()
