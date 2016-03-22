"""
plot_erm_shielding_v1.py

@author: wronk

Plot saved shielding factors
"""
import os
import os.path as op
import numpy as np
import pickle as pkl
import matplotlib

#######################################
# Define global vars and load data
code_root = os.environ['CODE_ROOT']
saved_erm_fname = op.join(code_root, 'sss_fine_cal_files', 'cache',
                          'erm_shielding_factors.pkl')
saved_params_fname = op.join(code_root, 'sss_fine_cal_files', 'cache',
                             'shielding_factor_params.pkl')

with open(saved_erm_fname, 'rb') as pkl_file:
    sf_list = pkl.load(pkl_file)
with open(saved_params_fname, 'rb') as pkl_file:
    params = pkl.load(pkl_file)

save_dir = op.join(code_root, 'sss_fine_cal_files', 'figures')
save_plots = True
#######################################
# Helper functions
#######################################


def get_sf_arr(sf_data, key):
    """Helper to get all data from shielding factor for one key"""
    sf_keyed_list = [day[key] for day in sf_data]
    sf_keyed_list_filter = [day_filt for day_filt in sf_keyed_list
                            if day_filt.shape[0] == 180001]

    return np.array(sf_keyed_list_filter)

#######################################
# Plot
#######################################
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
plot_prefix = 'ERM_'

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
sf_arr = {}

for ri, reg_key in enumerate(params['cal_keys']):
    sf_arr[reg_key] = get_sf_arr(sf_list, reg_key)

    mean_sf_list.append(np.mean(sf_arr[reg_key], axis=1))
    max_sf_list.append(np.max(sf_arr[reg_key], axis=1))
sf_arr['raw_norm'] = get_sf_arr(sf_list, 'raw_norm')

############################
# Plot mean/max shielding factor
plt.close('all')

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(2 * fig_width_1col, 4))

for ri, cal_key in enumerate(params['cal_keys']):
    axs[0].plot(range(mean_sf_list[ri].shape[0]), mean_sf_list[ri],
                color=colors3_hex[ri], label='Fine cal: ' + cal_key, lw=2)
    axs[1].plot(range(max_sf_list[ri].shape[0]), max_sf_list[ri],
                color=colors3_hex[ri], lw=2)


ax_ylims = axs[0].get_ylim()
#axs[0].set_ylim(ax_ylims[0], ax_ylims[1] + 0.1 * (ax_ylims[1] - ax_ylims[0]))
axs[0].set_ylim(ax_ylims[0], 140)
axs[0].legend(loc='best', fontsize=label_fontsize - 4)
axs[0].set_ylabel('Mean Shielding Factor', fontsize=label_fontsize)
axs[1].set_ylabel('Max Shielding Factor', fontsize=label_fontsize)

for ax in axs:
    ax.set_xlabel('Day', fontsize=label_fontsize)
    ax.set_xticks(range(mean_sf_list[0].shape[0]))
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=8)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

plt.tight_layout()

if save_plots:
    fig.savefig(op.join(save_dir, plot_prefix +
                        'shielding_mean_max.png'), dpi=150)
    #fig.savefig(op.join(save_dir) + 'shielding_mean_max.pdf', dpi=300)


#############################################
# Plot histogram of mean/max shielding factor
bins_mean = np.linspace(0, 120, 20)
bins_max = np.linspace(0, 500, 20)

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
    ax.set_ylabel('Count (# of days)')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

plt.tight_layout()

if save_plots:
    fig2.savefig(op.join(save_dir, plot_prefix +
                         'shielding_mean_max_hist.png'), dpi=150)

#############################################
# Plot violin plot of mean/max shielding factor

fig3, axs3 = plt.subplots(nrows=1, ncols=2, figsize=(2 * fig_width_1col, 6))
axs3[0].violinplot(mean_sf_list, showmeans=True, showmedians=False)
axs3[1].violinplot(max_sf_list, showmeans=True, showmedians=False)

axs3[0].set_ylabel('Mean Shielding Factor', fontsize=label_fontsize)
axs3[1].set_ylabel('Max Shielding Factor', fontsize=label_fontsize)

axs3[0].set_ylim([0, 120])
axs3[1].set_ylim([0, 500])

for ax in axs3:
    ax.set_xlabel('Fine calibration type', fontsize=label_fontsize)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([y + 1 for y in range(len(mean_sf_list))])
    ax.set_xticklabels(params['cal_keys'])

    ax.yaxis.grid(True)

plt.tight_layout()

if save_plots:
    fig3.savefig(op.join(save_dir, plot_prefix +
                         'shielding_mean_max_violin.png'), dpi=150)

#############################################
# Plot shielding factor over time

x_max = 180001
n_days = 4
key = 'None'

fig4, ax4 = plt.subplots(figsize=(3 * fig_width_1col, 6))
ax4.plot(np.arange(0, x_max * 0.001, step=0.001), sf_arr[key][0:n_days, :].T, alpha=0.7)

ax4.set_xlabel('Fine calibration type', fontsize=label_fontsize)
ax4.set_ylabel('Shielding Factor', fontsize=label_fontsize)

ax4.set_ylim([0, 70])
ax4.xaxis.set_ticks_position('bottom')
ax4.yaxis.set_ticks_position('left')

ax4.yaxis.grid(True)

plt.tight_layout()

if save_plots:
    fig4.savefig(op.join(save_dir, plot_prefix + key +
                         '_dynamic_shielding.png'), dpi=150)

#############################################
# Plot shielding factor over time

fig5, axs5 = plt.subplots(nrows=1, ncols=3, figsize=(5 * fig_width_1col, 8))
ylims = [65, 120, 350]
colors = matplotlib.colors.cnames.keys()
#colors = ['b', 'g', 'r', 'c']
for ai, (ax, key) in enumerate(zip(axs5, params['cal_keys'])):

    for row_i in range(sf_arr['raw_norm'].shape[0]):
        ax.scatter(sf_arr['raw_norm'][row_i, :], sf_arr[key][row_i, :], s=.2,
                   c='k', alpha=0.05, edgecolor='none')

    ax.set_xlim(0, 2e-9)
    ax.set_ylim(0, ylims[ai])
    ax.set_xlabel('Raw signal norm', fontsize=label_fontsize)
    ax.set_ylabel('Shielding factor', fontsize=label_fontsize)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(True)
fig5.suptitle('Data from all days')

plt.tight_layout(pad=2.)

if save_plots:
    fig5.savefig(op.join(save_dir, plot_prefix + 'norm_comparison.png'),
                 dpi=150)

plt.show()
