"""
shielding_func.py

@author: wronk

Extra functions for sss_fine_cal analysis
"""

import numpy as np
from numpy.linalg import norm
from mne.io.pick import pick_types


def coilNoise_error_checks(raw, lp=1200, hp=0.15, sfreq=3000, nsecs=120):
    """Helper function for error checking files with added coil noise"""

    if raw.info['lowpass'] < lp:
        raise RuntimeError('LP filter < ' + str(lp) + ' Hz')
    if raw.info['highpass'] > hp:
        raise RuntimeError('HP filter > ' + str(hp) + ' Hz')
    if raw.info['sfreq'] < sfreq:
        raise RuntimeError('Sampling Freq < ' + str(sfreq) + ' Hz')
    if raw.times[-1] < nsecs:
        raise RuntimeError('< ' + str(nsecs) + ' of data')

    if 'maxshield' in raw.info.keys():
        if raw.info['maxshield'] is True:
            raise RuntimeError('Data recorded with MaxShield on')


def erm_error_checks(raw, lp=300, hp=0.15, sfreq=1000, nsecs=120):
    """Helper function for error checking empty room files"""
    if raw.info['lowpass'] < lp:
        raise RuntimeError('LP filter < ' + str(lp) + ' Hz')
    if raw.info['highpass'] > hp:
        raise RuntimeError('HP filter > ' + str(hp) + ' Hz')
    if raw.info['sfreq'] < sfreq:
        raise RuntimeError('Sampling Freq < ' + str(sfreq) + ' Hz')
    if raw.times[-1] < nsecs:
        raise RuntimeError('< ' + str(nsecs) + ' secs of data')

    if 'maxshield' in raw.info.keys():
        if raw.info['maxshield'] is True:
            raise RuntimeError('Data recorded with MaxShield on')


def comp_shielding(raw, raw_sss):
    """Helper to computer shielding factor using raw and SSS processed data"""

    # Grab only magnetometers for calculation (though both mags and grads
    # affect shielding factor calculation)
    picks = pick_types(raw.info, meg='mag')
    assert np.all(picks == pick_types(raw_sss.info, meg='mag')), \
        'Channel mismatch'

    # Compute signal norm of both matrices for all time points
    raw_power = norm(raw[picks][0], axis=0)
    sss_power = norm(raw_sss[picks][0], axis=0)

    # Return shielding factor
    return raw_power / sss_power
