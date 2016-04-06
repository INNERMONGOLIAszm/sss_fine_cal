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
    """Helper to compute shielding factor using raw and SSS processed data"""

    # Compute signal norm of both matrices for all time points
    raw_power = get_power(raw)
    sss_power = get_power(raw_sss)

    # Return shielding factor
    return raw_power / sss_power


def get_power(raw, picks=None):
    """Helper to compute power of magnetometer channels in raw object"""

    # Grab only magnetometers for calculation (though both mags and grads
    # affect shielding factor calculation)
    if picks is None:
        picks = pick_types(raw.info, meg='mag')

    # Return norm along channel axis
    return norm(raw[picks, :][0], axis=0)


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2, metric='radian'):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    if metric == 'radian':
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    elif metric == 'degree':
        return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    else:
        raise RuntimeError('Input metric not recognized')
