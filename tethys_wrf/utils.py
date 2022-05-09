"""
Some useful functions
"""
from __future__ import division

import os
import numpy as np
from wrf import getvar
from functools import wraps
import pyproj

# Default proj
wgs84 = pyproj.Proj(proj='latlong', datum='WGS84')

def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""

    attr_name = '_lazy_' + fn.__name__

    @property
    @wraps(fn)
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property


# A series of variables and dimension names that Salem will understand
valid_names = dict()
valid_names['x_dim'] = ['west_east', 'lon', 'longitude', 'longitudes', 'lons',
                        'xlong', 'xlong_m', 'dimlon', 'x', 'lon_3', 'long',
                        'phony_dim_0', 'eastings', 'easting', 'nlon', 'nlong',
                        'grid_longitude_t']
valid_names['y_dim'] = ['south_north', 'lat', 'latitude', 'latitudes', 'lats',
                        'xlat', 'xlat_m', 'dimlat', 'y','lat_3', 'phony_dim_1',
                        'northings', 'northing', 'nlat', 'grid_latitude_t']
valid_names['z_dim'] = ['levelist','level', 'pressure', 'press', 'zlevel', 'z',
                        'bottom_top']
valid_names['t_dim'] = ['time', 'times', 'xtime']

valid_names['lon_var'] = ['lon', 'longitude', 'longitudes', 'lons', 'long']
valid_names['lat_var'] = ['lat', 'latitude', 'latitudes', 'lats']
valid_names['time_var'] = ['time', 'times']


def get_wrf_var(ncfile, varname, times):
    """

    """
    t = getvar(ncfile, varname)
    t = t.drop('Time').expand_dims({'Time': times}).copy()

    for i in range(len(times)):
        t[{'Time': i}] = getvar(ncfile, varname, timeidx=i)

    return t


# raw_param_encodings = {'T2': {'scale_factor': 0.01, 'dtype': 'int16', '_FillValue': -9999},
#                     'RAINNC': {'scale_factor': 0.1, 'dtype': 'int16', '_FillValue': -9999},
#                     'SNOWNC': {'scale_factor': 0.1, 'dtype': 'int16', '_FillValue': -9999},
#                     'SFROFF': {'scale_factor': 0.1, 'dtype': 'int16', '_FillValue': -9999},
#                     'UDROFF': {'scale_factor': 0.1, 'dtype': 'int16', '_FillValue': -9999},
#                     'PSFC': {'scale_factor': 0.01, 'dtype': 'int16', '_FillValue': -9999},
#                     'SWDOWN': {'scale_factor': 0.01, 'dtype': 'int16', '_FillValue': -9999},
#                     'GLW': {'scale_factor': 0.01, 'dtype': 'int16', '_FillValue': -9999},
#                     'GRDFLX': {'scale_factor': 0.01, 'dtype': 'int16', '_FillValue': -9999},
#                     'Q2': {'scale_factor': 0.01, 'dtype': 'int16', '_FillValue': -9999},
#                     'U10': {'scale_factor': 0.01, 'dtype': 'int16', '_FillValue': -9999},
#                     'V10': {'scale_factor': 0.01, 'dtype': 'int16', '_FillValue': -9999},
#                     # 'reference_et_at_0': ['T2', 'Q2', 'U10', 'V10', 'SWDOWN', 'GLW', 'GRDFLX', 'PSFC', 'ALBEDO']
#                     }


def str_in_list(l1, l2):
    """Check if one element of l1 is in l2 and if yes, returns the name of
    that element in a list (could be more than one.

    Examples
    --------
    >>> print(str_in_list(['time', 'lon'], ['temp','time','prcp']))
    ['time']
    >>> print(str_in_list(['time', 'lon'], ['temp','time','prcp','lon']))
    ['time', 'lon']
    """
    return [i for i in l1 if i.lower() in l2]


def nice_scale(mapextent, maxlen=0.15):
    """Returns a nice number for a legend scale of a map.

    Parameters
    ----------
    mapextent : float
        the total extent of the map
    maxlen : float
        from 0 to 1, the maximum relative length allowed for the scale

    Examples
    --------
    >>> print(nice_scale(140))
    20.0
    >>> print(nice_scale(140, maxlen=0.5))
    50.0
    """
    d = np.array([1, 2, 5])
    e = (np.ones(12) * 10) ** (np.arange(12)-5)
    candidates = np.matmul(e[:, None],  d[None, :]).flatten()
    return np.max(candidates[candidates / mapextent <= maxlen])


def reduce(arr, factor=1, how=np.mean):
    """Reduces an array's size by a given factor.

    The reduction can be done by any reduction function (default is mean).

    Parameters
    ----------
    arr : ndarray
        an array of at least 2 dimensions (the reduction is done on the two
        last dimensions).
    factor : int
        the factor to apply for reduction (must be a divider of the original
        axis dimension!).
    how : func
        the reduction function

    Returns
    -------
    the reduced array
    """
    arr = np.asarray(arr)
    shape = list(arr.shape)
    newshape = shape[:-2] + [np.round(shape[-2] / factor).astype(int), factor,
                             np.round(shape[-1] / factor).astype(int), factor]
    return how(how(arr.reshape(*newshape), axis=len(newshape)-3),
               axis=len(newshape)-2)
