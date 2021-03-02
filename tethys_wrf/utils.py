"""
Some useful functions
"""
from __future__ import division

import numpy as np


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
