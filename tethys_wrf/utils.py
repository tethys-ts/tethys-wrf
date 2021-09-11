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

ds_cols = ['feature', 'parameter', 'frequency_interval', 'aggregation_statistic', 'units', 'wrf_standard_name', 'cf_standard_name', 'scale_factor']


param_func_mappings = {'temp_at_2': ['T2'],
                       'precip_at_0': ['RAINNC'],
                       'snow_at_0': ['SNOWNC'],
                       'runoff_at_0': ['SFROFF'],
                       'recharge_at_0': ['UDROFF'],
                       'pressure_at_0': ['PSFC'],
                       'shortwave_rad_at_0': ['SWDOWN'],
                       'longwave_rad_at_0': ['GLW'],
                       'heat_flux_at_0': ['GRDFLX'],
                       'relative_humidity_at_2': ['T2', 'Q2', 'PSFC'],
                       'wind_speed_at_2': ['U10', 'V10'],
                       'wind_speed_at_10': ['U10', 'V10'],
                       # 'reference_et_at_0': ['T2', 'Q2', 'U10', 'V10', 'SWDOWN', 'GLW', 'GRDFLX', 'PSFC', 'ALBEDO']
                       }

# param_file_mappings = {'temp_at_2': ['2m_temperature_*.nc'],
#                        'precip_at_0': ['total_precipitation_*.nc'],
#                        'snow_at_0': ['snowfall_*.nc'],
#                        'runoff_at_0': ['surface_runoff_*.nc'],
#                        'recharge_at_0': ['sub_surface_runoff_*.nc'],
#                        'pressure_at_0': ['surface_pressure_*.nc'],
#                        'shortwave_rad_at_0': ['surface_net_solar_radiation_*.nc'],
#                        'longwave_rad_at_0': ['surface_net_thermal_radiation_*.nc'],
#                        'heat_flux_at_0': ['surface_latent_heat_flux_*.nc'],
#                        'relative_humidity_at_2': ['2m_temperature_*.nc', '2m_dewpoint_temperature_*.nc'],
#                        'wind_speed_at_2': ['10m_u_component_of_wind_*.nc', '10m_v_component_of_wind_*.nc'],
#                        'wind_speed_at_10': ['10m_u_component_of_wind_*.nc', '10m_v_component_of_wind_*.nc'],
#                        'reference_et_at_0': ['2m_temperature_*.nc', '2m_dewpoint_temperature_*.nc', '10m_u_component_of_wind_*.nc', '10m_v_component_of_wind_*.nc', 'surface_net_solar_radiation_*.nc', 'surface_net_thermal_radiation_*.nc', 'surface_latent_heat_flux_*.nc', 'surface_pressure_*.nc'],
#                        'pet_at_0': ['potential_evaporation_*.nc'],
#                        'evaporation_at_0': ['total_evaporation_*.nc']
#                        }

# param_height_mappings = {'t2m': 2,
#                          'd2m': 2,
#                          'tp': 0,
#                          'sf': 0,
#                          'sro': 0,
#                          'ssro': 0,
#                          'sp': 0,
#                          'ssr': 0,
#                          'str': 0,
#                          'slhf': 0,
#                          'u10': 10,
#                          'v10': 10,
#                          'pev': 0,
#                          'e': 0
#                          }

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
