#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:08:51 2021

@author: mike
"""
import xarray as xr
import pandas as pd
import numpy as np


###########################################
### Functions


def fix_accum(ds):
    """

    """
    ## Convert from accumultion to cumultive
    # ds2 = da.diff('time', label='lower')
    # ds3 = xr.where(ds2 < 0, da[1:], ds2)
    # ds3['time'] = da['time'][:-1]

    ds1 = ds.diff('time', label='lower')
    ds2 = xr.where(ds1 < 0, 0, ds1)

    return ds2


# def rh_2(wrf_xr, method=1):
#     """
#     Two methods to calc relative humidity at 2 meters. Copied from https://github.com/keenmisty/WRF/blob/master/matlab_scripts/functions/CalRH.m.
#     I do not know where the first method comes from, but the second method is the algorithm used in ncl (wrf_rh).

#     Parameters
#     ----------
#     wrf_xr : xr.Dataset
#         The complete WRF output dataset with Q2, T2, and PSFC.
#     method : int
#         The method to calc RH.

#     Returns
#     -------
#     xr.DataArray
#     """
#     ## Assign variables
#     qv = wrf_xr['Q2']
#     t = wrf_xr['T2']
#     pres = wrf_xr['PSFC']

#     if method == 1:
#         e = pres*qv/(0.622+qv)
#         E = 610.78*10**(7.5*(t-273.15)/(t-36.16))
#         rh = e/E*100
#     elif method == 2:
#         es = 6.112*np.exp(17.67*(t-273.15)/(t-29.65))
#         qvs = 0.622*es/(pres/100-(1-0.622)*es)
#         rh = 100*qv/qvs
#     else:
#         raise ValueError('method must be either 1 or 2')

#     rh = xr.where(rh > 100, 100, rh)

#     return rh


def relative_humidity(wrf_xr):
    """

    """
    rh = wrf_xr['relative_humidity']

    return rh


# def wind_speed(wrf_xr, height=10):
#     """
#     Estimate the mean wind speed at 10 or 2 m from the V and U WRF vectors of wind speed. The 2 m method is according to the FAO 56 paper.

#     Parameters
#     ----------
#     wrf_xr : xr.Dataset
#         The complete WRF output dataset with V10 and U10.
#     height : int
#         The height for the estimate.

#     Returns
#     -------
#     xr.DataArray
#     """
#     u10 = wrf_xr['U10']
#     v10 = wrf_xr['V10']

#     ws = np.sqrt(u10**2 + v10**2)

#     if height == 2:
#         ws = ws*4.87/(np.log(67.8*10 - 5.42))
#     elif height != 10:
#         raise ValueError('height must be either 10 or 2.')

#     return ws


def wind_speed(wrf_xr):
    """

    """
    u10 = wrf_xr['u_wind']
    v10 = wrf_xr['v_wind']

    ws = np.sqrt(u10**2 + v10**2)

    return ws


def wind_direction(wrf_xr):
    """

    """
    u = wrf_xr['u_wind']
    v = wrf_xr['v_wind']

    wd = np.mod(180 + (180/np.pi)*np.arctan2(v, u), 360)

    return wd

# def wind_speed_2(wrf_xr):
#     """
#     Estimate the mean wind speed at 10 or 2 m from the V and U WRF vectors of wind speed. The 2 m method is according to the FAO 56 paper.

#     Parameters
#     ----------
#     wrf_xr : xr.Dataset
#         The complete WRF output dataset with V10 and U10.
#     height : int
#         The height for the estimate.

#     Returns
#     -------
#     xr.DataArray
#     """
#     ws = wind_speed(wrf_xr, height=2)

#     return ws


# def wind_speed_10(wrf_xr):
#     """
#     Estimate the mean wind speed at 10 or 2 m from the V and U WRF vectors of wind speed. The 2 m method is according to the FAO 56 paper.

#     Parameters
#     ----------
#     wrf_xr : xr.Dataset
#         The complete WRF output dataset with V10 and U10.
#     height : int
#         The height for the estimate.

#     Returns
#     -------
#     xr.DataArray
#     """
#     ws = wind_speed(wrf_xr, height=10)

#     return ws


# def temp_2(wrf_xr, units='degC'):
#     """

#     """
#     t2 = wrf_xr['T2']

#     if units == 'degC':
#         t2 = t2 - 273.15
#     elif units != 'K':
#         raise ValueError('units must be either degC or K.')

#     return t2

def avi(data):
    """

    """
    u20 = data['u_wind'].sel(height=20)
    v20 = data['v_wind'].sel(height=20)

    ws = np.sqrt(u20**2 + v20**2)

    avi = (ws * data['pblh'])
    avi = xr.where(avi > 32000, 32000, avi)

    return avi.expand_dims('height', axis=3)


def soil_temperature(data, units='degC'):
    """

    """
    t2 = data['soil_temperature']

    if units == 'degC':
        t2 = t2 - 273.15
    elif units != 'K':
        raise ValueError('units must be either degC or K.')

    return t2


def soil_water(data):
    """

    """
    t2 = data['soil_water']

    return t2


def air_temperature(wrf_xr, units='degC'):
    """

    """
    t2 = wrf_xr['air_temperature']

    if units == 'degC':
        t2 = t2 - 273.15
    elif units != 'K':
        raise ValueError('units must be either degC or K.')

    return t2


def dew_temperature(wrf_xr, units='degC'):
    """

    """
    t2 = wrf_xr['dew_temperature']

    # if units == 'degC':
    #     t2 = t2 - 273.15
    # elif units != 'K':
    #     raise ValueError('units must be either degC or K.')

    return t2


def barometric_pressure(wrf_xr, units='hPa'):
    """

    """
    pres = wrf_xr['air_pressure']

    if units == 'hPa':
        pres = pres * 0.01
    elif units == 'kPa':
        pres = pres * 0.001
    elif units != 'Pa':
        raise ValueError('units must be kPa, hPa, or Pa.')

    return pres


# def eto_0(wrf_xr):
#     """

#     """
#     ## Assign variables
#     qv = wrf_xr['Q2']
#     pres = wrf_xr['PSFC']
#     gamma = (0.665*10**-3)*pres/1000
#     t2 = wrf_xr['T2'] - 273.15
#     G = wrf_xr['GRDFLX'] * 0.0036
#     R_n = (wrf_xr['SWDOWN']*wrf_xr['ALBEDO'] + wrf_xr['GLW']) * 0.0036
#     # R_nl = wrf_xr['GLW'] * 0.0036
#     # alb = wrf_xr['ALBEDO']
#     u10 = wrf_xr['U10']
#     v10 = wrf_xr['V10']
#     ws2 = np.sqrt(u10**2 + v10**2)*4.87/(np.log(67.8*10 - 5.42))

#     # Humidity
#     e_mean = 0.6108*np.exp(17.27*t2/(t2+237.3))
#     qvs = 0.622*e_mean/(pres/1000-(1-0.622)*e_mean)
#     rh = 100*qv/qvs
#     rh = xr.where(rh > 100, 100, rh)

#     # Vapor pressure
#     e_a = e_mean * rh/100
#     delta = 4098*(0.6108*np.exp(17.27*t2/(t2 + 237.3)))/((t2 + 237.3)**2)
#     # R_ns = (1 - alb)*R_s

#     # Calc ETo
#     ETo = (0.408*delta*(R_n - G) + gamma*37/(t2 + 273)*ws2*(e_mean - e_a))/(delta + gamma*(1 + 0.34*ws2))

#     return ETo


def precipitation(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['precipitation']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

    return precip2


def snowfall(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['snowfall']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

    return precip2


def surface_runoff(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['surface_runoff']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

    return precip2


def gw_recharge(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['gw_recharge']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

    return precip2


def downward_shortwave(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['downward_shortwave']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

    return precip2


def downward_longwave(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['downward_longwave']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

    return precip2


def ground_heat_flux(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['ground_heat_flux']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

    return precip2


# def latent_heat_flux(wrf_xr):
#     """

#     """
#     ## Assign variables
#     precip = wrf_xr['latent_heat_flux']

#     ## Convert from accumultion to cumultive
#     precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

#     return precip2


# def upward_heat_flux(wrf_xr):
#     """

#     """
#     ## Assign variables
#     precip = wrf_xr['upward_heat_flux']

#     ## Convert from accumultion to cumultive
#     precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

#     return precip2


def upward_moisture_flux(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['upward_moisture_flux']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip).assign_coords(height=0).expand_dims('height', axis=3)

    return precip2


def albedo(wrf_xr):
    """

    """
    ## Assign variables
    a = wrf_xr['albedo'].assign_coords(height=0).expand_dims('height', axis=3)

    return a


def specific_humidity(wrf_xr):
    """

    """
    ## Assign variables
    a = wrf_xr['water_vapor_mixing_ratio']
    q = (a/(1 + a)) * 1000

    return q


def surface_emissivity(wrf_xr):
    """

    """
    ## Assign variables
    a = wrf_xr['surface_emissivity'].assign_coords(height=0).expand_dims('height', axis=3)

    return a



########################################
### dicts

wrf_variables_dict = {'air_temperature': {'main': 'tk', 'surface': 'T2', 'surface_height': 1.999},
                 'u_wind': {'main': 'ua', 'surface': 'U10', 'surface_height': 9.999},
                 'v_wind': {'main': 'va', 'surface': 'V10', 'surface_height': 9.999},
                 'relative_humidity': {'main': 'rh', 'surface': 'rh2', 'surface_height': 1.999},
                 'dew_temperature': {'main': 'td', 'surface': 'td2', 'surface_height': 1.999},
                 'air_pressure': {'main': 'p', 'surface': 'PSFC', 'surface_height': 0},
                 'precipitation': {'main': 'RAINNC'},
                 'snowfall': {'main': 'SNOWNC'},
                 'surface_runoff': {'main': 'SFROFF'},
                 'gw_recharge': {'main': 'UDROFF'},
                 'downward_shortwave': {'main': 'SWDOWN'},
                 'downward_longwave': {'main': 'GLW'},
                 'ground_heat_flux': {'main': 'GRDFLX'},
                 'soil_temperature': {'main': 'TSLB'},
                 'soil_water': {'main': 'SMOIS'},
                 'pblh': {'main': 'PBLH'},
                 'albedo': {'main': 'ALBEDO'},
                 'surface_emissivity': {'main': 'EMISS'},
                 # 'terrain_height': {'main': 'HGT'},
                 # 'upward_heat_flux': {'main': 'HFX'},
                 # 'upward_moisture_flux': {'main': 'QFX'},
                 # 'latent_heat_flux': {'main': 'LH'},
                 'water_vapor_mixing_ratio': {'main': 'QVAPOR', 'surface': 'Q2', 'surface_height': 1.999},
                 }

func_dict = {
    'air_temperature': {'variables': ['air_temperature'],
             'function': air_temperature,
             'metadata':
                 {'feature': 'atmosphere',
                  'parameter': 'temperature',
                  'aggregation_statistic': 'instantaneous',
                  'units': 'degC',
                  'cf_standard_name': 'air_temperature',
                  'wrf_standard_name': 'T',
                  'precision': 0.01,
                  'properties':
                    {'encoding':
                      {'temperature':
                        {'scale_factor': 0.01,
                        'dtype': 'int16',
                        '_FillValue': -9999}
                        }
                          }
                        }
                  },
    'relative_humidity': {'variables': ['relative_humidity'],
                          'function': relative_humidity,
                          'metadata':
                              {'feature': 'atmosphere',
                               'parameter': 'relative_humidity',
                               'aggregation_statistic': 'instantaneous',
                               'units': '%',
                               'cf_standard_name': 'relative_humidity',
                               'wrf_standard_name': 'RH',
                               'precision': 0.01,
                               'properties':
                                 {'encoding':
                                   {'relative_humidity':
                                     {'scale_factor': 0.01,
                                     'dtype': 'int16',
                                     '_FillValue': -9999}
                                     }
                                       }
                                     }
                },
    'wind_speed': {'variables': ['u_wind', 'v_wind'],
                   'function': wind_speed,
                   'metadata':
                       {'feature': 'atmosphere',
                        'parameter': 'wind_speed',
                        'aggregation_statistic': 'instantaneous',
                        'units': 'm/s',
                        'cf_standard_name': 'wind_speed',
                        'wrf_standard_name': 'UV',
                        'precision': 0.01,
                        'properties':
                          {'encoding':
                            {'wind_speed':
                              {'scale_factor': 0.01,
                              'dtype': 'int16',
                              '_FillValue': -9999}
                              }
                                }
                              }
                        },
    'wind_direction': {'variables': ['u_wind', 'v_wind'],
                       'function': wind_direction,
                       'metadata':
                           {'feature': 'atmosphere',
                            'parameter': 'wind_direction',
                            'aggregation_statistic': 'instantaneous',
                            'units': 'deg',
                            'cf_standard_name': 'wind_from_direction',
                            'wrf_standard_name': 'UV',
                            'precision': 0.1,
                            'properties':
                              {'encoding':
                                {'wind_direction':
                                  {'scale_factor': 0.1,
                                  'dtype': 'int16',
                                  '_FillValue': -999}
                                  }
                                    }
                                  }
                        },
    'dew_temperature': {'variables': ['dew_temperature'],
                 'function': dew_temperature,
                 'metadata':
                     {'feature': 'atmosphere',
                      'parameter': 'temperature_dew_point',
                      'aggregation_statistic': 'instantaneous',
                      'units': 'degC',
                      'cf_standard_name': 'dew_point_temperature',
                      'wrf_standard_name': 'TD',
                      'precision': 0.01,
                      'properties':
                        {'encoding':
                          {'temperature_dew_point':
                            {'scale_factor': 0.01,
                            'dtype': 'int16',
                            '_FillValue': -9999}
                            }
                              }
                            }
                      },
    'barometric_pressure': {'variables': ['air_pressure'],
                         'function': barometric_pressure,
                         'metadata':
                             {'feature': 'atmosphere',
                              'parameter': 'barometric_pressure',
                              'aggregation_statistic': 'instantaneous',
                              'units': 'hPa',
                              'cf_standard_name': 'air_pressure',
                              'wrf_standard_name': 'P',
                              'precision': 0.1,
                              'properties':
                                {'encoding':
                                  {'barometric_pressure':
                                    {'scale_factor': 0.1,
                                    'dtype': 'int16',
                                    '_FillValue': -9999}
                                    }
                                      }
                                    }
                      },
    'precipitation': {'variables': ['precipitation'],
               'function': precipitation,
               'metadata':
                   {'feature': 'atmosphere',
                    'parameter': 'precipitation',
                    'aggregation_statistic': 'cumulative',
                    'units': 'mm',
                    'cf_standard_name': 'precipitation_amount',
                    'wrf_standard_name': 'RAINNC',
                    'precision': 0.1,
                    'properties':
                      {'encoding':
                        {'precipitation':
                          {'scale_factor': 0.1,
                          'dtype': 'int16',
                          '_FillValue': -9999}
                          }
                            }
                          }
                     },
    'snowfall': {'variables': ['snowfall'],
                 'function': snowfall,
                 'metadata':
                     {'feature': 'atmosphere',
                      'parameter': 'snow_depth',
                      'aggregation_statistic': 'cumulative',
                      'units': 'mm',
                      'cf_standard_name': 'thickness_of_snowfall_amount',
                      'wrf_standard_name': 'SNOWNC',
                      'precision': 0.1,
                      'properties':
                        {'encoding':
                          {'snow_depth':
                            {'scale_factor': 0.1,
                            'dtype': 'int16',
                            '_FillValue': -9999}
                            }
                              }
                            }
                   },
    'downward_longwave': {'variables': ['downward_longwave'],
                          'function': downward_longwave,
                          'metadata':
                              {'feature': 'atmosphere',
                               'parameter': 'radiation_incoming_longwave',
                               'aggregation_statistic': 'cumulative',
                               'units': 'W/m^2',
                               'cf_standard_name': 'surface_downwelling_longwave_flux_in_air',
                               'wrf_standard_name': 'GLW',
                               'precision': 0.1,
                               'properties':
                                 {'encoding':
                                   {'radiation_incoming_longwave':
                                     {'scale_factor': 0.1,
                                     'dtype': 'int16',
                                     '_FillValue': -9999}
                                     }
                                       }
                                     }
                       },
    'downward_shortwave': {'variables': ['downward_shortwave'],
                           'function': downward_shortwave,
                           'metadata':
                               {'feature': 'atmosphere',
                                'parameter': 'radiation_incoming_shortwave',
                                'aggregation_statistic': 'cumulative',
                                'units': 'W/m^2',
                                'cf_standard_name': 'surface_downwelling_shortwave_flux_in_air',
                                'wrf_standard_name': 'SWDOWN',
                                'precision': 0.1,
                                'properties':
                                  {'encoding':
                                    {'radiation_incoming_shortwave':
                                      {'scale_factor': 0.1,
                                      'dtype': 'int16',
                                      '_FillValue': -9999}
                                      }
                                        }
                                      }

                        },
    'surface_runoff': {'variables': ['surface_runoff'],
                       'function': surface_runoff,
                       'metadata':
                           {'feature': 'pedosphere',
                            'parameter': 'runoff',
                            'aggregation_statistic': 'cumulative',
                            'units': 'mm',
                            'cf_standard_name': 'runoff_amount',
                            'wrf_standard_name': 'SFROFF',
                            'precision': 0.1,
                            'properties':
                              {'encoding':
                                {'runoff':
                                  {'scale_factor': 0.1,
                                  'dtype': 'int16',
                                  '_FillValue': -9999}
                                  }
                                    }
                                  }
                        },
    'gw_recharge': {'variables': ['gw_recharge'],
                    'function': gw_recharge,
                    'metadata':
                        {'feature': 'pedosphere',
                         'parameter': 'recharge_groundwater',
                         'aggregation_statistic': 'cumulative',
                         'units': 'mm',
                         'cf_standard_name': 'subsurface_runoff_amount',
                         'wrf_standard_name': 'UDROFF',
                         'precision': 0.1,
                         'properties':
                           {'encoding':
                             {'recharge_groundwater':
                               {'scale_factor': 0.1,
                               'dtype': 'int16',
                               '_FillValue': -9999}
                               }
                                 }
                               }
                        },
    'ground_heat_flux': {'variables': ['ground_heat_flux'],
                         'function': ground_heat_flux,
                         'metadata':
                             {'feature': 'pedosphere',
                              'parameter': 'ground_heat_flux',
                              'aggregation_statistic': 'cumulative',
                              'units': 'W/m^2',
                              'cf_standard_name': 'downward_heat_flux_in_soil',
                              'wrf_standard_name': 'GRDFLX',
                              'precision': 0.1,
                              'properties':
                                {'encoding':
                                  {'ground_heat_flux':
                                    {'scale_factor': 0.1,
                                    'dtype': 'int16',
                                    '_FillValue': -9999}
                                    }
                                      }
                                    }
                        },
    # 'latent_heat_flux': {'variables': ['latent_heat_flux'],
    #                      'function': latent_heat_flux,
    #                      'metadata':
    #                          {'feature': 'pedosphere',
    #                           'parameter': 'latent_heat_flux',
    #                           'aggregation_statistic': 'cumulative',
    #                           'units': 'W/m^2',
    #                           'cf_standard_name': 'downward_heat_flux_in_soil',
    #                           'wrf_standard_name': 'GRDFLX',
    #                           'precision': 0.1,
    #                           'properties':
    #                             {'encoding':
    #                               {'ground_heat_flux':
    #                                 {'scale_factor': 0.1,
    #                                 'dtype': 'int16',
    #                                 '_FillValue': -9999}
    #                                 }
    #                                   }
    #                                 }
    #                     },
    # 'upward_heat_flux': {'variables': ['ground_heat_flux'],
    #                      'function': ground_heat_flux,
    #                      'metadata':
    #                          {'feature': 'pedosphere',
    #                           'parameter': 'ground_heat_flux',
    #                           'aggregation_statistic': 'cumulative',
    #                           'units': 'W/m^2',
    #                           'cf_standard_name': 'downward_heat_flux_in_soil',
    #                           'wrf_standard_name': 'GRDFLX',
    #                           'precision': 0.1,
    #                           'properties':
    #                             {'encoding':
    #                               {'ground_heat_flux':
    #                                 {'scale_factor': 0.1,
    #                                 'dtype': 'int16',
    #                                 '_FillValue': -9999}
    #                                 }
    #                                   }
    #                                 }
    #                     },
    # 'downward_heat_flux': {'variables': ['ground_heat_flux'],
    #                      'function': ground_heat_flux,
    #                      'metadata':
    #                          {'feature': 'pedosphere',
    #                           'parameter': 'ground_heat_flux',
    #                           'aggregation_statistic': 'cumulative',
    #                           'units': 'W/m^2',
    #                           'cf_standard_name': 'downward_heat_flux_in_soil',
    #                           'wrf_standard_name': 'GRDFLX',
    #                           'precision': 0.1,
    #                           'properties':
    #                             {'encoding':
    #                               {'ground_heat_flux':
    #                                 {'scale_factor': 0.1,
    #                                 'dtype': 'int16',
    #                                 '_FillValue': -9999}
    #                                 }
    #                                   }
    #                                 }
    #                     },
    'soil_temperature': {'variables': ['soil_temperature'],
                  'function': soil_temperature,
                  'metadata':
                      {'feature': 'pedosphere',
                       'parameter': 'temperature',
                       'aggregation_statistic': 'instantaneous',
                       'units': 'degC',
                       'cf_standard_name': 'soil_temperature',
                       'wrf_standard_name': 'TSLB',
                       'precision': 0.01,
                       'properties':
                         {'encoding':
                           {'temperature':
                             {'scale_factor': 0.01,
                             'dtype': 'int16',
                             '_FillValue': -9999}
                             }
                               }
                             }
                  },
    'soil_water': {'variables': ['soil_water'],
                   'function': soil_water,
                   'metadata':
                       {'feature': 'pedosphere',
                        'parameter': 'volumetric_water_content',
                        'aggregation_statistic': 'instantaneous',
                        'units': 'm^3/m^3',
                        'cf_standard_name': 'mass_content_of_water_in_soil',
                        'wrf_standard_name': 'SMOIS',
                        'precision': 0.0001,
                        'properties':
                          {'encoding':
                            {'volumetric_water_content':
                              {'scale_factor': 0.0001,
                              'dtype': 'int16',
                              '_FillValue': -9999}
                              }
                                }
                              }
                  },
    'avi': {'variables': ['u_wind', 'v_wind', 'pblh'],
            'function': avi,
            'metadata':
                {'feature': 'atmosphere',
                 'parameter': 'air_ventilation_index',
                 'aggregation_statistic': 'instantaneous',
                 'units': 'm^2/s',
                 # 'cf_standard_name': 'air_pressure',
                 # 'wrf_standard_name': 'PSFC',
                 'description': 'The air ventilation index is the product of the mixing height (m) and the transport wind speed (m/s) used as a tool for air quality forecasters to determine the potential of the atmosphere to disperse contaminants such as smoke or smog. We have used the product of the PBLH and 20m wind speed. This is comparible to the air ventilation index used by the University of Washington.',
                 'precision': 1,
                 'properties':
                   {'encoding':
                     {'air_ventilation_index':
                       {'scale_factor': 1,
                       'dtype': 'int16',
                       '_FillValue': -9999}
                       }
                         }
                       }
                  },
    'surface_emissivity': {'variables': ['surface_emissivity'],
                           'function': surface_emissivity,
                           'metadata':
                               {'feature': 'pedosphere',
                                'parameter': 'surface_emissivity',
                                'aggregation_statistic': 'instantaneous',
                                'units': '',
                                'cf_standard_name': 'surface_longwave_emissivity',
                                'wrf_standard_name': 'EMISS',
                                'precision': 0.0001,
                                'properties':
                                  {'encoding':
                                    {'surface_emissivity':
                                      {'scale_factor': 0.0001,
                                      'dtype': 'int16',
                                      '_FillValue': -9999}
                                      }
                                        }
                                      }
                  },
    'specific_humidity': {'variables': ['water_vapor_mixing_ratio'],
                           'function': specific_humidity,
                           'metadata':
                               {'feature': 'atmosphere',
                                'parameter': 'specific_humidity',
                                'aggregation_statistic': 'instantaneous',
                                'units': 'g/kg',
                                'cf_standard_name': 'specific_humidity',
                                # 'wrf_standard_name': 'QVAPOR',
                                'precision': 0.001,
                                'properties':
                                  {'encoding':
                                    {'specific_humidity':
                                      {'scale_factor': 0.001,
                                      'dtype': 'int16',
                                      '_FillValue': -9999}
                                      }
                                        }
                                      }
                  },
    'albedo': {'variables': ['albedo'],
                           'function': albedo,
                           'metadata':
                               {'feature': 'pedosphere',
                                'parameter': 'albedo',
                                'aggregation_statistic': 'instantaneous',
                                'units': '',
                                'cf_standard_name': 'surface_albedo',
                                'wrf_standard_name': 'ALBEDO',
                                'precision': 0.0001,
                                'properties':
                                  {'encoding':
                                    {'albedo':
                                      {'scale_factor': 0.0001,
                                      'dtype': 'int16',
                                      '_FillValue': -9999}
                                      }
                                        }
                                      }
                  },
    }
