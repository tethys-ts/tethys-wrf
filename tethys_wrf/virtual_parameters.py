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


def fix_accum(da):
    """

    """
    ## Convert from accumultion to cumultive
    ds2 = da.diff('time', label='lower')
    ds3 = xr.where(ds2 < 0, da[1:], ds2)
    ds3['time'] = da['time'][:-1]

    return ds3


def calc_rh_2(wrf_xr, method=1):
    """
    Two methods to calc relative humidity at 2 meters. Copied from https://github.com/keenmisty/WRF/blob/master/matlab_scripts/functions/CalRH.m.
    I do not know where the first method comes from, but the second method is the algorithm used in ncl (wrf_rh).

    Parameters
    ----------
    wrf_xr : xr.Dataset
        The complete WRF output dataset with Q2, T2, and PSFC.
    method : int
        The method to calc RH.

    Returns
    -------
    xr.DataArray
    """
    ## Assign variables
    qv = wrf_xr['Q2']
    t = wrf_xr['T2']
    pres = wrf_xr['PSFC']

    if method == 1:
        e = pres*qv/(0.622+qv)
        E = 610.78*10**(7.5*(t-273.15)/(t-36.16))
        rh = e/E*100
    elif method == 2:
        es = 6.112*np.exp(17.67*(t-273.15)/(t-29.65))
        qvs = 0.622*es/(pres/100-(1-0.622)*es)
        rh = 100*qv/qvs
    else:
        raise ValueError('method must be either 1 or 2')

    rh = xr.where(rh > 100, 100, rh)

    return rh


def calc_wind_speed(wrf_xr, height=10):
    """
    Estimate the mean wind speed at 10 or 2 m from the V and U WRF vectors of wind speed. The 2 m method is according to the FAO 56 paper.

    Parameters
    ----------
    wrf_xr : xr.Dataset
        The complete WRF output dataset with V10 and U10.
    height : int
        The height for the estimate.

    Returns
    -------
    xr.DataArray
    """
    u10 = wrf_xr['U10']
    v10 = wrf_xr['V10']

    ws = np.sqrt(u10**2 + v10**2)

    if height == 2:
        ws = ws*4.87/(np.log(67.8*10 - 5.42))
    elif height != 10:
        raise ValueError('height must be either 10 or 2.')

    return ws


def calc_wind_speed_2(wrf_xr):
    """
    Estimate the mean wind speed at 10 or 2 m from the V and U WRF vectors of wind speed. The 2 m method is according to the FAO 56 paper.

    Parameters
    ----------
    wrf_xr : xr.Dataset
        The complete WRF output dataset with V10 and U10.
    height : int
        The height for the estimate.

    Returns
    -------
    xr.DataArray
    """
    ws = calc_wind_speed(wrf_xr, height=2)

    return ws


def calc_wind_speed_10(wrf_xr):
    """
    Estimate the mean wind speed at 10 or 2 m from the V and U WRF vectors of wind speed. The 2 m method is according to the FAO 56 paper.

    Parameters
    ----------
    wrf_xr : xr.Dataset
        The complete WRF output dataset with V10 and U10.
    height : int
        The height for the estimate.

    Returns
    -------
    xr.DataArray
    """
    ws = calc_wind_speed(wrf_xr, height=10)

    return ws


def calc_temp_2(wrf_xr, units='degC'):
    """

    """
    t2 = wrf_xr['T2']

    if units == 'degC':
        t2 = t2 - 273.15
    elif units != 'K':
        raise ValueError('units must be either degC or K.')

    return t2


def calc_surface_pressure_0(wrf_xr, units='hPa'):
    """

    """
    pres = wrf_xr['PSFC']

    if units == 'hPa':
        pres = pres * 0.01
    elif units == 'kPa':
        pres = pres * 0.001
    elif units != 'Pa':
        raise ValueError('units must be kPa, hPa, or Pa.')

    return pres


def calc_eto_0(wrf_xr):
    """

    """
    ## Assign variables
    qv = wrf_xr['Q2']
    pres = wrf_xr['PSFC']
    gamma = (0.665*10**-3)*pres/1000
    t2 = wrf_xr['T2'] - 273.15
    G = wrf_xr['GRDFLX'] * 0.0036
    R_n = (wrf_xr['SWDOWN']*wrf_xr['ALBEDO'] + wrf_xr['GLW']) * 0.0036
    # R_nl = wrf_xr['GLW'] * 0.0036
    # alb = wrf_xr['ALBEDO']
    u10 = wrf_xr['U10']
    v10 = wrf_xr['V10']
    ws2 = np.sqrt(u10**2 + v10**2)*4.87/(np.log(67.8*10 - 5.42))

    # Humidity
    e_mean = 0.6108*np.exp(17.27*t2/(t2+237.3))
    qvs = 0.622*e_mean/(pres/1000-(1-0.622)*e_mean)
    rh = 100*qv/qvs
    rh = xr.where(rh > 100, 100, rh)

    # Vapor pressure
    e_a = e_mean * rh/100
    delta = 4098*(0.6108*np.exp(17.27*t2/(t2 + 237.3)))/((t2 + 237.3)**2)
    # R_ns = (1 - alb)*R_s

    # Calc ETo
    ETo = (0.408*delta*(R_n - G) + gamma*37/(t2 + 273)*ws2*(e_mean - e_a))/(delta + gamma*(1 + 0.34*ws2))

    return ETo


def calc_precip_0(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['RAINNC']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip)

    return precip2


def calc_snow_0(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['SNOWNC']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip)

    return precip2


def calc_runoff_0(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['SFROFF']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip)

    return precip2


def calc_recharge_0(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['UDROFF']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip)

    return precip2


def calc_shortwave_0(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['SWDOWN']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip)

    return precip2


def calc_longwave_0(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['GLW']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip)

    return precip2


def calc_ground_heat_flux_0(wrf_xr):
    """

    """
    ## Assign variables
    precip = wrf_xr['GRDFLX']

    ## Convert from accumultion to cumultive
    precip2 = fix_accum(precip)

    return precip2
