#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:32:40 2021

@author: mike
"""
import os
import xarray as xr
import pandas as pd
import numpy as np
# from netCDF4 import Dataset
# from wrf import getvar, interpline, CoordPair, xy_to_ll, ll_to_xy
# import wrf
import copy
import orjson
import tethys_utils as tu
import salem
from multiprocessing.pool import ThreadPool, Pool
import zstandard as zstd

##############################################
### Parameters

ds_cols = ['feature', 'parameter', 'frequency_interval', 'aggregation_statistic', 'units', 'wrf_standard_name', 'cf_standard_name', 'scale_factor']

base_dir = os.path.realpath(os.path.dirname(__file__))

########################################
### Main class


class WRF(object):
    """

    """

    ## Initialization
    def __init__(self, wrf_nc):
        """

        """
        xr1 = salem.open_mf_wrf_dataset(wrf_nc)
        xr1 = xr1.drop('xtime', errors='ignore')

        ### Pre-process the station data
        ## Station_ids
        lat = xr1['lat'].values
        lon = xr1['lon'].values

        def make_stn_id(x, y):
            stn_id = tu.assign_station_id(tu.create_geometry([x, y]))
            return stn_id

        vfunc = np.vectorize(make_stn_id)
        stn_ids = vfunc(lon, lat)

        xr1.coords['station_id'] = (('south_north', 'west_east'), stn_ids)

        ## Altitude
        alt = xr1['HGT'].isel(time=0)
        xr1.coords['altitude'] = (('south_north', 'west_east'), alt)

        ## Determine frequency interval
        freq = xr1['time'][:5].to_index()[:5].inferred_freq

        if freq is None:
            raise ValueError('The time frequency could not be determined from the netcdf file.')

        ### Read in mapping table
        wrf_mapping = pd.read_csv(os.path.join(base_dir, 'wrf_mappings.csv'))
        wrf_mapping.set_index('parameter_code', inplace=True)
        wrf_mapping['frequency_interval'] = freq

        ### Process base datasets
        dsb = wrf_mapping[ds_cols].rename(columns={'scale_factor': 'precision'}).to_dict('index')

        ### Set attrs
        setattr(self, 'data', xr1)
        setattr(self, 'mappings', wrf_mapping)
        setattr(self, 'datasets', dsb)

        pass


    def __repr__(self):
        return repr(self.data)


    @staticmethod
    def calc_rh2(wrf_xr, method=1):
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


    @staticmethod
    def calc_wind_speed(wrf_xr, height=2):
        """
        Estimate the mean wind speed at 10 or 2 m from the V and U WRF vectors of wind speed. The 2 m method is according to the FAO 56 paper.

        Parameters
        ----------
        wrf_xr : xr.Dataset
            The complete WRF output dataset with Q2, T2, and PSFC.
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


    @staticmethod
    def calc_temp2(wrf_xr, units='degC'):
        """

        """
        t2 = wrf_xr['T2']

        if units == 'degC':
            t2 = t2 - 273.15
        elif units != 'K':
            raise ValueError('units must be either degC or K.')

        return t2


    @staticmethod
    def calc_surface_pressure(wrf_xr, units='hPa'):
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


    def get_results(self, parameter_code=None, station_id_index=True):
        """

        """
        if (not hasattr(self, 'parameter_code')) and (parameter_code is None):
            raise ValueError('Either define a parameter_code or run the build_dataset method prior to the get_results method.')

        if isinstance(parameter_code, str):
            map1 = self.mappings.loc[parameter_code].copy()
        else:
            map1 = self.mappings.loc[self.parameter_code].copy()

        if isinstance(map1['function'], str):
            meth = getattr(self, map1['function'])
            res1 = meth(self.data)
        else:
            res1 = self.data[map1['wrf_standard_name']]

        res1.name = self.param_dataset['parameter']
        if station_id_index:
            res2 = res1.stack(id=['west_east', 'south_north']).set_index(id='station_id').rename(id='station_id').copy()
        else:
            res2 = res1.copy()

        setattr(self, 'param_data', res2)
        setattr(self, 'param_map', map1)

        return res2


    def build_dataset(self, parameter_code, owner, product_code, data_license, attribution, method='simulation', result_type="time_series_grid_simulation"):
        """

        """
        if parameter_code not in self.datasets:
            raise ValueError('parameter_code ' + parameter_code + ' is not available. Check the datasets dict for the available parameter codes.')

        ## Remove prior parameter_code
        if hasattr(self, 'parameter_code'):
            delattr(self, 'parameter_code')
        if hasattr(self, 'param_dataset'):
            delattr(self, 'param_dataset')
        if hasattr(self, 'param_data'):
            delattr(self, 'param_data')
        if hasattr(self, 'param_map'):
            delattr(self, 'param_map')

        ## Build the dataset
        datasets = self.datasets

        ds = datasets[parameter_code].copy()
        ds.update({'owner': owner, 'product_code': product_code, 'license': data_license, 'attribution': attribution, 'utc_offset': '0H', "result_type": result_type, 'method': method})

        ##  Assign the dataset_id
        ds1 = tu.assign_ds_ids([ds])[0]

        setattr(self, 'param_dataset', ds1)
        setattr(self, 'parameter_code', parameter_code)

        return ds1


    def save_results(self, conn_config, bucket, public_url=None, run_date=None, threads=30):
        """

        """
        start1 = pd.Timestamp.now('utc').round('s')
        print('start: ' + str(start1))

        ## prepare all of the input data
        if not hasattr(self, 'param_dataset'):
            raise ValueError('Run build dataset before save_results.')

        ds1 = self.param_dataset.copy()

        if not hasattr(self, 'param_data'):
            data1 = self.get_results().copy()
        else:
            data1 = self.param_data.copy()

        map1 = self.param_map.copy()
        data1['height']= map1['height']
        encoding = {ds1['parameter']: map1[['scale_factor', 'add_offset', 'dtype', '_FillValue']].dropna().to_dict()}
        attrs = {ds1['parameter']: ds1.copy()}

        run_date_key = tu.make_run_date_key(run_date)

        s3 = tu.s3_connection(conn_config, threads)

        ## Create the iterator for starmap
        stn_ids = data1.station_id.values.tolist()
        inputs = iter((data1.sel(station_id=s).copy(), attrs, encoding, run_date_key, conn_config, bucket, s3, public_url) for s in stn_ids)
        # inputs = iter((data1.sel(station_id=s).copy(), attrs, encoding, run_date_key, conn_config, bucket, s3, public_url) for s in stn_ids)

        ## Run the threadpool
        # output = ThreadPool(threads).imap_unordered(update_result, results)
        with ThreadPool(threads) as pool:
            # output = pool.starmap(self._save_results, inputs, chunksize=90)
            output = pool.imap_unordered(self._save_results, inputs, chunksize=900)
            pool.close()
            pool.join()

        ## Final station and dataset agg
        ds_new = tu.put_remote_dataset(s3, bucket, ds1)
        ds_stations = tu.put_remote_agg_stations(s3, bucket, ds1['dataset_id'])

        ### Aggregate all datasets for the bucket
        ds_all = tu.put_remote_agg_datasets(s3, bucket)

        end1 = pd.Timestamp.now('utc').round('s')
        print('End: ' + str(end1))


    @staticmethod
    def _save_results(val):
        """

        """
        data, attrs, encoding, run_date, conn_config, bucket, s3, public_url = val
        stn_id = str(data['station_id'].values)

        print(stn_id)

        lat = round(float(data['lat'].values), 6)
        lon = round(float(data['lon'].values), 6)
        alt = round(float(data['altitude'].values), 3)

        geo1 = {"coordinates": [lon, lat], "type": "Point"}

        stn_data = {'geometry': geo1, 'altitude': alt, 'station_id': stn_id, 'virtual_station': True}

        df1 = data.drop(['lat', 'lon', 'altitude', 'station_id']).to_dataframe().reset_index()
        df2 = df1.drop_duplicates('time', keep='last').set_index(['time', 'height'])
        parameter = df2.columns[0]
        ds_id = attrs[parameter]['dataset_id']

        new1 = tu.data_to_xarray(df2, stn_data, parameter, attrs, encoding, virtual_station=True)

        up1 = tu.compare_datasets_from_s3(conn_config, bucket, new1, add_old=True, last_run_date_key=run_date, public_url=public_url)

        ## Save results
        if isinstance(up1, xr.Dataset) and (len(up1[parameter].time) > 0):

            print('Save results')
            key_dict = {'dataset_id': ds_id, 'station_id': stn_id, 'run_date': run_date}

            new_key = tu.key_patterns['results'].format(**key_dict)

            cctx = zstd.ZstdCompressor(level=1)
            c_obj = cctx.compress(up1.to_netcdf())

            s3.put_object(Body=c_obj, Bucket=bucket, Key=new_key, ContentType='application/zstd', Metadata={'run_date': run_date})

            up1.close()

            ## Process stn data
            print('Save station data')

            stn_m = tu.process_station_summ(ds_id, stn_id, conn_config, bucket, mod_date=run_date, public_url=public_url)

            stn4 = orjson.loads(stn_m.json(exclude_none=True))
            up_stns = tu.put_remote_station(s3, bucket, stn4, run_date=run_date)

        else:
            print('No new data to update')

        ## Get rid of big objects
        data.close()
        new1.close()
        new1 = None
        up1 = None













# def open_wrf_nc(nc, parallel=True, chunks=None):
#     """
#     Parameters
#     ----------
#     nc : str
#         Path to one or more netcdf wrf files associated with a single wrf model run.
#         See xarray with dask (open_mfdataset) for more info.
#     """



# def make_run_date(run_date=None):
#     """

#     """
#     if run_date is None:
#         run_date1 = pd.Timestamp.today(tz='utc')
#     else:
#         run_date1 = pd.Timestamp(run_date)

#     run_date_key = run_date1.strftime('%Y%m%dT%H%M%SZ')

#     return run_date1, run_date_key


# def get_mapping(wrf_parameter):
#     """

#     """
#     ### Get necessary parameters
#     wrf_params = wrf_mapping['wrf_standard_name'].unique()

#     if wrf_parameter not in wrf_params:
#         raise ValueError('wrf_parameter must be one of ' + ', '.join(wrf_params))

#     ds_df = wrf_mapping[wrf_mapping['wrf_standard_name'] == wrf_parameter].iloc[0].copy()

#     return ds_df


# def get_encoding(wrf_parameter):
#     """

#     """
#     enc_cols = ['scale_factor', 'add_offset', 'dtype', '_FillValue']

#     ds_df = get_mapping(wrf_parameter)

#     encoding = ds_df[enc_cols].dropna().to_dict()

#     return encoding


# def convert_times(wrf_xr):
#     """

#     """
#     times = pd.to_datetime(np.char.replace(wrf_xr['Times'].values.astype(str), '_', ' ')).drop_duplicates().sort_values()

#     return times


# def get_freq(times):
#     """

#     """
#     ### Determine time interval
#     freq = times.inferred_freq

#     if freq is None:
#         raise ValueError('The time frequency could not be determined from the netcdf.')

#     return freq


# def get_dataset(wrf_parameter, owner, product_code, data_license, attribution, frequency_interval):
#     """

#     """
#     ds_cols = ['wrf_standard_name', 'parameter', 'feature', 'aggregation_statistic', 'units', 'cf_standard_name']

#     ds_df = get_mapping(wrf_parameter)

#     ds = ds_df[ds_cols].to_dict()
#     ds.update({'owner': owner, 'product_code': product_code, 'license': data_license, 'attribution': attribution, 'frequency_interval': frequency_interval, 'utc_offset': '0H', "result_type": "time_series_grid_simulation", 'method': 'simulation'})

#     ###  Assign the dataset_id
#     ds1 = tu.assign_ds_ids([ds])[0]

#     return ds1


# def process_stations(wrf_xr, to_dict=False):
#     """

#     """

#     if 'lat' in wrf_xr:
#         df1 = wrf_xr[['lat', 'lon', 'HGT']].to_dataframe().reset_index()[['lat', 'lon', 'HGT']].drop_duplicates(['lat', 'lon']).rename(columns={'HGT': 'altitude'})
#     else:
#         df1 = wrf_xr[['XLAT', 'XLONG', 'HGT']].to_dataframe().reset_index()[['XLAT', 'XLONG', 'HGT']].drop_duplicates(['XLAT', 'XLONG']).rename(columns={'XLAT': 'lat', 'XLONG': 'lon', 'HGT': 'altitude'})

#     df1['lat'] = df1['lat'].astype(float)
#     df1['lon'] = df1['lon'].astype(float)
#     df1['altitude'] = df1['altitude'].astype(float)

#     df2 = tu.process_stations_df(df1)

#     if to_dict:
#         stns_list = df2.drop(['lat', 'lon'], axis=1).to_dict('records')
#         o = [s.update({'virtual_station': True}) for s in stns_list]
#         stns_dict = tu.process_stations_base(stns_list)

#         return stns_dict
#     else:
#         return df2



# def wrf_process_results(nc, dataset, run_date):
#     """

#     """
#     ### Get necessary parameters
#     wrf_parameter = dataset['wrf_standard_name']

#     ds_df = get_wrf_mapping(wrf_parameter)


#     ds = copy.deepcopy(wrf_dataset_mapping[wrf_parameter])
#     ds.update(general_dataset_data)
#     height = ds.pop('height')
#     parameter = ds['parameter']

#     ### Get data
#     xr1 = xr.open_dataset(nc)

#     times = pd.to_datetime(np.char.replace(xr1['Times'].values.astype(str), '_', ' ')).drop_duplicates().sort_values()

#     df2 = xr1[['Times', wrf_parameter]].to_dataframe().reset_index()[['Times', 'XLAT', 'XLONG', wrf_parameter]]
#     df2['Times'] = pd.to_datetime(np.char.replace(df2.Times.values.astype(str), '_', ' '))

#     df2.rename(columns={'Times': 'time', 'XLAT': 'lat', 'XLONG': 'lon', wrf_parameter: parameter}, inplace=True)

#     ### Determine time interval
#     times = df2['time'].drop_duplicates().sort_values()

#     freq = times.dt.freq

#     if freq is not None:
#         ds.update({"frequency_interval": freq})
#     else:
#         ds.update({"frequency_interval": "H"})




#     conversion_eq = ds_df['conversion_eq']



# def process_wrf_nc(ncs, wrf_parameter, dataset, conn_config, run_date=None):
#     """

#     """
#     if wrf_parameter in wrf_python_parameters:

#     else:
#         ds1 = xr.open_dataset([os.path.join(base_path, nc) for nc in ncs])
