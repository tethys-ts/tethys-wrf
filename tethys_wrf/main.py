#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 10:32:40 2021

@author: mike
"""
import os
import glob
import xarray as xr
import pandas as pd
import numpy as np
import tethys_data_models as tdm
# from netCDF4 import Dataset
# from wrf import getvar, interpline, CoordPair, xy_to_ll, ll_to_xy
# import wrf
# from tethys_wrf import virtual_parameters as vp
# from tethys_utils.misc import parse_data_paths
# from . import virtual_parameters as vp
import copy
# import orjson
# import tethys_utils as tu
from tethys_wrf import sio, utils, virtual_parameters
# from hydrointerp import Interp
# from .utils import param_func_mappings
# from glob import glob
import concurrent.futures
import multiprocessing as mp
from netCDF4 import Dataset
from wrf import getvar, interplevel
import pathlib
import tethys_utils as tu
from typing import List, Optional, Dict, Union

##############################################
### Parameters

# ds_cols = ['feature', 'parameter', 'frequency_interval', 'aggregation_statistic', 'units', 'wrf_standard_name', 'cf_standard_name', 'scale_factor']

# base_dir = os.path.realpath(os.path.dirname(__file__))


#############################################
### Functions


def determine_heights(file):
    """

    """
    ncfile = Dataset(file)
    heights = getvar(ncfile, "height_agl").isel(south_north=0, west_east=0).data

    ncfile.close()
    del ncfile

    return heights


def preprocess_data_structure(nc_path, variables, heights, time_index_bool=None):
    """

    """
    new_paths = []
    # base_dims = ('time', 'south_north', 'west_east')
    xr1 = sio.open_wrf_dataset(nc_path)

    ## Get first timestamp for file naming
    times = xr1['time'].copy()
    time1 = pd.Timestamp(times.values[0])
    time1_str = time1.strftime('%Y%m%d%H%M%S')

    ## Get spatial dimensions
    x = xr1['west_east'].copy().load()
    y = xr1['south_north'].copy().load()

    ncfile = Dataset(nc_path)

    h = utils.get_wrf_var(ncfile, "height_agl", times)

    ## Iterate through variables
    for v in variables:
        var_dict = virtual_parameters.wrf_variables_dict[v]

        if 'soil' in v:
            xr2 = xr1[var_dict['main']].rename({'soil_layers': 'height'}).copy().load()
        else:
            xr2 = utils.get_wrf_var(ncfile, var_dict['main'], times)
            proj = xr2.attrs.pop('projection').proj4()

        if 'surface' in var_dict:
            v2 = utils.get_wrf_var(ncfile, var_dict['surface'], times).expand_dims('bottom_top', axis=1)
            v3 = xr.concat([v2, xr2], dim='bottom_top')
            h2 = v2.copy()
            h2[:] = var_dict['surface_height']
            h3 = xr.concat([h2, h], dim='bottom_top')

            sh = round(var_dict['surface_height'])

            heights2 = [h for h in heights if h >= sh]

            xr2 = interplevel(v3, h3, heights2, squeeze=False)

            del v2
            del v3
            del h2
            del h3

            ## Transpose, sort, and rename
            xr2.name = v
            _ = [xr2.attrs.pop(a) for a in ['projection', 'missing_value', '_FillValue', 'stagger'] if a in xr2.attrs]

            xr3 = xr2.to_dataset().drop(['XLONG', 'XLAT', 'XTIME'], errors='ignore')
            xr3 = xr3.assign_coords({'south_north': y, 'west_east': x})

            xr3 = xr3.transpose('Time', 'south_north', 'west_east', 'level')
            xr3 = xr3.rename({'Time': 'time', 'level': 'height', 'south_north': 'y', 'west_east': 'x'}).sortby(['time', 'y', 'x', 'height'])
        else:
            ## Transpose, sort, and rename
            xr2.name = v
            _ = [xr2.attrs.pop(a) for a in ['projection', 'missing_value', '_FillValue', 'stagger'] if a in xr2.attrs]

            xr3 = xr2.to_dataset().drop(['XLONG', 'XLAT', 'XTIME', 'lat', 'lon'], errors='ignore')
            if 'west_east' not in xr3:
                xr3 = xr3.assign_coords({'south_north': y, 'west_east': x}).rename({'Time': 'time'})
                xr3 = xr3.transpose('time', 'south_north', 'west_east')
                xr3 = xr3.rename({'south_north': 'y', 'west_east': 'x'}).sortby(['time', 'y', 'x'])
            else:
                xr3 = xr3.transpose('time', 'south_north', 'west_east', 'height')
                xr3 = xr3.rename({'south_north': 'y', 'west_east': 'x'}).sortby(['time', 'y', 'x', 'height'])

        ## Remove time duplicates if necessary
        if time_index_bool is not None:
            xr3 = xr3.sel(time=time_index_bool)

        xr3.attrs = {'proj4': proj}

        ## Save data
        new_file_name_str = '{var}_proj_{date}.nc'
        path1 = pathlib.Path(nc_path)
        base_path = path1.parent
        new_file_name = new_file_name_str.format(var=v, date=time1_str)
        new_path = base_path.joinpath(new_file_name)
        xr3.to_netcdf(new_path, unlimited_dims=['time'])
        new_paths.append(str(new_path))

        xr2.close()
        del xr2

        xr3.close()
        del xr3

    ncfile.close()
    del ncfile

    h.close()
    del h

    xr1.close()
    del xr1

    ## delete old file
    os.remove(nc_path)

    return new_paths


########################################
### wrf class

func_dict = virtual_parameters.func_dict

# variables = []
# for k, v in func_dict.items():
#     vars1 = v['variables']
#     variables.extend(vars1)

# variables = list(set(variables))
# variables.sort()

avail_datasets = list(func_dict.keys())
avail_datasets.sort()


class WRF(tu.Grid):
    """

    """
    _func_dict = func_dict

    available_dataset_codes = avail_datasets

    @staticmethod
    def parse_proj(file, **kwargs):
        """

        """
        xr1 = sio.open_wrf_dataset(file, **kwargs)

        proj = xr1.attrs['pyproj_srs']

        xr1.close()
        del xr1

        return proj


    @staticmethod
    def determine_duplicate_times(nc_paths):
        """

        """
        if isinstance(nc_paths, str):
            nc_paths1 = glob.glob(nc_paths)
        elif isinstance(nc_paths, list):
            nc_paths1 = nc_paths

        nc_paths1.sort()

        ## Determine duplicate times
        if len(nc_paths1) > 1:
            xr1 = sio.open_wrf_mfdataset(nc_paths1[:2])

            time_bool = xr1.get_index('time').duplicated(keep='first')

            xr1.close()
            del xr1

            time_len = int(len(time_bool)/2)
            time_index_bool = ~time_bool[time_len:]
        else:
            raise ValueError('nc_paths must have > 1 files.')

        return time_index_bool


    def variable_processing(self, nc_paths, heights, time_index_bool=None, max_workers=4):
        """

        """
        if isinstance(nc_paths, str):
            nc_paths1 = glob.glob(nc_paths)
        elif isinstance(nc_paths, list):
            nc_paths1 = nc_paths

        nc_paths1.sort()

        ## Determine the variables needed to be extracted
        dataset_codes = list(self.dataset_codes_dict.keys())

        variables_set = set()
        for d in dataset_codes:
            v1 = self._func_dict[d]['variables']
            variables_set.update(v1)

        variables = list(variables_set)
        variables.sort()

        ## Determine duplicate times
        if (len(nc_paths1) > 1) and (time_index_bool is None):
            time_index_bool = self.determine_duplicate_times(nc_paths1)

        ## Iterate through files
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as executor:
            futures = []
            for nc_path in nc_paths1:
                f = executor.submit(preprocess_data_structure, nc_path, variables, heights, time_index_bool)
                futures.append(f)
            runs = concurrent.futures.wait(futures)

        ## process output
        new_paths = [r.result() for r in runs[0]]
        new_paths1 = []
        for new_path in new_paths:
            new_paths1.extend(new_path)

        new_paths1.sort()

        return new_paths1


    # def calc_new_variables(self, source_paths):
    #     """

    #     """
    #     new_paths2 = tu.grid.multi_calc_new_variables(source_paths, self.dataset_list, self.max_version_date, self._func_dict)

    #     return new_paths2



# class WRF(object):
#     """

#     """
#     ## Initialization
#     def __init__(self):
#         """

#         """

#         pass


#     def parse_paths(self, glob_obj, date_format=None, from_date=None, to_date=None):
#         """

#         """
#         paths = parse_data_paths(glob_obj, date_format, from_date, to_date)

#         self.source_data_paths = paths

#         return paths


#     def load_wrf_grid(self, parameter_codes, chunks=None, param_func_mappings=param_func_mappings, process_altitude=False, preprocessor=None):
#         """

#         """
#         # if isinstance(wrf_nc, str):
#         #     data_path = wrf_nc

#         # elif isinstance(wrf_nc, list):
#         #     if isinstance(wrf_nc[0], str):
#         #         paths = []
#         #         [paths.extend(glob(p)) for p in wrf_nc]
#         #         paths.sort()
#         #         data_path = paths.copy()
#         #     else:
#         #         raise TypeError('If wrf_nc is a list, then it must be a list of str paths.')

#         # else:
#         #     raise TypeError('wrf_nc must be a str path that xr.open_mfdataset can open.')

#         ## Get base path
#         # if isinstance(wrf_nc, list):
#         #     base_path = os.path.split(wrf_nc[0])[0]
#         # elif isinstance(wrf_nc, str):
#         #     base_path = os.path.split(wrf_nc)[0]
#         # else:
#         #     raise TypeError('wrf_nc must be either a list of str or a str.')

#         xr1 = sio.open_mf_wrf_dataset(self.source_data_paths, chunks=chunks, preprocess=preprocessor)
#         xr1 = xr1.drop('xtime', errors='ignore')

#         ## Get data projection
#         source_crs = xr1.attrs['pyproj_srs']

#         ### Pre-process the station data
#         ## Station_ids
#         lat = xr1['lat'].values

#         ## Get approximate grid resolution
#         grid_res = np.quantile(np.abs(np.diff(lat.T)), 0.5).round(4)
#         # lon_res = np.quantile(np.abs(np.diff(lon)), 0.5).round(4)

#         ## Altitude
#         if process_altitude:
#             alt = xr1['HGT'].isel(time=0)
#             xr1.coords['altitude'] = (('south_north', 'west_east'), alt)

#         ## Determine frequency interval
#         # freq = xr1['time'][:5].to_index()[:5].inferred_freq

#         # if freq is None:
#         #     raise ValueError('The time frequency could not be determined from the netcdf file.')

#         ### Read in mapping table
#         # wrf_mapping = pd.read_csv(os.path.join(base_dir, 'wrf_mappings.csv'))
#         # wrf_mapping.set_index('parameter_code', inplace=True)
#         # wrf_mapping['frequency_interval'] = freq

#         ### Process base datasets
#         # dsb = wrf_mapping[ds_cols].rename(columns={'scale_factor': 'precision'}).to_dict('index')

#         ### Select only the parameters necessary
#         params = []
#         [params.extend(p) for pc, p in param_func_mappings.items() if pc in parameter_codes]
#         params = list(set(params))

#         xr1 = xr1[params]

#         ## Remove duplicate times
#         time_bool = xr1.get_index('time').duplicated(keep='first')
#         xr1 = xr1.sel(time=~time_bool)

#         ### Set attrs
#         setattr(self, 'data', xr1)
#         # setattr(self, 'mappings', wrf_mapping)
#         # setattr(self, 'datasets', dsb)
#         # setattr(self, 'vp', vp)
#         setattr(self, 'data_crs', source_crs)
#         setattr(self, 'grid_res', grid_res)

                # new_grid = i1.grid_to_grid(self.grid_res, 4326, order=order, bbox=bbox)
                # if isinstance(min_val, (int, float)):
                #     new_grid = xr.where(new_grid.precip <= min_val, min_val, new_grid.precip)
                # if isinstance(max_val, (int, float)):
                #     new_grid = xr.where(new_grid.precip >= max_val, max_val, new_grid.precip)

#     def __repr__(self):
#         return repr(self.data)


    # def _resample_to_wgs84_grid(self, data, order=2, min_val=None, max_val=None):
    #     """

    #     """
    #     data_name = data.name
    #     res2 = data.drop(['altitude', 'station_id'], errors='ignore').to_dataset().load()

    #     i1 = Interp(grid_data=res2, grid_time_name='time', grid_x_name='west_east', grid_y_name='south_north', grid_data_name=data_name, grid_crs=self.data_crs)

    #     new_grid = i1.grid_to_grid(self.grid_res, 4326, order=order)
    #     if isinstance(min_val, (int, float)):
    #         new_grid = xr.where(new_grid.precip <= min_val, min_val, new_grid.precip)
    #     if isinstance(max_val, (int, float)):
    #         new_grid = xr.where(new_grid.precip >= max_val, max_val, new_grid.precip)

    #     new_grid3 = new_grid.rename({'x': 'lon', 'y': 'lat', 'precip': data_name})
    #     # new_grid3.name = data_name

    #     return new_grid3


    # def save_results(self, output_path, order=2, min_val=None, max_val=None):
    #     """

    #     """
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)

    #     vars1 = [v for v in list(self.data.variables) if not v in list(self.data.coords)]
    #     vars1.sort()

    #     for v in vars1:
    #         file_name = v + '.nc'
    #         file_path = os.path.join(output_path, file_name)

    #         print(file_path)

    #         v_data = self.data[v].copy()

    #         v_data2 = self._resample_to_wgs84_grid(v_data, order, min_val, max_val)
    #         # v_data2[v].attrs = self.data[v].attrs.copy()
    #         # v_data2[v].encoding = self.data[v].encoding.copy()
    #         # v_data2.attrs = self.data.attrs.copy()

    #         v_data2.to_netcdf(file_path)

    #     print('-- Finished saving data.')


#####################################################
### Processors


# def preprocessor(ds):
#     """

#     """
#     ## Read variables
#     # vars1 = list(ds.variables)
#     # vars1 = [v for v in vars1 if v not in dims][0]

#     ## Determine which parameters can be converted
#     # height = param_height_mappings[vars1]

#     ## Restructure dims
#     # ds = ds.assign_coords({'height': height})
#     ds = ds.rename({'longitude': 'lon', 'latitude': 'lat'})
#     # ds = ds.expand_dims('height')

#     return ds


# def postprocessor(ds, parameter_code):
#     """

#     """
#     ## Read variables

#     ## Read in mapping table
#     mappings = pd.read_csv(os.path.join(base_dir, 'wrf_mappings.csv'))
#     mappings.set_index('parameter_code', inplace=True)

#     ## Select the conversion functions
#     m = mappings.loc[parameter_code]

#     meth = getattr(vp, m['function'])
#     res1 = meth(ds)

#     ds[m['parameter']] = res1

#     encoding = {'dtype': m['dtype'], '_FillValue': m['_FillValue']}
#     if not np.isnan(m['scale_factor']):
#         encoding['scale_factor'] = m['scale_factor']
#     if not np.isnan(m['add_offset']):
#         encoding['add_offset'] = m['add_offset']

#     ds[m['parameter']].encoding = encoding

#     ds = ds.assign_coords({'height': m['height']})

#     ds = ds.expand_dims('height')

#     return ds






    # def get_results(self):
    #     """

    #     """
    #     if not hasattr(self, 'parameter_code'):
    #         raise ValueError('Run the build_dataset method prior to the get_results method.')

    #     map1 = self.param_map

    #     if isinstance(map1['function'], str):
    #         meth = getattr(self.vp, map1['function'])
    #         res1 = meth(self.data)
    #     else:
    #         res1 = self.data[map1['wrf_standard_name']]

    #     res1.name = map1['parameter']

    #     setattr(self, 'param_data', res1)

    #     _, index = np.unique(res1['time'], return_index=True)

    #     res1 = res1.isel(time=index)

    #     ## Reproject data
    #     res2 = self._resample_to_wgs84_grid(res1)

    #     res1.close()
    #     del res1

    #     map1 = self.param_map.copy()
    #     res2 = res2.assign_coords({'height': map1['height']})
    #     res2 = res2.expand_dims('height')

    #     return res2
