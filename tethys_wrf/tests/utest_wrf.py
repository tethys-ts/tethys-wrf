# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:25:41 2019

@author: michaelek
"""
# import pytest
import xarray as xr
import numpy as np
import os
import yaml
import tethys_utils as tu
import pandas as pd
from tethys_wrf import sio, utils, virtual_parameters
# from tethys_utils.datasets import get_path

pd.options.display.max_columns = 10
d

###############################################
### Parameters

base_path = '/media/nvme1/data/UC/wrf'
nc1 = 'wrfout_d04_2020-02-01_00_00_00'
# nc1 = 'wrfout_d04_2014-02-22_00_00_00.nc'
# nc1 = 'wrfout_d04_*'

nc2 = '/media/nvme1/data/met-solutions/new_set2/MetSolutionsWRF_2022_07_28_06Z.nc'

# ncs1 = ['wrfout_d04_2014-02-22_00_00_00.nc', 'wrfout_d04_2014-03-01_00_00_00.nc']

# ncs = [os.path.join(base_path, nc) for nc in ncs1]

# nc = ncs[0]

# nc = Dataset(os.path.join(base_path, nc1))
wrf_nc = os.path.join(base_path, nc1)

base_dir = os.path.realpath(os.path.dirname(__file__))

ds_cols = ['feature', 'parameter', 'frequency_interval', 'aggregation_statistic', 'units', 'wrf_standard_name', 'cf_standard_name', 'scale_factor']

# with open(os.path.join(base_dir, 'parameters.yml')) as param:
#     param = yaml.safe_load(param)

# wrf_mapping = pd.read_csv(os.path.join(base_dir, 'wrf_mappings.csv'))

# general_dataset_data = {'utc_offset': '0H',
#                         "license": "https://creativecommons.org/licenses/by/4.0/",
#                         "attribution": "Data licensed by the NZ Open Data Consortium",
#                         'method': 'simulation',
#                         "result_type": "time_series_grid_simulation"}

owner = 'NZ Open Modelling Consortium'
product_code = 'NZ South Island 3km v01'
product_code = 'Test 1km v01'
data_license = "https://creativecommons.org/licenses/by/4.0/"
attribution = "Data licensed by the NZ Open Data Consortium"

parameter_codes = ['wind_speed_at_2', 'precip_at_0', 'temp_at_2']
# parameter_code = 'precip_at_0'

run_date = pd.Timestamp.now('utc').tz_localize(None).round('s')
# run_date = pd.Timestamp(param['source']['dataset_metadata']['run_date'])

# conn_config = param['remote']['s3']['connection_config']
# public_url = param['remote']['file']['connection_config']
# bucket = param['remote']['s3']['bucket']


# inputs = [(data1.sel(station_id=s).copy(), attrs, encoding, run_date_key, conn_config, bucket, s3, public_url) for s in stn_ids[:90]]


# data, attrs, encoding, run_date, conn_config, bucket, s3, public_url = inputs[0]

chunk_size = 1000000000
chunk_size = 1000000


def find_time_chunk_size(data, parameter):
    """

    """
    time_chunks = [c for c in data[parameter].chunks if isinstance(c, tuple)][0]
    time_chunk_size = time_chunks[0]

    return time_chunk_size, len(time_chunks)




output_path = os.path.join(base_path, 'temp')


########################################
### Tests

self = WRF(wrf_nc, parameter_codes)

self.save_results(output_path)

data = self.data

time_chunk_size, time_chunks = find_time_chunk_size(data, parameter)

total_nbytes = data[parameter].nbytes

nbytes_per = int(total_nbytes/time_chunks)

n_parts = int(np.ceil(total_nbytes/chunk_size))








dataset = self.build_dataset(parameter_code, owner, product_code, data_license, attribution)
data = self.get_results()

self.save_results(conn_config, bucket, public_url=public_url, run_date=None, threads=30)


def test_read_pkl_zstd():
    df1 = read_pkl_zstd(d_path1)

    assert df1.shape == (20000, 7)

df1 = read_pkl_zstd(d_path1)


def test_write_pkl_zstd():
    p_df1 = write_pkl_zstd(df1)
    len1 = round(len(p_df1), -3)

    assert (len1 < 200000) and (len1 > 100000)


def test_df_to_xarray():
    p_ds1 = df_to_xarray(df1, nc_type, param_name, attrs, encoding, run_date_key, ancillary_variables, compression)
    len2 = round(len(p_ds1), -3)

    ds1 = df_to_xarray(df1, nc_type, param_name, attrs, encoding, run_date_key, ancillary_variables)

    assert (len(ds1) == 6) and (len2 < 30000) and (len2 > 20000)

#############################################################
### Other tests

file = nc2
nc_paths = [nc2]
nc_path = nc2


x1a = sio.open_wrf_dataset(nc2)
x1b = xr.open_dataset(nc2)

x2a = sio.open_wrf_dataset(wrf_nc)
x2b = xr.open_dataset(wrf_nc)






























































