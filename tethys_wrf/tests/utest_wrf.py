# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:25:41 2019

@author: michaelek
"""
import pytest
from tethys_utils import *
import pandas as pd
from tethys_utils.datasets import get_path

pd.options.display.max_columns = 10


###############################################
### Parameters

base_path = '/media/sdb1/Data/UC/wrf'
nc1 = 'wrfout_d03_2017-01-07_00_00_00.nc'
# nc1 = 'wrfout_d04_2014-02-22_00_00_00.nc'
nc1 = 'wrfout_d03_*.nc'

# ncs1 = ['wrfout_d04_2014-02-22_00_00_00.nc', 'wrfout_d04_2014-03-01_00_00_00.nc']

# ncs = [os.path.join(base_path, nc) for nc in ncs1]

# nc = ncs[0]

# nc = Dataset(os.path.join(base_path, nc1))
nc = os.path.join(base_path, nc1)

base_dir = os.path.realpath(os.path.dirname(__file__))

ds_cols = ['feature', 'parameter', 'frequency_interval', 'aggregation_statistic', 'units', 'wrf_standard_name', 'cf_standard_name', 'scale_factor']

with open(os.path.join(base_dir, 'parameters.yml')) as param:
    param = yaml.safe_load(param)

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

parameter_code = 'temp_at_2'

run_date = pd.Timestamp.now('utc').tz_localize(None).round('s')

conn_config = param['remote']['s3']['connection_config']
public_url = param['remote']['file']['connection_config']
bucket = param['remote']['s3']['bucket']


# inputs = [(data1.sel(station_id=s).copy(), attrs, encoding, run_date_key, conn_config, bucket, s3, public_url) for s in stn_ids[:90]]


# data, attrs, encoding, run_date, conn_config, bucket, s3, public_url = inputs[0]


########################################
### Tests

self = WRF(nc)
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




# @pytest.mark.parametrize('input_sites', [input_sites1, input_sites2, input_sites3])
# def test_nat(input_sites):
#     f1 = FlowNat(from_date, to_date, input_sites=input_sites)
#
#     nat_flow = f1.naturalisation()
#
#     assert (len(f1.summ) >= 1) & (len(nat_flow) > 2900)
