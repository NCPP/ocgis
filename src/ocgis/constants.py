import numpy as np

#: Standard bounds name used when none is available from the input data.
ocgis_bounds = 'bounds'

#: Default netCDF4 output file type
netCDF_default_data_model = 'NETCDF4_CLASSIC'

#: Standard headers for subset operations.
raw_headers = ['did','vid','ugid','tid','lid','gid','variable','alias','time','year','month','day','level','value']
#: Standard headers for computation.
calc_headers = ['did','vid','cid','ugid','tid','lid','gid','variable','alias','calc_key','calc_alias','time','year','month','day','level','value']
#: Standard headers for multivariate calculations.
multi_headers = ['did','cid','ugid','tid','lid','gid','calc_key','calc_alias','time','year','month','day','level','value']

level_headers = ['lid','level']

#: Required headers for every request.
required_headers = ['did','ugid','gid']

#: Key identifiers for output formats.
output_formats = ['numpy','nc','csv','csv+','shp','geojson','meta']

# Download URL for test datasets.
test_data_download_url_prefix = None

#: The day value to use for month centroids.
calc_month_centroid = 16
#: The month value to use for year centroids.
calc_year_centroid_month = 7
#: The default day value for year centroids.
calc_year_centroid_day = 1

#: The number of values to use when calculating data resolution.
resolution_limit = 100

#: The data type to use for NumPy integers.
np_int = np.int32
#: The data type to use for NumPy floats.
np_float = np.float32

#: Function key prefix for the `icclim` indices library.
prefix_icclim_function_key = 'icclim'

#: NumPy functions enabled for functions evaluated from string representations.
enabled_numpy_ufuncs = ['exp','log','abs']

#: The value for the 180th meridian to use when wrapping.
meridian_180th = 179.9999999999999


test_run_long_tests = False
test_run_dev_tests = False
