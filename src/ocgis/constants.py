import numpy as np


# : Standard bounds name used when none is available from the input data.
OCGIS_BOUNDS = 'bounds'

#: Standard name for the unique identifier in GIS files.
OCGIS_UNIQUE_GEOMETRY_IDENTIFIER = 'UGID'

#: Default netCDF4 output file type
NETCDF_DEFAULT_DATA_MODEL = 'NETCDF4'

#: Default temporal calendar.
DEFAULT_TEMPORAL_CALENDAR = 'standard'

#: Default temporal units.
DEFAULT_TEMPORAL_UNITS = 'days since 0001-01-01 00:00:00'

#: Default name for coordinate systems in netCDF file if none is provided.
DEFAULT_COORDINATE_SYSTEM_NAME = 'coordinate_system'

#: Default sample size variable standard name.
DEFAULT_SAMPLE_SIZE_STANDARD_NAME = 'sample_size'

#: Default sample size variable long name.
DEFAULT_SAMPLE_SIZE_LONG_NAME = 'Statistical Sample Size'

#: Default row coordinate name.
DEFAULT_NAME_ROW_COORDINATES = 'yc'

#: Default column coordinate name.
DEFAULT_NAME_COL_COORDINATES = 'xc'

#: Default corners dimension name.
DEFAULT_NAME_CORNERS_DIMENSION = 'ncorners'

#: Standard headers for subset operations.
HEADERS_RAW = ['did', 'vid', 'ugid', 'tid', 'lid', 'gid', 'variable', 'alias', 'time', 'year', 'month', 'day', 'level',
               'value']
#: Standard headers for computation.
HEADERS_CALC = ['did', 'vid', 'cid', 'ugid', 'tid', 'lid', 'gid', 'variable', 'alias', 'calc_key', 'calc_alias', 'time',
                'year', 'month', 'day', 'level', 'value']
#: Standard headers for multivariate calculations.
HEADERS_MULTI = ['did', 'cid', 'ugid', 'tid', 'lid', 'gid', 'calc_key', 'calc_alias', 'time', 'year', 'month', 'day',
                 'level', 'value']

#: Required headers for every request.
HEADERS_REQUIRED = ['did', 'ugid', 'gid']

#: Key identifiers for output formats.
OUTPUT_FORMATS = ['numpy', 'nc', 'csv', 'csv+', 'shp', 'geojson', 'meta']

# Download URL for test datasets.
TEST_DATA_DOWNLOAD_PREFIX = None

#: The day value to use for month centroids.
CALC_MONTH_CENTROID = 16
#: The month value to use for year centroids.
CALC_YEAR_CENTROID_MONTH = 7
#: The default day value for year centroids.
CALC_YEAR_CENTROID_DAY = 1

#: The number of values to use when calculating data resolution.
RESOLUTION_LIMIT = 100

#: The data type to use for NumPy integers.
NP_INT = np.int32
#: The data type to use for NumPy floats.
NP_FLOAT = np.float32

#: Function key prefix for the `icclim` indices library.
ICCLIM_PREFIX_FUNCTION_KEY = 'icclim'

#: NumPy functions enabled for functions evaluated from string representations.
ENABLED_NUMPY_UFUNCS = ['exp', 'log', 'abs']

#: The value for the 180th meridian to use when wrapping.
MERIDIAN_180TH = 180.
# MERIDIAN_180TH = 179.9999999999999
