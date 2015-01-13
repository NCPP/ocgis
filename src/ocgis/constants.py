import numpy as np


# : Standard bounds name used when none is available from the input data.
OCGIS_BOUNDS = 'bounds'

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


class HEADERS(object):
    ID_DATASET = 'did'
    ID_VARIABLE = 'vid'
    ID_SELECTION_GEOMETRY = 'ugid'
    ID_TEMPORAL = 'tid'
    ID_LEVEL = 'lid'
    ID_GEOMETRY = 'gid'
    ID_CALCULATION = 'cid'

    VARIABLE = 'variable'
    VARIABLE_ALIAS = 'alias'

    TEMPORAL = 'time'
    TEMPORAL_YEAR = 'year'
    TEMPORAL_MONTH = 'month'
    TEMPORAL_DAY = 'day'

    LEVEL = 'level'

    VALUE = 'value'

    CALCULATION_KEY = 'calc_key'
    CALCULATION_ALIAS = 'calc_alias'


#: Standard headers for subset operations.
HEADERS_RAW = [HEADERS.ID_DATASET, HEADERS.ID_VARIABLE, HEADERS.ID_SELECTION_GEOMETRY, HEADERS.ID_TEMPORAL,
               HEADERS.ID_LEVEL, HEADERS.ID_GEOMETRY, HEADERS.VARIABLE, HEADERS.VARIABLE_ALIAS, HEADERS.TEMPORAL,
               HEADERS.TEMPORAL_YEAR, HEADERS.TEMPORAL_MONTH, HEADERS.TEMPORAL_DAY, HEADERS.LEVEL, HEADERS.VALUE]

#: Standard headers for computation.
HEADERS_CALC = [HEADERS.ID_DATASET, HEADERS.ID_VARIABLE, HEADERS.ID_CALCULATION, HEADERS.ID_SELECTION_GEOMETRY,
                HEADERS.ID_TEMPORAL, HEADERS.ID_LEVEL, HEADERS.ID_GEOMETRY, HEADERS.VARIABLE, HEADERS.VARIABLE_ALIAS,
                HEADERS.CALCULATION_KEY, HEADERS.CALCULATION_ALIAS, HEADERS.TEMPORAL, HEADERS.TEMPORAL_YEAR,
                HEADERS.TEMPORAL_MONTH, HEADERS.TEMPORAL_DAY, HEADERS.LEVEL, HEADERS.VALUE]

#: Standard headers for multivariate calculation.
HEADERS_MULTI = [HEADERS.ID_DATASET, HEADERS.ID_CALCULATION, HEADERS.ID_SELECTION_GEOMETRY,
                 HEADERS.ID_TEMPORAL, HEADERS.ID_LEVEL, HEADERS.ID_GEOMETRY, HEADERS.CALCULATION_KEY,
                 HEADERS.CALCULATION_ALIAS, HEADERS.TEMPORAL, HEADERS.TEMPORAL_YEAR, HEADERS.TEMPORAL_MONTH,
                 HEADERS.TEMPORAL_DAY, HEADERS.LEVEL, HEADERS.VALUE]

#: Required headers for every request.
HEADERS_REQUIRED = [HEADERS.ID_DATASET, HEADERS.ID_SELECTION_GEOMETRY, HEADERS.ID_GEOMETRY]

#: Standard name for the unique identifier in GIS files.
OCGIS_UNIQUE_GEOMETRY_IDENTIFIER = HEADERS.ID_SELECTION_GEOMETRY.upper()

OUTPUT_FORMAT_CSV = 'csv'
OUTPUT_FORMAT_CSV_SHAPEFILE = 'csv-shp'
OUTPUT_FORMAT_CSV_SHAPEFILE_OLD = 'csv+'
OUTPUT_FORMAT_ESMPY_GRID = 'esmpy'
OUTPUT_FORMAT_GEOJSON = 'geojson'
OUTPUT_FORMAT_METADATA = 'meta'
OUTPUT_FORMAT_NETCDF = 'nc'
OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH = 'nc-ugrid-2d-flexible-mesh'
OUTPUT_FORMAT_NUMPY = 'numpy'
OUTPUT_FORMAT_SHAPEFILE = 'shp'

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

# The standard key used to identify geometries in a dictionary.
DEFAULT_GEOMETRY_KEY = 'geom'

# Attributes to remove when a value is changed if they are present in the attributes dictionary. These attributes are
# tuned to specific value ranges and will not apply when a value is changed.
NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE = ('scale_value', 'add_offset')
