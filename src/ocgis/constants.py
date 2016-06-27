import numpy as np

from ocgis.util.enum import IntEnum

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

#: Default rotated pole ellipse for transformation.
PROJ4_ROTATED_POLE_ELLPS = 'sphere'


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
OUTPUT_FORMAT_METADATA_JSON = 'meta-json'
OUTPUT_FORMAT_METADATA_OCGIS = 'meta-ocgis'
OUTPUT_FORMAT_NETCDF = 'nc'
OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH = 'nc-ugrid-2d-flexible-mesh'
OUTPUT_FORMAT_NUMPY = 'numpy'
OUTPUT_FORMAT_SHAPEFILE = 'shp'

#: These output formats are considered vector output formats affected by operations manipulation vector GIS data. For
#: example, vector GIS outputs are always wrapped to -180 to 180 if there is a spherical coordinate system.
VECTOR_OUTPUT_FORMATS = [OUTPUT_FORMAT_GEOJSON, OUTPUT_FORMAT_SHAPEFILE]

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
DEFAULT_NP_INT = np.int
#: The data type to use for NumPy floats.
DEFAULT_NP_FLOAT = np.float

#: Function key prefix for the `icclim` indices library.
ICCLIM_PREFIX_FUNCTION_KEY = 'icclim'

#: NumPy functions enabled for functions evaluated from string representations.
ENABLED_NUMPY_UFUNCS = ('exp', 'log', 'abs', 'power')

#: The value for the 180th meridian to use when wrapping.
MERIDIAN_180TH = 180.
# MERIDIAN_180TH = 179.9999999999999

# The standard key used to identify geometries in a dictionary.
DEFAULT_GEOMETRY_KEY = 'geom'

# Attributes to remove when a value is changed if they are present in the attributes dictionary. These attributes are
# tuned to specific value ranges and will not apply when a value is changed.
NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE = ('scale_value', 'add_offset', 'actual_range', 'valid_range')

NAME_DIMENSION_REALIZATION = 'rlz'
NAME_DIMENSION_TEMPORAL = 'time'
NAME_DIMENSION_LEVEL = 'level'

NAME_BOUNDS_DIMENSION_LOWER = 'lb'
NAME_BOUNDS_DIMENSION_UPPER = 'ub'

NAME_UID_DIMENSION_REALIZATION = 'rid'
NAME_UID_DIMENSION_TEMPORAL = 'tid'
NAME_UID_DIMENSION_LEVEL = 'lid'
NAME_UID_FIELD = 'fid'

# calculation dictionary key defaults
CALC_KEY_KEYWORDS = 'kwds'
CALC_KEY_CLASS_REFERENCE = 'ref'
NAME_UID_FIELD = 'fid'


# Enumerations for wrapped states and actions. #########################################################################

class WrappedState(IntEnum):
    # Data is wrapped -180 to 180.
    WRAPPED = 1
    # Data is unwrapped 0 to 360.
    UNWRAPPED = 2
    # The wrapped state is unknown due to coordinate values only in the range 0 to 180.
    UNKNOWN = 3


class WrapAction(IntEnum):
    # Wrap the data to -180 to 180.
    WRAP = 1
    # Unwrap the data 0 to 360.
    UNWRAP = 2

########################################################################################################################
