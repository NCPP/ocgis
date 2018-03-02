import os
import tempfile

import ocgis

DIR_OUTPUT = tempfile.gettempdir()
DIR_DATA = '/home/local/WX/ben.koziol/links/ocgis/bin/nc'
# DIR_DATA = '/usr/local/climate_data/CanCM4'
FILENAME = 'rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc'
#          'rhsmin_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc'
#          'rhsmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc'
VARIABLE = 'rhs'
#          'rhsmin'
#          'rhsmax'
AGGREGATE = False  # True
SPATIAL_OPERATION = 'intersects'  # 'clip'
GEOM = 'state_boundaries'
STATES = {'CO': [32], 'CA': [25]}
OUTPUT_FORMAT = 'csv+'  # 'csv' #'nc' #'shp'
PREFIX = 'ocgis_output'
TIME_REGION = {'month': [6, 7], 'year': [2011]}

################################################################################

## construct the request dataset. time subsetting is also parameterized with 
## this object.
rd = ocgis.RequestDataset(uri=os.path.join(DIR_DATA, FILENAME),
                          variable=VARIABLE, time_range=None,
                          time_region=TIME_REGION)

## construct the operations call
ops = ocgis.OcgOperations(dataset=rd, geom=GEOM, select_ugid=STATES['CO'],
                          aggregate=AGGREGATE, spatial_operation=SPATIAL_OPERATION, prefix=PREFIX,
                          output_format=OUTPUT_FORMAT)

## return the path to the folder containing the output data
path = ops.execute()

print(path)
