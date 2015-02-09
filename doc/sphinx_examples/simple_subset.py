import os

import ocgis



# Directory holding climate data.
DATA_DIR = '/usr/local/climate_data/CanCM4'
# Location and variable name for a daily decadal temperature simulation.
URI_TAS = os.path.join(DATA_DIR, 'tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc')
VAR_TAS = 'tas'
# Make it easy to switch to non-snippet requests.
SNIPPET = True
# Set output directory for shapefile and keyed formats. (MAKE SURE IT EXISTS!)
ocgis.env.DIR_OUTPUT = '/tmp/foo'
os.mkdir('/tmp/foo')
# The bounding box coordinates [minx, miny, maxx, maxy] for the state of Colorado in WGS84 latitude/longitude
# coordinates.
BBOX = [-109.1, 36.9, -102.0, 41.0]


# Construct RequestDataset Object ######################################################################################

# This object will be reused so just build it once.
rd = ocgis.RequestDataset(URI_TAS, VAR_TAS)

# Returning NumPy Data Objects #########################################################################################

# The NumPy data type return is the default. Only the geometry and RequestDataset are required (except "snippet" of 
# course...). See the documentation for the OcgCollection object to understand the return structure.
ret = ocgis.OcgOperations(dataset=rd, geom=BBOX, snippet=SNIPPET).execute()

# Returning Converted Files ############################################################################################

output_formats = ['shp', 'csv', 'csv-shp', 'nc']
for output_format in output_formats:
    prefix = output_format + '_output'
    ops = ocgis.OcgOperations(dataset=rd, geom=BBOX, snippet=SNIPPET, output_format=output_format, prefix=prefix)
    ret = ops.execute()