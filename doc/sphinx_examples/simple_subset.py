import ocgis
import os.path


## Directory holding climate data.
DATA_DIR = '/usr/local/climate_data/CanCM4'
## Location and variable name for a daily decadal temperature simulation.
URI_TAS = os.path.join(DATA_DIR,'tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc')
VAR_TAS = 'tas'
## Make it easy to switch to non-snippet requests.
SNIPPET = True
## Set output directory for shapefile and keyed formats. (MAKE SURE IT EXISTS!)
ocgis.env.DIR_OUTPUT = '/tmp/foo'
## The bounding box coordinates [minx, miny, maxx, maxy] for the state of
## Colorado in WGS84 latitude/longitude coordinates.
BBOX = [-109.1, 36.9, -102.0, 41.0]


## Construct RequestDataset Object #############################################

## This object will be reused so just build it once.
rd = ocgis.RequestDataset(URI_TAS,VAR_TAS)

## Returning NumPy Data Objects ################################################

## The NumPy data type return is the default. Only the geometry and
## RequestDataset are required (except "snippet" of course...). See the
## documentation for the OcgCollection object to understand the return 
## structure.
ret = ocgis.OcgOperations(dataset=rd,geom=BBOX,snippet=SNIPPET).execute()

## Returning Converted Files ###################################################

## At this time, the software will create named temporary directories inside
## env.DIR_OUTPUT. This is to avoid the confusing process of managine overwrites
## etc. The support for managing output files will be improved in future 
## releases. The returned value is the absolute path to the file or folder
## depending on the requested format.
output_formats = ['shp','csv','keyed']
for output_format in output_formats:
    prefix = output_format + '_output'
    ops = ocgis.OcgOperations(dataset=rd,geom=BBOX,snippet=SNIPPET,
                        output_format=output_format,prefix=prefix)
    ret = ops.execute()