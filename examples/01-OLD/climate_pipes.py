import tempfile

import ocgis

'''
Usage:
    - Assumes repo is on current master
    - Check the globals below for major script parameters
    - The code for generating GeoJson directly from objects is below the code
       that using the OcgOperations API. It is commented out.
'''

## path to target file (may also be an OPeNDAP target)
URI = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data/CanCM4/tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'

## the target variable in the dataset to convert
VARIABLE = 'tas'

## this is the path to the shapefile containing state boundaries
SHAPEFILE_PATH = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/shp/state_boundaries/state_boundaries.shp'

## write data to a new temporary directory for each script start
DIR_OUTPUT = tempfile.mkdtemp()

## set the output format to convert to
OUTPUT_FORMAT = 'geojson'
# OUTPUT_FORMAT = 'shp'

## limit the headers in the output.
HEADERS = ['time', 'year', 'month', 'day', 'value']

## we can either subset the data by a geometry from a shapefile, or convert to
## geojson for the entire spatial domain. there are other options here (i.e. a
## bounding box for tiling or a Shapely geometry).
# GEOM = SHAPEFILE_PATH
GEOM = None

## connect to the dataset and load the data as a field object. this will be used
## to iterate over time coordinates during the conversion step.
rd = ocgis.RequestDataset(uri=URI, variable=VARIABLE)
field = rd.get()

## selecting specific geometries from a shapefile requires knowing the target
## geometry's UGID inside the shapefile. shapefile are required to have this
## identifier at this time as a full attribute search is not enabled. this code
## searches for TX to find the UGID associated with that state.
if GEOM is None:
    select_ugid = None
else:
    select_geom = filter(lambda x: x['properties']['STATE_ABBR'] == 'TX',
                         ocgis.GeomCabinetIterator(path=SHAPEFILE_PATH))
    ## this argument must always come in as a list
    select_ugid = [select_geom[0]['properties']['UGID']]

## USE OCGIS OPERATIONS ########################################################

## get an example time coordinate
centroid = field.temporal.value_datetime[0]
print('writing geojson for time slice: {0}'.format(centroid))
## this again is the target dataset with a time range for subsetting now
rd = ocgis.RequestDataset(uri=URI, variable=VARIABLE, time_range=[centroid, centroid])
## name of the output geojson file
prefix = 'ocgis_output_{0}'.format(OUTPUT_FORMAT)
## parameterize the operations to be performed on the target dataset
ops = ocgis.OcgOperations(dataset=rd, geom=SHAPEFILE_PATH, select_ugid=select_ugid,
                          output_format=OUTPUT_FORMAT, prefix=prefix, dir_output=DIR_OUTPUT,
                          headers=HEADERS)
ret = ops.execute()
print('path to output file: {0}'.format(ret))

## USE OCGIS OBJECTS DIRECTLY ##################################################

### get an example time coordinate
# centroid = field.temporal.value_datetime[0]
# print('writing geojson for time slice: {0}'.format(centroid))
#
### subset the field for that coordinate
# subsetted = field.get_between('temporal',lower=centroid,upper=centroid)
#
### alternatively you may simple slice the data (5-d slicing only - realization,
### time,level,row,column
##subsetted = field[:,0,:,:,:]
#
### collection objects are used the converters
# sc = SpatialCollection(headers=HEADERS)
### add the field. a geometry identifier value is required (GIS first always!).
### the geometry is empty however. the field needs to be given an alias for unique
### storage in the collection. the last argument is field object itself.
# sc.add_field(1,None,'tas',subsetted)
# conv = GeoJsonConverter([sc],DIR_OUTPUT,'ocgis_output_geojson')
# print(conv.write())
