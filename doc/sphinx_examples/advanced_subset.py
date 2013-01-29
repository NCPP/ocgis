from ocgis import OcgOperations, RequestDataset, RequestDatasetCollection
import os.path


## Directory holding climate data.
DATA_DIR = '/usr/local/climate_data/CanCM4'
## Filename to variable name mapping.
NCS = {'tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc':'tasmin',
       'tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc':'tas',
       'tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc':'tasmax'}
## Always start with a snippet.
SNIPPET = True


## RequestDatasetCollection ####################################################

rdc = RequestDatasetCollection([RequestDataset(
                os.path.join(DATA_DIR,uri),var) for uri,var in NCS.iteritems()])

## Return In-Memory ############################################################

## Data is returned as a dictionary with 51 keys (don't forget Puerto Rico...).
## A key in the returned dictionary corresponds to a geometry "ugid" with the
## value of type OcgCollection.
ops = OcgOperations(dataset=rdc,spatial_operation='clip',aggregate=True,
                    snippet=SNIPPET,geom='state_boundaries')
ret = ops.execute()

## Write to Shapefile ##########################################################

ops = OcgOperations(dataset=rdc,spatial_operation='clip',aggregate=True,
                    snippet=SNIPPET,geom='state_boundaries',output_format='shp')
path = ops.execute()

## Write All Data to Keyed Format ##############################################

## Without the snippet, we are writing all data to the linked CSV files. The
## operation will take considerably longer (~20 minutes).
ops = OcgOperations(dataset=rdc,spatial_operation='clip',aggregate=True,
                    snippet=False,geom='state_boundaries',output_format='keyed')
path = ops.execute()