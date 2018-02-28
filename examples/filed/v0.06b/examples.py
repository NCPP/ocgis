import os.path

import ocgis

CLIMATE_DATA_DIRECTORY = '/usr/local/climate_data/CanCM4'
NC1 = 'tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
VAR1 = 'tasmax'
NC2 = 'tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
VAR2 = 'tasmin'


def make_uri(filename):
    return (os.path.join(CLIMATE_DATA_DIRECTORY, filename))


rd1 = ocgis.RequestDataset(make_uri(NC1), VAR1)
rd2 = ocgis.RequestDataset(make_uri(NC2), VAR2)

## subset data by the boundaries of the USA ####################################

ops = ocgis.OcgOperations(dataset=rd1, geom='state_boundaries', agg_selection=True)
ret = ops.execute()
## access the subsetted values. returned data is stored as a dictionary with the
## key the geometry identifier and the value an OcgCollection.
values = ret[1].variables['tasmax'].value
## to get to the geometries
geoms = ret[1].variables['tasmax'].spatial.value
## time points
times = ret[1].variables['tasmax'].temporal.value

## return two variables ########################################################

rdc = ocgis.RequestDatasetCollection([rd1, rd2])
ops = ocgis.OcgOperations(rdc, snippet=True)
ret = ops.execute()

## aggregate the data by state boundary ########################################

ops = ocgis.OcgOperations(dataset=rd1, geom='state_boundaries', aggregate=True,
                          snippet=True, spatial_operation='clip')
ret = ops.execute()
## returned data has 51 keys. access the selection geometry for a particular 
## collection.
ugeom = ret[1].ugeom

## load a selection geometry from disk #########################################

sc = ocgis.ShpCabinet()
geoms = sc.get_geoms('state_boundaries')
## load only a particular state
geoms = sc.get_geoms('state_boundaries', attr_filter={'state_name': ['California']})
## use california to subset
ret = ocgis.OcgOperations(dataset=rd1, geom=geoms, spatial_operation='clip',
                          aggregate=True).execute()

################################################################################
