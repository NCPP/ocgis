from ocgis import OcgOperations, ShpCabinet


DATASET = {'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',
           'variable':'tasmax'}


## subset data by the boundaries of the USA
ops = OcgOperations(dataset=DATASET,geom='state_boundaries',agg_selection=True)
ret = ops.execute()
## access the subsetted values. returned data is stored as a dictionary with the
## key the geometry identifier and the value an OcgCollection.
values = ret[1].variables['tasmax'].value
## to get to the geometries
geoms = ret[1].variables['tasmax'].spatial.value
## time points
times = ret[1].variables['tasmax'].temporal.value

## aggregate the data by state boundary
ops = OcgOperations(dataset=DATASET,geom='state_boundaries',aggregate=True,
                    snippet=True,spatial_operation='clip')
ret = ops.execute()
## returned data has 51 keys. access the selection geometry for a particular 
## collection.
ugeom = ret[1].ugeom

## load a selection geometry from disk.
sc = ShpCabinet()
geoms = sc.get_geoms('state_boundaries')
## load only a particular state
geoms = sc.get_geoms('state_boundaries',attr_filter={'state_name':['California']})
## use california to subset
ret = OcgOperations(dataset=DATASET,geom=geoms,spatial_operation='clip',
                    aggregate=True).execute()
