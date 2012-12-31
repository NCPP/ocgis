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

#env.WORKSPACE = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/image/agu_for_luca'
#
#sc = ShpCabinet()
#gd = sc.get_geom_dict('state_boundaries')
#
#dataset = {'variable':'clt',
#           'uri':'http://esg-datanode.jpl.nasa.gov/thredds/dodsC/esg_dataroot/obs4MIPs/observations/atmos/clt/mon/grid/NASA-GSFC/MODIS/v20111130/clt_MODIS_L3_C5_200003-201109.nc'}
#
#ops = OcgOperations(geom=gd,dataset=dataset,snippet=True,output_format='shp',agg_selection=True,vector_wrap=True)
#ret1 = OcgInterpreter(ops).execute()
#print ret1
#
#ops = OcgOperations(geom=gd,dataset=dataset,snippet=True,output_format='shp',
#                    aggregate=True,spatial_operation='clip')
#ret2 = OcgInterpreter(ops).execute()
#print ret2