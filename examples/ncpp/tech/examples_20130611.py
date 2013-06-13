import ocgis
import itertools

ocgis.env.DIR_OUTPUT = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/presentation/ncpp-tech-20130611'
ocgis.env.OVERWRITE = True
ocgis.env.DIR_DATA = '/usr/local/climate_data'
ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True
#ocgis.env.VERBOSE = True


rd = ocgis.RequestDataset(uri='tas_TMSL_gfdl_1981010100.nc',
                          variable='tas',
                          time_region={'month':[6,7]})

# write geometry shapefile
geom_write = ocgis.RequestDataset(uri='tas_TMSL_gfdl_1981010100.nc',
                                  variable='tas')
ops = ocgis.OcgOperations(dataset=geom_write,prefix='tmsl_geometry',snippet=True,output_format='shp')
ops.execute()

# write subset files
select_ugid = [63,65]
spatial_operation = ['intersects','clip']
aggregate = [False,True]
output_format = ['csv+','nc','shp','csv']
for so,agg,of in itertools.product(spatial_operation,aggregate,output_format):
    if so == 'intersects' and agg is True:
        continue
    if so == 'clip' and (agg is False or of == 'shp'):
        continue
    if of == 'nc' and (agg is True or so == 'clip'):
        continue
    prefix = '{0}_{1}_{2}'.format(of,so,agg)
    print(prefix)
    ops = ocgis.OcgOperations(dataset=rd,spatial_operation=so,aggregate=agg,
                              output_format=of,prefix=prefix,geom='co_watersheds',
                              select_ugid=select_ugid)
    ops.execute()

#select_ugid = [63,65]
#calc = [{'func':'mean','name':'my_mean'},{'func':'median','name':'my_median'}]
#ops = ocgis.OcgOperations(dataset=rd,calc=calc,spatial_operation='clip',
#                          aggregate=True,prefix='calculation',output_format='csv+',
#                          geom='co_watersheds',select_ugid=select_ugid,
#                          calc_grouping=['month'])
#ops.execute()
