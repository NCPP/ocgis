import ocgis


ocgis.env.DIR_OUTPUT = '/tmp/graphics'


uri = '/home/local/WX/ben.koziol/climate_data/maurer/2010-concatenated/Maurer02new_OBS_tasmax_daily.1971-2000.nc'
variable = 'tasmax'


## the entire geographic region
ops = ocgis.OcgOperations(dataset={'uri':uri,'variable':variable},output_format='shp',
                          snippet=True,prefix='domain')
print(ops.execute())

## an intersects subset
ops = ocgis.OcgOperations(dataset={'uri':uri,'variable':variable},output_format='shp',
                          snippet=True,prefix='intersects',geom='state_boundaries',
                          select_ugid=[16])
print(ops.execute())

## a clip operation
ops = ocgis.OcgOperations(dataset={'uri':uri,'variable':variable},output_format='shp',
                          snippet=True,prefix='clip',geom='state_boundaries',
                          select_ugid=[16],spatial_operation='clip')
print(ops.execute())

## clip + aggregate
ops = ocgis.OcgOperations(dataset={'uri':uri,'variable':variable},output_format='shp',
                          snippet=True,prefix='clip+agg',geom='state_boundaries',
                          select_ugid=[16],spatial_operation='clip',aggregate=True)
print(ops.execute())