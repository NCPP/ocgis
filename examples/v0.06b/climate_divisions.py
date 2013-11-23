import ocgis
import os
import re


ocgis.env.DIR_DATA = '/usr/local/climate_data/bcsd'
ocgis.env.DIR_OUTPUT = '/tmp'


print('collecting files...')
ncfiles = filter(lambda x: x.endswith('.nc'),os.listdir(ocgis.env.DIR_DATA))
variables = map(lambda x: re.match('.*\.monthly.(.*)\.1950.*',x).group(1),ncfiles)
rds = [ocgis.RequestDataset(ncfile,variable) for ncfile,variable in zip(ncfiles,variables)]

print('building operations...')
select_ugid = [2001,2002,2003,2004,2005,2006,2007,2008,2009,2010]
geom = 'climate_divisions'
output_format = 'shp'
snippet = True

ops_agg = ocgis.OcgOperations(dataset=rds,aggregate=True,geom=geom,
 select_ugid=select_ugid,spatial_operation='clip',snippet=snippet,
 output_format=output_format,prefix='climdiv_agg')
ops_sel = ocgis.OcgOperations(dataset=rds,aggregate=False,geom=geom,
 select_ugid=select_ugid,spatial_operation='intersects',snippet=snippet,
 output_format=output_format,prefix='climdiv_sel')

print('executing aggregation...')
path_agg = ops_agg.execute()

print('executing select...')
path_sel = ops_sel.execute()

print('success.')