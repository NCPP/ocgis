import ocgis
import os


## directory holding data files. this directory and it subdirectories will be
## searched for data.
DIR_DATA = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data/CanCM4'
## where output data files are written.
DIR_OUTPUT = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/presentation/20130225_caspar_demo/output/01'
## state identifiers in the state shapefile that together comprise the
## southwestern united states.
SW_IDS = [25,23,24,37,42]
## set to true until final execution
SNIPPET = False


## update environmental variables.
ocgis.env.DIR_DATA = DIR_DATA
ocgis.env.DIR_OUTPUT = DIR_OUTPUT
## pull data files together into a single collections of request datasets.
rdc = ocgis.RequestDatasetCollection()
for ii,filename in enumerate(os.listdir(ocgis.env.DIR_DATA)):
    variable = filename.split('_')[0]
    alias = variable + '_' + str(ii)
    rd = ocgis.RequestDataset(filename,variable,alias=alias)
    rdc.update(rd)
    
## SUBSETTING ##################################################################
    
#ops = ocgis.OcgOperations(rdc,
#                          snippet=SNIPPET,
#                          geom='state_boundaries',
#                          select_ugid=SW_IDS,
#                          output_format='shp',
#                          agg_selection=False,
#                          spatial_operation='clip',
#                          aggregate=True)
#path = ops.execute()

## CALCULATION #################################################################

#calc = [{'func':'mean','name':'mean'},
#        {'func':'std','name':'std'},
#        {'func':'min','name':'min'},
#        {'func':'max','name':'max'},
#        {'func':'median','name':'median'}]
#
#ops = ocgis.OcgOperations(rdc,
#                          snippet=SNIPPET,
#                          geom='state_boundaries',
#                          select_ugid=SW_IDS,
#                          output_format='shp',
#                          agg_selection=False,
#                          spatial_operation='clip',
#                          aggregate=True,
#                          calc=calc,
#                          calc_grouping=['month','year'],
#                          calc_raw=False)
#path = ops.execute()
#
#ops = ocgis.OcgOperations(rdc,
#                          snippet=SNIPPET,
#                          geom='state_boundaries',
#                          select_ugid=SW_IDS,
#                          output_format='keyed',
#                          agg_selection=True,
#                          spatial_operation='intersects',
#                          aggregate=False,
#                          calc=calc,
#                          calc_grouping=['month','year'],
#                          calc_raw=False)
#path = ops.execute()

calc = [{'func':'threshold',
         'name':'gt285',
         'kwds':{'threshold':285,'operation':'gt'}}]
uri = 'tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
variable = 'tasmax'
calc_grouping = ['year']
rd = ocgis.RequestDataset(uri,variable)

ops = ocgis.OcgOperations(rd,calc=calc,geom='state_boundaries',agg_selection=True,
                          aggregate=False,spatial_operation='intersects',
                          calc_grouping=calc_grouping,snippet=False,output_format='shp')
path = ops.execute()