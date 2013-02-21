import ocgis
import os


## directory holding data files. this directory and it subdirectories will be
## searched for data.
DIR_DATA = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/climate_data/CanCM4'
## where output data files are written.
DIR_OUTPUT = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/presentation/20130225_caspar_demo/output/01'
## state identifiers in the state shapefile that together comprise the
## southwestern united states.
SW_IDS = [25,23,49,37,42]
## set to true until final execution
SNIPPET = True


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
## construct operations
ops = ocgis.OcgOperations(rdc,
                          snippet=SNIPPET,
                          geom='state_boundaries',
                          select_ugid=SW_IDS,
                          output_format='csv')