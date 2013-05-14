import ocgis
import os


ocgis.env.OVERWRITE = True


## set snippet to false to return all data
snippet = False
## city center coordinate
geom = [-97.74278,30.26694]
## output directory
ocgis.env.DIR_OUTPUT = '/tmp/narccap'
## the directory containing the target data
ocgis.env.DIR_DATA = '/usr/local/climate_data/narccap'
## push data to a common reference projection
ocgis.env.WRITE_TO_REFERENCE_PROJECTION = True

## get names of the data to subset
filenames = os.listdir(ocgis.env.DIR_DATA)
## construct aliases for the datasets
aliases = [fn.split('_')[1] for fn in filenames]
## files all use the same variable
variable = 'pr'
## make the request datasets
rds = [ocgis.RequestDataset(uri=fn,variable=variable,alias=alias) for fn,alias in zip(filenames,aliases)]

## these are the calculations to perform
calc = [{'func':'threshold','name':'gt_0','kwds':{'threshold':0.0,'operation':'gt'}}]
calc_grouping = ['month']

## the operations
ops = ocgis.OcgOperations(dataset=rds,snippet=snippet,calc=calc,calc_grouping=calc_grouping,
                          output_format='shp',geom=geom)
ret = ops.execute()
import ipdb;ipdb.set_trace()