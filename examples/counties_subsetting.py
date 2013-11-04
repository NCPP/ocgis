import ocgis
import os
import tempfile


## MODIFY: directory containing the CMIP netCDF data files
ocgis.env.DIR_DATA = '/home/local/WX/ben.koziol/wc'
## MODIFY: the directory to store the output
ocgis.env.DIR_OUTPUT = tempfile.mkdtemp()
## MODIFY: path to folder containing us_counties shapefile
ocgis.env.DIR_SHPCABINET = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/shp'

## these are the unique identifiers (UGID attribute) of the target counties
select_ugid = [1375,2416,1340]
## MODIFY: set this to False to extract all data - only first time slice pulled
## when True
snippet = True
## the key to the target shapefile
geom = 'us_counties'
## the output format
output_format = 'csv+'
## which headers to include in the output data file
headers = ['did','ugid','variable','time','value']
## the spatial operation to use when subsetting the data
spatial_operation = 'clip'
## option to aggregate the data after the spatial operation
aggregate = True

## this loop pulls the target datasets. we want two output files, one for precipitation
## and another for temperature
rds = {'tas':[],'pr':[]}
for ncfile in os.listdir(ocgis.env.DIR_DATA):
    ## in addition the variable name and file path, we need to include an alias
    ## to differentiate the variable origin - otherwise there would be duplicate
    ## variable names.
    rd = ocgis.RequestDataset(uri=ncfile,variable=ncfile.split('_')[0],alias=ncfile)
    ## choose which variable category the dataset belongs to
    if ncfile.startswith('pr'):
        rds['pr'].append(rd)
    else:
        rds['tas'].append(rd)

## generate the output for each variable by looping over the dictionary containing
## the request datasets
for variable_name,rds in rds.iteritems():
    print('processing: {0}'.format(variable_name))
    ## these are the operations to perform on the target datasets
    ops = ocgis.OcgOperations(dataset=rds,geom=geom,select_ugid=select_ugid,snippet=snippet,
                              output_format=output_format,headers=headers,prefix=variable_name,
                              aggregate=aggregate,spatial_operation=spatial_operation)
    ## the returned data value in this case is the path to the output data
    ret = ops.execute()
    print(' generated data file: {0}'.format(ret))

print('success!')
