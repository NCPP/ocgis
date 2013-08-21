import ocgis


ocgis.env.VERBOSE = True
ocgis.env.DIR_OUTPUT = '/tmp/gg'
ocgis.env.DIR_DATA = '/usr/local/climate_data/maurer/2010-concatenated'
OUTPUT_FORMAT = 'csv+'
SNIPPET = False
HEADERS = ['did','ugid','gid','year','month','day','value']


def get_request_datasets():
    filenames = [
#                 'Maurer02new_OBS_tasmax_daily.1971-2000.nc',
                 'Maurer02new_OBS_tasmin_daily.1971-2000.nc'
                 ]
    variables = [
#                 'tasmax',
                 'tasmin'
                 ]
    rds = []
    for filename,variable in zip(filenames,variables):
        rds.append(ocgis.RequestDataset(filename,variable))
    return(rds)

## get data for NorthCarolina ##################################################

select_ugid = [39]
geom = 'state_boundaries'

for rd in get_request_datasets():
    prefix = 'maurer_north_carolina_' + rd.variable
    ops = ocgis.OcgOperations(dataset=rd,output_format=OUTPUT_FORMAT,geom=geom,
           prefix=prefix,select_ugid=select_ugid,snippet=SNIPPET,
           headers=HEADERS)
    ret = ops.execute()
    print(ret)
    
## get data for city centroids #################################################

geom = 'gg_city_centroids'

for rd in get_request_datasets():
    prefix = 'maurer_city_centroids_' + rd.variable
    ops = ocgis.OcgOperations(dataset=rd,output_format=OUTPUT_FORMAT,geom=geom,
           prefix=prefix,snippet=SNIPPET,headers=HEADERS)
    ret = ops.execute()
    print(ret)