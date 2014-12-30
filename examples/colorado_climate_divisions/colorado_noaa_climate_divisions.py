import ocgis


ocgis.env.DIR_DATA = '/usr/local/climate_data/CanCM4'
ocgis.env.VERBOSE = True
SHP_PATH = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/bin/shp/climate_divisions/climate_divisions.shp'
FILENAMES = ['tas_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
             'tasmin_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',
             'tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc']
ocgis.env.DIR_OUTPUT = '/home/local/WX/ben.koziol/Dropbox/nesii/docs/NOAA-WriteUp-20140317'
ocgis.env.OVERWRITE = True


select = lambda x: x['properties']['STATE'] == 'Colorado'
rows = filter(select, ocgis.ShpCabinetIterator(path=SHP_PATH))
select_ugid = map(lambda x: x['properties']['UGID'], rows)
select_ugid.sort()

rds = []
for fn in FILENAMES:
    variable = fn.split('_')[0]
    rd = ocgis.RequestDataset(uri=fn, variable=variable, conform_units_to='Celsius')
    rds.append(rd)

ops = ocgis.OcgOperations(dataset=rds, select_ugid=select_ugid, spatial_operation='clip', aggregate=True,
                          output_format='csv-shp', geom=SHP_PATH)
ret = ops.execute()
print(ret)
