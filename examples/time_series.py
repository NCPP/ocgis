from tempfile import mkdtemp
import ocgis


ocgis.env.DIR_OUTPUT = mkdtemp()
uri = '/usr/local/climate_data/CanCM4/tas_day_CanCM4_decadal2011_r2i1p1_20120101-20211231.nc'
variable = 'tas'


rd = ocgis.RequestDataset(uri=uri, variable=variable)
ops = ocgis.OcgOperations(dataset=rd, geom=[-73.8, 42.7], aggregate=True)
ret = ops.execute()
# print(ret)
field = ret[1]['tas']
