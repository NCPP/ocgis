import ocgis
import os


ocgis.env.DIR_OUTPUT = '/tmp'
ocgis.env.OVERWRITE = True

## write maurer grid
prefix = 'maurer_grid'
uri = '/usr/local/climate_data/maurer/bcca/obs/tasmax/1_8deg/gridded_obs.tasmax.OBS_125deg.daily.1999.nc'
variable = 'tasmax'
rdm = ocgis.RequestDataset(uri,variable)
ops = ocgis.OcgOperations(dataset=rdm,prefix=prefix,output_format='shp',snippet=True)
print(ops.execute())

variable = 'perc_95'
uris = [os.path.join('/tmp/nctest/row_{0}'.format(ii),'row_{0}.nc'.format(ii)) for ii in [150,151]]
aliases = ['perc_95_{0}'.format(ii) for ii in [150,151]]
rds = [ocgis.RequestDataset(uri=uri,variable=variable,alias=alias) for uri,alias in zip(uris,aliases)]

ops = ocgis.OcgOperations(dataset=rds,output_format='shp',snippet=True)
ret = ops.execute()
print(ret)