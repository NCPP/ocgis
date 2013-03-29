import ocgis
import os


ocgis.env.DIR_OUTPUT = '/tmp'
ocgis.env.OVERWRITE = True

variable = 'perc_95'
uris = [os.path.join('/tmp/nctest/row_{0}'.format(ii),'row_{0}.nc'.format(ii)) for ii in [150]]
aliases = ['perc_95_{0}'.format(ii) for ii in [150]]
rds = [ocgis.RequestDataset(uri=uri,variable=variable,alias=alias) for uri,alias in zip(uris,aliases)]

ops = ocgis.OcgOperations(dataset=rds,output_format='shp',snippet=True)
ret = ops.execute()
print(ret)