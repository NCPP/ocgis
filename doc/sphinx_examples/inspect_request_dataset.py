from ocgis import RequestDataset


uri = 'http:://some.remote.dataset.nc'
variable = 'tas'

RequestDataset(uri,variable).inspect()