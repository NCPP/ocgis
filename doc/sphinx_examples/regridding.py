import os
import tempfile

import ESMF

import ocgis

# global model grid with ~3 degree resolution
URI1 = os.path.expanduser('~/climate_data/CanCM4/tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc')
# downscaled model grid covering the conterminous United States with ~1/8 degree resolution
URI2 = os.path.expanduser('~/climate_data/maurer/bcca/obs/tasmax/1_8deg/gridded_obs.tasmax.OBS_125deg.daily.1991.nc')
ocgis.env.DIR_OUTPUT = tempfile.gettempdir()

########################################################################################################################
# Simple regridding example with a bounding box subset writing to netCDF using conservative regridding

bbox = [-104, 36, -95, 44]
# Regrid the global dataset to the downscaled grid.
rd_global = ocgis.RequestDataset(uri=URI1)
rd_downscaled = ocgis.RequestDataset(uri=URI2)
ops = ocgis.OcgOperations(dataset=rd_global, regrid_destination=rd_downscaled, geom=bbox, output_format='nc',
                          prefix='with_corners')
ret = ops.execute()

########################################################################################################################
# Regrid using bilinear interpolation (i.e. without corners)

rd_global = ocgis.RequestDataset(uri=URI1)
rd_downscaled = ocgis.RequestDataset(uri=URI2)
regrid_options = {'regrid_method': ESMF.RegridMethod.BILINEAR}
ops = ocgis.OcgOperations(dataset=rd_global, regrid_destination=rd_downscaled, geom=bbox, output_format='nc',
                          regrid_options=regrid_options, prefix='without_corners')
ret = ops.execute()
