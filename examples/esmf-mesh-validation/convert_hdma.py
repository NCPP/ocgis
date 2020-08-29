import logging

import ocgis
from ocgis.constants import DriverKey
from ocgis.util.logging_ocgis import ocgis_lh

INPATH = '/home/ubuntu/data/hdma_global_catch_v2.gpkg'
# Path to the output ESMF unstructured NetCDF file.
PATH_OUT_NC = '/home/ubuntu/htmp/catchment.nc'
# Name of the feature class inside the file geodatabase to convert to ESMF Unstructured Format.
FEATURE_CLASS = 'Catchment'

ocgis_lh.configure(to_stream=True, level=logging.DEBUG)

# Create the request dataset and field identifying the target feature class in the process.
# rd = ocgis.RequestDataset(PATH_GDB, driver=DriverKey.VECTOR,
#                           driver_kwargs={'feature_class': FEATURE_CLASS})
rd = ocgis.RequestDataset(INPATH, driver=DriverKey.VECTOR,
                          driver_kwargs={'feature_class': FEATURE_CLASS})
field = rd.get()

# Convert the field geometry to an unstructured grid format based on the UGRID spec.
gc = field.geom.convert_to(use_geometry_iterator=True, pack=False,
                           node_threshold=None, split_interiors=False,
                           remove_self_intersects=False,
                           to_crs=ocgis.crs.Spherical(),
                           allow_splitting_excs=False,
                           add_center_coords=True)

ocgis_lh(msg="starting write", logger="convert_nhd")

# When writing the data to file, convert to ESMF unstructured format.
gc.parent.write(PATH_OUT_NC, driver=DriverKey.NETCDF_ESMF_UNSTRUCT)
