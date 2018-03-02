"""
Convert a polygon feature class inside an NHD File Geodatabase to ESMF Unstructured Format. This script should be run in
parallel using MPI to increase performance.
"""

import ocgis
from ocgis.constants import DriverKey

# Path to the input ESRI File Geodatabase.
PATH_GDB = '/home/ubuntu/data/NHDPlusNationalData/NHDPlusV21_National_Seamless.gdb'
# Path to the output ESMF unstructured NetCDF file.
PATH_OUT_NC = '/tmp/nhd.nc'
# Name of the feature class inside the file geodatabase to convert to ESMF Unstructured Format.
FEATURE_CLASS = 'Catchment'

# Create the request dataset and field identifying the target feature class in the process.
rd = ocgis.RequestDataset(PATH_GDB, driver=DriverKey.VECTOR, driver_kwargs={'feature_class': FEATURE_CLASS})
field = rd.get()

# Convert the field geometry to an unstructured grid format based on the UGRID spec.
gc = field.geom.convert_to(use_geometry_iterator=True, pack=False, node_threshold=5000, split_interiors=True)

# When writing the data to file, convert to ESMF unstructured format.
gc.parent.write(PATH_OUT_NC, driver=DriverKey.NETCDF_ESMF_UNSTRUCT)
