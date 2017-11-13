import ocgis
from ocgis.constants import DriverKey

# Path to the input ESRI File Geodatabase.
PATH_GDB = '/home/ubuntu/data/NHDPlusNationalData/NHDPlusV21_National_Seamless.gdb'
# Path to the output ESMF unstructured NetCDF file.
PATH_OUT_NC = '/tmp/nhd.nc'
# Name of the feature class inside the file geodatabase to convert to ESMF Unstructured Format.
FEATURE_CLASS = 'Catchment'

rd = ocgis.RequestDataset(PATH_GDB, driver=DriverKey.VECTOR, driver_kwargs={'feature_class': FEATURE_CLASS})
field = rd.get()
gc = field.geom.convert_to(use_geometry_iterator=True, pack=False, node_threshold=5000, split_interiors=True)
gc.parent.write(PATH_OUT_NC, driver=DriverKey.NETCDF_ESMF_UNSTRUCT)
