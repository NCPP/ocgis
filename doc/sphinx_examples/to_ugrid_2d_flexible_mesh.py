import tempfile

import ocgis
from ocgis.constants import OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH


# This is the input shapefile with no gaps between the polygons. Multipolygons not allowed!
SHP = '/path/to/no_gaps/shapefile.shp'
# Write the data to a temporary directory.
ocgis.env.DIR_OUTPUT = tempfile.gettempdir()


rd = ocgis.RequestDataset(uri=SHP)
ops = ocgis.OcgOperations(dataset=rd, output_format=OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH)
ret = ops.execute()
