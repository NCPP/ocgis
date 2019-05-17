import ocgis

# Path to the file of interest
PATH = "/home/benkoziol/Dropbox/dtmp/ctsm-grids/SCRIPgrid_4x5_nomask_c110308.nc"
# The metadata/IO driver for the data file.
DRIVER="netcdf-scrip"

# ----------------------------------------------------------------------------------------------------------------------

# The request dataset provides metadata for a field.
rd = ocgis.RequestDataset(PATH, driver=DRIVER)
# The field is similar to ESMF Field and provides access to grids, etc. It can also hold multiple data variables.
field = rd.create_field()
# Convert the coordinates into Shapely polygon objects.
print("This is the geometry abstraction that will be used: {}".format(field.grid.abstraction))
field.set_abstraction_geom()
print("Sending polygons to a shapefile...")
field.geom.write_vector("/tmp/shp_polygon.shp")

# Can also write the points in another way...
field.grid.abstraction = "point"
points = field.grid.get_abstraction_geometry()
points.write_vector("/tmp/shp_points.shp")

