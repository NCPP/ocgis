import ocgis

# Path to the file of interest
PATH = "/home/benkoziol/Dropbox/dtmp/ctsm-grids/SCRIPgrid_4x5_nomask_c110308.nc"
# The metadata/IO driver for the data file.
DRIVER="netcdf-scrip"

# The request dataset provides metadata for a field.
rd = ocgis.RequestDataset(PATH, driver=DRIVER)
# The field is similar to ESMF Field and provides access to grids, etc. It can also hold multiple data variables.
field = rd.create_field()
print("Average spatial resolution: {}".format(field.grid.resolution))
print("Maximum spatial resolution: {}".format(field.grid.resolution_max))

#-----------------------------------------------------------------------------------------------------------------------

print("\nDo a simple data file inspection...")
rd.inspect()
print("\nThis is the OCGIS interpretation of the metadata (can be customized)...")
rd.dimension_map.pprint(as_dict=True)