from ocgis import RequestDataset
from ocgis.test.base import create_gridxy_global, create_exact_field
from ocgis.variable.crs import Spherical

# Create synthetic data for this example.
grid = create_gridxy_global(resolution=5.0)
field = create_exact_field(grid, 'exact', ntime=31, crs=Spherical())
field.write('ocgis_example_inspect_request_dataset.nc')

# Create the request dataset object.
rd = RequestDataset('ocgis_example_inspect_request_dataset.nc')

# Provides a metadata dump for the request dataset.
rd.inspect()

# These are the auto-discovered data variable names.
assert rd.variable == 'exact'

# The dimension map provides information on how OCGIS will interpret the dataset.
rd.dimension_map.pprint()
