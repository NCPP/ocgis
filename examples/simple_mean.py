import ocgis
from ocgis.test.base import create_gridxy_global, create_exact_field

# Path to the output netCDf file.
PATH = '/tmp/foo.nc'

# Create a test grid.
grid = create_gridxy_global()
# Create an exact field on the grid.
field = create_exact_field(grid, 'foo')
# Write the field to disk.
field.write(PATH)

# Calculate a monthly mean.
ops = ocgis.OcgOperations(dataset={'uri': PATH}, calc=[{'func': 'mean', 'name': 'mean'}], calc_grouping=['month'])
# Exexcute the operations. This may be done in parallel with "mpirun".
ret = ops.execute()

# Inspect the data on rank 0 only.
if ocgis.vm.rank == 0:
    ocgis.RequestDataset(uri=PATH).inspect()
