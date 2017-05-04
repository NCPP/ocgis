from shapely.geometry import box

import ocgis
from ocgis.test.base import create_gridxy_global, create_exact_field

assert ocgis.vm.size_global == 2

# Create synthetic data for this example. Create and write the data on a single process. Subcommunicators are always
# given a name.
with ocgis.vm.scoped('serial write', [0]):
    # There will be null communicators for the duration of the context manager.
    if not ocgis.vm.is_null:
        grid = create_gridxy_global(resolution=5.0, dist=False)
        field = create_exact_field(grid, 'exact', ntime=31)
        field.write('ocgis_parallel_example.nc')

# Place a barrier so write can finish. Race conditions can occur with subcommunicators. Generally, barriers are managed
# by the operations.
ocgis.vm.barrier()

# This is our subset geometry.
bbox = [150, 30, 170, 50]
bbox = box(*bbox)

# Load the data from file.
rd = ocgis.RequestDataset('ocgis_parallel_example.nc')
field = rd.get()

# By default, data is distributed along the largest spatial dimension. The x or longitude dimension in this case.
distributed_dimension = field.grid.dimensions[1]
bounds_local = {0: (0, 36), 1: (36, 72)}
assert bounds_local[ocgis.vm.rank] == distributed_dimension.bounds_local
assert (0, 72) == distributed_dimension.bounds_global

# Subset by the geometry.
sub = field.grid.get_intersects(bbox).parent

# Empty objects are returned from spatial functions as the emptiness may be important for analysis. Empty objects do not
# have values and many functions will raise an exception if they are present.
if ocgis.vm.rank == 0:
    assert sub.is_empty
else:
    assert not sub.is_empty

# We can scope the VM by emptyable objects.
with ocgis.vm.scoped_by_emptyable('some math', sub):
    if ocgis.vm.is_null:
        print('Empty data on rank {}'.format(ocgis.vm.rank_global))
    else:
        the_shape = sub.data_variables[0].shape
        print('Not empty on rank {} and the shape is {}'.format(ocgis.vm.rank_global, the_shape))
        # Scoping is needed for writing.
        sub.write('ocgis_parallel_example_subset.nc')

print('Parallel example finished')
