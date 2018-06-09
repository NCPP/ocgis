from ocgis import OcgOperations
from ocgis.test import create_gridxy_global, create_exact_field
from ocgis.variable import stack

# Create data files using the CF-Grid metadata convention with three timesteps per file ################################

filenames = []
for ii in range(1, 4):
    grid = create_gridxy_global()
    field = create_exact_field(grid, 'data', ntime=3)
    field.time.v()[:] += 10 * ii
    field['data'].v()[:] += 10 * ii
    currfn = 'ocgis_example_stacking_subsets_{}.nc'.format(ii)
    filenames.append(currfn)
    field.write(currfn)

########################################################################################################################

# Subset each file created above using a bounding box and return the data as a spatial collection.
colls = [OcgOperations(dataset={'uri': fn}, geom=[40, 30, 50, 60]).execute() for fn in filenames]
# Extract the fields to stack from each spatial collection.
colls_to_stack = [coll.get_element() for coll in colls]
# Stack the data along the time dimension returning a field.
stacked = stack(colls_to_stack, 'time')
