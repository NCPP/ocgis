import os

import ocgis
from ocgis import OcgOperations, RequestDataset, env
from ocgis.test.base import create_gridxy_global, create_exact_field

# Only return the first time slice.
SNIPPET = True
# Data returns will overwrite in this case. Use with caution!!
env.OVERWRITE = True
# This is where to find the shapfiles.
ocgis.env.DIR_GEOMCABINET = os.path.join(os.getcwd(), os.path.split(ocgis.test.__file__)[0], 'bin')

########################################################################################################################
# Write example datasets for use in this example.

grid = create_gridxy_global(resolution=3.0)
vars = ['ocgis_example_tasmin', 'ocgis_example_tas', 'ocgis_example_tasmax']
paths = ['{}.nc'.format(ii) for ii in vars]
field_names = ['tasmin', 'tas', 'tasmax']
for idx, (path, var) in enumerate(zip(paths, vars)):
    field = create_exact_field(grid.copy(), var, ntime=3)
    field.data_variables[0].get_value()[:] = idx + 1
    field.write(path)
# for path in paths:
#     RequestDataset(path).inspect()

########################################################################################################################
# Create the request dataset objects for the file paths we'd like to subset. Variable will often be auto-discovered and
# default field names will be created. We'll just pass in everything here.

rds = [RequestDataset(uri=uri, variable=var, field_name=field_name) for uri, var, field_name in
       zip(paths, vars, field_names)]

########################################################################################################################
# Return in-memory as an OCGIS collection for a single geometry.

# Return a SpatialCollection, but only for a target state in a U.S. state boundaries shapefile. In this case, the UGID 
# attribute value of 23 is associated with Nebraska.
print('Returning OCGIS format for a state...')
ops = OcgOperations(dataset=rds, spatial_operation='clip', aggregate=True, snippet=SNIPPET, geom='state_boundaries',
                    geom_select_uid=[16])
ret = ops.execute()

# Find the geometry field in the returned collection.
nebraska = ret.children[16]
# Or...
nebraska = ret.groups[16]
# Assert some things about the geometry field.
assert nebraska['STATE_NAME'].get_value()[0] == 'Nebraska'
assert nebraska.geom.geom_type == 'Polygon'

# The Nebraska subset geometry field contains the subsetted and aggregated data for each request dataset. Note how we
# overloaded the field names.
assert nebraska.children.keys() == field_names

# Get the actual aggregated data value. With only one subset geometry, "container_ugid" is not really necessary. By
# default, the first field will be returned without the container identifier. Check the values are what we expect.
for idx, (field_name, var_name) in enumerate(zip(field_names, vars)):
    aggregated_variable = ret.get_element(container_ugid=16, field_name=field_name, variable_name=var_name)
    assert aggregated_variable.get_value().mean() == idx + 1

########################################################################################################################
# Return data as the default OCGIS spatial collection output for all state boundaries.

print('Returning OCGIS format for all states...')
ops = OcgOperations(dataset=rds, spatial_operation='clip', aggregate=True, snippet=SNIPPET, geom='state_boundaries')
ret = ops.execute()
assert len(ret.geoms) == 51

########################################################################################################################
# Write to shapefile

print('Creating ESRI Shapefile...')
ops = OcgOperations(dataset=rds, spatial_operation='clip', aggregate=True, snippet=SNIPPET, geom='state_boundaries',
                    output_format='shp')
path = ops.execute()
assert os.path.exists(path)

########################################################################################################################
# Write to linked CSV and ESRI Shapefile

# Without the snippet, we are writing all data to the linked CSV-Shapefile output format. The operation will take 
# considerably longer.
print('Creating linked CSV and ESRI Shapefile...')
ops = OcgOperations(dataset=rds, spatial_operation='clip', aggregate=True, snippet=False, geom='state_boundaries',
                    output_format='csv-shp')
path = ops.execute()
assert os.path.exists(path)
