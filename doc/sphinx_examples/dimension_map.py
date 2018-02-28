import os

import ocgis
from ocgis.constants import DimensionMapKey

OUT_NC = os.path.join(os.getcwd(), 'ocgis_example_dimension_map.nc')

########################################################################################################################
# Write some initial data.

var_x = ocgis.Variable(name='nonstandard_x', value=[10, 20, 30], dimensions='nsx')
var_y = ocgis.Variable(name='nonstandard_y', value=[40, 50, 60, 70], dimensions='nsy')
var_t = ocgis.Variable(name='nonstandard_time', value=[1, 2, 3], units='days since 2000-1-1',
                       attrs={'calendar': 'standard'}, dimensions='nst')
data_dimensions = [var_t.dimensions[0], var_x.dimensions[0], var_y.dimensions[0]]
var_data = ocgis.Variable(name='some_data', dimensions=data_dimensions)

vc = ocgis.VariableCollection(variables=[var_x, var_y, var_t, var_data])
vc.write(OUT_NC)

########################################################################################################################
# This metadata is not self-describing. Hence, no data or coordinate variables are interpretable. OpenClimateGIS will
# use standard names for coordinate variables, but these names are not standard!

rd = ocgis.RequestDataset(OUT_NC)

try:
    assert rd.variable
except ocgis.exc.NoDataVariablesFound:
    pass

########################################################################################################################
# Construct the dimension map as an object.

dmap = ocgis.DimensionMap()
dmap.set_variable(DimensionMapKey.TIME, var_t)
dmap.set_variable(DimensionMapKey.X, var_x)
dmap.set_variable(DimensionMapKey.Y, var_y)

rd = ocgis.RequestDataset(OUT_NC, dimension_map=dmap)
assert rd.variable == var_data.name

########################################################################################################################
# Construct the dimension map using a dictionary.

dmap = {'time': {'dimension': ['nst'],
                 'variable': 'nonstandard_time'},
        'x': {'dimension': ['nsx'],
              'variable': 'nonstandard_x'},
        'y': {'dimension': ['nsy'],
              'variable': 'nonstandard_y'}}

rd = ocgis.RequestDataset(OUT_NC, dimension_map=dmap)
assert rd.variable == var_data.name
