"""
Provides a general overview of the OCGIS spatial collection object.
"""

from ocgis import SpatialCollection, GeometryVariable, Variable, crs
from ocgis.test.base import create_gridxy_global, create_exact_field
from ocgis.util.helpers import pprint_dict
from shapely.geometry import box

# Spatial collections are hierarchical and may contain groups (children). In OCGIS, the first spatial collection groups
# are subset geometries. The children or groups within the subset geometry groups contain the data subset by the
# geometry. This is easiest to understand by walking through how OCGIS subsets and assembles a spatial collection inside
# operations.

# Create a test grid.
grid = create_gridxy_global()

# Create an exact field on the grid.
field = create_exact_field(grid, 'foo')

# Create a subset geometry.
subset_geom = box(30., 20., 40., 25.)

# Subset the field using the geometry. Note we subset the grid and return its parent (the field).
sub = field.grid.get_intersects(subset_geom).parent

# Subset geometries are themselves treated as fields. Convert the geometry to an OCGIS geometry variable.
gvar = GeometryVariable.from_shapely(subset_geom, ugid=11, crs=crs.Spherical())

# Add some descriptive variables.
info = Variable(name='desc', value=['random bbox'], dtype=str, dimensions=gvar.dimensions[0])
gvar.parent.add_variable(info, is_data=True)
height = Variable(name='height', value=[30], dimensions=gvar.dimensions[0], attrs={'made_up': 'yes'})
gvar.parent.add_variable(height, is_data=True)

# Create an empty spatial collection.
sc = SpatialCollection()

# Add the subsetted field with its containing geometry.
sc.add_field(sub, gvar.parent)

# These are the container geometries for the spatial collection.
pprint_dict(sc.geoms)

# These are the properties of those geometries.
pprint_dict(sc.properties)
print('')

# There are a number of ways to access data in the spatial collection. You can navigate the hierarchy directly..
var = sc.groups[11].groups[field.name]['foo']
print('Variable Mean:', var.v().mean())
print('')

# All the fields are accessible via an iterator...
for field, field_container in sc.iter_fields(yield_container=True):
    print(field_container.geom.ugid.v()[0], field.keys())

# A melted iterator may be used to access each element in the collection...
for element in sc.iter_melted():
    print(element)

# Lastly, the get element method is useful to access something specific...
target = sc.get_element(variable_name='foo', container_ugid=11)
print('')
print(target.name, target.dimensions)
