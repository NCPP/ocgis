from ocgis.test import create_gridxy_global, create_exact_field

# Create an in-memory field.
grid = create_gridxy_global()
field = create_exact_field(grid, 'a_var')

# Set the test variable's units.
field['a_var'].units = 'odd_source_units'

# Another way to set the source units...
# field['a_var'].attrs['units'] = 'odd_source_units'

# These calls retrieve the underlying data values without a mask.
# value = field['a_var'].v()
# value = field['a_var'].get_value()

# It is best to work with the masked values unless performance is an issue.
masked_value = field['a_var'].mv()
# masked_value = field['a_var'].get_masked_value()

# Convert the units in-place and update the units attribute on the target variable.
masked_value[:] *= 1.75
field['a_var'].units = 'odd_destination_units'
