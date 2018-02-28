import os
import tempfile

import ocgis
from ocgis.test.base import create_gridxy_global, create_exact_field

# Name of the variable to subset.
VAR_TAS = 'tas'
# Make it easy to switch to non-snippet requests.
SNIPPET = True
# Set output directory for shapefile and keyed formats. (MAKE SURE IT EXISTS!)
ocgis.env.DIR_OUTPUT = tempfile.mkdtemp()
print ocgis.env.DIR_OUTPUT
# The bounding box coordinates [minx, miny, maxx, maxy] for the state of Colorado in WGS84 latitude/longitude
# coordinates.
BBOX = [-109.1, 36.9, -102.0, 41.0]

# Create synthetic data for this example.
grid = create_gridxy_global(resolution=5.0)
field = create_exact_field(grid, VAR_TAS, ntime=31)
data_path = os.path.join(ocgis.env.DIR_OUTPUT, 'ocgis_example_simple_subset.nc')
field.write(data_path)

# This object will be reused so just build it once. Variable names are typically auto-discovered.
rd = ocgis.RequestDataset(data_path, VAR_TAS)

########################################################################################################################
# Returning an OCGIS spatial collection

ret = ocgis.OcgOperations(dataset=rd, geom=BBOX, snippet=SNIPPET).execute()

########################################################################################################################
# Returning conversions

output_formats = ['shp', 'csv', 'csv-shp', 'nc']
for output_format in output_formats:
    prefix = output_format + '_output'
    ops = ocgis.OcgOperations(dataset=rd, geom=BBOX, snippet=SNIPPET, output_format=output_format, prefix=prefix)
    ret = ops.execute()

print('simple_subset Example Finished')
