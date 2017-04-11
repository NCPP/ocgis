import ocgis
from ocgis import constants
from ocgis.constants import OutputFormatName

URI = '/usr/local/climate_data/CanCM4/tas_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc'
VARIABLE = 'tas'


rd = ocgis.RequestDataset(uri=URI, variable='tas')
ops = ocgis.OcgOperations(dataset=rd, output_format=constants.OutputFormatName.OCGIS, time_region={'month': [1]})
ret = ops.execute()
""":type: ocgis.driver.collection.SpatialCollection"""

# the first key corresponds to the UGID of the selection geometry(s). when there is no selection geometry, this defaults
# to 1.
field_dict = ret[1]

# "field_dict" is a dictionary with key(s) corresponding to the unique Field alias and the value being the associated
# Field object. in most cases, the Field alias is the variable alias unless this is output from a multivariate
# calculation.
field = field_dict['tas']
# metadata dictionary...
field.meta

# a Field object contains variables. these could be calculations outputs as well.
var = field.variables['tas']
# to get the value...the array will always have five dimensions: (realization, time, level, row, column). the extra
# dimensions can be removed with np.squeeze if desired.
var_value = var.value

# it also contains the dimension information...
field.temporal.value
field.temporal.value_datetime
field.temporal.bounds
field.temporal.bounds_datetime

field.spatial.crs
field.spatial.geom.polygon.value
field.spatial.geom.point.value
field.spatial.grid.value
