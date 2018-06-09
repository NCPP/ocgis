from ocgis import RequestDataset, OcgOperations
from ocgis.contrib.library_icclim import IcclimTG90p

########################################################################################################################
# Compute a custom percentile basis using ICCLIM.

# Path to CF climate dataset. This examples uses the same file for indice and percentile basis calculation.
in_file = '/path/to/cf_data.nc'

# Subset the input dataset to return the desired base period for the percentile basis.
variable = 'tas'
years = range(2001, 2003)  # A custom date range may be required for your data
time_region = {'year': years}
rd = RequestDataset(uri=in_file, variable=variable)
field = rd.create_field()
field = field.time.get_time_region(time_region).parent

# Calculate the percentile basis. The data values must be a three-dimensional array.
arr = field[variable].get_masked_value().squeeze()  # This is the field data to use for the calculation
dt_arr = field.temporal.value_datetime  # This is an array of datetime objects.
percentile = 90
window_width = 5
t_calendar, t_units = field.time.calendar, field.time.units  # ICCLIM requires calendar and units for the calculation
percentile_dict = IcclimTG90p.get_percentile_dict(arr, dt_arr, percentile, window_width, t_calendar, t_units)

########################################################################################################################
# Calculate indice using custom percentile basis.

# Depending on the size of the data, this computation may take some time...
calc = [{'func': 'icclim_TG90p', 'name': 'TG90p', 'kwds': {'percentile_dict': percentile_dict}}]
calc_grouping = 'month'
# Returns data as an in-memory spatial collection
ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping)
coll = ops.execute()
