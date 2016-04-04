from ocgis import RequestDataset, OcgOperations
from ocgis.contrib.library_icclim import IcclimTG90p

########################################################################################################################
# Compute a custom percentile basis using ICCLIM. ######################################################################
########################################################################################################################

# Path to CF climate dataset. This examples uses the same file for indice and percentile basis calculation.
in_file = '/path/to/cf_data.nc'

# Subset the input dataset to return the desired base period for the percentile basis.
variable = 'tas'
years = range(1971, 2001)
time_region = {'year': years}
rd = RequestDataset(uri=in_file, variable=variable)
field = rd.get()
field.get_time_region(time_region)

# Calculate the percentile basis. The data values must be a three-dimensional array.
arr = field.variables[variable].value.squeeze()
dt_arr = field.temporal.value_datetime
percentile = 90
window_width = 5
percentile_dict = IcclimTG90p.get_percentile_dict(arr, dt_arr, percentile, window_width)

########################################################################################################################
# Calculate indice using custom percentile basis. ######################################################################
########################################################################################################################

calc = [{'func': 'icclim_TG90p', 'name': 'TG90p', 'kwds': {'percentile_dict': percentile_dict}}]
calc_grouping = 'month'
ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping)
coll = ops.execute()
