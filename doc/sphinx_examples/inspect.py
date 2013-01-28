from ocgis import Inspect
import os.path


## Directory holding climate data.
DATA_DIR = '/usr/local/climate_data/CanCM4'
## Location and variable name for a daily decadal temperature simulation.
URI_TAS = os.path.join(DATA_DIR,'tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc')
VAR_TAS = 'tas'


## Inspect the dataset at the global level.
print(Inspect(URI_TAS))
## Inspect a variable contained in a dataset.
print(Inspect(URI_TAS,variable=VAR_TAS))