#: The directory where output data is written. OpenClimateGIS always creates 
#: temporary directories inside this directory path ensuring data is not 
#: overwritten. Also, many of the output formats have multiple output files 
#: making a single directory location potentially troubling in terms of file 
#: quantity. If `None`, it defaults to the system's temporary directory.
WORKSPACE = None
#: If `True`, execute in serial. Only set to `False` if you are confident in your grasp of the software and operation.
SERIAL = True
#: If operating in parallel (i.e. :attr:`~ocgis.env.SERIAL` = `False`), specify the number of cores to use.
CORES = 6
MODE = 'raw'
#: The default prefix to apply to output files.
BASE_NAME = 'ocg'
#: Location of the shapefile directory for use by :class:`~ocgis.ShpCabinet`.
import os.path
SHP_DIR = os.path.expanduser('~/links/ocgis/bin/shp')
#: The fill value for masked data in NetCDF output.
FILL_VALUE = 1e20
#: Indicate if additional output information should be printed to terminal. (Currently not very useful.)
VERBOSE = False