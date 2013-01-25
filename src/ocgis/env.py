#: The directory where output data is written.
WORKSPACE = None
#: If `True`, execute in serial.
SERIAL = True
#: If operating in parallel (i.e. :attr:`~ocgis.env.SERIAL` = `False`), specify the number of cores to use.
CORES = 6
MODE = 'raw'
#: The default prefix to apply to output files.
BASE_NAME = 'ocg'
#: Location of the shapefile directory for use by :class:`~ocgis.ShpCabinet`.
SHP_DIR = '~/links/project/ocg/bin/shp'
#: The fill value for masked data in NetCDF output.
FILL_VALUE = 1e20
#: Indicate if additional output information should be printed to terminal. (Currently not very useful.)
VERBOSE = False