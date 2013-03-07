import tempfile
import os
from ocgis.util import helpers


#: Set to `True` to overwrite existing output folders. This will remove the
#: folder if it exists!
OVERWRITE = False

#: The directory where output data is written. OpenClimateGIS always creates 
#: temporary directories inside this directory path ensuring data is not 
#: overwritten. Also, many of the output formats have multiple output files 
#: making a single directory location potentially troubling in terms of file 
#: quantity. If `None`, it defaults to the system's temporary directory.
DIR_OUTPUT = tempfile.gettempdir()

#: Location of the shapefile directory for use by :class:`~ocgis.ShpCabinet`.
DIR_SHPCABINET = os.path.expanduser('~/links/ocgis/bin/shp')

#: Directories to search through to find climate data. If specified, this should
#: be a sequence of directories. It may also be a single directory location. Note
#: that the search may take considerable time if a very high level directory is
#: chosen.
DIR_DATA = None

#: The fill value for masked data in NetCDF output.
#: If `True`, execute in serial. Only set to `False` if you are confident in your grasp of the software and operation.
SERIAL = True

#: If operating in parallel (i.e. :attr:`~ocgis.env.SERIAL` = `False`), specify the number of cores to use.
CORES = 6

MODE = 'raw'

#: The default prefix to apply to output files.
PREFIX = 'ocgis'

FILL_VALUE = 1e20

#: Indicate if additional output information should be printed to terminal. (Currently not very useful.)
VERBOSE = False


class Environment(object):
    
    def __init__(self):
        self.OVERWRITE = EnvParm('OVERWRITE',OVERWRITE,formatter=helpers.format_bool)
        self.DIR_OUTPUT = EnvParm('DIR_OUTPUT',DIR_OUTPUT)
        self.DIR_SHPCABINET = EnvParm('DIR_SHPCABINET',DIR_SHPCABINET)
        self.DIR_DATA = EnvParm('DIR_DATA',DIR_DATA)
        self.SERIAL = EnvParm('SERIAL',SERIAL,formatter=helpers.format_bool)
        self.CORES = EnvParm('CORES',CORES,formatter=int)
        self.MODE = EnvParm('MODE',MODE)
        self.PREFIX = EnvParm('PREFIX',PREFIX)
        self.FILL_VALUE = EnvParm('FILL_VALUE',FILL_VALUE,formatter=float)
        self.VERBOSE = EnvParm('VERBOSE',VERBOSE,formatter=helpers.format_bool)
        
    def __getattribute__(self,name):
        attr = object.__getattribute__(self,name)
        try:
            ret = attr.value
        except AttributeError:
            ret = attr
        return(ret)
    
    def __setattr__(self,name,value):
        if isinstance(value,EnvParm):
            object.__setattr__(self,name,value)
        else:
            attr = object.__getattribute__(self,name)
            attr.value = value
            
    def reset(self):
        for value in self.__dict__.itervalues():
            if isinstance(value,EnvParm):
                value._value = None


class EnvParm(object):
    
    def __init__(self,name,default,formatter=None):
        self.name = name.upper()
        self.env_name = 'OCGIS_{0}'.format(self.name)
        self.formatter = formatter
        self.default = default
        self._value = None
        
    @property
    def value(self):
        if self._value is None:
            ret = os.getenv(self.env_name)
            if ret is None:
                ret = self.default
            else:
                if self.formatter is not None:
                    ret = self.formatter(ret)
        else:
            ret = self._value
        return(ret)
    @value.setter
    def value(self,value):
        self._value = value


env = Environment()
