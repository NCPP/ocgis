import tempfile
import os
from ocgis.util import helpers


class Environment(object):
    
    def __init__(self):
        self.OVERWRITE = EnvParm('OVERWRITE',False,formatter=helpers.format_bool)
        self.DIR_OUTPUT = EnvParm('DIR_OUTPUT',tempfile.gettempdir())
        self.DIR_SHPCABINET = EnvParm('DIR_SHPCABINET',None)
        self.DIR_DATA = EnvParm('DIR_DATA',None)
        self.DIR_TEST_DATA = EnvParm('DIR_TEST_DATA',None)
        self.SERIAL = EnvParm('SERIAL',True,formatter=helpers.format_bool)
        self.CORES = EnvParm('CORES',6,formatter=int)
        self.MODE = EnvParm('MODE','raw')
        self.PREFIX = EnvParm('PREFIX','ocgis_output')
        self.FILL_VALUE = EnvParm('FILL_VALUE',1e20,formatter=float)
        self.VERBOSE = EnvParm('VERBOSE',False,formatter=helpers.format_bool)
        
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
        '''Reset values to defaults (Values will be read from any overloaded
        system environment variables.'''
        for value in self.__dict__.itervalues():
            if isinstance(value,EnvParm):
                value._value = 'use_env'


class EnvParm(object):
    
    def __init__(self,name,default,formatter=None):
        self.name = name.upper()
        self.env_name = 'OCGIS_{0}'.format(self.name)
        self.formatter = formatter
        self.default = default
        self._value = 'use_env'
        
    def __repr__(self):
        return(str(self.value))
        
    @property
    def value(self):
        if self._value == 'use_env':
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
