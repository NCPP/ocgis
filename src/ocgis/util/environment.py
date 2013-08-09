import tempfile
import os
from ocgis.interface.projection import WGS84
from ocgis.exc import OcgisEnvironmentError
from ocgis.util.logging_ocgis import ocgis_lh


class Environment(object):
    
    def __init__(self):
        self.OVERWRITE = EnvParm('OVERWRITE',False,formatter=self._format_bool_)
        self.DIR_OUTPUT = EnvParm('DIR_OUTPUT',os.getcwd())
        self.DIR_SHPCABINET = EnvParm('DIR_SHPCABINET',None)
        self.DIR_DATA = EnvParm('DIR_DATA',None)
        self.DIR_TEST_DATA = EnvParm('DIR_TEST_DATA',None)
        self.SERIAL = EnvParm('SERIAL',True,formatter=self._format_bool_)
        self.CORES = EnvParm('CORES',6,formatter=int)
        self.MODE = EnvParm('MODE','raw')
        self.PREFIX = EnvParm('PREFIX','ocgis_output')
        self.FILL_VALUE = EnvParm('FILL_VALUE',1e20,formatter=float)
        self.VERBOSE = EnvParm('VERBOSE',False,formatter=self._format_bool_)
        self.OPTIMIZE_FOR_CALC = EnvParm('OPTIMIZE_FOR_CALC',False,formatter=self._format_bool_)
        self.WRITE_TO_REFERENCE_PROJECTION = EnvParm('WRITE_TO_REFERENCE_PROJECTION',False,formatter=self._format_bool_)
        self.ENABLE_FILE_LOGGING = EnvParm('ENABLE_FILE_LOGGING',True,formatter=self._format_bool_)
        self.DEBUG = EnvParm('DEBUG',False,formatter=self._format_bool_)
        self.REFERENCE_PROJECTION = ReferenceProjection()
        self.USE_CACHING = EnvParm('USE_CACHING',False,formatter=self._format_bool_)
        self.DIR_CACHE = EnvParm('DIR_CACHE',None)
        self.DIR_BIN = EnvParm('DIR_BIN',None)
        
        self.ops = None
        
    def __str__(self):
        msg = []
        for value in self.__dict__.itervalues():
            if isinstance(value,EnvParm):
                msg.append(str(value))
        msg.sort()
        return('\n'.join(msg))
        
    def __getattribute__(self,name):
        attr = object.__getattribute__(self,name)
        try:
            ret = attr.value
        except AttributeError:
            ret = attr
        return(ret)
    
    def __setattr__(self,name,value):
        if isinstance(value,EnvParm) or name in ['ops','_use_logging']:
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
                getattr(value,'value')
        env.ops = None
                
    def _format_bool_(self,value):
        '''Format a string to boolean.
        
        :param value: The value to convert.
        :type value: int or str'''
        from ocgis.util.helpers import format_bool
        return(format_bool(value))


class EnvParm(object):
    
    def __init__(self,name,default,formatter=None):
        self.name = name.upper()
        self.env_name = 'OCGIS_{0}'.format(self.name)
        self.formatter = formatter
        self.default = default
        self._value = 'use_env'
        
    def __str__(self):
        return('{0}={1}'.format(self.name,self.value))
        
    @property
    def value(self):
        if self._value == 'use_env':
            ret = os.getenv(self.env_name)
            if ret is None:
                ret = self.default
            else:
                ## attempt to use the parameter's format method.
                try:
                    ret = self.format(ret)
                except NotImplementedError:
                    if self.formatter is not None:
                        ret = self.formatter(ret)
        else:
            ret = self._value
        return(ret)
    @value.setter
    def value(self,value):
        self._value = value
        
    def format(self,value):
        raise(NotImplementedError)
    
    
class ReferenceProjection(EnvParm):
    
    def __init__(self):
        EnvParm.__init__(self,'REFERENCE_PROJECTION',WGS84())
        
    def format(self,value):
        if os.environ.get(self.env_name) is not None:
            msg = 'REFERENCE_PROJECTION may not be set as a system environment variable. It must be parameterized at runtime.'
            e = OcgisEnvironmentError(self,msg)
            ocgis_lh(exc=e,logger='env')


env = Environment()
