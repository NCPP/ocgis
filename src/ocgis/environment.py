import os
import subprocess
import sys
from importlib import import_module
from warnings import warn

from ocgis.base import AbstractOcgisObject
from ocgis.exc import OcgWarning

# HACK!! on some systems, there are issues with loading a parallel ESMF installation if this import occurs in a
# different location. it is unclear what mechanism causes the import issue. ESMF is not a required package, so a failed
# import is okay (if it is not installed).
try:
    import ESMF
except ImportError:
    pass

# HACK!! the gdal data is often not read correctly by the osgeo installation. remove the necessity for users to set this
# variable when installing.

# Need to check for bad GDAL data directories.
if 'GDAL_DATA' in os.environ:
    if not os.path.exists(os.environ['GDAL_DATA']):
        msg = 'The GDAL_DATA directory set in environment does not exist!: {}. It will be overloaded.'.format(
            os.environ['GDAL_DATA'])
        warn(OcgWarning(msg))
        os.environ.pop('GDAL_DATA')

if 'GDAL_DATA' not in os.environ:
    try:
        if os.name == 'nt':
            # Try setting the value assuming an Anaconda environment on Windows.
            datadir = os.path.join(os.path.split(sys.executable)[0], 'share', 'gdal')
            if not os.path.exists(datadir):
                datadir = os.path.join(os.path.split(sys.executable)[0], 'Library', 'share', 'gdal')
        else:
            # Try using the gdal-config executable to find the data directory.
            datadir = subprocess.check_output(['gdal-config', '--datadir']).decode().strip()
            datadir = datadir.strip('\\n')
    except:
        pass
    else:
        os.environ['GDAL_DATA'] = datadir
        msg = 'Consider setting the system environment variable "GDAL_DATA={0}" to improve load performance'. \
            format(datadir)
        warn(OcgWarning(msg))
        from osgeo import gdal

        gdal.SetConfigOption('GDAL_DATA', str(datadir))

from osgeo import osr, ogr
import numpy as np
from ocgis import constants
from ocgis.util.helpers import get_iter

# tell ogr/osr to raise exceptions
ogr.UseExceptions()
osr.UseExceptions()


class Environment(AbstractOcgisObject):
    def __init__(self):
        self.OVERWRITE = EnvParm('OVERWRITE', False, formatter=self._format_bool_)
        self.DIR_OUTPUT = EnvParm('DIR_OUTPUT', os.getcwd())
        self.DIR_GEOMCABINET = EnvParm('DIR_GEOMCABINET', None)
        # Left in for backwards compatibility. Performs the same function as DIR_GEOMCABINET.
        self.DIR_SHPCABINET = EnvParm('DIR_SHPCABINET', None)
        self.DIR_DATA = EnvParm('DIR_DATA', None)
        self.DIR_TEST_DATA = EnvParm('DIR_TEST_DATA', None)
        self.MELTED = EnvParm('MELTED', None, formatter=self._format_bool_)
        self.MODE = EnvParm('MODE', 'raw')
        self.PREFIX = EnvParm('PREFIX', 'ocgis_output')
        self.FILL_VALUE = EnvParm('FILL_VALUE', 1e20, formatter=float)
        self.VERBOSE = EnvParm('VERBOSE', False, formatter=self._format_bool_)
        self.OPTIMIZE_FOR_CALC = EnvParm('OPTIMIZE_FOR_CALC', False, formatter=self._format_bool_)
        self.ENABLE_FILE_LOGGING = EnvParm('ENABLE_FILE_LOGGING', False, formatter=self._format_bool_)
        self.DEBUG = EnvParm('DEBUG', False, formatter=self._format_bool_)
        self.DIR_BIN = EnvParm('DIR_BIN', None)
        self.USE_SPATIAL_INDEX = EnvParmImport('USE_SPATIAL_INDEX', None, 'rtree')
        self.USE_CFUNITS = EnvParmImport('USE_CFUNITS', None, ('cf_units', 'cfunits'))
        self.USE_ESMF = EnvParmImport('USE_ESMF', None, 'ESMF')
        self.USE_ICCLIM = EnvParmImport('USE_ICCLIM', None, 'icclim')
        self.CONF_PATH = EnvParm('CONF_PATH', os.path.expanduser('~/.config/ocgis.conf'))
        self.SUPPRESS_WARNINGS = EnvParm('SUPPRESS_WARNINGS', True, formatter=self._format_bool_)
        self.DEFAULT_GEOM_UID = EnvParm('DEFAULT_GEOM_UID', constants.OCGIS_UNIQUE_GEOMETRY_IDENTIFIER, formatter=str)
        self.NETCDF_FILE_FORMAT = EnvParm('NETCDF_FILE_FORMAT', constants.NETCDF_DEFAULT_DATA_MODEL, formatter=str)
        self.NP_INT = EnvParm('NP_INT', constants.DEFAULT_NP_INT)
        self.NP_FLOAT = EnvParm('NP_FLOAT', constants.DEFAULT_NP_FLOAT)
        self.ADD_OPS_MPI_BARRIER = EnvParm('ADD_OPS_MPI_BARRIER', True, formatter=self._format_bool_)
        self.PREFER_NETCDFTIME = EnvParm('PREFER_NETCDFTIME', None, formatter=self._format_bool_)

        if self.PREFER_NETCDFTIME is None:
            self.PREFER_NETCDFTIME = get_netcdftime_preference()

        from ocgis.variable.crs import CFSpherical

        self.DEFAULT_COORDSYS = EnvParm('DEFAULT_COORDSYS', CFSpherical())

        self.ops = None
        self._optimize_store = {}

    def __str__(self):
        msg = []
        for value in self.__dict__.values():
            if isinstance(value, EnvParm):
                msg.append(str(value))
        msg.sort()
        return '\n'.join(msg)

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        try:
            ret = attr.value
        except AttributeError:
            ret = attr
        return ret

    def __setattr__(self, name, value):
        if isinstance(value, EnvParm) or name in ['ops'] or name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            attr = object.__getattribute__(self, name)
            attr.value = value
            if attr.on_change is not None:
                attr.on_change()

    def get_geomcabinet_path(self):
        return self.DIR_GEOMCABINET or self.DIR_SHPCABINET

    def set_geomcabinet_path(self, value):
        self.DIR_SHPCABINET = value
        self.DIR_GEOMCABINET = value

    def reset(self):
        """
        Reset values to defaults (Values will be read from any overloaded system environment variables.
        """

        for value in self.__dict__.values():
            if isinstance(value, EnvParm):
                value._value = 'use_env'
                getattr(value, 'value')
        env.ops = None
        self._optimize_store = {}

    @staticmethod
    def _format_bool_(value):
        """
        Format a string to boolean.

        :param value: The value to convert.
        :type value: int or str
        """

        from ocgis.util.helpers import format_bool

        return format_bool(value)

    def _get_property_dtype_(self, name_private, name_dtype):
        attr_value = getattr(self, name_private)
        if attr_value is None:
            netcdf_file_format = self.NETCDF_FILE_FORMAT
            """:type netcdf_file_format: str"""
            dtype = get_dtype(name_dtype, data_model=netcdf_file_format)
            setattr(self, name_private, EnvParm(name_private[1:], dtype))
        return object.__getattribute__(self, name_private)


class EnvParm(object):
    def __init__(self, name, default, formatter=None, on_change=None):
        self.name = name.upper()
        self.env_name = 'OCGIS_{0}'.format(self.name)
        self.formatter = formatter
        self.default = default
        self.on_change = on_change
        self._value = 'use_env'

    def __str__(self):
        return '{0}={1}'.format(self.name, self.value)

    @property
    def value(self):
        if self._value == 'use_env':
            ret = os.getenv(self.env_name)
            if ret is None:
                ret = self.default
            else:
                # attempt to use the parameter's format method.
                try:
                    ret = self.format(ret)
                except NotImplementedError:
                    if self.formatter is not None:
                        ret = self.formatter(ret)
        else:
            ret = self._value
        return ret

    @value.setter
    def value(self, value):
        self._value = value

    def format(self, value):
        raise NotImplementedError


class EnvParmImport(EnvParm):
    def __init__(self, name, default, module_names):
        self.module_names = module_names
        super(EnvParmImport, self).__init__(name, default)

    @property
    def value(self):
        if self._value == 'use_env':
            ret = os.getenv(self.env_name)
            if ret is None:
                if self.default is None:
                    ret = self._get_module_available_()
                else:
                    ret = self.default
            else:
                ret = Environment._format_bool_(ret)
        else:
            ret = self._value
        return ret

    @value.setter
    def value(self, value):
        self._value = value

    def _get_module_available_(self):
        results = []
        for m in get_iter(self.module_names):
            try:
                import_module(m)
                app = True
            except ImportError:
                app = False
            results.append(app)
        return any(results)


def get_dtype(string_name, data_model=None):
    """
    :param str string_name: The name of the data type: ``'int'`` or ``'float'``.
    :param str data_model: The target data model string designation.
    :return: The appropriate data type for the ``string_name`` and ``netcdf_file_format``.
    :rtype: type
    """

    # The classic format does not support 64-bit data.
    if data_model == 'NETCDF3_CLASSIC':
        mp = {'int': np.int32,
              'float': np.float32}
    elif data_model in ('NETCDF3_64BIT_OFFSET', 'NETCDF3_64BIT'):
        mp = {'int': np.int32,
              'float': constants.DEFAULT_NP_FLOAT}
    else:
        mp = {'int': constants.DEFAULT_NP_INT,
              'float': constants.DEFAULT_NP_FLOAT}
    return mp[string_name]


def get_netcdftime_preference():
    # netcdftime added more advanced operation in later versions. Check if __add__ is available on the class to
    # determine usage preference.
    from netCDF4 import netcdftime

    if hasattr(netcdftime.datetime, '__add__'):
        ret = True
    else:
        ret = False
    return ret


# Always keep this later to make sure functions may be used in environment set-up.
env = Environment()
