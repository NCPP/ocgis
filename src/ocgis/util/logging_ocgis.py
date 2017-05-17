import datetime
import logging
import os
import sys
from warnings import warn

import six

import ocgis
from ocgis import env
from ocgis.exc import OcgWarning
from ocgis.vmachine.mpi import get_standard_comm_state

fiona_logger = logging.getLogger('Fiona')
fiona_logger.setLevel(logging.ERROR)


class ProgressOcgOperations(object):
    """
    :param function callback: A function taking two parameters: ``percent_complete`` ``message``.
    :param int n_subsettables: The number of data objects to subset and/or manipulate.
    :param int n_geometries: The number of geometries to use for subsetting.
    :param int n_calculations: The number of calculations to apply.
    """

    def __init__(self, callback=None, n_subsettables=1, n_geometries=1, n_calculations=0):
        assert (n_subsettables > 0)
        assert (n_geometries > 0)

        self.callback = callback
        self.n_subsettables = n_subsettables
        self.n_geometries = n_geometries
        self.n_calculations = n_calculations
        self.n_completed_operations = 0

    def __call__(self, message=None):
        if self.callback is not None:
            return self.callback(self.percent_complete, message)

    @property
    def n_operations(self):
        if self.n_calculations == 0:
            nc = 1
        else:
            nc = self.n_calculations
        return self.n_subsettables * self.n_geometries * nc

    @property
    def percent_complete(self):
        return 100 * (self.n_completed_operations / float(self.n_operations))

    def mark(self):
        self.n_completed_operations += 1


class OcgisLogging(object):
    def __init__(self, comm=None):
        _, rank, _ = get_standard_comm_state(comm=comm)

        self.level = None
        self.null = True  # pass through if not configured
        self.parent = None
        self.callback = None
        self.callback_level = None
        self.loggers = None
        self.rank = rank

        logging.captureWarnings(None)

    def __call__(self, msg=None, logger=None, level=logging.INFO, alias=None, ugid=None, exc=None):
        # attach a default exception to messages to handle warnings if an exception is not provided
        if level == logging.WARN:
            if exc is None:
                exc = OcgWarning(msg)
            if not env.SUPPRESS_WARNINGS:
                warn(exc)

        if self.callback is not None and self.callback_level <= level:
            if msg is not None:
                self.callback(msg)
            elif exc is not None:
                callback_msg = '{0}: {1}'.format(exc.__class__.__name__, exc)
                self.callback(callback_msg)

        if self.null:
            if exc is None or level == logging.WARN:
                pass
            else:
                raise exc
        else:
            dest_level = level or self.level
            # get the logger by string name
            if isinstance(logger, six.string_types):
                dest_logger = self.get_logger(logger)
            else:
                dest_logger = logger or self.parent
            if alias is not None:
                msg = self.get_formatted_msg(msg, alias, ugid=ugid)
            if exc is None:
                dest_logger.log(dest_level, msg)
            else:
                if level == logging.WARN:
                    wmsg = '{0}: {1}'.format(exc.__class__.__name__, str(exc))
                    dest_logger.warn(wmsg)
                else:
                    dest_logger.exception(msg)
                    raise exc

    def configure(self, to_file=None, to_stream=False, level=logging.INFO, callback=None, callback_level=logging.INFO):
        self.callback = callback
        self.callback_level = callback_level

        # No logging to file or stdout. There is nothing to configure.
        if to_file is None and not to_stream:
            self.null = True
        else:
            self.level = level
            self.null = False
            # Log to null if there is no file present.
            if to_file is None:
                filename = os.devnull
            else:
                filename = to_file
            # This is the root logger.
            self.loggers = {}
            self.parent = logging.getLogger('ocgis')
            self.parent.parent = None
            self.parent.setLevel(level)
            self.parent.handlers = []
            # Always configure the file handler. This goes to to null if there is no file path.
            fh = logging.FileHandler(filename, 'w')
            fh.setLevel(level)
            fh.setFormatter(
                logging.Formatter(fmt='%(name)s: rank={}: %(levelname)s: %(asctime)s: %(message)s'.format(self.rank),
                                  datefmt='%Y-%m-%d %H:%M:%S'))
            self.parent.addHandler(fh)
            # This is the stdout logger.
            if to_stream:
                console = logging.StreamHandler()
                console.setLevel(level)
                console.setFormatter(logging.Formatter(
                    fmt='[%(asctime)s]: rank={}: %(levelname)s: %(name)s: %(message)s'.format(self.rank),
                    datefmt='%Y-%m-%d %H:%M:%S'))
                self.parent.addHandler(console)

            # Capture warnings if requested by the environment.
            logging.captureWarnings(env.SUPPRESS_WARNINGS)

            # Insert the current software version into the logging file.
            self.parent.log(logging.INFO, 'Initialized at {0} (UTC)'.format(datetime.datetime.utcnow()))
            self.parent.log(logging.INFO, 'OpenClimateGIS v{0}'.format(ocgis.__release__))

            # If we are debugging, print some additional information.
            if self.level == logging.DEBUG:
                self._log_versions_()

    @staticmethod
    def get_formatted_msg(msg, alias, ugid=None):
        if ugid is None:
            ret = 'alias={0}: {1}'.format(alias, msg)
        else:
            ret = 'alias={0}, ugid={1}: {2}'.format(alias, ugid, msg)
        return ret

    def get_logger(self, name):
        if self.null:
            ret = None
        else:
            ret = logging.getLogger('ocgis').getChild(name)
        return ret

    def shutdown(self):
        self.__init__()
        try:
            logging.shutdown()
            logging.captureWarnings(None)
        except:
            pass

    def _log_versions_(self):
        versions = get_versions()
        versions = ', '.join(['{0}={1}'.format(k, v) for k, v in versions.items()])

        self.parent.log(logging.DEBUG, 'Dependency versions: {0}'.format(versions))


def get_versions():
    try:
        import cfunits
    except ImportError:
        v_cfunits = None
    else:
        v_cfunits = cfunits.__version__
    try:
        import cf_units
    except ImportError:
        v_cf_units = None
    else:
        v_cf_units = cf_units.__version__
    try:
        import ESMF
    except ImportError:
        v_esmf = None
    else:
        v_esmf = ESMF.__release__
    try:
        import icclim
    except ImportError:
        v_icclim = None
    else:
        v_icclim = icclim.__version__
    try:
        import rtree
    except ImportError:
        v_rtree = None
    else:
        v_rtree = rtree.__version__
    try:
        import mpi4py
    except ImportError:
        v_mpi4py = None
    else:
        v_mpi4py = mpi4py.__version__
    import fiona
    v_fiona = fiona.__version__
    import netCDF4
    v_netcdf4 = netCDF4.__version__
    import numpy
    v_numpy = numpy.__version__
    from ocgis import osgeo
    v_gdal = osgeo.__version__
    import six
    v_six = six.__version__
    import pyproj
    v_pyproj = pyproj.__version__
    try:
        import nose
    except ImportError:
        v_nose = None
    else:
        v_nose = nose.__version__

    versions = dict(esmf=v_esmf, cfunits=v_cfunits, rtree=v_rtree, gdal=v_gdal, numpy=v_numpy, netcdf4=v_netcdf4,
                    icclim=v_icclim, fiona=v_fiona, cf_units=v_cf_units, mpi4py=v_mpi4py, six=v_six, pyproj=v_pyproj,
                    python=sys.version_info, nose=v_nose)
    return versions


ocgis_lh = OcgisLogging()
