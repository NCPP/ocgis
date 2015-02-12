import logging
import os
from warnings import warn


# try to turn off fiona logging except for errors
from ocgis import env
from ocgis.exc import OcgWarning

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
    def __init__(self):
        self.level = None
        self.null = True  # pass through if not configured
        self.parent = None
        self.callback = None
        self.callback_level = None
        self.loggers = None

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
            if isinstance(logger, basestring):
                dest_logger = self.get_logger(logger)
            else:
                dest_logger = logger or self.parent
            if alias is not None:
                msg = self.get_formatted_msg(msg, alias, ugid=ugid)
            if exc is None:
                dest_logger.log(dest_level, msg)
            else:
                if level == logging.WARN:
                    wmsg = '{0}: {1}'.format(exc.__class__.__name__, exc.message)
                    dest_logger.warn(wmsg)
                else:
                    dest_logger.exception(msg)
                    raise exc

    def configure(self, to_file=None, to_stream=False, level=logging.INFO, callback=None, callback_level=logging.INFO):
        # set the callback arguments
        self.callback = callback
        self.callback_level = callback_level

        # no need to configure loggers
        if to_file is None and not to_stream:
            self.null = True
        else:
            self.level = level
            self.null = False
            # add the filehandler if request
            if to_file is None:
                filename = os.devnull
            else:
                filename = to_file
            # create the root logger
            self.loggers = {}
            self.parent = logging.getLogger('ocgis')
            self.parent.parent = None
            self.parent.setLevel(level)
            self.parent.handlers = []
            # add the file handler
            fh = logging.FileHandler(filename, 'w')
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt='%(name)s: %(levelname)s: %(asctime)s: %(message)s',
                                              datefmt='%Y-%m-%d %H:%M'))
            self.parent.addHandler(fh)
            # add the stream handler if requested
            if to_stream:
                console = logging.StreamHandler()
                console.setLevel(level)
                console.setFormatter(logging.Formatter('%(name)s: %(levelname)s: %(message)s'))
                self.parent.addHandler(console)

            # tell logging to capture warnings
            logging.captureWarnings(env.SUPPRESS_WARNINGS)

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


ocgis_lh = OcgisLogging()
