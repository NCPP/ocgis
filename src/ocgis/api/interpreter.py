import logging
import os
import shutil

from ocgis.util.logging_ocgis import ocgis_lh, ProgressOcgOperations
from ocgis import exc, env
from ocgis.conv.meta import MetaConverter
from subset import SubsetOperation
from ocgis.conv.base import AbstractConverter, AbstractTabularConverter


class Interpreter(object):
    """Superclass for custom interpreter frameworks.

    :param ops: The input operations object to interpret.
    :type ops: :class:`ocgis.OcgOperations`
    """

    def __init__(self, ops):
        self.ops = ops

    @classmethod
    def get_interpreter(cls, ops):
        """Select interpreter class."""

        imap = {'ocg': OcgInterpreter}
        try:
            return imap[ops.backend](ops)
        except KeyError:
            raise exc.InterpreterNotRecognized

    def check(self):
        """Validate operation definition dictionary."""

        raise NotImplementedError

    def execute(self):
        """
        Run requested operations and return a path to the output file or a NumPy-based output object depending on
        specification.
        """

        raise NotImplementedError


class OcgInterpreter(Interpreter):
    """The OCGIS interpreter and execution framework."""

    def check(self):
        pass

    def execute(self):
        # check for a user-supplied output prefix
        prefix = self.ops.prefix

        # do directory management #

        # flag to indicate a directory is made. mostly a precaution to make sure the appropriate directory is is
        # removed.
        made_output_directory = False

        if self.ops.output_format in ['numpy', 'esmpy', 'meta']:
            # no output directory for numpy output
            outdir = None
        else:
            # directories or a single output file(s) is created for the other cases
            if self.ops.add_auxiliary_files:
                # auxiliary files require that a directory be created
                outdir = os.path.join(self.ops.dir_output, prefix)
                if os.path.exists(outdir):
                    if env.OVERWRITE:
                        shutil.rmtree(outdir)
                    else:
                        raise (IOError('The output directory exists but env.OVERWRITE is False: {0}'.format(outdir)))
                os.mkdir(outdir)
                # on an exception, the output directory needs to be removed
                made_output_directory = True
            else:
                # with no auxiliary files the output directory will do just fine
                outdir = self.ops.dir_output

        try:
            # # configure logging ###################################################

            # if file logging is enable, perform some logic based on the operational parameters.
            if env.ENABLE_FILE_LOGGING and self.ops.add_auxiliary_files is True:
                if self.ops.output_format in ['numpy', 'esmpy', 'meta']:
                    to_file = None
                else:
                    to_file = os.path.join(outdir, prefix + '.log')
            else:
                to_file = None

            # flags to determine streaming to console
            if env.VERBOSE:
                to_stream = True
            else:
                to_stream = False

            # configure the logger
            if env.DEBUG:
                level = logging.DEBUG
            else:
                level = logging.INFO
            # this wraps the callback function with methods to capture the completion of major operations.
            progress = ProgressOcgOperations(callback=self.ops.callback)
            ocgis_lh.configure(to_file=to_file, to_stream=to_stream, level=level, callback=progress,
                               callback_level=level)

            # create local logger
            interpreter_log = ocgis_lh.get_logger('interpreter')

            ocgis_lh('Initializing...', interpreter_log)

            # # set up environment ##############################################

            # run validation - doesn't do much now
            self.check()

            # do not perform vector wrapping for NetCDF output
            if self.ops.output_format == 'nc':
                ocgis_lh('"vector_wrap" set to False for netCDF output',
                         interpreter_log, level=logging.WARN)
                self.ops.vector_wrap = False

            # if the requested output format is "meta" then no operations are run and only the operations dictionary is
            # required to generate output.
            if self.ops.output_format == 'meta':
                ret = MetaConverter(self.ops).write()
            # this is the standard request for other output types.
            else:
                # the operations object performs subsetting and calculations
                ocgis_lh('initializing subset', interpreter_log, level=logging.DEBUG)
                so = SubsetOperation(self.ops, progress=progress)
                # if there is no grouping on the output files, a singe converter is is needed
                if self.ops.output_grouping is None:
                    ocgis_lh('initializing converter', interpreter_log, level=logging.DEBUG)
                    conv = self._get_converter_(outdir, prefix, so)
                    ocgis_lh('starting converter write loop: {0}'.format(self.ops.output_format), interpreter_log,
                             level=logging.DEBUG)
                    ret = conv.write()
                else:
                    raise NotImplementedError

            ocgis_lh('Operations successful.'.format(self.ops.prefix), interpreter_log)

            return ret
        except:
            # on an exception, the output directory needs to be removed if one was created. once the output directory is
            # removed, reraise.
            if made_output_directory:
                shutil.rmtree(outdir)
            raise
        finally:
            # shut down logging
            ocgis_lh.shutdown()

    def _get_converter_(self, outdir, prefix, so):
        """
        :param str outdir: The output directory to contain converted files.
        :param str prefix: The file prefix for the file outputs.
        :param so: The subset operation object doing all the work.
        :type so: :class:`ocgis.api.subset.SubsetOperation`
        :returns: A converter object.
        :rtype: :class:`ocgis.conv.base.AbstractConverter`
        """

        conv_klass = AbstractConverter.get_converter(self.ops.output_format)
        kwargs = dict(outdir=outdir, prefix=prefix, ops=self.ops, add_auxiliary_files=self.ops.add_auxiliary_files,
                      overwrite=env.OVERWRITE)
        if issubclass(conv_klass, AbstractTabularConverter):
            kwargs['melted'] = self.ops.melted
        conv = conv_klass(so, **kwargs)
        return conv
