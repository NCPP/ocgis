import logging
import os
import shutil
from abc import ABCMeta

import six

from ocgis import constants, vm
from ocgis import exc, env
from ocgis.base import AbstractOcgisObject
from ocgis.conv.base import AbstractTabularConverter
from ocgis.conv.meta import AbstractMetaConverter
from ocgis.ops.engine import OperationsEngine
from ocgis.ops.parms.definition import OutputFormat
from ocgis.util.logging_ocgis import ocgis_lh, ProgressOcgOperations


@six.add_metaclass(ABCMeta)
class Interpreter(AbstractOcgisObject):
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

    # todo: creating a directory should be a feature of the output format not something handled by the interpreter
    _no_directory = [constants.OutputFormatName.OCGIS, constants.OutputFormatName.ESMPY_GRID,
                     constants.OutputFormatName.METADATA_JSON, constants.OutputFormatName.METADATA_OCGIS]

    def check(self):
        pass

    def execute(self):
        # check for a user-supplied output prefix
        prefix = self.ops.prefix

        # do directory management #

        # flag to indicate a directory is made. mostly a precaution to make sure the appropriate directory is is
        # removed.
        made_output_directory = False

        if self.ops.output_format in self._no_directory:
            # No output directory for some formats.
            outdir = None
        else:
            # Directories or a single output file(s) is created for the other cases.
            if self.ops.add_auxiliary_files:
                # Auxiliary files require that a directory be created.
                outdir = os.path.join(self.ops.dir_output, prefix)
                # Create and/or remove the output directory.
                if vm.rank == 0:
                    if os.path.exists(outdir):
                        if env.OVERWRITE:
                            shutil.rmtree(outdir)
                        else:
                            raise IOError('The output directory exists but env.OVERWRITE is False: {0}'.format(outdir))
                    os.mkdir(outdir)
                # Block until output directory is created. Most often the zero rank manages writing, but this is not a
                # requirement.
                vm.Barrier()
                # On an exception, the output directory needs to be removed.
                made_output_directory = True
            else:
                # with no auxiliary files the output directory will do just fine
                outdir = self.ops.dir_output

        try:
            # configure logging ########################################################################################

            progress = self._get_progress_and_configure_logging_(outdir, prefix)

            # create local logger
            interpreter_log = ocgis_lh.get_logger('interpreter')

            ocgis_lh('Initializing...', interpreter_log)

            # set up environment #######################################################################################

            # run validation - doesn't do much now
            self.check()

            # do not perform vector wrapping for NetCDF output
            if self.ops.output_format == 'nc':
                ocgis_lh('"vector_wrap" set to False for netCDF output',
                         interpreter_log, level=logging.WARN)
                self.ops.vector_wrap = False

            # if the requested output format is "meta" then no operations are run and only the operations dictionary is
            # required to generate output.
            Converter = self.ops._get_object_(OutputFormat.name).get_converter_class()
            if issubclass(Converter, AbstractMetaConverter):
                ret = Converter(self.ops).write()
            # this is the standard request for other output types.
            else:
                # the operations object performs subsetting and calculations
                ocgis_lh('initializing subset', interpreter_log, level=logging.DEBUG)
                so = OperationsEngine(self.ops, progress=progress)
                # if there is no grouping on the output files, a singe converter is is needed
                if self.ops.output_grouping is None:
                    ocgis_lh('initializing converter', interpreter_log, level=logging.DEBUG)
                    conv = self._get_converter_(Converter, outdir, prefix, so)
                    ocgis_lh('starting converter write loop: {0}'.format(self.ops.output_format), interpreter_log,
                             level=logging.DEBUG)
                    ret = conv.write()
                else:
                    raise NotImplementedError

            ocgis_lh('Operations successful.'.format(self.ops.prefix), interpreter_log)

            return ret
        except:
            # The output directory needs to be removed if one was created. Shutdown logging before to make sure there
            # is no file lock (Windows).
            ocgis_lh.shutdown()
            if vm.rank == 0 and made_output_directory:
                shutil.rmtree(outdir)
            raise
        finally:
            ocgis_lh.shutdown()

            if env.ADD_OPS_MPI_BARRIER:
                vm.Barrier()

    def _get_converter_(self, conv_klass, outdir, prefix, so):
        """
        :param conv_klass: The target converter class.
        :type conv_klass: :class:`ocgis.conv.base.AbstractCollectionConverter`
        :param str outdir: The output directory to contain converted files.
        :param str prefix: The file prefix for the file outputs.
        :param so: The subset operation object doing all the work.
        :type so: :class:`ocgis.driver.subset.OperationsEngine`
        :returns: A converter object.
        :rtype: :class:`ocgis.conv.base.AbstractCollectionConverter`
        """

        kwargs = dict(outdir=outdir, prefix=prefix, ops=self.ops, add_auxiliary_files=self.ops.add_auxiliary_files,
                      overwrite=env.OVERWRITE, options=self.ops.output_format_options)
        if issubclass(conv_klass, AbstractTabularConverter):
            kwargs['melted'] = self.ops.melted
        conv = conv_klass(so, **kwargs)
        return conv

    def _get_progress_and_configure_logging_(self, outdir, prefix):
        """
        :param str outdir: The output directory for the operations.
        :param str prefix: The file prefix to use when creating the output files.
        :returns: A progress object to use when executing the operations.
        :rtype: :class:`ocgis.util.logging_ocgis.ProgressOcgOperations`
        """

        # If file logging is enabled, check where or if the log should be written.
        if env.ENABLE_FILE_LOGGING and self.ops.add_auxiliary_files is True:
            if self.ops.output_format in self._no_directory:
                to_file = None
            else:
                if vm.rank == 0:
                    os.makedirs(os.path.join(outdir, 'logs'))
                vm.Barrier()
                to_file = os.path.join(outdir, 'logs', '{prefix}-rank-{rank}.log'.format(prefix=prefix, rank=vm.rank))
        else:
            to_file = None

        # Flags to determine streaming to console.
        if env.VERBOSE:
            to_stream = True
        else:
            to_stream = False

        # Configure the logger.
        if env.DEBUG:
            level = logging.DEBUG
        else:
            level = logging.INFO
        # This wraps the callback function with methods to capture the completion of major operations.
        progress = ProgressOcgOperations(callback=self.ops.callback)
        ocgis_lh.configure(to_file=to_file, to_stream=to_stream, level=level, callback=progress, callback_level=level)

        return progress
