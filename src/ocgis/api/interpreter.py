from ocgis import exc, env, constants
from ocgis.api.parms import definition
from ocgis.conv.meta import MetaConverter
from ocgis.conv.base import OcgConverter
from subset import SubsetOperation
import os
import shutil
import logging
import tempfile
from ocgis.util.logging_ocgis import configure_logging, ocgis_lh

## TODO: add method to estimate request size

class Interpreter(object):
    '''Superclass for custom interpreter frameworks.
    
    ops :: OcgOperations'''
    
    def __init__(self,ops):
        self.ops = ops
        
    @classmethod
    def get_interpreter(cls,ops):
        '''Select interpreter class.'''
        
        imap = {'ocg':OcgInterpreter}
        try:
            return(imap[ops.backend](ops))
        except KeyError:
            raise(exc.InterpreterNotRecognized)
        
    def check(self):
        '''Validate operation definition dictionary.'''
        raise(NotImplementedError)
    
    def execute(self):
        '''Run requested operations and return a path to the output file or a
        NumPy-based output object depending on specification.'''
        raise(NotImplementedError)


class OcgInterpreter(Interpreter):
    '''The OCGIS interpreter and execution framework.'''
    
    def check(self):
        definition.identify_iterator_mode(self.ops)
    
    def execute(self):
        ## check for a user-supplied output prefix
        prefix = self.ops.prefix
            
        ## do directory management.
        if self.ops.output_format == 'numpy':
            outdir = None
        else:
            outdir = os.path.join(self.ops.dir_output,prefix)
            if os.path.exists(outdir):
                if env.OVERWRITE:
                    shutil.rmtree(outdir)
                else:
                    raise(IOError('The output directory exists but env.OVERWRITE is False: {0}'.format(outdir)))
            os.mkdir(outdir)
            
        ## configure logging ###################################################
        
        ## if file logging is enable, perform some logic based on the operational
        ## parameters.
        if env.ENABLE_FILE_LOGGING:
            ## for numpy output, we do not need need to write the log file, unless
            ## the level is DEBUG.
            if constants.logging_level == logging.DEBUG and self.ops.output_format == 'numpy':
                add_filehandler = True
                filename = os.path.join(tempfile.gettempdir(),prefix+'.log')
            ## if the output is numpy and we are not logging at the debug level,
            ## then do not create the filehandler.
            elif constants.logging_level != logging.DEBUG and self.ops.output_format == 'numpy':
                add_filehandler = False
                filename = None
            ## add a log file to the output directory
            else:
                add_filehandler = True
                filename = os.path.join(outdir,prefix+'.log')
        else:
            add_filehandler = False
            filename = None
        ## configure the logger
        configure_logging(add_filehandler=add_filehandler,filename=filename)
        
        ## create local logger
        if add_filehandler:
            interpreter_log = logging.getLogger('interpreter')
        else:
            interpreter_log = None
        
        ocgis_lh('executing: {0}'.format(self.ops.prefix),interpreter_log)
        
        ## add operations to environment
        env.ops = self.ops
            
        self.check() ## run validation
            
        ## determine if data is remote or local. if the data is remote, it needs
        ## to be downloaded to a temporary location then the calculations
        ## performed on the local data. the downloaded data should be removed
        ## when computations have finished.
        ## TODO: add single download
            
        ## in the case of netcdf output, geometries must be unioned. this is
        ## also true for the case of the selection geometry being requested as
        ## aggregated.
        if (self.ops.output_format == 'nc' or self.ops.agg_selection is True) and self.ops.geom is not None and len(self.ops.geom) > 1:
            ocgis_lh('aggregating selection geometry',interpreter_log)
            self.ops.geom.aggregate()
            
        ## do not perform vector wrapping for NetCDF output
        if self.ops.output_format == 'nc':
            ocgis_lh('"vector_wrap" set to False for netCDF output',interpreter_log,level=logging.WARN)
            self.ops.vector_wrap = False

        ## if the requested output format is "meta" then no operations are run
        ## and only the operations dictionary is required to generate output.
        if self.ops.output_format == 'meta':
            ret = MetaConverter(self.ops).write()
        ## this is the standard request for other output types.
        else:
            ## the operations object performs subsetting and calculations
            ocgis_lh('initializing subset',interpreter_log)
            so = SubsetOperation(self.ops,serial=env.SERIAL,nprocs=env.CORES,validate=True)
            ## if there is no grouping on the output files, a singe converter is
            ## is needed
            if self.ops.output_grouping is None:
                Conv = OcgConverter.get_converter(self.ops.output_format)
                ocgis_lh('initializing converter',interpreter_log)
                conv = Conv(so,outdir,prefix,mode=self.ops.mode,ops=self.ops)
                ocgis_lh('starting converter write loop',interpreter_log)
                ret = conv.write()
            else:
                raise(NotImplementedError)
        
        ocgis_lh('execution complete: {0}'.format(self.ops.prefix),interpreter_log)
        
        try:
            return(ret)
        finally:
            logging.shutdown()
