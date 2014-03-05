from ocgis.util.logging_ocgis import ocgis_lh
import logging
from ocgis import exc, env
from ocgis.conv.meta import MetaConverter
from subset import SubsetOperation
import os
import shutil
from ocgis.conv.base import AbstractConverter

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
        pass
    
    def execute(self):
        ## check for a user-supplied output prefix
        prefix = self.ops.prefix
            
        ## do directory management.
        if self.ops.output_format == 'numpy':
            outdir = None
        else:
            if self.ops.add_auxiliary_files:
                outdir = os.path.join(self.ops.dir_output,prefix)
                if os.path.exists(outdir):
                    if env.OVERWRITE:
                        shutil.rmtree(outdir)
                    else:
                        raise(IOError('The output directory exists but env.OVERWRITE is False: {0}'.format(outdir)))
                os.mkdir(outdir)
            else:
                outdir = self.ops.dir_output
            
        try:
            ## configure logging ###################################################
            
            ## if file logging is enable, perform some logic based on the operational
            ## parameters.
            if env.ENABLE_FILE_LOGGING and self.ops.add_auxiliary_files == True:
                if self.ops.output_format == 'numpy':
                    to_file = None
                else:
                    to_file = os.path.join(outdir,prefix+'.log')
            else:
                to_file = None
            
            ## flags to determine streaming to console
            if env.VERBOSE:
                to_stream = True
            else:
                to_stream = False
    
            ## configure the logger
            if env.DEBUG:
                level = logging.DEBUG
            else:
                level = logging.INFO
            ocgis_lh.configure(to_file=to_file,to_stream=to_stream,level=level)
            
            ## create local logger
            interpreter_log = ocgis_lh.get_logger('interpreter')
            
            ocgis_lh('executing: {0}'.format(self.ops.prefix),interpreter_log)
            
            ## set up environment ##############################################
                
            self.check() ## run validation - doesn't do much now
                
            ## do not perform vector wrapping for NetCDF output
            if self.ops.output_format == 'nc':
                ocgis_lh('"vector_wrap" set to False for netCDF output',
                         interpreter_log,level=logging.WARN)
                self.ops.vector_wrap = False
    
            ## if the requested output format is "meta" then no operations are run
            ## and only the operations dictionary is required to generate output.
            if self.ops.output_format == 'meta':
                ret = MetaConverter(self.ops).write()
            ## this is the standard request for other output types.
            else:
                ## the operations object performs subsetting and calculations
                ocgis_lh('initializing subset',interpreter_log,level=logging.DEBUG)
                so = SubsetOperation(self.ops,serial=env.SERIAL,nprocs=env.CORES)
                ## if there is no grouping on the output files, a singe converter is
                ## is needed
                if self.ops.output_grouping is None:
                    Conv = AbstractConverter.get_converter(self.ops.output_format)
                    ocgis_lh('initializing converter',interpreter_log,
                             level=logging.DEBUG)
                    conv = Conv(so,outdir,prefix,ops=self.ops,add_auxiliary_files=self.ops.add_auxiliary_files)
                    ocgis_lh('starting converter write loop: {0}'.format(self.ops.output_format),interpreter_log,
                             level=logging.DEBUG)
                    ret = conv.write()
                else:
                    raise(NotImplementedError)
            
            ocgis_lh('execution complete: {0}'.format(self.ops.prefix),interpreter_log)

            return(ret)
        finally:
            ## shut down logging
            ocgis_lh.shutdown()
