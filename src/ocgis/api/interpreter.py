from ocgis import exc, env
from ocgis.api.parms import definition
from ocgis.conv.meta import MetaConverter
from ocgis.conv.converter import OcgConverter
from subset import SubsetOperation
from ocgis.util.helpers import union_geoms
import os
import shutil

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
        ## perform basic validation checks
#        definition.validate_update_definition(self.ops)
        ## by parsing operational commands, determine iterator mode for
        ## converters
        definition.identify_iterator_mode(self.ops)
    
    def execute(self):
        self.check() ## run validation
        
        ## check for a user-supplied output prefix
        prefix = self.ops.prefix
            
        ## do directory management.
        if self.ops.output_format == 'numpy':
            outdir = None
        else:
            outdir = os.path.join(env.DIR_OUTPUT,prefix)
            if os.path.exists(outdir):
                if env.OVERWRITE:
                    shutil.rmtree(outdir)
                else:
                    raise(IOError('The output directory exists but env.OVERWRITE is False: {0}'.format(outdir)))
            os.mkdir(outdir)
            
        ## determine if data is remote or local. if the data is remote, it needs
        ## to be downloaded to a temporary location then the calculations
        ## performed on the local data. the downloaded data should be removed
        ## when computations have finished.
        ## TODO: add single download
            
        ## in the case of netcdf output, geometries must be unioned. this is
        ## also true for the case of the selection geometry being requested as
        ## aggregated.
        if self.ops.output_format == 'nc' or self.ops.agg_selection is True:
            self.ops.geom = union_geoms(self.ops.geom)
            
        ## limited operations available for netcdf
        if self.ops.output_format == 'nc':
#            ## computations are not supported currently for netcdf output
#            if self.ops.calc is not None:
#                raise(NotImplementedError('Computational NetCDF output is currently not supported.'))
            self.ops.spatial_operation = 'intersects'
            self.ops.aggregate = False
            self.ops.calc_raw = False
#            self.ops.mode = 'raw'
            self.ops.vector_wrap = False
        
        ## if the requested output format is "meta" then no operations are run
        ## and only the operations dictionary is required to generate output.
        if self.ops.output_format == 'meta':
            ret = MetaConverter(self.ops).write()
        ## this is the standard request for other output types.
        else:
            ## the operations object performs subsetting and calculations
            so = SubsetOperation(self.ops,serial=env.SERIAL,nprocs=env.CORES)
            ## if there is no grouping on the output files, a singe converter is
            ## is needed
            if self.ops.output_grouping is None:
                Conv = OcgConverter.get_converter(self.ops.output_format)
                conv = Conv(so,outdir,prefix,mode=self.ops.mode,ops=self.ops)
                ret = conv.write()
            else:
                raise(NotImplementedError)
        return(ret)
