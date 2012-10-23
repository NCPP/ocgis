from ocgis.api.interp.interpreter import Interpreter
from ocgis.api.interp import definition
from ocgis.conv.meta import MetaConverter
from ocgis.api.interp.iocg.processes import SubsetOperation
from ocgis.conv.converter import OcgConverter
from ocgis import env


class OcgInterpreter(Interpreter):
    '''The OCGIS interpreter and execution framework.'''
    
    def check(self):
        ## perform basic validation checks
        definition.validate_update_definition(self.desc)
        ## by parsing operational commands, determine iterator mode for
        ## converters
        definition.identify_iterator_mode(self.desc)
    
    def execute(self):
        self.check() ## run validation
        
        ## check for a user-supplied output prefix
        request_prefix = self.desc.get('request_prefix')
        if request_prefix is not None:
            env.BASE_NAME = request_prefix
        
        ## if the requested output format is "meta" then no operations are run
        ## and only the operations dictionary is required to generate output.
        if self.desc['output_format'] == 'meta':
            ## attempt to pull the request url
            request_url = self.desc.get('request_url')
            ret = MetaConverter(self.desc,request_url=request_url).write()
        ## this is the standard request for other output types.
        else:
            ## the operations object performs subsetting and calculations
            so = SubsetOperation(self.desc,serial=True,nprocs=7)
            ## if there is no grouping on the output files, a singe converter is
            ## is needed
            if self.desc['output_grouping'] is None:
                Conv = OcgConverter.get_converter(self.desc['output_format'])
                conv = Conv(so,wd=env.WORKSPACE,base_name=env.BASE_NAME,
                            mode=self.desc['mode'])
                ret = conv.write()
            else:
                raise(NotImplementedError)
        return(ret)
        