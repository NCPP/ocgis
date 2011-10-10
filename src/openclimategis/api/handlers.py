from piston.handler import BaseHandler
from emitters import *
from piston.utils import rc
from climatedata import models
import inspect
from slug import *
import climatedata.models
from climatedata.models import Archive
from climatedata.models import ClimateModel
from climatedata.models import Scenario
from climatedata.models import Variable
#from util.ncconv.in_memory_oo_multi_core import multipolygon_multicore_operation
import netCDF4
from experimental.in_memory_oo import multipolygon_operation


class ocg(object):
    """Structure class to hold keyword arguments."""
    
    def __repr__(self):
        prints = []
        mems = inspect.getmembers(self)
        for mem in mems:
            if not mem[0].startswith('__'):
                prints.append('{0}={1}\n'.format(*mem))
        return(''.join(prints))


class OpenClimateHandler(BaseHandler):
    """Superclass for all OpenClimate handlers."""
    
    allowed_methods = ('GET',)
    
    def __init__(self,*args,**kwds):
        ## set some default parameters for the handlers
        self.ocg = ocg()
        
        super(OpenClimateHandler,self).__init__(*args,**kwds)
    
    def read(self,request,**kwds):
        """
        Subclasses should not overload this method. Each return will be checked
        for basic validity.
        """
        ## save the URL argument keywords
        request.url_args = kwds
        ## parse URL arguments
        self._parse_slugs_(kwds)
        ## add OCG object to request object for use by emitters
        request.ocg = self.ocg
        ## call the subclass read methods
        return self.check(self._read_(request))
    
    def check(self,payload):
        """Basic checks on returned data."""
        
        if len(payload) == 0:
            return rc.NOT_FOUND
        else:
            return payload
        
    def _read_(self,request,**kwds):
        """Overload in subclasses."""
        
        raise NotImplementedError
    
    def _parse_slugs_(self,kwds):
        
        self.ocg.temporal = TemporalSlug('temporal',possible=kwds).value
        self.ocg.aoi = PolygonSlug('aoi',possible=kwds).value
        self.ocg.aggregate = AggregateSlug('aggregate',possible=kwds).value
        self.ocg.operation = OperationSlug('operation',possible=kwds).value
        self.ocg.html_template = OcgSlug('html_template',possible=kwds).value
        self.ocg.run = IntegerSlug('run',possible=kwds).value
        
        self.ocg.variable = IExactQuerySlug(models.Variable,'variable',possible=kwds,one=True).value
        self.ocg.scenario = IExactQuerySlug(models.Scenario,'scenario',possible=kwds,one=True,code_field='urlslug').value
        self.ocg.archive = IExactQuerySlug(models.Archive,'archive',possible=kwds,one=True,code_field='urlslug').value
        self.ocg.climate_model = IExactQuerySlug(models.ClimateModel,'model',possible=kwds,one=True,code_field='urlslug').value
        
        ## return the dataset object if all components passed
        if all([self.ocg.scenario,self.ocg.archive,self.ocg.climate_model,self.ocg.run,self.ocg.variable]):
            fkwds = dict(archive=self.ocg.archive,
                         scenario=self.ocg.scenario,
                         climate_model=self.ocg.climate_model,
                         run=self.ocg.run,
                         variable=self.ocg.variable)
            qs = climatedata.models.SimulationOutput.objects.filter(**fkwds)
            assert(len(qs) == 1)
            self.ocg.simulation_output = qs[0]
        else:
            self.ocg.simulation_output = None
            

class NonSpatialHandler(OpenClimateHandler):
    __data_kwds__ = {}
     
    def _read_(self,request):
        
        if not self.model:
            return None
        try:
            urlslug = request.url_args['urlslug']
            query = self.model.objects.filter(urlslug__iexact=str(urlslug))
        except:
            query = self.model.objects.all()
        return query


class ApiHandler(NonSpatialHandler):
    model = None


class ArchiveHandler(NonSpatialHandler):
    model = Archive


class ClimateModelHandler(NonSpatialHandler):
    model = ClimateModel


class ScenarioHandler(NonSpatialHandler):
    model = Scenario


class VariableHandler(NonSpatialHandler):
    model = Variable
    
    
class SpatialHandler(OpenClimateHandler):
    
    def _read_(self,request):
        dataset = self.ocg.simulation_output.netcdf_variable.netcdf_dataset
        
        ## arguments for the dataset object
        kwds = dict(rowbnds_name=dataset.rowbnds_name,
                    colbnds_name=dataset.colbnds_name,
                    time_name=dataset.time_name,
                    time_units=dataset.time_units,
                    calendar=dataset.calendar,)
#                    level_name=self.ocg.dataset.level_name,) TODO: add level variable name

        ## construct arguments to clip operation
        if self.ocg.operation == 'clip':
            clip = True
        else:
            clip = False
        
#        elements = multipolygon_multicore_operation(dataset.uri,
#                                      self.ocg.simulation_output.netcdf_variable.code,
#                                      [self.ocg.aoi],
#                                      time_range=self.ocg.temporal,
#                                      clip=clip,
#                                      dissolve=self.ocg.aggregate,
#                                      levels=None,
#                                      ocgOpts=kwds,
#                                      subdivide=True,
#                                      #subres = 90
#                                      )
        
#        import ipdb;ipdb.set_trace()
        
        try:
            dataset = netCDF4.Dataset(dataset.uri,'r')
            elements = multipolygon_operation(dataset,
                                              self.ocg.simulation_output.netcdf_variable.code,
                                              [self.ocg.aoi],
                                              time_range=self.ocg.temporal,
                                              clip=clip,
                                              dissolve=self.ocg.aggregate,
                                              ocg_kwds=kwds
                                              )
#            ## pull the elements
#            elements = d.extract_elements(self.ocg.variable.title(), # TODO: variable formatting
#                                          dissolve=self.ocg.aggregate,
#                                          polygon=self.ocg.aoi,
#                                          time_range=self.ocg.temporal,
#                                          clip=self.ocg.operation)
        ## close the connection...
#        except RuntimeError:
#            sys.stdout.write()
        finally:
            if hasattr(dataset, 'close'):
                dataset.close()
            
        return(elements)
