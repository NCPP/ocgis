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
from climatedata.models import SimulationOutput

import logging
from util.ncconv.experimental.wrappers import multipolygon_operation
logger = logging.getLogger(__name__)

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
        
    def create(self,request,**kwds):
        self._prepare_(request,kwds)
        return(self._create_(request))
    
    def read(self,request,**kwds):
        """
        Subclasses should not overload this method. Each return will be checked
        for basic validity.
        """
        self._prepare_(request,kwds)
        return(self._read_(request))
    
    def check(self,payload):
        """Basic checks on returned data."""
        
        if len(payload) == 0:
            return rc.NOT_FOUND
        else:
            return payload
        
    def _prepare_(self,request,kwds):
        """Called by all response method to look for common URL arguments."""
        ## save the URL argument keywords
        request.url_args = kwds
        ## parse URL arguments
        self._parse_slugs_(kwds)
        ## add OCG object to request object for use by emitters
        request.ocg = self.ocg
        
        
    def _create_(self,request,**kwds):
        raise NotImplementedError
        
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
        if all([self.ocg.scenario,
                self.ocg.archive,
                self.ocg.climate_model,
                self.ocg.run,
                self.ocg.variable]):
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
            
#        pdb.set_trace()


class NonSpatialHandler(OpenClimateHandler):
     
    def _read_(self,request):
        try:
            urlslug = request.url_args['urlslug']
            query = self.model.objects.filter(urlslug__iexact=str(urlslug))
        except:
            query = self.model.objects.all()
        return query


class ApiHandler(NonSpatialHandler):
    model = None
    
    def _read_(self,request):
        return None

class ArchiveHandler(NonSpatialHandler):
    model = Archive

    def _read_(self,request):
        try:
            urlslug = request.url_args['urlslug']
            query = self.model.objects.filter(urlslug__iexact=str(urlslug))
        except:
            query = self.model.objects.all()
        return query


class ClimateModelHandler(NonSpatialHandler):
    model = ClimateModel
    
    def _read_(self,request):
        try:
            urlslug = request.url_args['urlslug']
            query = self.model.objects.filter(urlslug__iexact=str(urlslug))
        except:
            query = self.model.objects.all()
        return query


class ScenarioHandler(NonSpatialHandler):
    model = Scenario

    def _read_(self,request):
        try:
            urlslug = request.url_args['urlslug']
            query = self.model.objects.filter(urlslug__iexact=str(urlslug))
        except:
            query = self.model.objects.all()
        return query


class VariableHandler(NonSpatialHandler):
    model = Variable

    def _read_(self,request):
        try:
            urlslug = request.url_args['urlslug']
            query = self.model.objects.filter(urlslug__iexact=str(urlslug))
        except:
            query = self.model.objects.all()
        return query


class SimulationOutputHandler(NonSpatialHandler):
    model = SimulationOutput
    exclude = ()
    def _read_(self,request):
        try:
            id = request.url_args['id']
            query = self.model.objects.filter(id__exact=int(id))
        except:
            query = self.model.objects.all()
        return query


class QueryHandler(NonSpatialHandler):
    model = None
    allowed_methods = ('GET','POST')
    
    def _read_(self,request):
        return(request.ocg.simulation_output)
    
    def _create_(self,request):
        return(request.ocg.simulation_output)


class SpatialHandler(OpenClimateHandler):
    __mode__ = 'single' # or 'multi'
    
    def _read_(self,request):
        
        logger.debug("starting SpatialHandler._read_()...")
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
        
        ## check polygon sequence   
        if type(self.ocg.aoi) not in (list,tuple):
                polygons = [self.ocg.aoi]
        else:
            polygons = self.ocg.aoi
        
        ## choose extraction mode and pull data appropriately.
        if self.__mode__ == 'single':
            sub = multipolygon_operation(dataset.uri,
                                         self.ocg.simulation_output.netcdf_variable.code,
                                         ocg_opts=kwds,
                                         polygons=polygons,
                                         time_range=self.ocg.temporal,
                                         level_range=None, 
                                         clip=clip,
                                         union=self.ocg.aggregate,
                                         in_parallel=False, 
                                         max_proc=8,
                                         max_proc_per_poly=1)
#            elements = multipolygon_singlecore_operation(dataset.uri, 
#                         self.ocg.simulation_output.netcdf_variable.code, 
#                         polygons, 
#                         time_range=self.ocg.temporal, 
#                         clip=clip, 
#                         dissolve=self.ocg.aggregate, 
#                         levels=None,
#                         ocgOpts=kwds)
        elif self.__mode__ == 'multi':
            sub = multipolygon_operation(dataset.uri,
                                         self.ocg.simulation_output.netcdf_variable.code,
                                         ocg_opts=kwds,
                                         polygons=polygons,
                                         time_range=self.ocg.temporal,
                                         level_range=None, 
                                         clip=clip,
                                         union=self.ocg.aggregate,
                                         in_parallel=True, 
                                         max_proc=8,
                                         max_proc_per_poly=1)
#            elements = multipolygon_multicore_operation(dataset.uri,
#                                          self.ocg.simulation_output.netcdf_variable.code,
#                                          [self.ocg.aoi],
#                                          time_range=self.ocg.temporal,
#                                          clip=clip,
#                                          dissolve=self.ocg.aggregate,
#                                          levels=None,
#                                          ocgOpts=kwds,
#                                          subdivide=True,
#                                          #subres = 90
#                                          )
#        pdb.set_trace()
        ########################################################################
        
        ## OLD APPROACH WITH SINGLE CORE #######################################
            
#        try:
#            dataset = netCDF4.Dataset(dataset.uri,'r')
#            elements = multipolygon_operation(dataset,
#                                              self.ocg.simulation_output.netcdf_variable.code,
#                                              [self.ocg.aoi],
#                                              time_range=self.ocg.temporal,
#                                              clip=clip,
#                                              dissolve=self.ocg.aggregate,
#                                              ocg_kwds=kwds
#                                              )
##            ## pull the elements
##            elements = d.extract_elements(self.ocg.variable.title(), # TODO: variable formatting
##                                          dissolve=self.ocg.aggregate,
##                                          polygon=self.ocg.aoi,
##                                          time_range=self.ocg.temporal,
##                                          clip=self.ocg.operation)
#        ## close the connection...
#        finally:
#            dataset.close()

        ########################################################################
        logger.debug("...ending SpatialHandler._read_()")
        return(sub)
