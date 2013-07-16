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
from django.conf import settings
from util.ncconv.experimental.wrappers import multipolygon_operation
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import time

import logging
from exc import OcgUrlError, MalformedSimulationOutputSelection,\
    AggregateFunctionError, AoiError, UncaughtRuntimeError
from util.ncconv.experimental.ocg_dataset.dataset import EmptyDataNotAllowed
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
        self.ocg.query = ocg()
        ## mark the request start time
        self.ocg.start_time = time.time()
        
        super(OpenClimateHandler,self).__init__(*args,**kwds)
        
    def create(self,request,**kwds):
        try:
            self._prepare_(request,kwds)
            return(self._create_(request))
        except Exception, e:
            return(self.error_handler(e))
    
    def read(self,request,**kwds):
        """
        Subclasses should not overload this method. Each return will be checked
        for basic validity.
        """
        try:
            self._prepare_(request,kwds)
            return(self._read_(request))
        except Exception, e:
            return(self.error_handler(e))
    
    def error_handler(self,e):
        if isinstance(e,OcgUrlError):
            return(e.response())
        else:
            raise(e)
    
#    def check(self,payload):
#        """Basic checks on returned data."""
#        
#        if len(payload) == 0:
#            return rc.NOT_FOUND
#        else:
#            return payload
        
    def _prepare_(self,request,kwds):
        """Called by all response method to look for common URL arguments."""
        ## save the URL argument keywords
        request.url_args = kwds
        ## parse URL arguments
        self._parse_slugs_(kwds)

        ## parse the query string for filters
        self._parse_query_dict_(request.GET)
#        self._parse_query_dict_(request.POST)
        
        ## add OCG object to request object for use by emitters
        request.ocg = self.ocg
        
    def _create_(self,request,**kwds):
        raise NotImplementedError
        
    def _read_(self,request,**kwds):
        """Overload in subclasses."""
        
        raise NotImplementedError
    
    def _parse_query_dict_(self, qdict):
        '''extracts filters from the query dict'''
        self.ocg.query.archive = InQuerySlug(climatedata.models.Archive,'archive',possible=qdict,code_field='urlslug').value
        self.ocg.query.climate_model = InQuerySlug(climatedata.models.ClimateModel,'model',possible=qdict,code_field='urlslug').value
        self.ocg.query.scenario = InQuerySlug(climatedata.models.Scenario,'scenario',possible=qdict,code_field='urlslug').value
        self.ocg.query.variable = InQuerySlug(climatedata.models.Variable,'variable',possible=qdict,code_field='urlslug').value
        self.ocg.query.functions = FunctionSlug('stat',possible=qdict).value
        self.ocg.query.grouping = GroupingSlug('grouping',possible=qdict).value
        
        ## check for any raw aggregate functions
        if self.ocg.query.functions is not None:
            if any([f['raw'] for f in self.ocg.query.functions]):
                if self.ocg.aggregate is False:
                    raise(AggregateFunctionError)
        
        ## if functions are passed, calculate statistics.
        if self.ocg.query.functions is not None:
            self.ocg.query.use_stat = True
        else:
            self.ocg.query.use_stat = False
        
    def _parse_slugs_(self,kwds):
        self.ocg.temporal = TemporalSlug('temporal',possible=kwds).value
        self.ocg.aoi = PolygonSlug('aoi',possible=kwds).value
        self.ocg.aggregate = BooleanSlug('aggregate',possible=kwds).value
        self.ocg.operation = OperationSlug('operation',possible=kwds).value
        self.ocg.html_template = OcgSlug('html_template',possible=kwds).value
        self.ocg.run = IntegerSlug('run',possible=kwds).value
        
        self.ocg.variable = IExactQuerySlug(models.Variable,'variable',possible=kwds,one=True).value
        self.ocg.scenario = IExactQuerySlug(models.Scenario,'scenario',possible=kwds,one=True,code_field='urlslug').value
        #self.ocg.useraoi = IExactQuerySlug(models.UserGeometryData,'aoi',possible=kwds,one=False).value
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
            if len(qs) == 0:
                raise(MalformedSimulationOutputSelection)
            self.ocg.simulation_output = qs[0]
        else:
            self.ocg.simulation_output = None


class MetacontentHandler(OpenClimateHandler):
    model = None
    
    def _read_(self,request):
        return(request)


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
        if 'id' in request.url_args:
            id = request.url_args['id']
            ret = self.model.objects.filter(id__exact=int(id))
        else:
            filter_kwds = {}
            for key,value in request.ocg.query.__dict__.iteritems():
                if key not in ['functions','grouping','use_stat']:
                    if value is not None:
                        filter_kwds.update({key+'__in':value})
            if len(filter_kwds) == 0:
                ret = self.model.objects.all()
            else:
                ret = self.model.objects.filter(**filter_kwds)
        return(ret)


class AoiHandler(NonSpatialHandler):
    model = UserGeometryMetadata
    exclude = ()
    
    def _read_(self,request):
        try:
            code = request.url_args['code']
            query = self.model.objects.filter(code__iexact=str(code))
        except:
            query = self.model.objects.all()
        #temp = query[0]
        return(query)


class QueryHandler(NonSpatialHandler):
    model = None
    allowed_methods = ('GET','POST')
    
    def _read_(self,request):
        return(request.ocg.simulation_output)
    
    def _create_(self,request):
        return(request.ocg.simulation_output)
    
    
class AoiUploadHandler(NonSpatialHandler):
    model = None
    allowed_methods = ('GET','POST')
    
    def _read_(self,request):
        return(request)
    
    def _create_(self,request):
        return(request)


class SpatialHandler(OpenClimateHandler):
    
    def _read_(self,request):
#        import ipdb;ipdb.set_trace()
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
        
        ## choose extraction mode and pull data appropriately.
        if settings.MAXPROCESSES == 1:
            in_parallel = False
        else:
            in_parallel = True
        
        ## TODO: for netcdf outputs, we will need to initialize the dataset again.
        ## not ideal and should be fixed in the future.
        request.ocg.ocg_opts = kwds
        request.ocg.dataset_uri = dataset.uri
        
        try:
            sub = multipolygon_operation(dataset.uri,
                                         self.ocg.simulation_output.netcdf_variable.code,
                                         ocg_opts=kwds,
                                         polygons=self.ocg.aoi,
                                         time_range=self.ocg.temporal,
                                         level_range=None, 
                                         clip=clip,
                                         union=self.ocg.aggregate,
                                         in_parallel=in_parallel, 
                                         max_proc=settings.MAXPROCESSES,
                                         max_proc_per_poly=settings.MAXPROCESSES_PER_POLY,
                                         allow_empty=False)
        except EmptyDataNotAllowed:
            raise(AoiError)
        except RuntimeError:
            raise(UncaughtRuntimeError)
        except:
            raise

        logger.debug("...ending SpatialHandler._read_()")
        return(sub)
