from piston.handler import BaseHandler
from climatedata.models import ClimateModel, Archive, Experiment, Variable,\
    SpatialGridCell, TemporalGridCell
from emitters import *
from piston.utils import rc
from django.contrib.gis.geos.collections import MultiPolygon
from util.ncconv import NetCdfAccessor
import urlparse
from util.helpers import parse_polygon_wkt
from django.contrib.gis.geos.polygon import Polygon
from django.contrib.gis.geos.geometry import GEOSGeometry
import datetime
from climatedata import models


class ocg(object):
    """Structure class to hold keyword arguments."""
    pass


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
#        import ipdb;ipdb.set_trace()
        ## parse query
#        self._query_string_(request)
        ## parse URL arguments
        self._parse_kwds_(kwds)
#        import ipdb;ipdb.set_trace()
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
    
    def _parse_kwds_(self,kwds):
        """Parser and formatter for potential URL keyword arguments."""
        
        def _format_date_(start,end):
            return([datetime.datetime.strptime(d,'%Y-%m-%d') for d in [start,end]])
        def _get_iexact_(model,code):
            "Return a single record from the database. Otherwise raise exception."
            ## this is the null case and should be treated as such
            if code == None:
                ret = None
            else:
                obj = model.objects.filter(code__iexact=code)
                if len(obj) != 1:
                    msg = '{0} records returned for model {1} with code query {2}'.format(len(obj),model,code)
                    raise ValueError(msg)
                else:
                    ret = obj[0]
            return(ret)
        
        ## name of the scenario
        self.ocg.scenario = kwds.get('scenario')
        ## the temporal arguments
        t = kwds.get('temporal')
        if t != None:
            if '+' in t:
                start,end = t.split('+')
            else:
                start = t
                end = t
        self.ocg.temporal = _format_date_(start,end)
        ## the polygon overlay
        aoi = kwds.get('aoi')
        self.ocg.aoi = GEOSGeometry(parse_polygon_wkt(aoi)) or aoi
        ## the model archive
        self.ocg.archive = kwds.get('archive')
        ## target variable
        self.ocg.variable = kwds.get('variable')
        ## aggregation boolean
        agg = kwds.get('aggregate')
        ## the None case is different than 'true' or 'false'
        if agg:
            self.ocg.aggregate = bool(agg)
        else:
            self.ocg.aggregate = None
        ## the model designation
        self.ocg.model = kwds.get('model')
        ## the overlay operation
        self.ocg.operation = kwds.get('operation')
        
        ## these queries return objects from the database classifying the NetCDF.
        ## the goal is to return the prediction.
        self.ocg.archive_obj = _get_iexact_(models.Archive,self.ocg.archive)
        self.ocg.model_obj = _get_iexact_(models.ClimateModel,self.ocg.model)
        self.ocg.scenario_obj = _get_iexact_(models.Experiment,self.ocg.scenario)
        self.ocg.variable_obj = _get_iexact_(models.Variable,self.ocg.variable)
        ## if we have data for each component, we can return a prediction
        if all([self.ocg.archive,self.ocg.model,self.ocg.scenario,self.ocg.variable]):
            fkwds = dict(archive=self.ocg.archive_obj,
                         climate_model=self.ocg.model_obj,
                         experiment=self.ocg.scenario_obj,
                         variable=self.ocg.variable_obj)
            self.ocg.prediction_obj = models.Prediction.objects.filter(**fkwds)
            if len(self.ocg.prediction_obj) != 1:
                raise ValueError('prediction query should return 1 record.')
            self.ocg.prediction_obj = self.ocg.prediction_obj[0]
        else:
            self.ocg.prediction_obj = None
        
        
#    def _query_string_(self,request):
#        """Parse URL query string and store as attributes."""
#        
#        qstr = request.META['QUERY_STRING']
#        if qstr:
#            url = urlparse.parse_qs(qstr)
#            for key,value in url.iteritems():
#                key = key.lower()
#                value = value[0]
#                if key == 'spatial':
#                    self._spatial = GEOSGeometry(parse_polygon_wkt(value))
##                elif key == 'intersection':
##                    self._intersection = bool(int(value))
#                else:
#                    raise KeyError('The query parameters "{0}" was not recognized by the handler.'.format(key))        
        

#class HelloWorldHandler(OpenClimateHandler):
#    allowed_methods = ('GET',)
#    model = ClimateModel
#    
#    def read(self,request,model_name=None):
#        if model_name != None:
#            query = self.model.objects.filter(name=str(model_name))
#            if len(query) == 0:
#                return rc.NOT_FOUND
#        else:
#            query = self.model.objects.all()
#        return query


class NonSpatialHandler(OpenClimateHandler):
     
    def _read_(self,request,code=None):
        if code:
            query = self.model.objects.filter(code__iexact=str(code))
        else:
            query = self.model.objects.all()
        return query


class ArchiveHandler(NonSpatialHandler):
    model = Archive
    
    
class ClimateModelHandler(NonSpatialHandler):
    model = ClimateModel
    
    
class ExperimentHandler(NonSpatialHandler):
    model = Experiment
    
    
class VariableHandler(NonSpatialHandler):
    model = Variable
    
    
class SpatialHandler(OpenClimateHandler):
    
    def _read_(self,request):
        
        from tests import get_example_netcdf
        
        attrs = get_example_netcdf()
        
        ## SPATIAL QUERYING ----------------------------------------------------
        
        ## perform the spatial operation
        if self.ocg.operation in ['intersects','intersect']:
            qs = SpatialGridCell.objects.filter(geom__intersects=self.ocg.aoi)
        ## this is synonymous with an intersection
        elif self.ocg.operation == 'clip':
            raise NotImplementedError
        else:
            raise NotImplementedError
        qs = qs.order_by('row','col')
        ## transform the grid geometries to MultiPolygon
        geom_list = [MultiPolygon(obj.geom) for obj in qs]
        ## if a spatial query is provided select the correct indices
#        if self._spatial:
        y_indices = [obj.row for obj in qs]
        x_indices = [obj.col for obj in qs]
        
        ## TEMPORAL QUERYING ---------------------------------------------------
        
#        import ipdb;ipdb.set_trace()
        ti = TemporalGridCell.objects.filter(grid_temporal=self.ocg.prediction_obj.grid_temporal)\
                                     .filter(date_ref__range=self.ocg.temporal)
        ti = ti.order_by('index').values_list('index',flat=True)
#        time_indices = [obj.index for obj in ti]
        
        ## RETRIEVE NETCDF DATA ------------------------------------------------
#        else:
#            y_indices = []
#            x_indices = []
        ## access the netcdf
#        print('here')
#        import ipdb;ipdb.set_trace()
        na = NetCdfAccessor(attrs['rootgrp'],attrs['var'])
        ## extract a dictionary representation of the netcdf
        dl = na.get_dict(geom_list,time_indices=ti,y_indices=y_indices,x_indices=x_indices)
#        print(dl)
#        import ipdb;ipdb.set_trace()        
        return(dl)