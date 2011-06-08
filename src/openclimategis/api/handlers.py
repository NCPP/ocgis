from piston.handler import BaseHandler
from climatedata.models import ClimateModel, Archive, Experiment, Variable,\
    SpatialGridCell
from emitters import *
from piston.utils import rc
from django.contrib.gis.geos.collections import MultiPolygon
from util.ncconv import NetCdfAccessor
import urlparse
from util.helpers import parse_polygon_wkt
from django.contrib.gis.geos.polygon import Polygon
from django.contrib.gis.geos.geometry import GEOSGeometry


class OpenClimateHandler(BaseHandler):
    """Superclass for all OpenClimate handlers."""
    
    allowed_methods = ('GET',)
    
    def __init__(self,*args,**kwds):
        ## set some default parameters for the handlers
        self._intersection = False ## perform a full intersection
        self._spatial = None ## wkt representation of intersects geometry
        self._drange = None ## date range to select
        
        super(OpenClimateHandler,self).__init__(*args,**kwds)
    
    def read(self,request,**kwds):
        """
        Subclasses should not overload this method. Each return will be checked
        for basic validity.
        """

        ## parse query
        self._query_string_(request)
        ## call the subclass read methods
        return self.check(self._read_(request,**kwds))
    
    def check(self,payload):
        """Basic checks on returned data."""
        
        if len(payload) == 0:
            return rc.NOT_FOUND
        else:
            return payload
        
    def _read_(self,request,**kwds):
        """Overload in subclasses."""
        
        raise NotImplementedError
    
    def _query_string_(self,request):
        """Parse URL query string and store as attributes."""
        
        qstr = request.META['QUERY_STRING']
        if qstr:
            url = urlparse.parse_qs(qstr)
            for key,value in url.iteritems():
                key = key.lower()
                value = value[0]
                if key == 'spatial':
                    self._spatial = GEOSGeometry(parse_polygon_wkt(value))
                elif key == 'intersection':
                    self._intersection = bool(int(value))
                else:
                    raise KeyError('The query parameters "{0}" was not recognized by the handler.'.format(key))        
        

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
        ## check for a geometry
        if self._spatial != None:
            ## query is different if an intersection is requested
            if self._intersection:
                raise NotImplementedError
            else:
                qs = SpatialGridCell.objects.filter(geom__intersects=self._spatial)
        else:
            ## if not spatial or date query is provided, return all the objects
            qs = SpatialGridCell.objects.all()
        qs = qs.order_by('row','col')
        ## transform the grid geometries to MultiPolygon
        geom_list = [MultiPolygon(obj.geom) for obj in qs]
        ## if a spatial query is provided select the correct indices
        if self._spatial:
            y_indices = [obj.row for obj in qs]
            x_indices = [obj.col for obj in qs]
        else:
            y_indices = []
            x_indices = []
        ## access the netcdf
        na = NetCdfAccessor(attrs['rootgrp'],attrs['var'])
        ## extract a dictionary representation of the netcdf
        dl = na.get_dict(geom_list,y_indices=y_indices,x_indices=x_indices)
        return(dl)