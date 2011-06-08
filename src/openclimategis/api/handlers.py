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

        if request.META['QUERY_STRING']:
            url = urlparse.parse_qs(request.META['QUERY_STRING'])
            for key,value in url.iteritems():
#                import ipdb;ipdb.set_trace()
                key = key.lower()
                value = value[0]
#                import ipdb;ipdb.set_trace()
                if key == 'spatial':
#                    import ipdb;ipdb.set_trace()
#                    poly = Polygon('POLYGON ((30 10, 10 20, 20 40, 40 40, 30 10))')
#                    import ipdb;ipdb.set_trace()
                    self._spatial = GEOSGeometry(parse_polygon_wkt(value))
                elif key == 'intersection':
                    self._intersection = bool(int(value))
                else:
                    raise KeyError('The query parameters "{0}" was not recognized by the handler.'.format(key))
#            import ipdb;ipdb.set_trace()
        
        


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
        
#        import ipdb;ipdb.set_trace()
        attrs = get_example_netcdf()
        qs = SpatialGridCell.objects.all().order_by('row','col')
        geom_list = [MultiPolygon(obj.geom) for obj in qs]
        na = NetCdfAccessor(attrs['rootgrp'],attrs['var'])
        dl = na.get_dict(geom_list)
#        import ipdb;ipdb.set_trace()
        return dl