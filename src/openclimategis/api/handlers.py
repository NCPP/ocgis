from piston.handler import BaseHandler
from climatedata.models import ClimateModel, Archive, Experiment, Variable,\
    SpatialGridCell
from emitters import *
from piston.utils import rc
from django.contrib.gis.geos.collections import MultiPolygon
from util.ncconv import NetCdfAccessor


class OpenClimateHandler(BaseHandler):
    """Superclass for all OpenClimate handlers."""
    
    allowed_methods = ('GET',)
    
    def read(self,request,**kwds):
        """
        Subclasses should not overload this method. Each return will be checked
        for basic validity.
        """
        
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
    
    def _read_(self,request,
                    wkt_like=None,
                    drange_like=None,
                    intersection=False):
        
        from tests import get_example_netcdf
        
        attrs = get_example_netcdf()
        qs = SpatialGridCell.objects.all().order_by('row','col')
        geom_list = [MultiPolygon(obj.geom) for obj in qs]
        na = NetCdfAccessor(attrs['rootgrp'],attrs['var'])
        dl = na.get_dict(geom_list)
#        import ipdb;ipdb.set_trace()
        return dl