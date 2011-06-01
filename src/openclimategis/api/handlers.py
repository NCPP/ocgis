from piston.handler import BaseHandler
from climatedata.models import ClimateModel, SpatialGridCell
from emitters import *
from piston.utils import rc
from django.contrib.gis.geos.collections import MultiPolygon
from util.ncconv import GeoQuerySetFactory


class OpenClimateHandler(BaseHandler):
    pass
    
    
class HelloWorldHandler(OpenClimateHandler):
    allowed_methods = ('GET',)
    model = ClimateModel
    
    def read(self,request,model_name=None):
        if model_name != None:
            query = self.model.objects.filter(name=str(model_name))
            if len(query) == 0:
                return rc.NOT_FOUND
        else:
            query = self.model.objects.all()
        return query
    
    
class IntersectsHandler(OpenClimateHandler):
    allowed_methods = ('GET',)

    def read(self,request):
        qs = SpatialGridCell.objects.all().order_by('row','col')
        geom_list = [MultiPolygon(obj.geom) for obj in qs]
        gf = GeoQuerySetFactory(self.rootgrp,self.var)
        gqs = gf.get_queryset(geom_list)
        return gqs


class IntersectionHandler(IntersectsHandler):
    pass