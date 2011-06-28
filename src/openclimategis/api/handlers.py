from piston.handler import BaseHandler
from climatedata.models import ClimateModel, Archive, Experiment, Variable,\
    SpatialGridCell, TemporalGridCell
from emitters import *
from piston.utils import rc
from util.ncconv import NetCdfAccessor
from util.helpers import parse_polygon_wkt
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
        if agg.lower() == 'true':
            self.ocg.aggregate = True
        elif agg.lower() == 'false':
            self.ocg.aggregate = False
        else:
            msg = '"{0}" aggregating boolean operation not recognized.'
            raise(ValueError(msg.format(agg)))
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
        
        ocgeom = OpenClimateGeometry(self.ocg.aoi,
                                     self.ocg.operation,
                                     self.ocg.aggregate)
        geom_list = ocgeom.get_geom_list()
        row,col = ocgeom.get_indices()
        weights = ocgeom.get_weights()
        
        ## TEMPORAL QUERYING ---------------------------------------------------
        
        ti = TemporalGridCell.objects.filter(grid_temporal=self.ocg.prediction_obj.grid_temporal)\
                                     .filter(date_ref__range=self.ocg.temporal)
        ti = ti.order_by('index').values_list('index',flat=True)
        
        ## RETRIEVE NETCDF DATA ------------------------------------------------

        na = NetCdfAccessor(attrs['rootgrp'],attrs['var'])
        ## extract a dictionary representation of the netcdf
        dl = na.get_dict(geom_list,
                         time_indices=ti,
                         row=row,
                         col=col,
                         aggregate=self.ocg.aggregate,
                         weights=weights)       
        return(dl)
    
    
class OpenClimateGeometry(object):
    """
    Perform OpenClimateGIS geometry operations. Manages clip v. intersect
    operations and spatial unioning in the case of an aggregation.
    
    aoi -- GEOSGeometry Polygon object acting as the geometric selection overlay.
    op -- 'intersect(s)' or 'clip'
    aggregate -- set to True to union the geometries.
    """
    
    def __init__(self,aoi,op,aggregate):
        self.aoi = aoi
        self.op = op
        self.aggregate = aggregate
        
        self.__qs = None ## queryset with correct spatial operations
        self.__geoms = None ## list of geometries with correct attribute selected
        
        ## set the geometry attribute depending on the operation
        if op in ['intersect','intersects']:
            self._gattr = 'geom'
        elif op == 'clip':
            self._gattr = 'intersection'
        else:
            msg = 'spatial operation "{0}" not recognized.'.format(op)
            raise NotImplementedError(msg)
                                          
    def get_indices(self):
        "Returning row and column indices used to index into NetCDF."
        
        row = [obj.row for obj in self._qs]
        col = [obj.col for obj in self._qs]
        return((row,col))
    
    def get_weights(self):
        "Returns weights for each polygon in the case of an aggregation."
        
        if self.aggregate:
            ## calculate weights for each polygon
            areas = [obj.area for obj in self._geoms]
            asum = sum(areas)
            weights = [a/asum for a in areas]
            ret = weights
        else:
            ret = None
        return(ret)
    
    def get_geom_list(self):
        "Return the list of geometries accounting for processing parms."
        
        if self.aggregate:
            ret = self._union_geoms_()
        else:
            ret = self._geoms
        return(ret)
    
    @property
    def _qs(self):
        if self.__qs == None:
            ## always perform the spatial select to limit returned records.
            self.__qs = SpatialGridCell.objects.filter(geom__intersects=self.aoi)\
                                              .order_by('row','col')
            ## intersection operations require element-by-element intersection
            ## operations.
            if self.op == 'clip':
                self.__qs = self.__qs.intersection(self.aoi)
        return(self.__qs)
    
    @property
    def _geoms(self):
        if self.__geoms == None:
            self.__geoms = [getattr(obj,self._gattr) for obj in self._qs]
        return(self.__geoms)
    
    def _union_geoms_(self):
        "Returns the union of geometries in the case of an aggregation."
        
        first = True
        for geom in self._geoms:
            if first:
                union = geom
                first = False
            else:
                union = union.union(geom)
        return(union)
