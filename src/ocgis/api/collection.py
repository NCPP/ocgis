from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from ocgis import constants
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from copy import deepcopy


class AbstractCollection(object):
    __metaclass__ = ABCMeta
    
    @property
    def _archetype(self):
        return(self.variables[self.variables.keys()[0]])
    
    @property
    def projection(self):
        return(self._archetype.spatial.projection)
    
    @property
    def ugid(self):
        try:
            ret = self.ugeom.spatial.uid[0]
        ## the geometry may be empty
        except AttributeError:
            if self.ugeom is None:
                ret = 1
            else:
                raise
        return(ret)
    
    @abstractmethod
    def get_headers(self): list
    
    
class RawCollection(AbstractCollection):
    
    def __init__(self,ugeom=None):
        self.ugeom = ugeom
        self.variables = OrderedDict()
        
    def get_headers(self):
        return(constants.raw_headers)
    
    def get_iter(self):
        headers = self.get_headers()
        vid = 1
        ugid = self.ugid
        for var_name,ds in self.variables.iteritems():
            for geom,attrs in ds.get_iter_value():
                attrs['var_name'] = var_name
                attrs['vid'] = vid
                attrs['ugid'] = ugid
                row = [attrs[key] for key in headers]
                if type(geom) == Polygon:
                    geom = MultiPolygon([geom])
                yield(geom,row)
            vid += 1
        
        
class CalcCollection(AbstractCollection):
    
    def __init__(self,raw_collection):
        self.ugeom = raw_collection.ugeom
        self.variables = raw_collection.variables
        self.calc = OrderedDict()
        
    def get_headers(self):
        return(constants.calc_headers)
    
    def get_iter(self):
        headers = self.get_headers()
        vid = 1
        cid = 1
        ugid = self.ugid
        for var_name,calc in self.calc.iteritems():
            ds = self.variables[var_name]
            for calc_name,calc_value in calc.iteritems():
                for geom,attrs in ds.get_iter_value(value=calc_value,temporal_group=True):
                    attrs['var_name'] = var_name
                    attrs['calc_name'] = calc_name
                    attrs['vid'] = vid
                    attrs['cid'] = cid
                    attrs['ugid'] = ugid
                    row = [attrs[key] for key in headers]
                    if type(geom) == Polygon:
                        geom = MultiPolygon([geom])
                    yield(geom,row)
                cid += 1
            vid += 1
            
            
class MultivariateCalcCollection(CalcCollection):
    
    def get_headers(self):
        ## get the representative dataset
        arch = self._archetype
        ## determine if there is a temporal grouping
        temporal_group = False if arch.temporal.group is None else True
        if temporal_group:
            headers = deepcopy(super(MultivariateCalcCollection,self).get_headers())
            ## the variable name is not relevant for multivariate calculations
            headers.remove('var_name')
            headers.remove('vid')
        else:
            headers = constants.multi_headers
        return(headers)
    
    def get_iter(self):
        arch = self._archetype
        ## determine if there is a temporal grouping
        temporal_group = False if arch.temporal.group is None else True
        headers = self.get_headers()
        cid = 1
        ugid = self.ugid
        for calc_name,calc_value in self.calc.iteritems():
            for geom,attrs in arch.get_iter_value(value=calc_value,temporal_group=temporal_group,
                                                  add_masked=True):
                attrs['calc_name'] = calc_name
                attrs['cid'] = cid
                attrs['ugid'] = ugid
                row = [attrs[key] for key in headers]
                if type(geom) == Polygon:
                    geom = MultiPolygon([geom])
                yield(geom,row)
            cid += 1
