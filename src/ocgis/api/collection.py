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
    
    def get_headers(self,upper=False):
        headers = self._get_headers_()
        if upper:
            ret = [h.upper() for h in headers]
        else:
            ret = headers
        return(ret)
    
    @abstractmethod
    def _get_headers_(self): list
    
    
class RawCollection(AbstractCollection):
    
    def __init__(self,ugeom=None):
        self.ugeom = ugeom
        self.variables = OrderedDict()
    
    def get_iter(self):
        headers = self.get_headers()
        vid = 1
        ugid = self.ugid
        for alias,ds in self.variables.iteritems():
            did = ds.request_dataset.did
            variable = ds.request_dataset.variable
            for geom,attrs in ds.get_iter_value():
                attrs['did'] = did
                attrs['alias'] = alias
                attrs['variable'] = variable
                attrs['vid'] = vid
                attrs['ugid'] = ugid
                row = [attrs[key] for key in headers]
                if type(geom) == Polygon:
                    geom = MultiPolygon([geom])
                yield(geom,row)
            vid += 1
            
    def _get_headers_(self):
        return(constants.raw_headers)
        
        
class CalcCollection(AbstractCollection):
    
    def __init__(self,raw_collection):
        self.ugeom = raw_collection.ugeom
        self.variables = raw_collection.variables
        self.calc = OrderedDict()
    
    def get_iter(self):
        headers = self.get_headers()
        vid = 1
        cid = 1
        ugid = self.ugid
        for alias,calc in self.calc.iteritems():
            ds = self.variables[alias]
            did = ds.request_dataset.did
            variable = ds.request_dataset.variable
            for calc_name,calc_value in calc.iteritems():
                for geom,attrs in ds.get_iter_value(value=calc_value,temporal_group=True):
                    attrs['did'] = did
                    attrs['variable'] = variable
                    attrs['alias'] = alias
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
            
    def _get_headers_(self):
        return(constants.calc_headers)
            
            
class MultivariateCalcCollection(CalcCollection):
    
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

    def _get_headers_(self):
        ## get the representative dataset
        arch = self._archetype
        ## determine if there is a temporal grouping
        temporal_group = False if arch.temporal.group is None else True
        if temporal_group:
            headers = [h.upper() for h in deepcopy(super(MultivariateCalcCollection,self)._get_headers_())]
            ## the variable name is not relevant for multivariate calculations
            headers.remove('variable')
            headers.remove('vid')
        else:
            headers = constants.multi_headers
        return(headers)
