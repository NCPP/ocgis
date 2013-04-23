from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from ocgis import constants


class AbstractCollection(object):
    __metaclass__ = ABCMeta
    
    @property
    def ugid(self):
        try:
            return(self.ugeom.ugid)
        except AttributeError:
            return(1)
    
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
                yield(geom,row)
            vid += 1
        
        
class CalcCollection(AbstractCollection):
    
    def __init__(self,raw_collection):
        self.ugeom = raw_collection.ugeom
        self.variables = raw_collection.variables
        self.calc = OrderedDict()
        
    def get_headers(self):
        raise(NotImplementedError)
    
    def get_iter(self):
        raise(NotImplementedError)