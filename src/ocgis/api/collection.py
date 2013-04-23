from abc import ABCMeta
from collections import OrderedDict


class AbstractCollection(object):
    __metaclass__ = ABCMeta
    
    @property
    def ugid(self):
        try:
            return(self.ugeom.ugid)
        except AttributeError:
            return(1)
    
    
class RawCollection(AbstractCollection):
    
    def __init__(self,ugeom=None):
        self.ugeom = ugeom
        self.variables = OrderedDict()
        
        
class CalcCollection(AbstractCollection):
    
    def __init__(self,raw_collection):
        self.ugeom = raw_collection.ugeom
        self.variables = raw_collection.variables
        self.calc = OrderedDict()