from abc import ABCMeta
from collections import OrderedDict


class AbstractCollection(object):
    __metaclass__ = ABCMeta
    
    
class RawCollection(AbstractCollection):
    
    def __init__(self,ugeom=None):
        self.ugeom = ugeom
        self.variables = OrderedDict()
        
    @property
    def ugid(self):
        try:
            ret = self.ugeom.uid
        except AttributeError:
            ret = 1
        return(ret)