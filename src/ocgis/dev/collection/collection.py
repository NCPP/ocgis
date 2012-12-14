from collections import OrderedDict
import numpy as np


class OcgVariable(object):
    
    def __init__(self,name,value,temporal,spatial,level=None):
        assert(value.shape[0] == len(temporal.value))
        if level is None:
            assert(value.shape[1] == 1)
        assert(np.all(value.shape[2:] == spatial.value.shape))
        
        self.name = name
        self.raw_value = value
        self.temporal = temporal
        self.spatial = spatial
        self.level = level
        
        ## hold aggregated values separate from raw
        self.agg_value = None


class OcgIdentifier(OrderedDict):
    
    def __init__(self,*args,**kwds):
        self._curr = 1
        super(OcgIdentifier,self).__init__(*args,**kwds)
    
    def add(self,value):
        if self._get_is_unique_(value):
            self.update({value:self._get_current_identifier_()})
        
    def _get_is_unique_(self,value):
        if value in self:
            ret = False
        else:
            ret = True
        return(ret)
    
    def _get_current_identifier_(self):
        try:
            return(self._curr)
        finally:
            self._curr += 1
