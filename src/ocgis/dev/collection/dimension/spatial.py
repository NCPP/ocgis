from dimension import OcgDimension
import numpy as np
from ocgis.util.helpers import iter_array


class SpatialDimension(OcgDimension):
    
    def __init__(self,uid,value,value_mask):
        super(SpatialDimension,self).__init__('gid',uid,None,value)
        
        self.value_mask = value_mask
        
    def get_masked(self):
        return(np.ma.array(self.value,mask=self.value_mask))
    
    @staticmethod
    def _iter_values_idx_(value):
        for idx in iter_array(value):
            yield(idx)
