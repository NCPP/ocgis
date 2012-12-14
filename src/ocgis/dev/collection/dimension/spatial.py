from dimension import OcgDimension
import numpy as np
from ocgis.util.helpers import iter_array
from shapely.geometry.point import Point


class SpatialDimension(OcgDimension):
    
    def __init__(self,value,value_mask,weights=None):
        super(SpatialDimension,self).__init__(value)
        
        self.value_mask = value_mask
        if weights is None:
            if isinstance(self.value[0,0],Point):
                weights = np.ones(value.shape,dtype=float)
                weights = np.ma.array(weights,mask=value_mask)
            else:
                weights = np.empty(value.shape,dtype=float)
                masked = self.get_masked()
                for idx,geom in iter_array(masked,return_value=True):
                    weights[idx] = geom.area
                weights = weights/weights.max()
                weights = np.ma.array(weights,mask=value_mask)
        else:
            assert(weights.shape == value.shape)
        self.weights = weights
        
    def get_masked(self):
        return(np.ma.array(self.value,mask=self.value_mask))
    
    @staticmethod
    def _iter_values_idx_(value):
        for idx in iter_array(value):
            yield(idx)
