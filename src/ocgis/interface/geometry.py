import base
from ocgis.interface.projection import WGS84
import numpy as np


class GeometrySpatialDimension(base.AbstractSpatialDimension):
    
    def __init__(self,uid=None,geom=None,projection=WGS84,attrs=None):
        super(GeometrySpatialDimension,self).__init__(projection=projection)
        self._geom = self._as_numpy_(geom)
        self._uid = self._as_numpy_(uid)
        self.attrs = attrs or {}
    
    @property
    def geom(self):
        if len(self._geom) == 1:
            return(self._geom[0])
        else:
            return(self._geom)
    
    @property
    def shape(self):
        return(self.geom.shape)
    
    @property
    def uid(self):
        if len(self._uid) == 1:
            return(self._uid[0])
        else:
            return(self._uid)
        
    @property
    def weights(self):
        raise(NotImplementedError)
    
    def get_iter(self):
        raise(NotImplementedError)
    
    def _as_numpy_(self,element):
        ret = np.array(element)
        if len(ret.shape) == 0:
            ret = ret.reshape(1,)
        return(ret)
    
    
class GeometryDataset(base.AbstractDataset):
    _dspatial = GeometrySpatialDimension
    _dlevel = None
    _dtemporal = None
    
    def __init__(self,*args,**kwds):
        self._spatial = kwds.pop('spatial',None)
        if self._spatial is None:
            self._spatial = GeometrySpatialDimension(*args,**kwds)
        
    def __getitem__(self,slc):
        raise(NotImplementedError)
    
    def __len__(self):
        return(len(self.spatial._geom))
        
    @property
    def metadata(self):
        raise(NotImplementedError)
    
    @property
    def get_subset(self):
        raise(NotImplementedError)
