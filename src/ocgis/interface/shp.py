import base
from ocgis.interface.projection import WGS84
from ocgis.util.shp_cabinet import ShpCabinet
import numpy as np


class ShpSpatialDimension(base.AbstractSpatialDimension):
    
    def __init__(self,uid,geom,projection=WGS84,attrs=None):
        super(self.__class__,self).__init__(projection=projection)
        self.geom = geom
        self.uid = uid
        self.attrs = attrs or {}
        
    @property
    def shape(self):
        return(self.geom.shape)
        
    @property
    def weights(self):
        raise(NotImplementedError)
    
    @classmethod
    def _load_(cls,gi):
        geoms = gi._sc.get_geoms(gi.key)
        lgeoms = len(geoms)
        fill_geoms = np.empty(lgeoms,dtype=object)
        uid = np.empty(lgeoms,dtype=int)
        attrs = {}
        for ii,geom in enumerate(geoms):
            fill_geoms[ii] = geom.pop('geom')
            uid[ii] = geom.pop('ugid')
            for k,v in geom.iteritems():
                if k not in attrs:
                    attrs[k] = np.empty(lgeoms,dtype=object)
                attrs[k][ii] = v
        ret = cls(uid,fill_geoms,attrs=attrs)
        return(ret)

class ShpDataset(base.AbstractDataset):
    _dlevel = None
    _dtemporal = None
    _dspatial = ShpSpatialDimension
    
    def __init__(self,key):
        self.key = key
        self._spatial = None
        self._sc = ShpCabinet()
        self._temporal = None
        self._level = None
    
    @property
    def metadata(self):
        raise(NotImplementedError)
    
    @property
    def get_subset(self):
        raise(NotImplementedError)