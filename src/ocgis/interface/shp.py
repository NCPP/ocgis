import base
from ocgis.interface.projection import WGS84
from ocgis.util.shp_cabinet import ShpCabinet
import numpy as np
from copy import deepcopy


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
    
    def get_iter(self):
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
    
    def __init__(self,key=None,spatial=None):
        self.key = key
        self._spatial = spatial
        self._temporal = None
        self._level = None
        self.__sc = None
        
    def __getitem__(self,slc):
        geom = self.spatial.geom[slc]
        uid = self.spatial.uid[slc]
        new_attrs = {}
        for k,v in self.spatial.attrs.iteritems():
            new_attrs[k] = v[slc]
        
        spatial = ShpSpatialDimension(uid,geom,projection=self.spatial.projection,
                                      attrs=new_attrs)
        ret = self.__class__(key=self.key,spatial=spatial)
        return(ret)
    
    @property
    def metadata(self):
        raise(NotImplementedError)
    
    @property
    def get_subset(self):
        raise(NotImplementedError)
    
    @property
    def _sc(self):
        if self.__sc is None:
            self.__sc = ShpCabinet()
        return(self.__sc)