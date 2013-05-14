import base
from ocgis.interface.projection import WGS84
import numpy as np
from shapely import wkb
from osgeo.ogr import CreateGeometryFromWkb
from ocgis.util.spatial.wrap import Wrapper
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point


class GeometrySpatialDimension(base.AbstractSpatialDimension):
    
    def __init__(self,uid=None,geom=None,projection=None,attrs=None):
        _projection = projection or WGS84()
        super(GeometrySpatialDimension,self).__init__(projection=_projection)
        self.geom = self._as_numpy_(geom)
        self.uid = self._as_numpy_(uid)
        self.attrs = attrs or {}
    
    @property
    def shape(self):
        return(self.geom.shape)
        
    @property
    def weights(self):
        raise(NotImplementedError)
    
    def get_iter(self):
        raise(NotImplementedError)
    
    def unwrap_geoms(self,axis=0.0):
        axis = float(axis)
        w = Wrapper(axis=axis)
        geom = self.geom
        for idx in range(geom.shape[0]):
            geom[idx] = w.unwrap(geom[idx])
    
    def _as_numpy_(self,element):
        ## check for multipolygons to avoid array confusion
        if isinstance(element,MultiPolygon) or isinstance(element,MultiPoint) or isinstance(element,Point):
            ret = np.ma.array([None],dtype=object)
            ret[0] = element
        else:
            ret = np.ma.array(element)
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
            
    def __iter__(self):
        for idx in range(len(self)):
            yield(self[idx])
        
    def __getitem__(self,slc):
        geom = self.spatial.geom[slc]
        uid = self.spatial.uid[slc]
        new_attrs = {}
        for k,v in self.spatial.attrs.iteritems():
            new_attrs[k] = v[slc]
        
        spatial = GeometrySpatialDimension(uid,geom,projection=self.spatial.projection,
                                           attrs=new_attrs)
        ret = self.__class__(spatial=spatial)
        return(ret)
    
    def __len__(self):
        return(len(self.spatial.geom))
        
    @property
    def metadata(self):
        raise(NotImplementedError)
    
    def get_subset(self):
        raise(NotImplementedError)
    
    def project(self,projection):
        to_sr = projection.sr
        from_sr = self.spatial.projection.sr
        se = self.spatial.geom
        len_se = len(se)
        loads = wkb.loads
        
        for idx in range(len_se):
            geom = CreateGeometryFromWkb(se[idx].wkb)
            geom.AssignSpatialReference(from_sr)
            geom.TransformTo(to_sr)
            se[idx] = loads(geom.ExportToWkb())
        
        self.spatial.projection = projection