import geometry
from ocgis.util.shp_cabinet import ShpCabinet
import numpy as np
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import cascaded_union
from ocgis.util.helpers import iter_array


class ShpSpatialDimension(geometry.GeometrySpatialDimension):
    
    @classmethod
    def _load_(cls,gi):
        geoms = gi._sc.get_geoms(gi.key,select_ugid=gi._select_ugid)
        lgeoms = len(geoms)
        fill_geoms = np.empty(lgeoms,dtype=object)
        uid = np.empty(lgeoms,dtype=int)
        attrs = {}
        for ii,geom in enumerate(geoms):
            fill_geoms[ii] = geom.pop('geom')
            uid[ii] = geom.pop('UGID')
            for k,v in geom.iteritems():
                if k not in attrs:
                    attrs[k] = np.empty(lgeoms,dtype=object)
                attrs[k][ii] = v
        ret = cls(uid,fill_geoms,attrs=attrs)
        return(ret)

class ShpDataset(geometry.GeometryDataset):
    _dlevel = None
    _dtemporal = None
    _dspatial = ShpSpatialDimension
    
    def __init__(self,key=None,spatial=None,select_ugid=None):
        self.key = key
        self._spatial = spatial
        self._temporal = None
        self._level = None
        self.__sc = None
        self._select_ugid = select_ugid
        
    def __getitem__(self,slc):
        geom = self.spatial.geom[slc]
        uid = self.spatial.uid[slc]
        new_attrs = {}
        try:
            for k,v in self.spatial.attrs.iteritems():
                new_attrs[k] = v[slc]
        ## there may be not attrs
        except AttributeError:
            if self.spatial.attrs is None:
                new_attrs = None
            else:
                raise
        
        spatial = ShpSpatialDimension(uid,geom,projection=self.spatial.projection,
                                      attrs=new_attrs)
        ret = self.__class__(key=self.key,spatial=spatial)
        return(ret)
    
    @property
    def _sc(self):
        if self.__sc is None:
            self.__sc = ShpCabinet()
        return(self.__sc)
    
    def aggregate(self,new_id=1):
        to_union = []
        for geom in self.spatial.geom.compressed():
            if isinstance(geom,MultiPolygon):
                for poly in geom:
                    to_union.append(poly)
            else:
                to_union.append(geom)
        ugeom = cascaded_union(to_union)
        new_geom = np.ma.array([0],dtype=object)
        new_geom[0] = ugeom
        new_uid = np.ma.array([new_id],dtype=int)
        self.spatial.geom = new_geom
        self.spatial.uid = new_uid
        self.spatial.attrs = {}

    def write(self,path):
        geoms = []
        uid = self.spatial.uid
        attrs = self.spatial.attrs
        for ii,geom in iter_array(self.spatial.geom,return_value=True):
            dct = {'geom':geom,'UGID':uid[ii]}
            for k,v in attrs.iteritems():
                dct[k] = v[ii]
            geoms.append(dct)
        sc = ShpCabinet()
        sc.write(geoms,path,sr=self.spatial.projection.sr)