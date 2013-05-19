from osgeo.ogr import CreateGeometryFromWkb
from shapely import wkb
from copy import deepcopy
from osgeo.osr import SpatialReference


class ocgis(object):
    
    def __init__(self,selection_geometry):
        self.selection_geometry = selection_geometry
    
    @property
    def geom_type(self):
        raise(NotImplementedError)
    
    @property
    def sr(self):
        sr = SpatialReference()
        sr.ImportFromProj4(self._proj4_str)
        return(sr)
    
    def get_aggregated(self):
        raise(NotImplementedError)
        
    def get_projected(self,to_sr):
        from_sr = self.sr
        se = self.selection_geometry
        len_se = len(se)
        loads = wkb.loads
        
        ret = [None]*len_se
        for idx in range(len_se):
            gc = deepcopy(se[idx])
            geom = CreateGeometryFromWkb(gc['geom'].wkb)
            geom.AssignSpatialReference(from_sr)
            geom.TransformTo(to_sr)
            gc['geom'] = loads(geom.ExportToWkb())
            ret[idx] = gc
        
        return(SelectionGeometry(ret,sr=to_sr))
    
    def get_unwrapped(self,axis):
        raise(NotImplementedError)
    
    def get_wrapped(self,axis):
        raise(NotImplementedError)


class SelectionGeometry(list):
    
    def __init__(self,*args,**kwds):
        self.ocgis = ocgis(self)
        sr = kwds.pop('sr',None)
        if sr is None:
            sr = SpatialReference()
            sr.ImportFromEPSG(4326)
        self.ocgis._proj4_str = sr.ExportToProj4()
        
        super(SelectionGeometry,self).__init__(*args,**kwds)
