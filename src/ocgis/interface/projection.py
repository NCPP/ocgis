from warnings import warn
from osgeo.osr import SpatialReference
from ocgis.util.helpers import itersubclasses
from osgeo.ogr import CreateGeometryFromWkb
from shapely.geometry.point import Point
from osgeo import osr
from shapely import wkb


def get_projection(dataset):
    projs = []
    for r in itersubclasses(DatasetSpatialReference):
        try:
            projs.append(r(dataset))
        except NoProjectionFound:
            continue
    if len(projs) == 1:
        ret = projs[0]
    elif len(projs) == 0:
        warn('no projection information found assuming WGS84')
        ret = WGS84()
    else:
        raise(MultipleProjectionsFound)
    return(ret)
    

class NoProjectionFound(Exception):
    pass


class MultipleProjectionsFound(Exception):
    pass


class OcgSpatialReference(object):
    _build = None
    
    def __init__(self,*args,**kwds):
        self.sr = self.get_sr(*args,**kwds)
    
    @classmethod
    def _build_(cls,build_str=None):
        raise(NotImplementedError)
        
    def get_area_km2(self,to_sr,geom):
        if isinstance(geom,Point):
            ret = None
        else:
            geom = CreateGeometryFromWkb(geom.wkb)
            geom.AssignSpatialReference(self.sr)
            geom.TransformTo(to_sr)
            ret = geom.GetArea()*1e-6
        return(ret)
    
    def project_to_match(self,geoms,in_sr=None):
        """
        Args:
          geoms: A sequence of geometry dictionaries.
          
        Returns:
          A projected copy of the input geometry sequence.
        """
        if in_sr is None:
            in_sr = osr.SpatialReference()
            in_sr.ImportFromEPSG(4326)
        
        ret = [None]*len(geoms)
        for idx in range(len(geoms)):
            gc = geoms[idx].copy()
            geom = CreateGeometryFromWkb(gc['geom'].wkb)
            geom.AssignSpatialReference(in_sr)
            geom.TransformTo(self.sr)
            gc['geom'] = wkb.loads(geom.ExportToWkb())
            ret[idx] = gc
#            import ipdb;ipdb.set_trace()
#        try:
#            if self._srid == to_sr._srid:
#                ret = geom.wkb
#        except AttributeError:
#            geom = CreateGeometryFromWkb(geom.wkb)
#            geom.AssignSpatialReference(self.sr)
#            geom.TransformTo(to_sr)
#            ret = geom.ExportToWkb()
        return(ret)
    
    def get_sr(self,proj4_str):
        sr = osr.SpatialReference()
        sr.ImportFromProj4(proj4_str)
        return(sr)
    
    
class SridSpatialReference(OcgSpatialReference):
    _srid = None
    
    def __init__(self,*args,**kwds):
        assert(self._srid is not None)
        super(SridSpatialReference,self).__init__(*args,**kwds)
        
    @classmethod
    def _build_(cls):
        return(cls())
        
    def get_sr(self):
        sr = SpatialReference()
        sr.ImportFromEPSG(self._srid)
        return(sr)


class DatasetSpatialReference(OcgSpatialReference):
    _proj_str = None
        
    def get_sr(self,dataset=None):
        if dataset is not None:
            proj_str = self._get_proj4_(dataset)
            self._proj_str = proj_str
        ret = self._get_sr_from_proj4_(self._proj_str)
        return(ret)
    
    @classmethod
    def _build_(cls):
        return(cls(dataset=None))
    
    @staticmethod
    def _get_sr_from_proj4_(proj_str):
        sr = SpatialReference()
        sr.ImportFromProj4(proj_str)
        return(sr)
        
    def _get_proj4_(self,dataset):
        raise(NotImplementedError)
    
    
class HostetlerProjection(DatasetSpatialReference):
    
    def _get_proj4_(self,dataset):
        try:
            var = dataset.variables['Lambert_Conformal']
            proj = ('+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} '
                    '+lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 '
                    '+to_meter=1000.0 +no_defs')
            lat1,lat2 = var.standard_parallel[0],var.standard_parallel[1]
            lat0 = var.latitude_of_projection_origin
            lon0 = var.longitude_of_central_meridian
            proj = proj.format(lat1=lat1,lat2=lat2,lat0=lat0,lon0=lon0)
            return(proj)
        except KeyError:
            raise(NoProjectionFound)
        
        
class LambertConformalConic(DatasetSpatialReference):
    
    def _get_proj4_(self,dataset):
        try:
            var = dataset.variables['lambert_conformal_conic']
            proj = ('+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} '
                    '+lon_0={lon0} +x_0={false_easting} +y_0={false_northing} +datum=WGS84 '
                    '+units=km +no_defs ')
            lat1,lat2 = var.standard_parallel[0],var.standard_parallel[1]
            lat0 = var.latitude_of_projection_origin
            lon0 = var.longitude_of_central_meridian
            false_easting = var.false_easting
            false_northing = var.false_northing
            proj = proj.format(lat1=lat1,lat2=lat2,lat0=lat0,lon0=lon0,false_easting=false_easting,false_northing=false_northing)
            return(proj)
        except KeyError:
            raise(NoProjectionFound)
        
        
class WGS84(SridSpatialReference):
    _srid = 4326
    
    
class UsNationalEqualArea(SridSpatialReference):
    _srid = 2163