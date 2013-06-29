from osgeo.osr import SpatialReference
from ocgis.util.helpers import itersubclasses
from osgeo.ogr import CreateGeometryFromWkb
from shapely.geometry.point import Point
import abc
from ocgis.util.logging_ocgis import ocgis_lh
import logging


def get_projection(dataset):
    projs = []
    for r in itersubclasses(DatasetSpatialReference):
        try:
            projs.append(r.init_from_dataset(dataset))
        except NoProjectionFound:
            continue
    if len(projs) == 1:
        ret = projs[0]
    elif len(projs) == 0:
        try:
            files = dataset._files
        except AttributeError:
            files = 'NA'
        ocgis_lh('No projection information found assuming WGS84: {0}'.format(files),
                 'projection',
                 logging.WARN)
        ret = WGS84()
    else:
        raise(MultipleProjectionsFound)
    return(ret)
    

class NoProjectionFound(Exception):
    pass


class MultipleProjectionsFound(Exception):
    pass


class OcgSpatialReference(object):
    __metaclass__ = abc.ABCMeta
    _build = None
    
    @abc.abstractmethod
    def write_to_rootgrp(self,rootgrp): return(None)
        
    @abc.abstractproperty
    def sr(self): pass
        
    def get_area_km2(self,to_sr,geom):
        if isinstance(geom,Point):
            ret = None
        else:
            geom = CreateGeometryFromWkb(geom.wkb)
            geom.AssignSpatialReference(self.sr)
            geom.TransformTo(to_sr)
            ret = geom.GetArea()*1e-6
        return(ret)
    
#    def project_to_match(self,geoms,in_sr=None):
#        """
#        Args:
#          geoms: A sequence of geometry dictionaries.
#          
#        Returns:
#          A projected copy of the input geometry sequence.
#        """
#        if in_sr is None:
#            in_sr = osr.SpatialReference()
#            in_sr.ImportFromEPSG(4326)
#        
#        ret = [None]*len(geoms)
#        for idx in range(len(geoms)):
#            gc = geoms[idx].copy()
#            geom = CreateGeometryFromWkb(gc['geom'].wkb)
#            geom.AssignSpatialReference(in_sr)
#            geom.TransformTo(self.sr)
#            gc['geom'] = wkb.loads(geom.ExportToWkb())
#            ret[idx] = gc
##            import ipdb;ipdb.set_trace()
##        try:
##            if self._srid == to_sr._srid:
##                ret = geom.wkb
##        except AttributeError:
##            geom = CreateGeometryFromWkb(geom.wkb)
##            geom.AssignSpatialReference(self.sr)
##            geom.TransformTo(to_sr)
##            ret = geom.ExportToWkb()
#        return(ret)
#    
##    def get_sr(self,proj4_str):
##        sr = osr.SpatialReference()
##        sr.ImportFromProj4(proj4_str)
##        return(sr)
    
    
class SridSpatialReference(OcgSpatialReference):
    
    @abc.abstractproperty
    def _srid(self): int
        
    @property
    def sr(self):
        sr = SpatialReference()
        sr.ImportFromEPSG(self._srid)
        return(sr)


class DatasetSpatialReference(OcgSpatialReference):
    @abc.abstractproperty
    def _names(self): [str]
    @abc.abstractproperty
    def _template(self): str
    
    def __init__(self):
        self._proj4_str = None
        
    @property
    def sr(self):
        sr = SpatialReference()
        sr.ImportFromProj4(self.proj4_str)
        return(sr)
    
    @classmethod
    def init_from_dataset(cls,dataset):
        chk = set(cls._names).intersection(dataset.variables.keys())
        try:
            var = dataset.variables[list(chk)[0]]
            ret = cls._init_from_variable_(var)
        except IndexError:
            raise(NoProjectionFound)
        return(ret)
        
    def write_to_rootgrp(self,rootgrp,ncmeta):
        found = False
        for key in self._names:
            try:
                ref = ncmeta['variables'][key]
                found = True
                lc = rootgrp.createVariable(key,ref['dtype'])
                for k,v in ref['attrs'].iteritems():
                    setattr(lc,k,v)
            except KeyError:
                continue
        if not found:
            raise(KeyError)
        
#    @classmethod
    @abc.abstractmethod
    def _init_from_variable_(cls,var):
        pass
    
#    @abc.abstractmethod
#    def _get_proj4_(self,dataset): str
    
    
#class HostetlerProjection(DatasetSpatialReference):
#    
#    def write_to_rootgrp(self):
#        raise(NotImplementedError)
#    
#    def _get_proj4_(self,dataset):
#        try:
#            var = dataset.variables['Lambert_Conformal']
#            proj = ('+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} '
#                    '+lon_0={lon0} +x_0=0 +y_0=0 +datum=WGS84 '
#                    '+to_meter=1000.0 +no_defs')
#            lat1,lat2 = var.standard_parallel[0],var.standard_parallel[1]
#            lat0 = var.latitude_of_projection_origin
#            lon0 = var.longitude_of_central_meridian
#            proj = proj.format(lat1=lat1,lat2=lat2,lat0=lat0,lon0=lon0)
#            return(proj)
#        except KeyError:
#            raise(NoProjectionFound)
        
        
class LambertConformalConic(DatasetSpatialReference):
    _names = ['lambert_conformal_conic','Lambert_Conformal']
    _template = ('+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat_0} '
     '+lon_0={lon_0} +x_0={x_0} +y_0={y_0} +datum={datum} '
     '+units={units} +no_defs ')
    
    def __init__(self,standard_parallel,longitude_of_central_meridian,
                 latitude_of_projection_origin,false_easting,false_northing,
                 units='km',datum='WGS84'):
        self.standard_parallel = standard_parallel
        self.longitude_of_central_meridian = longitude_of_central_meridian
        self.latitude_of_projection_origin = latitude_of_projection_origin
        self.false_easting = false_easting
        self.false_northing = false_northing
        self.units = units
        self.datum = datum
        super(LambertConformalConic,self).__init__()
        
    @property
    def proj4_str(self):
        if self._proj4_str is None:
            kwds = {}
            lat1,lat2 = self.standard_parallel
            kwds['lat1'] = lat1
            kwds['lat2'] = lat2
            kwds['datum'] = self.datum
            kwds['units'] = self.units
            kwds['lat_0'] = self.latitude_of_projection_origin
            kwds['lon_0'] = self.longitude_of_central_meridian
            kwds['x_0'] = self.false_easting
            kwds['y_0'] = self.false_northing
            self._proj4_str = self._template.format(**kwds)
        return(self._proj4_str)
    
    @classmethod
    def _init_from_variable_(cls,var):
        ret = cls(var.standard_parallel,var.longitude_of_central_meridian,
                 var.latitude_of_projection_origin,var.false_easting,var.false_northing)
        return(ret)
        

class NarccapObliqueMercator(DatasetSpatialReference):
    _names = ['Transverse_Mercator']
    _template = ('+proj=omerc +lat_0={lat_0} +lonc={lonc} +k_0={k_0} '
                 '+x_0={x_0} +y_0={y_0} +alpha={alpha}')
    
    def __init__(self,latitude_of_projection_origin,longitude_of_central_meridian,
                 scale_factor_at_central_meridian,false_easting,false_northing,
                 alpha=360):
        self.latitude_of_projection_origin = latitude_of_projection_origin
        self.longitude_of_central_meridian = longitude_of_central_meridian
        self.scale_factor_at_central_meridian = scale_factor_at_central_meridian
        self.false_easting = false_easting
        self.false_northing = false_northing
        self.alpha = alpha
        super(NarccapObliqueMercator,self).__init__()
        
    @property
    def proj4_str(self):
        if self._proj4_str is None:
            kwds = {}
            kwds['lat_0'] = self.latitude_of_projection_origin
            kwds['lonc'] = self.longitude_of_central_meridian
            kwds['k_0'] = self.scale_factor_at_central_meridian
            kwds['x_0'] = self.false_easting
            kwds['y_0'] = self.false_northing
            kwds['alpha'] = self.alpha
            self._proj4_str = self._template.format(**kwds)
        return(self._proj4_str)
    
    @classmethod
    def _init_from_variable_(cls,var):
        ret = cls(var.latitude_of_projection_origin,var.longitude_of_central_meridian,
                 var.scale_factor_at_central_meridian,var.false_easting,var.false_northing)
        return(ret)


class PolarStereographic(DatasetSpatialReference):
    _names = ['polar_stereographic']
    _template = ('+proj=stere +lat_ts={latitude_natural} +lat_0={lat_0} +lon_0={longitude_natural} '
                 '+k_0={scale_factor} +x_0={false_easting} +y_0={false_northing}')
    
    def __init__(self,standard_parallel,latitude_of_projection_origin,straight_vertical_longitude_from_pole,
                 false_easting,false_northing,scale_factor=1.0):
        self.standard_parallel = standard_parallel
        self.latitude_of_projection_origin = latitude_of_projection_origin
        self.straight_vertical_longitude_from_pole = straight_vertical_longitude_from_pole
        self.false_easting = false_easting
        self.false_northing = false_northing
        self.scale_factor = scale_factor
        super(PolarStereographic,self).__init__()
        
    @property
    def proj4_str(self):
        if self._proj4_str is None:
            kwds = {}
            kwds['latitude_natural'] = self.standard_parallel
            kwds['lat_0'] = self.latitude_of_projection_origin
            kwds['longitude_natural'] = self.straight_vertical_longitude_from_pole
            kwds['scale_factor'] = self.scale_factor
            kwds['false_easting'] = self.false_easting
            kwds['false_northing'] = self.false_northing
            self._proj4_str = self._template.format(**kwds)
        return(self._proj4_str)
    
    @classmethod
    def _init_from_variable_(cls,var):
        ret = cls(var.standard_parallel,var.latitude_of_projection_origin,
                  var.straight_vertical_longitude_from_pole,var.false_easting,
                  var.false_northing)
        return(ret)
        
        
class RotatedPole(DatasetSpatialReference):
    _names = ['rotated_pole']
    _template = '+proj=ob_tran +o_proj=latlon +o_lon_p={lon_pole} +o_lat_p={lat_pole} +lon_0=180'
    
    def __init__(self,grid_north_pole_longitude,grid_north_pole_latitude):
        self.grid_north_pole_longitude = grid_north_pole_longitude
        self.grid_north_pole_latitude = grid_north_pole_latitude
        self._trans_proj = self._template.format(lon_pole=self.grid_north_pole_longitude,
                                                 lat_pole=self.grid_north_pole_latitude)
        super(RotatedPole,self).__init__()
        
    @property
    def proj4_str(self):
        if self._proj4_str is None:
            self._proj4_str = WGS84().sr.ExportToProj4()
        return(self._proj4_str)
    
    @classmethod
    def _init_from_variable_(cls,var):
        ret = cls(var.grid_north_pole_longitude,var.grid_north_pole_latitude)
        return(ret)
        
        
class WGS84(SridSpatialReference):
    _srid = 4326
    
    def write_to_rootgrp(self,rootgrp):
        pass


class UsNationalEqualArea(SridSpatialReference):
    _srid = 2163
