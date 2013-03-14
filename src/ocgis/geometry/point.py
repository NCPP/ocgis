from shapely.geometry.multipoint import MultiPoint
import abc
from osgeo import osr
from shapely.geometry.point import Point


class SelectionGeometry(object):
    '''
    :param ugid: Unique geometry identifier.
    :type ugid: int
    :param geom: The stored geometry.
    :type geom: shapely.geometry
    :param attrs: A dictionary of attributes. Key is the attribute name and the value is the attribute value.
    :type attrs: dict
    '''
    
    def __init__(self,ugid,geom,attrs=None):
        self.ugid = ugid
        self.geom = geom
        self.attrs = None
        
class SelectionGeometryCollection(object):
    
    def __init__(self,sgeoms,srs=None):
        self.sgeoms = sgeoms
        
        if srs is None:
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(4326)
            self._srs_proj4 = sr.ExportToProj4()
        else:
            self._srs_proj4 = srs.ExportToProj4()
            
    @property
    def srs(self):
        sr = osr.SpatialReference()
        sr.ImportFromProj4(self._srs_proj4)
        return(sr)

pt1 = Point(1,1)
pt2 = Point(2,2)

sg1 = SelectionGeometry(1,pt1,attrs={'foo':1})
sg2 = SelectionGeometry(2,pt2,attrs={'foo':2})

sgc = SelectionGeometryCollection([sg1,sg2])