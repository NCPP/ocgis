from ocgis.conv.base import OcgConverter
import datetime
from osgeo import ogr
import numpy as np
from types import NoneType
from shapely.geometry.multipolygon import MultiPolygon
from ocgis import constants, env

    
class ShpConverter(OcgConverter):
    _ext = 'shp'
    _add_ugeom = True
    _add_ugeom_nest = False
    
    def __init__(self,*args,**kwds):
        self.layer = kwds.pop('layer','lyr')
        self.srid = kwds.pop('srid',4326)
        
        super(ShpConverter,self).__init__(*args,**kwds)
        
        ## create shapefile base attributes
        self.fcache = FieldCache()
        self.ogr_fields = []
        
        ## get the geometry in order
#        self.ogr_geom = OGRGeomType(self.sub_ocg_dataset.geometry[0].geometryType()).num
#        self.ogr_geom = 6 ## assumes multipolygon
#        self.srs = self.ocg_dataset.i.spatial.projection.sr
    
    def _write_(self):
        dr = ogr.GetDriverByName('ESRI Shapefile')
        ds = dr.CreateDataSource(self.path)
        if ds is None:
            raise IOError('Could not create file on disk. Does it already exist?')
                
        build = True
        for coll in self:
            for geom,row in coll.get_iter():
                if build:
                    if isinstance(geom,MultiPolygon):
                        geom_type = ogr.wkbMultiPolygon
                    else:
                        geom_type = ogr.wkbPoint
                    if env.WRITE_TO_REFERENCE_PROJECTION:
                        srs = constants.reference_projection.sr
                    else:
                        srs = coll.projection.sr
                    layer = ds.CreateLayer(self.layer,srs=srs,geom_type=geom_type)
                    headers = coll.get_headers(upper=True)
                    self._set_ogr_fields_(headers,row)
                    for ogr_field in self.ogr_fields:
                        layer.CreateField(ogr_field.ogr_field)
                        feature_def = layer.GetLayerDefn()
                    build = False
                feat = ogr.Feature(feature_def)
                for ii,o in enumerate(self.ogr_fields):
                    args = [o.ogr_name,o.convert(row[ii])]
                    try:
                        feat.SetField(*args)
                    except NotImplementedError:
                        args[1] = str(args[1])
                        feat.SetField(*args)
#                wkb = self.ocg_dataset.i.projection.project(self.to_sr,row[-1])
                feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
                layer.CreateFeature(feat)
        
        ds = None
        
    def _set_ogr_fields_(self,headers,row):
        ## do not want to have a geometry field
        for h,r in zip(headers,row):
            self.ogr_fields.append(OgrField(self.fcache,h,type(r)))


class OgrField(object):
    """
    Manages OGR fields mapping to correct Python types and configuring field
    definitions.
    """
    
    _mapping = {int:[ogr.OFTInteger,None],
                datetime.date:[ogr.OFTDate,str],
                datetime.datetime:[ogr.OFTDateTime,str],
                float:[ogr.OFTReal,None],
                str:[ogr.OFTString,None],
                np.int64:[ogr.OFTReal,float],
                NoneType:[ogr.OFTInteger,None],
                np.int32:[ogr.OFTInteger,int],
                np.float64:[ogr.OFTReal,float],
                np.float32:[ogr.OFTReal,float],
                np.int16:[ogr.OFTInteger,int]}
    
    def __init__(self,fcache,name,data_type,precision=6,width=255):
        self.orig_name = name
        self._data_type = data_type
        
        self.ogr_name = fcache.add(name)
        
        try:
            self.ogr_type = self._mapping[data_type][0]
            self._conv = self._mapping[data_type][1]
        except KeyError:
            try:
                self.ogr_type = self._mapping[type(data_type)][0]
                self._conv = self._mapping[type(data_type)][1]
            except KeyError:
                raise(ValueError(('no type mapping for data type {0},' 
                      'field name {1}').format(data_type,name)))
        except:
            raise
        
        self.ogr_field = ogr.FieldDefn(self.ogr_name,self.ogr_type)
        if self.ogr_type == ogr.OFTReal: self.ogr_field.SetPrecision(precision)
        if self.ogr_type == ogr.OFTString: self.ogr_field.SetWidth(width)

    def convert(self,val):
        if self._conv is None:
            return(val)
        else:
            return(self._conv(val))
        
class FieldCache(object):
    """Manage shapefile fields names."""
    
    def __init__(self):
        self._cache = []
    
    def add(self,name):
        if len(name) > 10:
            name = name[0:10]
        if name not in self._cache:
            self._cache.append(name)
        else:
            raise ValueError('"{0}" is not a unique name.'.format(name))
        return name
