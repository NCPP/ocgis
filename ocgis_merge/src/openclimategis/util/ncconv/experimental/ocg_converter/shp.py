from util.ncconv.experimental.ocg_converter.ocg_converter import OcgConverter
from util.helpers import get_temp_path
from osgeo import osr, ogr
from django.contrib.gis.gdal.error import check_err
import io
import zipfile
import os
from util.ncconv.experimental.ocg_converter.csv_ import LinkedCsvConverter
import datetime
from sqlalchemy.types import Float, Integer, Date, DateTime
from util.ncconv.experimental.ocg_converter.subocg_converter import SubOcgConverter


class ShpConverter(SubOcgConverter):
    __exts__ = ['shp','shx','prj','dbf']
    """
    layer='lyr' -- Name of the layer to create in the shapefile.
    srid=4326 -- Destination SRID for the data.
    """
    
    def __init__(self,*args,**kwds):
        self.layer = kwds.pop('layer','lyr')
        self.srid = kwds.pop('srid',4326)
        
        ## call the superclass
        super(ShpConverter,self).__init__(*args,**kwds)
        
        ## generate dataset path
        self.path = get_temp_path(name=self.base_name,nest=True)
        
        ## create shapefile base attributes
        self.fcache = FieldCache()
        self.ogr_fields = []
        self._set_ogr_fields_()
        
        ## get the geometry in order
#        self.ogr_geom = OGRGeomType(self.sub_ocg_dataset.geometry[0].geometryType()).num
        self.ogr_geom = 6 ## assumes multipolygon
        self.srs = osr.SpatialReference()
        self.srs.ImportFromEPSG(self.srid)
    
    def _set_ogr_fields_(self):
        ## create shapefile base attributes
        for key,value in self.archetype.iteritems():
            if key == 'WKT':
                continue
            self.ogr_fields.append(OgrField(self.fcache,key,type(value)))
#            if c.name == 'tid' and not self.use_stat:
#                self.ogr_fields.append(OgrField(self.fcache,'time',datetime.datetime))
#        self.ogr_fields.append(OgrField(self.fcache,'ocgid',int))
#        self.ogr_fields.append(OgrField(self.fcache,'gid',int))
#        self.ogr_fields.append(OgrField(self.fcache,'time',datetime.datetime))
#        self.ogr_fields.append(OgrField(self.fcache,'level',int))
#        self.ogr_fields.append(OgrField(self.fcache,'value',float))
#        self.ogr_fields.append(OgrField(self.fcache,'area_m2',float))
        
    def _get_iter_(self):
        ## returns an iterator instance that generates a dict to match
        ## the ogr field mapping. must also return a geometry wkt attribute.
        
        def wrap():
            for row in self.get_iter(wkt=True,keep_geom=False):
                yield(row)
            
        return(wrap())
        
    def _convert_(self):
        dr = ogr.GetDriverByName('ESRI Shapefile')
        ds = dr.CreateDataSource(self.path)
        if ds is None:
            raise IOError('Could not create file on disk. Does it already exist?')
        
        layer = ds.CreateLayer(self.layer,srs=self.srs,geom_type=self.ogr_geom)
        
        for ogr_field in self.ogr_fields:
            check_err(layer.CreateField(ogr_field.ogr_field))
                
        feature_def = layer.GetLayerDefn()
        
        for attr in (self._get_iter_()):
            feat = ogr.Feature(feature_def)
            ## pull values 
            for o in self.ogr_fields:
                val = attr[o.orig_name.upper()]
                args = [o.ogr_name,val]
                try:
                    feat.SetField(*args)
                except NotImplementedError:
                    args[1] = str(args[1])
                    feat.SetField(*args)
            check_err(feat.SetGeometry(ogr.CreateGeometryFromWkt(attr['WKT'])))
            check_err(layer.CreateFeature(feat))
            
        return(self.path)
    
    def _response_(self,payload):
        buffer = io.BytesIO()
        zip = zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)
        for item in self.__exts__:
            filepath = payload.replace('shp',item)
            zip.write(filepath,arcname=os.path.split(filepath)[1])
        self.write_meta(zip)
        zip.close()
        buffer.flush()
        zip_stream = buffer.getvalue()
        buffer.close()
        return(zip_stream)
    
    
class LinkedShpConverter(ShpConverter):
    
    def __init__(self,*args,**kwds):
        args = list(args)
        args[0] = os.path.splitext(args[0])[0]+'.shp'
        super(LinkedShpConverter,self).__init__(*args,**kwds)
        
    def write(self):
        zip_stream = self.response()
        path = get_temp_path(suffix='.zip')
        with open(path,'wb') as f:
            f.write(zip_stream)
        return(path)
    
    def _convert_(self):
        ## get the payload dictionary from the linked csv converter. we also
        ## want to store the database module.
        lcsv = LinkedCsvConverter(os.path.splitext(self.base_name)[0],self.sub)
        info = lcsv._convert_()
        ## get the shapefile path (and write in the process)
        path = super(LinkedShpConverter,self)._convert_()
        return(info,path)
        
    def _response_(self,payload):
        info,path = payload
        buffer = io.BytesIO()
        zip = zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)
        for item in self.__exts__:
            filepath = path.replace('shp',item)
            zip.write(filepath,arcname='shp/'+os.path.split(filepath)[1])
        for ii in info:
            zip.writestr('csv/'+ii['arcname'],ii['buffer'].getvalue())
        self.write_meta(zip)
        zip.close()
        buffer.flush()
        zip_stream = buffer.getvalue()
        buffer.close()
        return(zip_stream)

    def _get_iter_(self):
        
        def wrap():
            if self.use_stat:
                iter = self.sub.sub.iter_geom_with_area
            else:
                iter = self.sub.iter_geom_with_area
            for row in iter():
                geom = row.pop('geometry')
                row.update({'WKT':geom.wkt})
                yield(row)
                
        return(wrap())

    def _set_ogr_fields_(self):
        ## create shapefile base attributes
        self.ogr_fields.append(OgrField(self.fcache,'gid',int))
        self.ogr_fields.append(OgrField(self.fcache,'area_m2',float))


class OgrField(object):
    """
    Manages OGR fields mapping to correct Python types and configuring field
    definitions.
    """
    
    _mapping = {int:ogr.OFTInteger,
                datetime.date:ogr.OFTDate,
                datetime.datetime:ogr.OFTDateTime,
                float:ogr.OFTReal,
                str:ogr.OFTString,
                Float:ogr.OFTReal,
                Integer:ogr.OFTInteger,
                Date:ogr.OFTDate,
                DateTime:ogr.OFTDateTime}
    
    def __init__(self,fcache,name,data_type,precision=6,width=255):
        self.orig_name = name
        self._data_type = data_type
        
        self.ogr_name = fcache.add(name)
        
        try:
            self.ogr_type = self._mapping[data_type]
        except KeyError:
            self.ogr_type = self._mapping[type(data_type)]
        except:
            raise
        
        self.ogr_field = ogr.FieldDefn(self.ogr_name,self.ogr_type)
        if self.ogr_type == ogr.OFTReal: self.ogr_field.SetPrecision(precision)
        if self.ogr_type == ogr.OFTString: self.ogr_field.SetWidth(width)
        
        
class FieldCache(object):
    """Manage shapefile fields names."""
    
    def __init__(self):
        self._cache = []
    
    def add(self,name):
        name = str(name).upper()
        if len(name) > 10:
            name = name[0:10]
        if name not in self._cache:
            self._cache.append(name)
        else:
            raise ValueError('"{0}" is not a unique name.'.format(name))
        return name