from osgeo import ogr, osr
import datetime
from django.contrib.gis.gdal.geomtype import OGRGeomType
from django.contrib.gis.gdal.error import check_err
import zipfile
from django.http import HttpResponse
import os
from io import BytesIO

from ipdb import set_trace as tr


class OpenClimateShp(object):
    """
    Write ESRI Shapefiles from dictionary list.
    
    path -- absolute path with filename. i.e. '/tmp/foo.shp'
    attrs -- dictionary list
    geom='geom' -- key to the geometry field
    layer='lyr' -- name of the shapefile layer containing the features
    id='id' -- name of the unique identifier field
    """
    
    def __init__(self,path,attrs,geom='geometry',layer='lyr',id='id',srid=4326):
        self.path = path
        self.filename = os.path.split(path)[1]
        self.attrs = attrs
        self.geom = geom
        self.layer = layer
        self.id = id
        self.srid = srid
        
        ## CREATE SHAPEFILE BASE ATTRS -----------------------------------------
        
        ## manages field names in the shapefile
        self.fcache = FieldCache()
        ## need the first row to assess data types
        template = self.attrs[0]
        ## the OGR fields
#        tr()
        self.ogr_fields = [OgrField(self.fcache,id,template[id])]
        self.ogr_fields += [OgrField(self.fcache,key,value) 
                           for key,value in template['properties'].iteritems()]
#        self.ogr_fields = [OgrField(self.fcache,key,value) 
#                           for key,value in template.iteritems() 
#                           if key != self.geom]
        ## create the geometry type
        bgeom = template[self.geom]
#        tr()
#        self.ogr_geom = OGRGeomType(bgeom.ogr.geom_type).num
        self.ogr_geom = OGRGeomType(bgeom.geometryType()).num
#        tr()
        ## the spatial reference system
#        import ipdb;ipdb.set_trace()
#        try:
#        try:
#            self.srs = osr.SpatialReference(bgeom.srid)
        self.srs = osr.SpatialReference()
        self.srs.ImportFromEPSG(srid)
#        except:
#            import ipdb;ipdb.set_trace()
#        except TypeError:
#            import ipdb;ipdb.set_trace()
#            self.srs = osr.SpatialReference(int(bgeom.srid))
        
    def write(self):
        """Write the shapefile to disk."""
        
        dr = ogr.GetDriverByName('ESRI Shapefile')
        ds = dr.CreateDataSource(self.path)
        if ds is None:
            raise IOError('Could not create file on disk. Does it already exist?')
        
        layer = ds.CreateLayer(self.layer,srs=self.srs,geom_type=self.ogr_geom)
        
        for ogr_field in self.ogr_fields:
            check_err(layer.CreateField(ogr_field.ogr_field))
                
        feature_def = layer.GetLayerDefn()
        
        for attr in self.attrs:
            feat = ogr.Feature(feature_def)
            for o in self.ogr_fields:
                if o.orig_name == self.id:
                    args = (o.ogr_name,attr[o.orig_name])
                else:
                    args = (o.ogr_name,attr['properties'][o.orig_name])
                try:
                    feat.SetField(*args)
                except NotImplementedError:
                    args = list(args)
                    args[1] = str(args[1])
                    feat.SetField(*args)
            check_err(feat.SetGeometry(ogr.CreateGeometryFromWkt(attr[self.geom].wkt)))
            check_err(layer.CreateFeature(feat))
        
        ds.Destroy()
        
    def zip_response(self):
        buffer = BytesIO()
        zip = zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED)
        files = ['shp','shx','prj','dbf']
        for item in files:
            filepath = self.path.replace('shp',item)
#            import ipdb;ipdb.set_trace()
#            filename = '%s.%s' % (self.path.replace('.shp',''), item)
            zip.write(filepath)#, arcname='%s.%s' % (file_name.replace('.shp',''), item))
#        if readme:
#            zip.writestr('README.txt',readme)
        zip.close()
        buffer.flush()
        zip_stream = buffer.getvalue()
        buffer.close()
        
        # Stick it all in a django HttpResponse
        response = HttpResponse()
        response['Content-Disposition'] = 'attachment; filename={0}.shz'.format(self.filename)# % file_name.replace('.shp','')
        response['Content-length'] = str(len(zip_stream))
        response['Content-Type'] = 'application/zip'
        response.write(zip_stream)
        return response


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


class OgrField(object):
    """
    Manages OGR fields mapping to correct Python types and configuring field
    definitions.
    """
    
    _mapping = {int:ogr.OFTInteger,
                datetime.date:ogr.OFTDate,
                datetime.datetime:ogr.OFTDateTime,
                float:ogr.OFTReal,
                str:ogr.OFTString}
    
    def __init__(self,fcache,name,data_type,precision=6,width=255):
        self.orig_name = name
        self._data_type = data_type
        
        self.ogr_name = fcache.add(name)
        
        try:
            self.ogr_type = self._mapping[data_type]
        except KeyError:
            self.ogr_type = self._mapping[type(data_type)]
        else:
            raise
        
        self.ogr_field = ogr.FieldDefn(self.ogr_name,self.ogr_type)
        if self.ogr_type == ogr.OFTReal: self.ogr_field.SetPrecision(precision)
        if self.ogr_type == ogr.OFTString: self.ogr_field.SetWidth(width)
        