import os
from ocgis import env
from ConfigParser import ConfigParser
from ocgis.util.helpers import get_shp_as_multi
import ogr
import osr
from shapely.geometry.multipolygon import MultiPolygon
from ocgis.conv.shp import OgrField, FieldCache


class ShpCabinet(object):
    '''
    >>> sc = ShpCabinet()
    >>> path = sc.get_shp_path('mi_watersheds')
    >>> assert(path.endswith('mi_watersheds.shp'))
    >>> geom_dict = sc.get_geom_dict('mi_watersheds')
    >>> len(geom_dict)
    60
    >>> sc.get_headers(geom_dict)
    ['ID', 'HUC', 'HUCCODE', 'HUCNAME']
    >>> path = '/tmp/foo.shp'
    >>> sc.write(geom_dict,path)
    '/tmp/foo.shp'
    '''
#    >>> it = sc.get_converter_iterator(geom_dict)
#    >>> print(it.next())
#    '''
    
    def __init__(self,path=None):
        self.path = path or env.SHP_DIR
        
    def get_shp_path(self,key):
        return(os.path.join(self.path,key,'{0}.shp'.format(key)))
    
    def get_cfg_path(self,key):
        return(os.path.join(self.path,key,'{0}.cfg'.format(key)))
    
    def get_geom_dict(self,key):
        shp_path = self.get_shp_path(key)
        cfg_path = self.get_cfg_path(key)
        config = ConfigParser()
        config.read(cfg_path)
        id_attr = config.get('mapping','id')
        other_attrs = config.get('mapping','attributes').split(',')
        geom_dict = get_shp_as_multi(shp_path,
                                     uid_field=id_attr,
                                     attr_fields=other_attrs)
        return(geom_dict)
    
    def get_headers(self,geom_dict):
        ret = ['id']
        keys = geom_dict[0].keys()
        keys.remove('id')
        keys.remove('geom')
        keys.sort()
        ret += keys
        ret = [h.upper() for h in ret]
        return(ret)
    
    def get_converter_iterator(self,geom_dict):
        for dct in geom_dict:
            geom = dct.pop('geom')
            if not isinstance(geom,MultiPolygon):
                geom = MultiPolygon([geom])
            yield(dct,geom)
            
    def write(self,geom_dict,path):
#        path = self.get_path()

        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        
        dr = ogr.GetDriverByName('ESRI Shapefile')
        ds = dr.CreateDataSource(path)
        if ds is None:
            raise IOError('Could not create file on disk. Does it already exist?')
        
        layer = ds.CreateLayer('lyr',srs=sr,geom_type=ogr.wkbMultiPolygon)
        headers = self.get_headers(geom_dict)
        
        build = True
        for dct,geom in self.get_converter_iterator(geom_dict):
            if build:
                ogr_fields = self._get_ogr_fields_(headers,dct)
                for of in ogr_fields:
                    layer.CreateField(of.ogr_field)
                    feature_def = layer.GetLayerDefn()
                build = False
            feat = ogr.Feature(feature_def)
            for o in ogr_fields:
                args = [o.ogr_name,o.convert(dct[o.ogr_name.lower()])]
                try:
                    feat.SetField(*args)
                except NotImplementedError:
                    args[1] = str(args[1])
                    feat.SetField(*args)
            feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
            layer.CreateFeature(feat)
        
        ds = None
        
        return(path)

    def _get_ogr_fields_(self,headers,row):
        ## do not want to have a geometry field
        ogr_fields = []
        fcache = FieldCache()
        for h,r in zip(headers,row):
            ogr_fields.append(OgrField(fcache,h,type(r)))
        return(ogr_fields)
        
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    print 'doctest success'