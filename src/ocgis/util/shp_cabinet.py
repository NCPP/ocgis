import os
from ocgis import env
from ConfigParser import ConfigParser
from ocgis.util.helpers import get_shp_as_multi
import ogr
import osr
from shapely.geometry.multipolygon import MultiPolygon
from ocgis.conv.shp import OgrField, FieldCache
import csv
from ocgis.conv.csv_ import OcgDialect


class ShpCabinet(object):
    '''
    >>> sc = ShpCabinet()
    >>> path = sc.get_shp_path('mi_watersheds')
    >>> assert(path.endswith('mi_watersheds.shp'))
    >>> geom_dict = sc.get_geom_dict('mi_watersheds')
    >>> len(geom_dict)
    60
    >>> sc.get_headers(geom_dict)
    ['UGID', 'HUC', 'HUCCODE', 'HUCNAME']
    >>> attr_filter = {'ugid':['1','2']}
    >>> filtered = sc.get_geom_dict('mi_watersheds',attr_filter=attr_filter)
    >>> len(filtered)
    2
    '''
    
    def __init__(self,path=None):
        self.path = path or env.SHP_DIR
        
    def keys(self):
        ret = []
        for dirpath,dirnames,filenames in os.walk(self.path):
            for dn in dirnames:
                for fn in os.listdir(os.path.join(dirpath,dn)):
                    if fn.endswith('shp'):
                        ret.append(os.path.splitext(fn)[0])
        return(ret)
        
    def get_shp_path(self,key):
        return(os.path.join(self.path,key,'{0}.shp'.format(key)))
    
    def get_cfg_path(self,key):
        return(os.path.join(self.path,key,'{0}.cfg'.format(key)))
    
    def get_geom_dict(self,*args,**kwds):
        return(self.get_geoms(*args,**kwds))
    
    def get_geoms(self,key,attr_filter=None):
        shp_path = self.get_shp_path(key)
        ## make sure requested geometry exists
        if not os.path.exists(shp_path):
            raise(RuntimeError('requested geometry with identifier "{0}" does not exists in the file system.'.format(key)))
        cfg_path = self.get_cfg_path(key)
        config = ConfigParser()
        config.read(cfg_path)
        id_attr = config.get('mapping','ugid')
        ## adjust the id attribute name for auto-generation in the shapefile
        ## reader.
        if id_attr == 'none':
            id_attr = None
            make_id = True
        else:
            make_id = False
        other_attrs = config.get('mapping','attributes').split(',')
        geom_dict = get_shp_as_multi(shp_path,
                                     uid_field=id_attr,
                                     attr_fields=other_attrs,
                                     make_id=make_id)

        ## filter the returned geometries if an attribute filter is passed
        if attr_filter is not None:
            ## get the attribute
            attr = attr_filter.keys()[0].lower()
#            ## rename ugid to id to prevent confusion on the front end.
#            if attr == 'ugid':
#                attr = 'id'
            ## get the target attribute data type
            dtype = type(geom_dict[0][attr])
            ## attempt to convert the filter values to that data type
            fvalues = [dtype(ii) for ii in attr_filter.values()[0]]
            ## if the filter data type is a string, do a conversion
            if dtype == str:
                fvalues = [f.lower() for f in fvalues]
            ## filter function
            def _filter_(x):
                ref = x[attr]
                ## attempt to lower the string value, otherwise move on
                try:
                    ref = ref.lower()
                except AttributeError:
                    pass
                if ref in fvalues: return(True)
            ## filter the geometry dictionary
            geom_dict = filter(_filter_,geom_dict)
        return(geom_dict)
    
    def get_headers(self,geom_dict):
        ret = ['ugid']
        keys = geom_dict[0].keys()
        for key in ['id','geom']:
            try:
                keys.remove(key)
            except ValueError:
                keys.remove(key.upper())
        keys.sort()
        ret += keys
        ret = [h.upper() for h in ret]
        return(ret)
    
    def get_converter_iterator(self,geom_dict):
        for dct in geom_dict:
            geom = dct.pop('geom')
            if not isinstance(geom,MultiPolygon):
                geom = MultiPolygon([geom])
            try:
                dct['ugid'] = dct.pop('id')
            except KeyError:
                dct['ugid'] = dct.pop('ID')
            yield(dct,geom)
            
    def write(self,geom_dict,path):
#        path = self.get_path()

        ##tdk
        sr = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
#        sr = osr.SpatialReference()
#        sr.ImportFromProj4('+proj=longlat +datum=WGS84 +pm=180dW ')
        ##tdk
        
        dr = ogr.GetDriverByName('ESRI Shapefile')
        ds = dr.CreateDataSource(path)
        if ds is None:
            raise IOError('Could not create file on disk. Does it already exist?')
        
        layer = ds.CreateLayer('lyr',srs=sr,geom_type=ogr.wkbMultiPolygon)
        headers = self.get_headers(geom_dict)
        
        build = True
        for dct,geom in self.get_converter_iterator(geom_dict):
            if build:
                csv_path = path.replace('shp','csv')
                csv_f = open(csv_path,'w')
                writer = csv.writer(csv_f,dialect=OcgDialect)
                writer.writerow(headers)
                
                ogr_fields = self._get_ogr_fields_(headers,dct)
                for of in ogr_fields:
                    layer.CreateField(of.ogr_field)
                    feature_def = layer.GetLayerDefn()
                build = False
            try:
                row = [dct[h.lower()] for h in headers]
            except KeyError:
                row = []
                for h in headers:
                    try:
                        x = dct[h]
                    except KeyError:
                        x = dct[h.lower()]
                    row.append(x)
            writer.writerow(row)
            feat = ogr.Feature(feature_def)
            for o in ogr_fields:
                args = [o.ogr_name,None]
                try:
                    args[1] = dct[o.ogr_name.lower()]
                except KeyError:
                    args[1] = dct[o.ogr_name]
#                args = [o.ogr_name,o.convert(dct[o.ogr_name.lower()])]
                try:
                    feat.SetField(*args)
                except NotImplementedError:
                    args[1] = str(args[1])
                    feat.SetField(*args)
            feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
            layer.CreateFeature(feat)
        
        ds = None
        csv_f.close()
        
        return(path)
    
    def _get_(self,dct,key):
        try:
            ret = dct[key]
        except KeyError:
            ret = dct[key.lower()]
        return(ret)

    def _get_ogr_fields_(self,headers,row):
        ## do not want to have a geometry field
        ogr_fields = []
        fcache = FieldCache()
        for header in headers:
            ogr_fields.append(OgrField(fcache,header,type(self._get_(row,header))))
        return(ogr_fields)
        
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    print 'doctest success'