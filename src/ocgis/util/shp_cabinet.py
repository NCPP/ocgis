import os
from ocgis import env
from ConfigParser import ConfigParser
from ocgis.util.helpers import get_shp_as_multi
import ogr
import osr
from shapely.geometry.multipolygon import MultiPolygon
import csv
from osgeo.ogr import CreateGeometryFromWkb
from shapely.geometry.polygon import Polygon
from osgeo.osr import SpatialReference
from copy import deepcopy
from shapely import wkb


class ShpCabinet(object):
    '''A utility object designed for accessing shapefiles stored in a locally
    accessible location.
    
    >>> # Adjust location of :class:`ocgis.ShpCabinet` search directory.
    >>> import ocgis
    ...
    >>> ocgis.env.DIR_SHPCABINET = '/path/to/local/shapefile/directory'
    >>> sc = ShpCabinet()
    >>> # List the shapefiles available.
    >>> sc.keys()
    ['state_boundaries', 'mi_watersheds', 'world_countries']
    >>> # Load geometries from the shapefile.
    >>> geoms = sc.get_geoms('state_boundaries')
    
    :param path: Absolute path the directory holding shapefile folders. Defaults to :attr:`ocgis.env.DIR_SHPCABINET`.
    :type path: str
    '''
    
    def __init__(self,path=None):
        self._path = path or env.DIR_SHPCABINET
    
    @property
    def path(self):
        if self._path is None:
            raise(ValueError('A path value is required. Either pass a path to the constructor or set ocgis.env.DIR_SHPCABINET.'))
        elif not os.path.exists(self._path):
            raise(ValueError('Specified path to ShpCabinet folder does not exist: {0}'.format(self._path)))
        return(self._path)
    
    def keys(self):
        """Return a list of the shapefile keys contained in the search directory.
        
        :rtype: list of str
        """
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
        """Return geometries from a shapefile specified by `key`.
        
        :param key: The shapefile identifier.
        :type key: str
        :param attr_filter: A dict containing attribute filters. Keys indicate attribute fields and values should be lists that will match attribute values `exactly`.
        :type attr_filter: dict
        """
        
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
        if id_attr.lower() == 'none':
            id_attr = None
            make_id = True
        else:
            make_id = False
        other_attrs = config.get('mapping','attributes').split(',')
        ## allow for no attributes to be loaded.
        if len(other_attrs) == 1 and other_attrs[0].lower() == 'none':
            other_attrs = []
        ## allow for all attributes to be loaded
        elif len(other_attrs) == 1 and other_attrs[0].lower() == 'all':
            other_attrs = 'all'
        ## get the geometry objects.
        geoms = get_shp_as_multi(shp_path,
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
            dtype = type(geoms[0][attr])
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
            geoms = filter(_filter_,geoms)
        
        return(SelectionGeometry(geoms))
    
    def get_headers(self,geoms):
        ret = ['ugid']
        keys = geoms[0].keys()
        for key in ['ugid','id','geom']:
            try:
                keys.remove(key)
            except ValueError:
                try:
                    keys.remove(key.upper())
                except ValueError:
                    pass
        keys.sort()
        ret += keys
        ret = [h.upper() for h in ret]
        return(ret)
    
    def get_converter_iterator(self,geom_dict):
        for dct in geom_dict:
            dct_copy = dct.copy()
            geom = dct_copy.pop('geom')
            if isinstance(geom,Polygon):
                geom = MultiPolygon([geom])
            yield(dct_copy,geom)
            
    def write(self,geom_dict,path,sr=None):
        """Write a list of geometry dictionaries (similar to that returned by :func:`~ocgis.ShpCabinet.get_geoms`) to disk.
        
        :param geom_dict: The list of geometry dictionaries.
        :type geom_dict: list of dict
        :param path: The absolute path to the output file.
        :type path: str
        :param sr: The spatial reference for the output. Defaults to WGS84.
        :type sr: :class:`osgeo.osr.SpatialReference`
        :rtype: str path to output file.
        """
        
        from ocgis.conv.csv_ import OcgDialect
#        path = self.get_path()

        if sr is None:
            sr = osr.SpatialReference()
            sr.ImportFromEPSG(4326)
        
        dr = ogr.GetDriverByName('ESRI Shapefile')
        ds = dr.CreateDataSource(path)
        if ds is None:
            raise IOError('Could not create file on disk. Does it already exist?')
        
        arch = CreateGeometryFromWkb(geom_dict[0]['geom'].wkb)
        layer = ds.CreateLayer('lyr',srs=sr,geom_type=arch.GetGeometryType())
        headers = self.get_headers(geom_dict)
        
        build = True
        for dct,geom in self.get_converter_iterator(geom_dict):
            if build:
                csv_path = path.replace('.shp','.csv')
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
        from ocgis.conv.shp import OgrField, FieldCache
        ## do not want to have a geometry field
        ogr_fields = []
        fcache = FieldCache()
        for header in headers:
            ogr_fields.append(OgrField(fcache,header,type(self._get_(row,header))))
        return(ogr_fields)
    
    
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
