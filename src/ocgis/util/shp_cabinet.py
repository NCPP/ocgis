import os
from ocgis import env
import ogr
import osr
from shapely.geometry.multipolygon import MultiPolygon
import csv
from osgeo.ogr import CreateGeometryFromWkb
from shapely.geometry.polygon import Polygon
from shapely import wkb
import fiona
from ocgis.interface.base.crs import CoordinateReferenceSystem


class ShpCabinetIterator(object):
    '''
    Iterate over a geometry selected by `key`.
    '''
    
    def __init__(self,key,select_ugid=None):
        self.key = key
        self.select_ugid = select_ugid
        self.sc = ShpCabinet()
        
    def __iter__(self):
        '''
        Return an iterator as from :meth:`ocgis.ShpCabinet.iter_geoms`
        '''
        for row in self.sc.iter_geoms(self.key,select_ugid=self.select_ugid):
            yield(row)


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
            for fn in filenames:
                if fn.endswith('shp'):
                    ret.append(os.path.splitext(fn)[0])
        return(ret)
    
    def get_meta(self,key):
        path = self.get_shp_path(key)
        with fiona.open(path,'r') as source:
            return(source.meta)
        
    def get_shp_path(self,key):
        return(self._get_path_(key,ext='shp'))
    
    def get_cfg_path(self,key):
        return(self._get_path_(key,ext='cfg'))
    
    def _get_path_(self,key,ext='shp'):
        ret = None
        for dirpath,dirnames,filenames in os.walk(self.path):
            for filename in filenames:
                if filename.endswith(ext) and os.path.splitext(filename)[0] == key:
                    ret = os.path.join(dirpath,filename)
                    return(ret)
        if ret is None:
            raise(ValueError('a shapefile with key "{0}" was not found under the directory: {1}'.format(key,self.path)))
    
    def iter_geoms(self,key,select_ugid=None):
        """Iterate over geometries from a shapefile specified by `key`.
        
        >>> sc = ShpCabinet()
        >>> geoms = sc.iter_geoms('state_boundaries',select_ugid=[1,48])
        >>> len(list(geoms))
        2
        
        :param key: The shapefile identifier.
        :type key: str
        :param select_ugid: Sequence of unique identifiers matching values from the shapefile's UGID attribute.
        :type select_ugid: sequence
        :yields: dict
        """
        
        ## path to the target shapefile
        shp_path = self.get_shp_path(key)
        ## make sure requested geometry exists
        if not os.path.exists(shp_path):
            raise(RuntimeError('requested geometry with identifier "{0}" does not exist in the file system.'.format(key)))
        
        ## get the source CRS
        meta = self.get_meta(key)
        crs = CoordinateReferenceSystem(crs=meta['crs'])
        
        ## get the geometries
        ds = ogr.Open(shp_path)
        try:
            lyr = ds.GetLayerByIndex(0)
            lyr.ResetReading()
            if select_ugid is not None:
                lyr_name = lyr.GetName()
                ## format where statement different for singletons
                if len(select_ugid) == 1:
                    sql_where = 'UGID = {0}'.format(select_ugid[0])
                else:
                    sql_where = 'UGID IN {0}'.format(tuple(select_ugid))
                sql = 'SELECT * FROM {0} WHERE {1}'.format(lyr_name,sql_where)
                features = ds.ExecuteSQL(sql)
            else:
                features = lyr
            
            for feature in features:
                ## TODO: lowercase all properties, add crs definition to each geometry
                yld = {'geom':wkb.loads(feature.geometry().ExportToWkb()),'properties':feature.items(),'crs':crs,
                       'meta':meta}
                assert('UGID' in yld['properties'])
                yield(yld)
        finally:
            ds.Destroy()
            ds = None
    
    def get_headers(self,geoms):
        ret = ['UGID']
        keys = geoms.keys()
        for key in ['UGID']:
            try:
                keys.remove(key)
            ## perhaps it is lower case
            except ValueError:
                keys.remove(key.lower())
        ret += keys
        return([r.upper() for r in ret])
    
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
        
#        arch = CreateGeometryFromWkb(geom_dict[0]['geom'].wkb)
#        layer = ds.CreateLayer('lyr',srs=sr,geom_type=arch.GetGeometryType())
#        headers = self.get_headers(geom_dict)
        
        build = True
        for dct,geom in self.get_converter_iterator(geom_dict):
            if build:
                arch = CreateGeometryFromWkb(geom.wkb)
                layer = ds.CreateLayer('lyr',srs=sr,geom_type=arch.GetGeometryType())
                headers = self.get_headers(dct)
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
                    x = self._get_(dct,h)
                    row.append(x)
            writer.writerow(row)
            feat = ogr.Feature(feature_def)
            for o in ogr_fields:
                args = [o.ogr_name,None]
                args[1] = self._get_(dct,o.ogr_name)
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
            try:
                ret = dct[key.lower()]
            except KeyError:
                mp = {key.lower():key for key in dct.keys()}
                ret = dct[mp[key.lower()]]
        return(ret)

    def _get_ogr_fields_(self,headers,row):
        from ocgis.conv.shp import OgrField, FieldCache
        ## do not want to have a geometry field
        ogr_fields = []
        fcache = FieldCache()
        for header in headers:
            ogr_fields.append(OgrField(fcache,header,type(self._get_(row,header))))
        return(ogr_fields)
    