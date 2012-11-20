from ocgis.util.helpers import get_temp_path, vprint
from ocgis.meta.interface.projection import UsNationalEqualArea
from shapely.geometry.point import Point
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkb


class OcgConverter(object):
    '''Base converter object. Intended for subclassing.
    
    so :: SubsetOperation
    mode="raw" :: str :: Iterator mode.
    base_name="ocg" :: str :: Prefix for data outputs.
    wd="/tmp" :: str :: Working directory for data outputs. Outputs are nested
        in temporary folders creating in this directory.
    use_dir=None :: str :: If provided, forces outputs into this directory.
    alt_it :: iterator :: Yields same as so iterator. Used for caching.
    '''
    _ext = None
    
    def __init__(self,so,mode='raw',base_name='ocg',wd='/tmp',use_dir=None,alt_it=None):
        self.so = so
        self.ops = so.ops
        self.base_name = base_name
        self.wd = wd
        self.use_dir = use_dir
        self.mode = mode
        self.alt_it = alt_it
        
        ## reference to calculation engine for convenience
        self.cengine = self.so.cengine
        ## reference dataset object
        self.ocg_dataset = so.ops.dataset[0]['ocg_dataset']
        self.projection = self.ocg_dataset.i.projection
        ## destination projection for area calculations.
        ## TODO: consider moving this to the "get_collection" method inside the
        ## subset operation.
        self.to_sr = UsNationalEqualArea().sr
    
    def write(self):
        raise(NotImplementedError)
    
    def run(self):
        return(self.write())
        
    @classmethod
    def get_converter(cls,output_format):
        '''Return the converter based on output extensions or key.
        
        output_format :: str
        
        returns
        
        OcgConverter'''
        
        from ocgis.conv.shp import ShpConverter
        from ocgis.conv.csv_ import CsvConverter
        from ocgis.conv.numpy_ import NumpyConverter
        from ocgis.conv.shpidx import ShpIdxConverter
        from ocgis.conv.keyed import KeyedConverter
        from ocgis.conv.nc import NcConverter
        
        mmap = {'shp':ShpConverter,
                'csv':CsvConverter,
                'numpy':NumpyConverter,
                'shpidx':ShpIdxConverter,
                'keyed':KeyedConverter,
                'nc':NcConverter}
        
        return(mmap[output_format])
    
    def get_path(self):
        '''Generate the output path.
        
        returns
        
        str'''
        
        if self.use_dir is not None:
            nest = False
            wd = self.use_dir
        else:
            nest = True
            wd = self.wd
        tpath = get_temp_path(suffix='.'+self._ext,
                              nest=nest,
                              wd=wd,
                              name=self.base_name)
        return(tpath)

    def __iter__(self):
        '''Iterator over collections stored in the SubsetOperation object.
        
        yields
        
        OcgCollection
        dict'''
        
        if self.alt_it is not None:
            it = self.alt_it()
        else:
            it = self.so
        
        for coll,geom_dict in it:
            #tdk
            try:
                vprint('geom id processed: {0}'.format(geom_dict['id']))
            except TypeError:
                pass
            #tdk
            yield(coll,geom_dict)
            
    def get_headers(self,upper=False):
        '''Return headers depending on iterator mode.
        
        upper=False :: bool :: Set to True to make the headers uppercase.
        
        returns
        
        []str'''
        
        if self.mode in ['raw','agg']:
            ret = ['tid','ugid','gid','vid','vlid','lid','time','variable','level','value']
        elif self.mode == 'calc':
            ret = ['tgid','ugid','gid','vid','vlid','lid','cid','year','month',
                   'day','variable','level','calc_name','value']
        elif self.mode == 'multi':
            raise(NotImplementedError)
#            ret = ['tgid','gid','vlid','lid','year','month','day','level']\
#                   + self.calc_multi.keys()
        else:
            raise(NotImplementedError)
        if upper:
            ret = [ii.upper() for ii in ret]
        return(ret)
        
    def get_iter(self,coll):
        '''Get the row iterator.
        
        coll :: OcgCollection
        
        yields
        
        []str
        Shapely Polygon or MultiPolygon'''
        
        headers = self.get_headers(upper=False)
        it = coll.get_iter(self.mode)
        for row,geom in it.iter_rows(headers):
            geom,area_km2 = self._process_geom_(geom)
            yield(row,geom)
            
    def _process_geom_(self,geom):
        '''Geometry conversion and area calculation.
        
        geom :: Shapely Polygon or MultiPolygon
        
        returns
        
        Shapely Polygon or MultiPolygon
        float
        '''
        
        geom = self._conv_to_multi_(geom)
        area_km2 = self.projection.get_area_km2(self.to_sr,geom)
        return(geom,area_km2)
    
    def _conv_to_multi_(self,geom):
        '''Geometry conversion to single type.'''
        
        if isinstance(geom,Point):
            pass
        else:
            try:
                geom = MultiPolygon(geom)
            except TypeError:
                geom = MultiPolygon([geom])
            except AssertionError:
                geom = wkb.loads(geom.wkb)
        return(geom)
