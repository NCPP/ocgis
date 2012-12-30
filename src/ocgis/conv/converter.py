from ocgis.util.helpers import get_temp_path, vprint
from ocgis.api.dataset.collection.iterators import MeltedIterator


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
    
    def __init__(self,so,mode='raw',base_name='ocg',wd='/tmp',use_dir=None):#,alt_it=None):
        self.so = so
        try:
            self.ops = so.ops
        except AttributeError:
            self.ops = None
        self.base_name = base_name
        self.wd = wd
        self.use_dir = use_dir
        self.mode = mode
    
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
        
        for coll in self.so:
            try:
                if coll.is_empty:
                    continue
            except AttributeError:
                if type(coll) == dict:
                    pass
            #tdk
            try:
                vprint('geom id processed: {0}'.format(coll.ugeom['ugid']))
            except:
                pass
            #tdk
            yield(coll)
            
    def get_headers(self,coll):
        it = MeltedIterator(coll,mode=self.mode)
        return(it.get_headers(upper=True))
    
    def get_iter(self,coll):
        return(MeltedIterator(coll,mode=self.mode).iter_list())
