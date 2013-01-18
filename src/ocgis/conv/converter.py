from ocgis.util.helpers import get_temp_path, vprint
from ocgis.api.dataset.collection.iterators import MeltedIterator
from ocgis.conv.meta import MetaConverter
import os.path
import tempfile


class OcgConverter(object):
    '''Base converter object. Intended for subclassing.
    
    so :: SubsetOperation
    mode="raw" :: str :: Iterator mode.
    base_name="ocg" :: str :: Prefix for data outputs.
    wd="/tmp" :: str :: Working directory for data outputs. Outputs are nested
        in temporary folders creating in this directory.
    use_dir=None :: str :: If provided, forces outputs into this directory.
    '''
    _ext = None
    
    def __init__(self,so,mode='raw',prefix='ocg',wd=None,ops=None,add_meta=True):
        self.so = so
        self.ops = ops
        self.wd = wd or get_temp_path(wd=tempfile.gettempdir(),nest=True,only_dir=True)
        self.mode = mode
        self.add_meta = add_meta
        
        if self._ext is None:
            self.path = self.wd
        else:
            self.path = os.path.join(self.wd,prefix+'.'+self._ext)
    
    def write(self):
        if self.add_meta:
            lines = MetaConverter(self.ops).write()
            out_path = os.path.join(self.wd,MetaConverter._meta_filename)
            with open(out_path,'w') as f:
                f.write(lines)
        self._write_()
        return(self.path)
    
    def _write_(self):
        raise(NotImplementedError)
    
#    def run(self):
#        return(self.write())
        
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
