from ocgis.conv.meta import MetaConverter
import os.path
import abc
from abc import ABCMeta


class OcgConverter(object):
    __metaclass__ = ABCMeta
    '''Base converter object. Intended for subclassing.
    
    :param colls: A sequence of `~ocgis.OcgCollection` objects.
    :type colls: sequence of `~ocgis.OcgCollection` objects
    
    so :: SubsetOperation
    mode="raw" :: str :: Iterator mode.
    base_name="ocg" :: str :: Prefix for data outputs.
    wd="/tmp" :: str :: Working directory for data outputs. Outputs are nested
        in temporary folders creating in this directory.
    use_dir=None :: str :: If provided, forces outputs into this directory.
    '''
    __metaclass__ = abc.ABCMeta
    _ext = None
    
    @abc.abstractmethod
    def _write_(self): pass # string path or data
    
    def __init__(self,colls,outdir,prefix,mode='raw',ops=None,add_meta=True):
        self.colls = colls
        self.ops = ops
        self.prefix = prefix
        self.outdir = outdir
        self.mode = mode
        self.add_meta = add_meta
        
        if self._ext is None:
            self.path = self.outdir
        else:
            self.path = os.path.join(self.outdir,prefix+'.'+self._ext)
    
    def write(self):
        if self.add_meta:
            lines = MetaConverter(self.ops).write()
            out_path = os.path.join(self.outdir,self.prefix+'_'+MetaConverter._meta_filename)
            with open(out_path,'w') as f:
                f.write(lines)
        ret = self._write_()
        
        ## return anything from the overloaded _write_ method. otherwise return
        ## the internal path.
        if ret is None:
            ret = self.path
        
        return(ret)
        
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
        
        for coll in self.colls:
            yield(coll)
#            try:
#                if coll.is_empty:
#                    continue
#            except AttributeError:
#                if type(coll) == dict:
#                    pass
#            yield(coll)
            
#    def get_headers(self,coll):
#        it = MeltedIterator(coll,mode=self.mode)
#        return(it.get_headers(upper=True))
#    
#    def get_iter(self,coll):
#        return(MeltedIterator(coll,mode=self.mode).iter_list())
