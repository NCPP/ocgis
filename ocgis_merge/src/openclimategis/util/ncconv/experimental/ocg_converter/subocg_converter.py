from util.ncconv.experimental.helpers import timing
from util.ncconv.experimental.ocg_dataset.sub import SubOcgDataset
from util.ncconv.experimental.ocg_dataset.stat import SubOcgStat


class SubOcgConverter(object):
    """
    db -- Database module.
    base_name -- Name of the output file to create (i.e. 'foo.shp').
    use_stat=False -- Set to True to use data from the statistics table.
    meta=None -- MetaConverter object.
    use_geom=False -- Set to True to only write the geometry table. Flag for
        subclasses to alter operations.
    """
    
    def __init__(self,base_name,sub,meta=None,use_geom=False):
        self.sub = sub
        self.base_name = base_name
        self.meta = meta
        self.use_geom = use_geom

        if isinstance(sub,SubOcgDataset):
            self.use_stat = False
            self._true_sub = self.sub
        elif isinstance(sub,SubOcgStat):
            self.use_stat = True
            self._true_sub = self.sub.sub
            
        self._archetype = None
        
    @property
    def archetype(self):
        if self._archetype is None:
            self._archetype = self.get_iter().next()
        return(self._archetype)
    
    def get_iter(self,**kwds):
        if not kwds.get('keep_geom'):
            kwds.update({'keep_geom':False})
        if self.use_stat:
            return(self.sub.iter_stats(**kwds))
        else:
            if self.use_geom:
                return(self.sub.iter_geom_with_area(**kwds))
            else:
                return(self.sub.iter_with_area(**kwds))
            
    def get_headers(self,adds=[],keys=None):
        if keys is None:
            keys = self.archetype.keys()
        exclude = ['geometry']
        headers = [h.upper() for h in keys if h not in exclude] + adds
        return(headers)
    
    @timing
    def convert(self,*args,**kwds):
        return(self._convert_(*args,**kwds))
    
    def _convert_(self,*args,**kwds):
        raise(NotImplementedError)
    
    def response(self,*args,**kwds):
        payload = self.convert(*args,**kwds)
        try:
            return(self._response_(payload))
        finally:
            self.cleanup()
    
    def _response_(self,payload):
        return(payload)
    
    def cleanup(self):
        pass
    
    def write(self):
        raise(NotImplementedError)

    def write_meta(self,zip):
        if self.meta is not None:
            zip.writestr('meta.txt',self.meta.response())