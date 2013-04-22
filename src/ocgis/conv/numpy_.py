from ocgis.conv.base import OcgConverter

    
class NumpyConverter(OcgConverter):
    _create_directory = False
        
    def __iter__(self):
        for coll in self.colls:
            yield(coll)
    
    def write(self):
        ret = {}
        for coll in self:
            ret.update({coll.ugid:coll})
        return(ret)

    def _write_(self): pass