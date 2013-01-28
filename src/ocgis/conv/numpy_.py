from ocgis.conv.converter import OcgConverter
from ocgis.util.helpers import vprint

    
class NumpyConverter(OcgConverter):
    _create_directory = False
        
    def __iter__(self):
        for coll in self.so:
            #tdk
            try:
                vprint('geom id processed: {0}'.format(coll.ugeom['ugid']))
            except:
                pass
            #tdk
            yield(coll)
    
    def write(self):
        ret = {}
        for coll in self:
            ret.update({coll.ugeom['ugid']:coll})
        return(ret)
