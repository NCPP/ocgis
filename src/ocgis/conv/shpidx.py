from ocgis.util.helpers import iter_array
from ocgis.conv.shp import ShpConverter
from ocgis.api.dataset.collection.collection import Identifier

    
class ShpIdxConverter(ShpConverter):
    
    def __init__(self,*args,**kwds):
        super(self.__class__,self).__init__(*args,**kwds)
        
        self._id = Identifier()
    
    def get_headers(self):
        return(['UGID'])
    
    def get_iter(self,coll):
        import ipdb;ipdb.set_trace()
        for gidx,gid in iter_array(coll.gid,return_value=True):
            geom,area_km2 = self._process_geom_(coll.geom[gidx])
            yield([gid],geom)
