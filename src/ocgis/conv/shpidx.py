from ocgis.util.helpers import iter_array
from ocgis.conv.shp import ShpConverter

    
class ShpIdxConverter(ShpConverter):
    
    def get_headers(self,upper=None):
        return(['GID'])
    
    def get_iter(self,coll):
        for gidx,gid in iter_array(coll.gid,return_value=True):
            geom,area_km2 = self._process_geom_(coll.geom[gidx])
            yield([gid],geom)
