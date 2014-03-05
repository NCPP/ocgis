from ocgis.conv.base import AbstractConverter
from ocgis.api.collection import SpatialCollection

    
class NumpyConverter(AbstractConverter):
    _create_directory = False
        
    def __iter__(self):
        for coll in self.colls:
            yield(coll)
    
    def write(self):
        build = True
        for coll in self:
            if build:
                ret = SpatialCollection(meta=coll.meta,key=coll.key,crs=coll.crs,headers=coll.headers)
                build = False
            for k,v in coll.iteritems():
                ret.add_field(k,coll.geoms[k],v.keys()[0],v.values()[0],properties=coll.properties[k])
        return(ret)

    def _write_(self): pass