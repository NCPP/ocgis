from ocgis.conv.shp import ShpConverter
from ocgis.api.dataset.collection.dimension import SpatialDimension

    
class ShpIdxConverter(ShpConverter):
    
    def get_headers(self,coll):
        return(['UGID','GID'])
    
    def get_iter(self,coll):
        arch = coll._arch
        for row in arch.spatial:
            yield([coll.ugeom['ugid'],row[1]['gid']],row[1]['geom'])


class ShpIdxIdentifierConverter(ShpIdxConverter):
    
    def get_iter(self,dct):
        self.projection = dct['projection']
        coll = dct['data']
        for idx in range(coll.uid.shape[0]):
            yield([coll.ugid[idx],coll.uid[idx]],
                  SpatialDimension._conv_to_multi_(coll.value[idx]))