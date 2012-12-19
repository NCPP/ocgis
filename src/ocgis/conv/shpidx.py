from ocgis.conv.shp import ShpConverter

    
class ShpIdxConverter(ShpConverter):
    
    def get_headers(self,coll):
        return(['UGID'])
    
    def get_iter(self,coll):
        arch = coll._arch
        for row in arch.spatial:
            yield([row[1]['gid']],row[1]['geom'])
