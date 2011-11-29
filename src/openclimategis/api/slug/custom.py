from generic import *
from util.helpers import parse_polygon_wkt
from shapely import wkt
import pdb
from climatedata.models import UserGeometryData, UserGeometryMetadata


class PolygonSlug(OcgSlug):
    
    def _get_(self):
        try:
            ret = [wkt.loads(parse_polygon_wkt(self.url_arg))]
        except ValueError:
            meta = UserGeometryMetadata.objects.filter(code=self.url_arg)
            assert(len(meta) == 1)
            geoms = UserGeometryData.objects.filter(user_meta=meta)
            assert(len(geoms) > 0)
            ret = [dict(gid=geom.gid,geom=(wkt.loads(geom.geom.wkt))) for geom in geoms]
#            ret = wkt.loads(obj[0].geom.wkt)
        return(ret)
    
    
class OperationSlug(OcgSlug):
    
    def _get_(self):
        if self.url_arg in ('intersect','intersects'):
            ret = 'intersect'
        elif self.url_arg in ('intersection','clip'):
            ret = 'clip'
        else:
            self.exception_()
        return(ret)
