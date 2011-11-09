from generic import *
from util.helpers import parse_polygon_wkt
from shapely import wkt
import pdb
from climatedata.models import UserGeometryData


class PolygonSlug(OcgSlug):
    
    def _get_(self):
        try:
            ret = wkt.loads(parse_polygon_wkt(self.url_arg))
        except ValueError:
            pk = int(self.url_arg)
            obj = UserGeometryData.objects.filter(pk=pk)
            ret = wkt.loads(obj[0].geom.wkt)
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


class AggregateSlug(BooleanSlug):
    """
    >>> agg = AggregateSlug('aggregate',url_arg='T')
    >>> print(agg)
    aggregate=True
    """
    pass