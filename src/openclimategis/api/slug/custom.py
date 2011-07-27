from generic import *
from django.contrib.gis.geos.geometry import GEOSGeometry
from util.helpers import parse_polygon_wkt


class PolygonSlug(OcgSlug):
    
    def _get_(self):
        return(GEOSGeometry(parse_polygon_wkt(self.url_arg)))
    
    
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