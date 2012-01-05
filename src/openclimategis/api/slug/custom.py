from generic import *
from util.helpers import parse_polygon_wkt
from shapely import wkt
import pdb
from climatedata.models import UserGeometryData, UserGeometryMetadata
from util.ncconv.experimental.ocg_stat import OcgStatFunction
from util.ncconv.experimental.helpers import check_function_dictionary


class PolygonSlug(OcgSlug):
    
    def _get_(self):
        try:
            ret = [dict(geom=(wkt.loads(parse_polygon_wkt(self.url_arg))),gid=None)]
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


class FunctionSlug(OcgSlug):
    
    def _get_(self):
        stat = OcgStatFunction()
        function_list = stat.get_function_list(self.url_arg[0].split(' '))
        check_function_dictionary(function_list)
        return(function_list)
    
    
class GroupingSlug(OcgSlug):
    
    def _get_(self):
        allowed = ['day','month','year']
        groups = self.url_arg[0].split(' ')
        for group in groups:
            if group not in allowed:
                raise(ValueError('"{0}" not allowed as grouping attribute'))
        return(groups)
