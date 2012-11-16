import numpy as np
from ocgis.util.helpers import make_poly
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from ocgis.util.shp_cabinet import ShpCabinet
import time


def wrap_geoms(geoms,left_max_x_bound):
    clip1 = make_poly((-90,90),(-180,left_max_x_bound))
    clip2 = make_poly((-90,90),(left_max_x_bound,180))
#        clip = MultiPolygon([clip1,clip2])
#    lon_cutoff = -1.40625
    
    def _get_iter_(geom):
        try:
            it = iter(geom)
        except TypeError:
            it = [geom]
        return(it)
    
    def _shift_(polygon):
        coords = np.array(polygon.exterior.coords)
        coords[:,0] = coords[:,0] + 360
        return(Polygon(coords))
    
    def _transform_(geom,lon_cutoff):
        ## return the geometry iterator
        it = _get_iter_(geom)
        ## loop through the polygons determining if any coordinates need to be
        ## shifted and flag accordingly.
        adjust = False
        for polygon in it:
            coords = np.array(polygon.exterior.coords)
            if np.any(coords[0,:] < lon_cutoff):
                adjust = True
                break
#            import ipdb;ipdb.set_trace()
#            for coords in polygon.exterior.coords:
#                if any([c < lon_cutoff for c in coords]):
#                    adjust = True
#                    break
        ## wrap the polygon if requested
        if adjust:
            ## intersection with the two regions
            left = geom.intersection(clip1)
            right = geom.intersection(clip2)
            
            ## pull out the right side polygons
            right_polygons = [poly for poly in _get_iter_(right)]
#            import ipdb;ipdb.set_trace()
#            if right.is_empty:
#                right_polygons = []
#            else:
#                if isinstance(right,MultiPolygon):
#                    right_polygons = [poly for poly in right]
#                else:
#                    right_polygons = [right]
            
#                shapely_to_shp(left,'left')
#                shapely_to_shp(right,'right')
#                tdk
#                import ipdb;ipdb.set_trace()
#                sc.write([{'geom':new_geom,'id':1}],'/tmp/spain3.shp')
#                import ipdb;ipdb.set_trace()
            if isinstance(left,Polygon):
                left_polygons = [_shift_(left)]
#                left_polygons = [Polygon([_shift_(ctup) for ctup in left.exterior.coords])]
            else:
                left_polygons = []
                for polygon in left:
#                    new_geom = Polygon(_shift_(polygon))
#                    new_geom = Polygon([_shift_(ctup) for ctup in polygon.exterior.coords])
                    left_polygons.append(_shift_(polygon))
#                if isinstance(right,MultiPolygon):
#                    right_polygons = [poly for poly in right]
#                else:
#                    right_polygons = [right]
#                    left = MultiPolygon(polygons)
            ret = MultiPolygon(left_polygons + right_polygons)
        else:
            ret = geom
        return(ret)
    
    for geom in geoms:
        geom['geom'] = _transform_(geom['geom'],left_max_x_bound)
    
#    sc = ShpCabinet()
#    sc.write(geoms,'/tmp/remapped{0}.shp'.format(time.time()))
#    import ipdb;ipdb.set_trace()