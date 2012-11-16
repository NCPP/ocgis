import numpy as np
from ocgis.util.helpers import make_poly
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon


def wrap_geoms(geoms,left_max_x_bound):
    clip1 = make_poly((-90,90),(-180,left_max_x_bound))
    clip2 = make_poly((-90,90),(left_max_x_bound,180))
    
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

        ## wrap the polygon if requested
        if adjust:
            ## intersection with the two regions
            left = geom.intersection(clip1)
            right = geom.intersection(clip2)
            
            ## pull out the right side polygons
            right_polygons = [poly for poly in _get_iter_(right)]
            
            ## adjust polygons falling the left window
            if isinstance(left,Polygon):
                left_polygons = [_shift_(left)]
            else:
                left_polygons = []
                for polygon in left:
                    left_polygons.append(_shift_(polygon))
            
            ## merge polygons into single unit
            ret = MultiPolygon(left_polygons + right_polygons)
        
        ## if polygon does not need adjustment, just return it.
        else:
            ret = geom
        return(ret)
    
    ## update the polygons in place
    for geom in geoms:
        geom['geom'] = _transform_(geom['geom'],left_max_x_bound)