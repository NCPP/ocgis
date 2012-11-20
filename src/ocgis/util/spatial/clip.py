from shapely import prepared
from ocgis.util.helpers import keep, iter_array
import numpy as np


def clip(coll,igeom):
    '''Do an intersects + intersection and set weights based on geometry
    areas.
    
    coll :: OcgCollection
    igeom :: Shapely Polygon or MultiPolygon
    
    returns
    
    OcgCollection'''
    
    ## logic for convenience. just return the provided collection if a NoneType
    ## is passed for the 'igeom' arugment
    if igeom is not None:
        ## take advange of shapely speedups
        prep_igeom = prepared.prep(igeom)
        ## the weight array
        weights = np.empty(coll.gid.shape,dtype=float)
        weights = np.ma.array(weights,mask=coll.gid.mask)
        ## do the spatial operation
        for idx,geom in iter_array(coll.geom_masked,
                                   return_value=True):
            if keep(prep_igeom,igeom,geom):
                new_geom = igeom.intersection(geom)
                weights[idx] = new_geom.area
                coll.geom[idx] = new_geom
        ## set maximum weight to one
        coll.weights = weights/weights.max()
    return(coll)