import numpy as np
from shapely.ops import cascaded_union
from shapely.geometry.multipolygon import MultiPolygon
from copy import deepcopy
from shapely.geometry.point import Point
from shapely.geometry.multipoint import MultiPoint
from ocgis.api.dataset.collection.collection import Identifier


def union(coll):
    """
    Union the object's geometries.
    
    new_id :: int :: the new geometry's unique identifier
    coll :: OcgCollection
    
    returns
    
    OcgCollection
    """
    
    ## reset the geometry identifier
    coll.gid = Identifier(dtype=object)
    
    for ocg_variable in coll.variables.itervalues():
        ## will hold the unioned geometry
        new_geometry = np.empty((1,1),dtype=object)
        ## get the masked geometries
        geoms = ocg_variable.spatial.get_masked().compressed()
        if coll.geomtype == 'point':
            pts = MultiPoint([pt for pt in geoms.flat])
            new_geometry[0,0] = Point(pts.centroid.x,pts.centroid.y)
        else:
            ## break out the MultiPolygon objects. inextricable geometry errors
            ## sometimes occur otherwise
            ugeom = []
            for geom in geoms:
                if isinstance(geom,MultiPolygon):
                    for poly in geom:
                        ugeom.append(poly)
                else:
                    ugeom.append(geom)
            ## execute the union
            new_geometry[0,0] = cascaded_union(ugeom)
        ## overwrite the original geometry
        ocg_variable.spatial.value = new_geometry
        ocg_variable.spatial.value_mask = np.array([[False]])
        coll.gid.add(new_geometry[0,0].wkb)
        ## aggregate the values
        ocg_variable.raw_value = ocg_variable.value.copy()
        ocg_variable.value = union_sum(ocg_variable.spatial.weights,ocg_variable.raw_value)
#    import ipdb;ipdb.set_trace()
        
#    ## update the collection. mask for masked object arrays are kept separate
#    ## in case the data needs to be pickled. know bug in numpy
#    import ipdb;ipdb.set_trace()
#    coll.geom = new_geometry
#    coll.geom_mask = np.array([[False]])
#    coll.gid = np.ma.array([[new_id]],mask=[[False]],dtype=int)
#    ## aggregate the values and store in the aggregated attribute of the
#    ## collection
#    for ocg_variable in coll.variables.itervalues():
#        ocg_variable.agg_value = union_sum(coll.weights,
#                                           ocg_variable.raw_value)
#    return(coll)

def union_sum(weight,value):
    ## TODO: this should be replaced with actual aggregation by built-in numpy
    ## functions
    '''Weighted sum for the geometry.
    
    weight :: nd.array
    value :: nd.array
    
    returns
    
    nd.array
    '''
    
    ## make the output array
    wshape = (value.shape[0],value.shape[1],1,1)
    weighted = np.ma.array(np.empty(wshape,dtype=float),
                            mask=np.zeros(wshape,dtype=bool))
    ## next, weight and sum the data accordingly
    for dim_time in range(value.shape[0]):
        for dim_level in range(value.shape[1]):
            weighted[dim_time,dim_level,0,0] = np.ma.average(value[dim_time,dim_level,:,:],weights=weight)
    return(weighted)

def union_geoms(geoms):
    if len(geoms) == 1:
        ret = deepcopy(geoms)
    else:
        ugeom = []
        for dct in geoms:
            geom = dct['geom']
            if isinstance(geom,MultiPolygon):
                for poly in geom:
                    ugeom.append(poly)
            else:
                ugeom.append(geom)
        ugeom = cascaded_union(ugeom)
        ret = [{'id':1,'geom':ugeom}]
    return(ret)