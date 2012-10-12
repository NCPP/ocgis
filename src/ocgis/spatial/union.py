import numpy as np
from shapely.ops import cascaded_union
from shapely.geometry.multipolygon import MultiPolygon


def union(new_id,coll):
    """
    Union the object's geometries.
    
    new_id :: int :: the new geometry's unique identifier
    coll :: OcgCollection
    
    returns
    
    OcgCollection
    """

    ## will hold the unioned geometry
    new_geometry = np.empty((1,1),dtype=object)
    ## break out the MultiPolygon objects. inextricable geometry errors
    ## sometimes occur otherwise
    ugeom = []
    for geom in coll.geom_masked.compressed():
        if isinstance(geom,MultiPolygon):
            for poly in geom:
                ugeom.append(poly)
        else:
            ugeom.append(geom)
    ## execute the union
    new_geometry[0,0] = cascaded_union(ugeom)
    ## update the collection. mask for masked object arrays are kept separate
    ## in case the data needs to be pickled. know bug in numpy
    coll.geom = new_geometry
    coll.geom_mask = np.array([[False]])
    coll.gid = np.ma.array([[new_id]],mask=[[False]],dtype=int)
    ## aggregate the values and store in the aggregated attribute of the
    ## collection
    for ocg_variable in coll.variables.itervalues():
        ocg_variable.agg_value = union_sum(coll.weights,
                                           ocg_variable.raw_value)
    return(coll)

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