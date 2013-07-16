from shapely.geometry.polygon import Polygon
import numpy as np
#from shapely.prepared import prep


def make_shapely_grid(poly,res,as_numpy=False):
    """
    Return a list or NumPy matrix of shapely Polygon objects.
    
    poly -- shapely Polygon to discretize
    res -- target grid resolution in the same units as |poly|
    as_numpy=False -- set to True to return a 2-d object matrix of shapely
        polygons
    
    >>> poly = Polygon(((-90,30),(-70,30),(-70,50),(-90,50)))
    >>> res = 1
    >>> grid = make_shapely_grid(poly,res,as_numpy=False)
    >>> assert(len(grid) == 441)
    >>> grid = make_shapely_grid(poly,res,as_numpy=True)
    >>> assert(grid.shape == (21,21))
    >>> from shapely import wkt
    >>> poly = wkt.loads('POLYGON ((-85.324076923076916 44.028020242914977,-84.280765182186229 44.16008502024291,-84.003429149797569 43.301663967611333,-83.607234817813762 42.91867611336032,-84.227939271255053 42.060255060728736,-84.941089068825903 41.307485829959511,-85.931574898785414 41.624441295546553,-85.588206477732783 43.011121457489871,-85.324076923076916 44.028020242914977))')
    >>> res = 0.3
    >>> grid = make_shapely_grid(poly,res,as_numpy=True)
    >>> assert(grid.shape == (11,9))
    """
    
    ## ensure we have a floating point resolution
    res = float(res)
    ## check that the target polygon is a valid geometry
    assert(poly.is_valid)
    ## vectorize the polygon creation
    vfunc_poly = np.vectorize(make_poly_array)#,otypes=[np.object])
    ## prepare the geometry for faster spatial relationship checking. throws a
    ## a warning so leaving out for now.
#    prepped = prep(poly)
    
    ## extract bounding coordinates of the polygon
    min_x,min_y,max_x,max_y = poly.envelope.bounds
    ## convert to matrices
    X,Y = np.meshgrid(np.arange(min_x,max_x+res,res),
                      np.arange(min_y,max_y+res,res))
    ## shift by the resolution
    pmin_x = X - res
    pmax_x = X + res
    pmin_y = Y - res
    pmax_y = Y + res
    ## make the 2-d array
    poly_array = vfunc_poly(pmin_y,pmin_x,pmax_y,pmax_x,poly)
    ## format according to configuration arguments
    if as_numpy:
        ret = poly_array
    else:
        ret = list(poly_array.reshape(-1))
    
    return(ret)

    
def make_poly_array(min_row,min_col,max_row,max_col,polyint=None):
    ret = Polygon(((min_col,min_row),
                    (max_col,min_row),
                    (max_col,max_row),
                    (min_col,max_row),
                    (min_col,min_row)))
    if polyint is not None:
        if polyint.intersects(ret) == False:
            ret = None
    return(ret)


if __name__ == "__main__":
    import doctest
    doctest.testmod()