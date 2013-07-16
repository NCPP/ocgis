import numpy as np
from shapely.geometry.polygon import Polygon
import os
from osgeo import osr, ogr


def make_shapely_grid(poly,res,as_numpy=False,clip=True):
    """
    Return a list or NumPy matrix of shapely Polygon objects.
    
    poly -- shapely Polygon to discretize
    res -- target grid resolution in the same units as |poly|
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
    X,Y = np.meshgrid(np.arange(min_x,max_x,res),
                      np.arange(min_y,max_y,res))
    #print X,Y

    ## shift by the resolution
    pmin_x = X
    pmax_x = X + res
    pmin_y = Y
    pmax_y = Y + res
    ## make the 2-d array
#    print pmin_y,pmin_x,pmax_y,pmax_x,poly.wkt
    if clip:
        poly_array = vfunc_poly(pmin_y,pmin_x,pmax_y,pmax_x,poly)
    else:
        poly_array = vfunc_poly(pmin_y,pmin_x,pmax_y,pmax_x)
    #print poly_array
    #sys.exit()
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
        else:
            ret = polyint.intersection(ret)
    return(ret)
        
def shapely_to_shp(obj,outname):
    path = os.path.join('/tmp',outname+'.shp')
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ogr_geom = 3
    
    dr = ogr.GetDriverByName('ESRI Shapefile')
    ds = dr.CreateDataSource(path)
    try:
        if ds is None:
            raise IOError('Could not create file on disk. Does it already exist?')
            
        layer = ds.CreateLayer('lyr',srs=srs,geom_type=ogr_geom)
        feature_def = layer.GetLayerDefn()
        feat = ogr.Feature(feature_def)
        feat.SetGeometry(ogr.CreateGeometryFromWkt(obj.wkt))
        layer.CreateFeature(feat)
    finally:
        ds.Destroy()