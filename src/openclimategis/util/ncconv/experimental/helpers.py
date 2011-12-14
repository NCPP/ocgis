import itertools
import numpy as np
from shapely.geometry.polygon import Polygon
import pdb
from osgeo import osr
from osgeo import ogr
import warnings
import os
from shapely import wkt
from shapely.geometry.multipolygon import MultiPolygon
import time

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]
        
def array_split(ary,n):
    """
    >>> data = [1,2,3,4,5]
    >>> array_split(data,3)
    [[1, 2], [3, 4], [5]]
    """
    step = int(round(len(ary)/float(n)))
    if step == 0:
        step = 1
    ret = []
    idx = 0
    for ii in range(0,n):
        try:
            app = ary[idx:idx+step]
            if len(app) > 0:
                ret.append(app)
        except IndexError:
            ret.append(ary[idx:-1])
        idx = idx + step
    return(ret)

def timing(f):
    def wrapf(*args,**kwds):
        t1 = time.time()
        try:
            return(f(*args,**kwds))
        finally:
            t = time.time() - t1
            print('{0} - {1} secs'.format(f.__name__,t))
    return(wrapf)
  
def itr_array(a):
    "a -- 2-d ndarray"
    ix = a.shape[0]
    jx = a.shape[1]
    for ii,jj in itertools.product(range(ix),range(jx)):
        yield ii,jj
        
def contains(grid,lower,upper,res=0.0):
    
    ## small ranges on coordinates requires snapping to closest coordinate
    ## to ensure values are selected through logical comparison.
    ugrid = np.unique(grid)
    lower = ugrid[np.argmin(np.abs(ugrid-(lower-0.5*res)))]
    upper = ugrid[np.argmin(np.abs(ugrid-(upper+0.5*res)))]
    
    s1 = grid >= lower
    s2 = grid <= upper
    ret = s1*s2

    return(ret)

def approx_resolution(vec):
    """
    >>> vec = [1,2,3,4,5]
    >>> approx_resolution(vec)
    1.0
    """
    diff = []
    for i in range(len(vec)):
        curr = vec[i]
        try:
            next = vec[i+1]
            diff.append(abs(curr-next))
        except IndexError:
            break
    return(np.mean(diff))

def make_poly(rtup,ctup):
    """
    rtup = (row min, row max)
    ctup = (col min, col max)
    """
    return Polygon(((ctup[0],rtup[0]),
                    (ctup[0],rtup[1]),
                    (ctup[1],rtup[1]),
                    (ctup[1],rtup[0])))
    
def sub_range(a):
    """
    >>> vec = np.array([2,5,9])
    >>> sub_range(vec)
    array([2, 3, 4, 5, 6, 7, 8, 9])
    """
    a = np.array(a)
    return(np.arange(a.min(),a.max()+1))

def keep(prep_igeom=None,igeom=None,target=None):
    test_geom = prep_igeom or igeom
    if test_geom.intersects(target) and not target.touches(igeom):
        ret = True
    else:
        ret = False
    return(ret)

def reduce_to_multipolygon(geoms):
    if type(geoms) not in (list,tuple): geoms = [geoms]
    polys = []
    for geom in geoms:
        if isinstance(geom,MultiPolygon):
            for poly in geom:
                polys.append(poly)
        else:
            polys.append(geom)
    return(MultiPolygon(polys))

def union_sum(weight,value,normalize=True):
    ## renormalize the weights
    if normalize: weight = weight/weight.sum()
    ## this will hold the weighted data
    weighted = np.empty((value.shape[0],value.shape[1],1))
    ## next, weight and sum the data accordingly
    for dim_time in range(value.shape[0]):
        for dim_level in range(value.shape[1]):
            weighted[dim_time,dim_level,0] = np.sum(weight*value[dim_time,dim_level,:])
    return(weighted)

def merge_subsets(subsets):
    for ii,sub in enumerate(subsets):
        if ii == 0:
            base = sub.copy()
        else:
            pass
        
def get_sr(srid):
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(srid)
    return(sr)

def get_area(geom,sr_orig,sr_dest):
    geom = ogr.CreateGeometryFromWkb(geom.wkb)
    geom.AssignSpatialReference(sr_orig)
    geom.TransformTo(sr_dest)
    return(geom.GetArea())

def get_area_srid(geom,srid_orig,srid_dest):
    sr = get_sr(srid_orig)
    sr2 = get_sr(srid_dest)
    return(get_area(geom,sr,sr2))

def get_wkt_from_shp(path,objectid,layer_idx=0):
    """
    >>> path = '/home/bkoziol/git/OpenClimateGIS/bin/shp/state_boundaries.shp'
    >>> objectid = 10
    >>> wkt = get_wkt_from_shp(path,objectid)
    >>> assert(wkt.startswith('POLYGON ((-91.730366281818348 43.499571367976877,'))
    """
    ds = ogr.Open(path)
    try:
        lyr = ds.GetLayerByIndex(layer_idx)
        lyr_name = lyr.GetName()
        if objectid is None:
            sql = 'SELECT * FROM {0}'.format(lyr_name)
        else:
            sql = 'SELECT * FROM {0} WHERE ObjectID = {1}'.format(lyr_name,objectid)
        data = ds.ExecuteSQL(sql)
        #import pdb; pdb.set_trace()
        feat = data.GetNextFeature()
        geom = feat.GetGeometryRef()
        wkt = geom.ExportToWkt()
        return(wkt)
    finally:
        ds.Destroy()
        
class ShpIterator(object):
    
    def __init__(self,path):
        assert(os.path.exists(path))
        self.path = path
        
    def iter_features(self,fields,lyridx=0,geom='geom',skiperrors=False):
        ds = ogr.Open(self.path)
        try:
            lyr = ds.GetLayerByIndex(lyridx)
            lyr.ResetReading()
            for feat in lyr:
                ## get the values
                values = []
                for field in fields:
                    try:
                        values.append(feat.GetField(field))
                    except:
                        try:
                            if skiperrors is True:
                                warnings.warn('Error in GetField("{0}")'.format(field))
                            else:
                                raise
                        except ValueError:
                            msg = 'Illegal field requested in GetField("{0}")'.format(field)
                            raise ValueError(msg)
#                values = [feat.GetField(field) for field in fields]
                attrs = dict(zip(fields,values))
                ## get the geometry
                attrs.update({geom:feat.GetGeometryRef().ExportToWkt()})
                yield attrs
        finally:
            ds.Destroy()
            
def get_shp_as_multi(path,uid_field=None):
    """
    >>> path = '/home/bkoziol/git/OpenClimateGIS/bin/shp/state_boundaries.shp'
    >>> uid_field = 'objectid'
    >>> ret = get_shp_as_multi(path,uid_field)
    """
    if uid_field is None or uid_field == '':
        uid_field = []
    else:
        uid_field = [str(uid_field)]
    shpitr = ShpIterator(path)
    data = [feat for feat in shpitr.iter_features(uid_field)]
    ## check the WKT is a polygon and the unique identifier is a unique integer
    uids = []
    for feat in data:
        assert('POLYGON' in feat['geom'])
        if len(uid_field) > 0:
            assert(isinstance(feat[uid_field[0]],int))
            uids.append(feat[uid_field[0]])
    assert(len(uids) == len(set(uids)))
    return(data)
        

if __name__ == '__main__':
    import doctest
    doctest.testmod()