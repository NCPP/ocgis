import itertools
import numpy as np
from shapely.geometry.polygon import Polygon
from osgeo import osr
from osgeo import ogr
import warnings
import os
from shapely import wkt
from shapely.geometry.multipolygon import MultiPolygon
import time
from collections import namedtuple
from util.ncconv.experimental.exc import FunctionNameError,\
    FunctionNotNamedError
import re
from util.helpers import get_temp_path
import datetime
import django.contrib.gis.geos.polygon as geos
import copy


def merge_dict_list(dl):
    for ii,dd in enumerate(dl):
        if ii == 0:
            arch = copy.copy(dd)
        else:
            for key,value in dd.iteritems():
                arch[key] += value
    return(arch)
        

def get_django_attrs(obj):
    from django.contrib.gis.db import models

    attrs = {}
    fields = [f.name for f in obj._meta.fields]
    for field in fields:
        attr = getattr(obj,field)
        if type(attr) in [datetime.date,datetime.datetime]:
            attr = str(attr)
        if isinstance(attr,geos.Polygon):
            attr = attr.wkt
        if isinstance(attr,models.Model):
            attrs.update({field:get_django_attrs(attr)})
        else:
            attrs.update({field:attr})
    return(attrs)

def itersubclasses(cls, _seen=None):
    """
    itersubclasses(cls)

    Generator over all subclasses of a given class, in depth first order.

    >>> list(itersubclasses(int)) == [bool]
    True
    >>> class A(object): pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>> 
    >>> for cls in itersubclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL (new-style) classes currently defined
    >>> [cls.__name__ for cls in itersubclasses(object)] #doctest: +ELLIPSIS
    ['type', ...'tuple', ...]
    """
    
    if not isinstance(cls, type):
        raise TypeError('itersubclasses must be called with '
                        'new-style classes, not %.100r' % cls)
    if _seen is None: _seen = set()
    try:
        subs = cls.__subclasses__()
    except TypeError: # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub

def user_geom_to_db(user_meta_id,to_disk=False):
    """
    >>> user_meta_id = 1
    >>> db = user_geom_to_db(user_meta_id)
    """
    from climatedata.models import UserGeometryMetadata
    from django.conf import settings
    
    ## get the user metadata and geometries
    user_meta = UserGeometryMetadata.objects.filter(pk=user_meta_id)
    geoms = user_meta[0].usergeometrydata_set.all()
    ## initialize the database
    db = init_db(to_disk=to_disk,procs=settings.MAXPROCESSES)
    ## load the geometries
    s = db.Session()
    try:
        for geom in geoms:
            obj = db.Geometry(wkt=geom.geom.wkt,gid=geom.gid)
            s.add(obj)
        s.commit()
    finally:
        s.close()
    
    return(db)

def init_db(engine=None,to_disk=False,procs=1):
    from sqlalchemy import create_engine
    from sqlalchemy.orm.session import sessionmaker
    from util.ncconv.experimental import db
    from sqlalchemy.pool import NullPool

    if engine is None:
        path = 'sqlite://'
        if to_disk or procs > 1:
            path = path + '/' + get_temp_path('.sqlite',nest=True)
            db.engine = create_engine(path,
                                      poolclass=NullPool)
        else:
            db.engine = create_engine(path,
#                                      connect_args={'check_same_thread':False},
#                                      poolclass=StaticPool
                                      )
    else:
        db.engine = engine
    
    db.metadata.bind = db.engine
    db.Session = sessionmaker(bind=db.engine)
    db.metadata.create_all()
    
    return(db)


def check_function_dictionary(funcs):
    """
    Perform common checks on a list of function definition dictionaries.
    """
    
    for f in funcs:
        if 'name' in f:
            if len(f['name']) >= 11:
                raise(FunctionNameError(f))
            if f['name'][0] in '0123456789':
                raise(FunctionNameError(f))
            if re.search('\W',f['name']) is not None:
                raise(FunctionNameError(f))
        else:
            ## function with parameters must have a name
            keys = ['args','kwds']
            lens = []
            for key in keys:
                val = f.get(key)
                if val is not None:
                    if len(val) > 0:
                        lens.append(True)
                else:
                    lens.append(False)
            if any(lens):
                raise(FunctionNotNamedError(f))


def bounding_coords(polygon):
    min_x,min_y,max_x,max_y = polygon.bounds
    Bounds = namedtuple('Bounds',['min_x','min_y','max_x','max_y'])
    return(Bounds(min_x=min_x,
                  max_x=max_x,
                  min_y=min_y,
                  max_y=max_y))

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
    >>> grps = array_split(range(0,1121),8)
    >>> total_len = 0
    >>> for grp in grps: total_len += len(grp)
    >>> assert(total_len == len(range(0,1121)))
    """
    step = int(round(len(ary)/float(n)))
    if step == 0:
        step = 1
    ret = []
    idx = 0
    for ii in range(0,n):
        if ii == n-1:
            app = ary[idx:]
        else:
            app = ary[idx:idx+step]
        if len(app) > 0:
            ret.append(app)
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
    assert(len(a.shape) == 2)
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