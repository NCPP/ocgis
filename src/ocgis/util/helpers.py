import numpy as np
import itertools
from shapely.geometry.polygon import Polygon
from collections import namedtuple
import os
import tempfile
import warnings
from osgeo import ogr, osr
from shapely import wkt, wkb
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import cascaded_union
import re
from ocgis.exc import DefinitionValidationError
import sys
import datetime
from copy import deepcopy
from ocgis.util.logging_ocgis import ocgis_lh
from osgeo.ogr import CreateGeometryFromWkb
from shapely.wkb import loads as wkb_loads
import fiona
from shapely.geometry.geo import mapping


def get_is_date_between(lower,upper,month=None,year=None):
    if month is not None:
        attr = 'month'
        to_test = month
    else:
        attr = 'year'
        to_test = year
        
    part_lower,part_upper = getattr(lower,attr),getattr(upper,attr)
    if part_lower != part_upper:
        ret = np.logical_and(to_test >= part_lower,to_test < part_upper)
    else:
        ret = np.logical_and(to_test >= part_lower,to_test <= part_upper)
    return(ret)


class FionaMaker(object):
            
            def __init__(self,path,epsg=4326,driver='ESRI Shapefile',geometry='Polygon'):
                assert(not os.path.exists(path))
                self.path = path
                self.crs = fiona.crs.from_epsg(epsg)
                self.properties = {'UGID':'int','NAME':'str'}
                self.geometry = geometry
                self.driver = driver
                self.schema = {'geometry':self.geometry,
                               'properties':self.properties}
                                
            def __enter__(self):
                self._ugid = 1
                self._collection = fiona.open(self.path,'w',driver=self.driver,schema=self.schema,crs=self.crs)
                return(self)
            
            def __exit__(self,*args,**kwds):
                self._collection.close()
                
            def make_record(self,dct):
                properties = dct.copy()
                geom = wkt.loads(properties.pop('wkt'))
                properties.update({'UGID':self._ugid})
                self._ugid += 1
                record = {'geometry':mapping(geom),
                          'properties':properties}
                return(record)
            
            def write(self,sequence_or_dct):
                if isinstance(sequence_or_dct,dict):
                    itr = [sequence_or_dct]
                else:
                    itr = sequence_or_dct
                for element in itr:
                    record = self.make_record(element)
                    self._collection.write(record)
    

def project_shapely_geometry(geom,from_sr,to_sr):
    if from_sr.IsSame(to_sr) == 1:
        ret = geom
    else:
        ogr_geom = CreateGeometryFromWkb(geom.wkb)
        ogr_geom.AssignSpatialReference(from_sr)
        ogr_geom.TransformTo(to_sr)
        ret = wkb_loads(ogr_geom.ExportToWkb())
    return(ret)

def assert_raise(test,**kwds):
    try:
        assert(test)
    except AssertionError:
        ocgis_lh(**kwds)
        
def get_iter(element,dtype=None):
    if dtype is not None:
        if isinstance(element,dtype):
            element = [element]
            
    if isinstance(element,basestring) or isinstance(element,np.ndarray):
        it = [element]
    else:
        try:
            it = iter(element)
        except TypeError:
            it = [element]
            
    return(it)

def get_default_or_apply(target,f,default=None):
    if target is None:
        ret = default
    else:
        ret = f(target)
    return(ret)

def get_none_or_1d(target):
    if target is None:
        ret = None
    else:
        ret = np.atleast_1d(target)
    return(ret)

def get_none_or_2d(target):
    if target is None:
        ret = None
    else:
        ret = np.atleast_2d(target)
    return(ret)

def get_none_or_slice(target,slc):
    if target is None:
        ret = None
    else:
        ret = target[slc]
    return(ret)

def get_reduced_slice(arr):
    arr_min,arr_max = arr.min(),arr.max()
    assert(arr_max-arr_min+1 == arr.shape[0])
    ret = slice(arr_min,arr_max+1)
    return(ret)

def get_formatted_slice(slc,n_dims):
    
    def _format_(slc):
        if isinstance(slc,int):
            ret = slice(slc,slc+1)
        elif isinstance(slc,slice):
            ret = slc
        elif isinstance(slc,np.ndarray):
            ret = slc
        else:
            if len(slc) == 1:
                ret = slice(slc[0])
            elif len(slc) > 1:
                ret = np.array(slc)
            else:
                raise(NotImplementedError(slc,n_dims))
        return(ret)
    
    if isinstance(slc,slice) and slc == slice(None):
        if n_dims == 1:
            ret = slc
        else:
            ret = [slice(None)]*n_dims
    elif n_dims == 1:
        ret = _format_(slc)
    elif n_dims > 1:
        try:
            assert(len(slc) == n_dims)
        except (TypeError,AssertionError):
            raise(IndexError("Only {0}-d slicing allowed.".format(n_dims)))
        ret = map(_format_,slc)
    else:
        raise(NotImplementedError((slc,n_dims)))
    
    return(ret)

def iter_arg(arg):
    if isinstance(arg,basestring):
        itr = [arg]
    else:
        try:
            itr = iter(arg)
        except TypeError:
            itr = iter([arg])
    for element in itr:
        yield(element)

def get_date_list(start,stop,days):
    ret = []
    delta = datetime.timedelta(days=days)
    check = start
    while check <= stop:
        ret.append(check)
        check += delta
    return(ret)

def bbox_poly(minx,miny,maxx, maxy):
    rtup = (miny,maxy)
    ctup = (minx,maxx)
    return(make_poly(rtup,ctup))

def validate_time_subset(time_range,time_region):
    '''
    Ensure `time_range` and `time_region` overlap. If one of the values is `None`, the
    function always returns `True`. Function will return `False` if the two time range
    descriptions do not overlap.
    
    :param time_range: Sequence with two datetime elements.
    :type time_range: sequence
    :param time_region: Dictionary with two keys 'month' and 'year' each containing
    an integer sequence corresponding to the respective time parts. For example:
    >>> time_region = {'month':[1,2],'year':[2013]}
    If a 'month' or 'year' key is missing, the key will be added with a default of `None`.
    :type time_region: dict
    :rtype: bool
    '''
    
    def _between_(target,lower,upper):
        if target >= lower and target <= upper:
            ret = True
        else:
            ret = False
        return(ret)
    
    def _check_months_(targets,months):
        check = [target in months for target in targets]
        if all(check):
            ret = True
        else:
            ret = False
        return(ret)
    
    def _check_years_(targets,min_range_year,max_range_year):
        if all([_between_(year_bound,min_range_year,max_range_year) for year_bound in targets]):
            ret = True
        else:
            ret = False
        return(ret)
    
    ## by default we return that it does not validate
    ret = False
    ## if any of the parameters are none, then it will validate True
    if any([t is None for t in [time_range,time_region]]):
        ret = True
    else:
        ## ensure time region has the necessary keys
        copy_time_region = deepcopy(time_region)
        for key in ['month','year']:
            if key not in copy_time_region:
                copy_time_region[key] = None
        ## pull basic date information from the time range
        min_range_year,max_range_year = time_range[0].year,time_range[1].year
        delta = datetime.timedelta(days=29,hours=12)
        months = set()
        current = time_range[0]
        while current <= time_range[1]:
            current += delta
            months.update([current.month])
            if len(months) == 12:
                break
        ## construct boundaries from time region. first, the case of only months.
        if copy_time_region['month'] is not None and copy_time_region['year'] is None:
            month_bounds = min(copy_time_region['month']),max(copy_time_region['month'])
            ret = _check_months_(month_bounds,months)
        ## case of only years
        elif copy_time_region['month'] is None and copy_time_region['year'] is not None:
            year_bounds = min(copy_time_region['year']),max(copy_time_region['year'])
            ret = _check_years_(year_bounds,min_range_year,max_range_year)
        ## case with both years and months
        else:
            month_bounds = min(copy_time_region['month']),max(copy_time_region['month'])
            year_bounds = min(copy_time_region['year']),max(copy_time_region['year'])
            ret_months = _check_months_(month_bounds,months)
            ret_years = _check_years_(year_bounds,min_range_year,max_range_year)
            if all([ret_months,ret_years]):
                ret = True
    return(ret)


def format_bool(value):
    '''Format a string to boolean.
    
    :param value: The value to convert.
    :type value: int or str'''
    
    try:
        ret = bool(int(value))
    except ValueError:
        value = value.lower()
        if value in ['t','true']:
            ret = True
        elif value in ['f','false']:
            ret = False
        else:
            raise(ValueError('String not recognized for boolean conversion: {0}'.format(value)))
    return(ret)

class ProgressBar(object):
    
    def __init__(self,title):
        sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
        sys.stdout.flush()
        self.px = 0
#        globals()["progress_x"] = 0
    
#    def startProgress(title):
#        sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
#        sys.stdout.flush()
#        globals()["progress_x"] = 0

    def progress(self,x):
        x = x*40//100
        sys.stdout.write("#"*(x - self.px))
        sys.stdout.flush()
        self.px = x
#        globals()["progress_x"] = x
    
    def endProgress(self):
        sys.stdout.write("#"*(40 - self.px))
        sys.stdout.write("]\n")
        sys.stdout.flush()

def locate(pattern, root=os.curdir, followlinks=True):
    '''Locate all files matching supplied filename pattern in and below
    supplied root directory.'''
    for path, dirs, files in os.walk(os.path.abspath(root),followlinks=followlinks):
        for filename in filter(lambda x: x == pattern,files):
            yield os.path.join(path, filename)

#def reduce_query(query):
#    ## parse keys into groups.
#    groups = {}
#    ungrouped = {}
#    exp_key = re.compile('^(\D+)\d+$')
#    exp_number = re.compile('^\D+(\d+)$')
#    for key,value in query.iteritems():
#        try:
#            m_key = re.match(exp_key,key).groups()[0]
#            m_number = int(re.match(exp_number,key).groups()[0])
#            if m_key not in groups:
#                groups[m_key] = []
#            groups[m_key].append(m_number)
#        except AttributeError:
#            ungrouped[key] = value
#    ## sort the groups
#    for value in groups.itervalues():
#        value.sort()
#    ## ensure the groups are the same. only applicable if there are any
#    ## grouped variables.
#    if len(groups) > 0:
#        arch = groups['uri']
#        for key,value in groups.iteritems():
#            try:
#                assert(arch == value)
#            except AssertionError:
#                if key in ['uri','variable']:
#                    raise(DefinitionValidationError('reduce_query','Integer group indicators are not consistent.'))
#                else:
#                    fill = [None]*len(arch)
#                    for integer in value:
#                        idx = arch.index(integer)
#                        fill[idx] = integer
#                    groups[key] = fill
#    ## replace integers with actual values
#    for key,value in groups.iteritems():
#        for idx in range(len(value)):
#            if value[idx] is None:
#                continue
#            pull_key = key + str(value[idx])
#            value[idx] = query[pull_key][0]
#        groups[key] = [value]
#    ## merge the grouped and ungrouped parameters
#    groups.update(ungrouped)
#    return(groups)
#        
#def union_geoms(ugeoms,new_id=1):
#    if len(ugeoms) == 1:
#        ret = deepcopy(ugeoms)
#    else:
#        to_union = []
#        for dct in ugeoms:
#            geom = dct['geom']
#            if isinstance(geom,MultiPolygon):
#                for poly in geom:
#                    to_union.append(poly)
#            else:
#                to_union.append(geom)
#        ugeom = cascaded_union(to_union)
#        ret = [{'ugid':new_id,'geom':ugeom}]
#    return(ret)
#
#
#def get_bounded(value,bounds=None,uid=None,names={'uid':'uid','value':'value'}):
#    if uid is None:
#        uid = np.arange(1,value.shape[0]+1,dtype=int)
#    ret = np.empty(value.shape[0],dtype=[(names['uid'],int),(names['value'],value.dtype,3)])
#    ret[names['uid']] = uid
#    ref = ret[names['value']]
#    ref[:,1] = value
#    if bounds is None:
#        ref[:,0] = ref[:,1]
#        ref[:,2] = ref[:,1]
#    else:
#        ref[:,0] = bounds[:,0]
#        ref[:,2] = bounds[:,1]
#    return(ret)

#def append(arr,value):
#    arr.resize(arr.shape[0]+1,refcheck=False)
#    arr[arr.shape[0]-1] = value

def iter_array(arr,use_mask=True,return_value=False):
    try:
        shp = arr.shape
    ## assume array is not a numpy array
    except AttributeError:
        arr = np.array(arr,ndmin=1)
        shp = arr.shape
    iter_args = [range(0,ii) for ii in shp]
    if use_mask and not np.ma.isMaskedArray(arr):
        use_mask = False
    else:
        try:
            mask = arr.mask
        ## array is not masked
        except AttributeError:
            pass
        
    for ii in itertools.product(*iter_args):
        if use_mask:
            try:
                if mask[ii]:
                    continue
                else:
                    idx = ii
            ## occurs with singleton dimension of masked array
            except IndexError:
                if mask:
                    continue
                else:
                    idx = ii
        else:
            idx = ii
        if return_value:
            ret = (idx,arr[ii])
        else:
            ret = idx
        yield(ret)

#def geom_to_mask(coll):
#    coll['geom'] = np.ma.array(coll['geom'],mask=coll['geom_mask'])
#    return(coll)
#
#def mask_to_geom(coll):
#    coll['geom'] = np.array(coll['geom'])
#    return(coll)
    
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
            nxt = vec[i+1]
            diff.append(abs(curr-nxt))
        except IndexError:
            break
    return(np.mean(diff))

def keep(prep_igeom=None,igeom=None,target=None):
    test_geom = prep_igeom or igeom
    if test_geom.intersects(target) and not target.touches(igeom):
        ret = True
    else:
        ret = False
    return(ret)

def prep_keep(prep_igeom,igeom,target):
    if prep_igeom.intersects(target) and not target.touches(igeom):
        ret = True
    else:
        ret = False
    return(ret)

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

#def itr_array(a):
#    "a -- 2-d ndarray"
#    assert(len(a.shape) == 2)
#    ix = a.shape[0]
#    jx = a.shape[1]
#    for ii,jj in itertools.product(range(ix),range(jx)):
#        yield ii,jj
        
def make_poly(rtup,ctup):
    """
    rtup = (row min, row max)
    ctup = (col min, col max)
    """
    
    return Polygon(((ctup[0],rtup[0]),
                    (ctup[0],rtup[1]),
                    (ctup[1],rtup[1]),
                    (ctup[1],rtup[0])))
    
#def get_sub_range(a):
#    """
#    >>> vec = np.array([2,5,9])
#    >>> sub_range(vec)
#    array([2, 3, 4, 5, 6, 7, 8, 9])
#    """
#    a = np.array(a)
##    ## for the special case of the array with one element
##    if len(a) == 1:
##        ret = np.arange(a[0],a[0]+1)
##    else:
#    ret = np.arange(a.min(),a.max()+1)
#    return(ret)
#
#def bounding_coords(polygon):
#    min_x,min_y,max_x,max_y = polygon.bounds
#    Bounds = namedtuple('Bounds',['min_x','min_y','max_x','max_y'])
#    return(Bounds(min_x=min_x,
#                  max_x=max_x,
#                  min_y=min_y,
#                  max_y=max_y))
#    
#def shapely_to_shp(obj,path,srs=None):
#    from osgeo import osr, ogr
#    
##    path = os.path.join('/tmp',outname+'.shp')
#    if srs is None:
#        srs = osr.SpatialReference()
#        srs.ImportFromEPSG(4326)
#        
#    if isinstance(obj,MultiPoint):
#        test = ogr.CreateGeometryFromWkb(obj[0].wkb)
#        ogr_geom = test.GetGeometryType()
#    else:
#        ogr_geom = 3
#    
#    dr = ogr.GetDriverByName('ESRI Shapefile')
#    ds = dr.CreateDataSource(path)
#    try:
#        if ds is None:
#            raise IOError('Could not create file on disk. Does it already exist?')
#            
#        layer = ds.CreateLayer('lyr',srs=srs,geom_type=ogr_geom)
#        try:
#            feature_def = layer.GetLayerDefn()
#        except:
#            import ipdb;ipdb.set_trace()
#        feat = ogr.Feature(feature_def)
#        try:
#            iterator = iter(obj)
#        except TypeError:
#            iterator = iter([obj])
#        for geom in iterator:
#            feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
#            layer.CreateFeature(feat)
#    finally:
#        ds.Destroy()

def get_temp_path(suffix='',name=None,nest=False,only_dir=False,wd=None,dir_prefix=None):
    """Return absolute path to a temporary file."""

    if dir_prefix is not None:
        if not dir_prefix.endswith('_'):
            dir_prefix = dir_prefix + '_'
    else:
        dir_prefix = ''

    def _get_wd_():
        if wd is None:
            return(tempfile.gettempdir())
        else:
            return(wd)

    if nest:
        f = tempfile.NamedTemporaryFile()
        f.close()
        dir = os.path.join(_get_wd_(),dir_prefix+os.path.split(f.name)[-1])
        os.mkdir(dir)
    else:
        dir = _get_wd_()
    if only_dir:
        ret = dir
    else:
        if name is not None:
            ret = os.path.join(dir,name+suffix)
        else:
            f = tempfile.NamedTemporaryFile(suffix=suffix,dir=dir)
            f.close()
            ret = f.name
    return(str(ret))

#def get_wkt_from_shp(path,objectid,layer_idx=0):
#    """
#    >>> path = '/home/bkoziol/git/OpenClimateGIS/bin/shp/state_boundaries.shp'
#    >>> objectid = 10
#    >>> wkt = get_wkt_from_shp(path,objectid)
#    >>> assert(wkt.startswith('POLYGON ((-91.730366281818348 43.499571367976877,'))
#    """
#    ds = ogr.Open(path)
#    try:
#        lyr = ds.GetLayerByIndex(layer_idx)
#        lyr_name = lyr.GetName()
#        if objectid is None:
#            sql = 'SELECT * FROM {0}'.format(lyr_name)
#        else:
#            sql = 'SELECT * FROM {0} WHERE ObjectID = {1}'.format(lyr_name,objectid)
#        data = ds.ExecuteSQL(sql)
#        #import pdb; pdb.set_trace()
#        feat = data.GetNextFeature()
#        geom = feat.GetGeometryRef()
#        wkt = geom.ExportToWkt()
#        return(wkt)
#    finally:
#        ds.Destroy()
#
#   
#class ShpIterator(object):
#    
#    def __init__(self,path):
#        assert(os.path.exists(path))
#        self.path = path
#        
#    def get_fields(self):
#        ds = ogr.Open(self.path)
#        try:
#            lyr = ds.GetLayerByIndex(0)
#            lyr.ResetReading()
#            feat = lyr.GetNextFeature()
#            return(feat.keys())
#        finally:
#            ds.Destroy()
#        
#    def iter_features(self,fields,lyridx=0,geom='geom',skiperrors=False,
#                      to_shapely=False):
#        
#        ds = ogr.Open(self.path)
#        try:
#            lyr = ds.GetLayerByIndex(lyridx)
#            lyr.ResetReading()
#            for feat in lyr:
#                ## get the values
#                values = []
#                for field in fields:
#                    try:
#                        values.append(feat.GetField(field))
#                    except:
#                        try:
#                            if skiperrors is True:
#                                warnings.warn('Error in GetField("{0}")'.format(field))
#                            else:
#                                raise
#                        except ValueError:
#                            msg = 'Illegal field requested in GetField("{0}")'.format(field)
#                            raise ValueError(msg)
##                values = [feat.GetField(field) for field in fields]
#                attrs = dict(zip(fields,values))
#                ## get the geometry
#                
#                wkt_str = feat.GetGeometryRef().ExportToWkt()
##                geom_obj = feat.GetGeometryRef()
##                geom_obj.TransformTo(to_sr)
##                wkt_str = geom_obj.ExportToWkt()
#                
#                if to_shapely:
#                    ## additional load to clean geometries
#                    geom_data = wkt.loads(wkt_str)
#                    geom_data = wkb.loads(geom_data.wkb)
#                else:
#                    geom_data = wkt_str
#                attrs.update({geom:geom_data})
#                yield attrs
#        finally:
#            ds.Destroy()
#
#
#def get_shp_as_multi(path,uid_field=None,attr_fields=[],make_id=False,id_name='ugid'):
#    """
#    >>> path = '/home/bkoziol/git/OpenClimateGIS/bin/shp/state_boundaries.shp'
#    >>> uid_field = 'objectid'
#    >>> ret = get_shp_as_multi(path,uid_field)
#    """
#    ## the iterator object instantiated here to make sure the shapefile exists
#    ## and there is access to the field acquisition.
#    shpitr = ShpIterator(path)
#    
#    if uid_field is None or uid_field == '':
#        uid_field = []
#    else:
#        uid_field = [str(uid_field)]
#    try:
#        fields = uid_field + attr_fields
#    except TypeError:
#        if attr_fields.lower() == 'all':
#            fields = shpitr.get_fields()
#            fields = [f.lower() for f in fields]
#            try:
#                if uid_field[0].lower() in fields:
#                    fields.pop(uid_field[0].lower())
#            except IndexError:
#                if len(uid_field) == 0:
#                    pass
#                else:
#                    raise
#            fields = uid_field + fields
#        else:
#            raise
#    data = [feat for feat in shpitr.iter_features(fields,to_shapely=True)]
#    ## add unique identifier if requested and the passed uid field is none
#    for ii,gd in enumerate(data,start=1):
#        if len(uid_field) == 0 and make_id is True:
#            gd[id_name] = ii
#        else:
#            geom_id = gd.pop(uid_field[0])
#            gd[id_name] = int(geom_id)
#    
#    ## check the WKT is a polygon and the unique identifier is a unique integer
#    uids = []
#    for feat in data:
#        if len(uid_field) > 0:
#            feat[uid_field[0]] = int(feat[uid_field[0]])
#            uids.append(feat[uid_field[0]])
#    assert(len(uids) == len(set(uids)))
#    return(data)
#
#def get_sr(srid):
#    sr = osr.SpatialReference()
#    sr.ImportFromEPSG(srid)
#    return(sr)

def get_area(geom,sr_orig,sr_dest):
    geom = ogr.CreateGeometryFromWkb(geom.wkb)
    geom.AssignSpatialReference(sr_orig)
    geom.TransformTo(sr_dest)
    return(geom.GetArea())

#def get_area_srid(geom,srid_orig,srid_dest):
#    sr = get_sr(srid_orig)
#    sr2 = get_sr(srid_dest)
#    return(get_area(geom,sr,sr2))