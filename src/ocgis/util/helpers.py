from collections import OrderedDict
import itertools
import os
import tempfile
import sys
import datetime
from copy import deepcopy
from tempfile import mkdtemp

import numpy as np
from shapely.geometry.polygon import Polygon
from osgeo import ogr
from osgeo.ogr import CreateGeometryFromWkb
from shapely.wkb import loads as wkb_loads
import fiona
from shapely.geometry.geo import mapping
from fiona.crs import from_epsg

from ocgis.exc import SingleElementError, ShapeError


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


def format_bool(value):
    """
    Format a string to boolean.

    :param value: The value to convert.
    :type value: int or str
    """

    try:
        ret = bool(int(value))
    except ValueError:
        value = value.lower()
        if value in ['t', 'true']:
            ret = True
        elif value in ['f', 'false']:
            ret = False
        else:
            raise ValueError('String not recognized for boolean conversion: {0}'.format(value))
    return ret


def get_added_slice(slice1, slice2):
    '''
    :param slice slice1:
    :param slice slice2:
    :raises AssertionError:
    :returns slice:
    '''
    assert(slice1.step == None)
    assert(slice2.step == None)

    def _add_(a,b):
        a = a or 0
        b = b or 0
        return(a+b)

    start = _add_(slice1.start,slice2.start)
    stop = _add_(slice1.stop,slice2.stop)

    return(slice(start,stop))


def get_bbox_poly(minx, miny, maxx, maxy):
    rtup = (miny, maxy)
    ctup = (minx, maxx)
    return make_poly(rtup, ctup)


def get_bounds_from_1d(centroids):
    """
    :param centroids: Vector representing center coordinates from which to interpolate bounds.
    :type centroids: :class:`numpy.ndarray`
    :returns: A *n*-by-2 array with *n* equal to the shape of ``centroids``.

    >>> import numpy as np
    >>> centroids = np.array([1,2,3])
    >>> get_bounds_from_1d(centroids)
    np.array([[0, 1],[1, 2],[2, 3]])

    :rtype: :class:`numpy.ndarray`
    :raises: NotImplementedError, ValueError
    """

    mids = get_bounds_vector_from_centroids(centroids)

    # loop to fill the bounds array
    bounds = np.zeros((centroids.shape[0], 2), dtype=centroids.dtype)
    for ii in range(mids.shape[0]):
        try:
            bounds[ii, 0] = mids[ii]
            bounds[ii, 1] = mids[ii + 1]
        except IndexError:
            break

    return bounds


def get_bounds_vector_from_centroids(centroids):
    """
    :param centroids: Vector representing center coordinates from which to interpolate bounds.
    :type centroids: :class:`numpy.ndarray`
    :returns: Vector representing upper and lower bounds for centroids with edges extrapolated.
    :rtype: :class:`numpy.ndarray` with shape ``centroids.shape[0]+1``
    :raises: NotImplementedError, ValueError
    """

    if len(centroids) < 2:
        raise ValueError('Centroid arrays must have length >= 2.')

    # will hold the mean midpoints between coordinate elements
    mids = np.zeros(centroids.shape[0] - 1, dtype=centroids.dtype)
    # this is essentially a two-element span moving average kernel
    for ii in range(mids.shape[0]):
        try:
            mids[ii] = np.mean(centroids[ii:ii + 2])
        # if the data type is datetime.datetime raise a more verbose error message
        except TypeError:
            if isinstance(centroids[ii], datetime.datetime):
                raise NotImplementedError('Bounds interpolation is not implemented for datetime.datetime objects.')
            else:
                raise
    # account for edge effects by averaging the difference of the midpoints. if there is only a single value, use the
    # different of the original values instead.
    if len(mids) == 1:
        diff = np.diff(centroids)
    else:
        diff = np.mean(np.diff(mids))
    # appends for the edges shifting the nearest coordinate by the mean difference
    mids = np.append([mids[0] - diff], mids)
    mids = np.append(mids, [mids[-1] + diff])

    return mids


def get_date_list(start, stop, days):
    ret = []
    delta = datetime.timedelta(days=days)
    check = start
    while check <= stop:
        ret.append(check)
        check += delta
    return ret


def get_default_or_apply(target,f,default=None):
    if target is None:
        ret = default
    else:
        ret = f(target)
    return ret


def get_extrapolated_corners_esmf(arr):
    """
    :param arr: Array of centroids.
    :type arr: :class:`numpy.ndarray`
    :returns: A two-dimensional array of extrapolated corners with dimension ``(arr.shape[0]+1, arr.shape[1]+1)``.
    :rtype: :class:`numpy.ndarray`
    """

    # if this is only a single element, we cannot make corners
    if all([element == 1 for element in arr.shape]):
        msg = 'At least two elements required to extrapolate corners.'
        raise SingleElementError(msg)

    # if one of the dimensions has only a single element, the fill approach is different
    if any([element == 1 for element in arr.shape]):
        ret = get_extrapolated_corners_esmf_vector(arr.reshape(-1))
        if arr.shape[1] == 1:
            ret = ret.swapaxes(0, 1)
        return ret

    # the corners array has one additional row and column
    corners = np.zeros((arr.shape[0]+1, arr.shape[1]+1), dtype=arr.dtype)

    # fill the interior of the array first with a 2x2 moving window. then do edges.
    for ii in range(arr.shape[0]-1):
        for jj in range(arr.shape[1]-1):
            window_values = arr[ii:ii+2, jj:jj+2]
            corners[ii+1, jj+1] = np.mean(window_values)

    # flag to determine if rows are increasing in value
    row_increasing = get_is_increasing(arr[:, 0])
    # flag to determine if columns are increasing in value
    col_increasing = get_is_increasing(arr[0, :])

    # the absolute difference of row and column elements
    row_diff = np.mean(np.abs(np.diff(arr[:, 0])))
    col_diff = np.mean(np.abs(np.diff(arr[0, :])))

    # fill the rows accounting for increasing flag
    for ii in range(1, corners.shape[0]-1):
        if col_increasing:
            corners[ii, 0] = corners[ii, 1] - col_diff
            corners[ii, -1] = corners[ii, -2] + col_diff
        else:
            corners[ii, 0] = corners[ii, 1] + col_diff
            corners[ii, -1] = corners[ii, -2] - col_diff

    # fill the columns accounting for increasing flag
    for jj in range(1, corners.shape[1]-1):
        if row_increasing:
            corners[0, jj] = corners[1, jj] - row_diff
            corners[-1, jj] = corners[-2, jj] + row_diff
        else:
            corners[0, jj] = corners[1, jj] + row_diff
            corners[-1, jj] = corners[-2, jj] - row_diff

    # fill the extreme corners accounting for increasing flag
    for row_idx in [0, -1]:
        if col_increasing:
            corners[row_idx, 0] = corners[row_idx, 1] - col_diff
            corners[row_idx, -1] = corners[row_idx, -2] + col_diff
        else:
            corners[row_idx, 0] = corners[row_idx, 1] + col_diff
            corners[row_idx, -1] = corners[row_idx, -2] - col_diff

    return corners


def get_extrapolated_corners_esmf_vector(vec):
    """
    :param vec: A vector.
    :type vec: :class:`numpy.ndarray`
    :returns: A two-dimensional corners array with dimension ``(2, vec.shape[0]+1)``.
    :rtype: :class:`numpy.ndarray`
    :raises: ShapeError
    """

    if len(vec.shape) > 1:
        msg = 'A vector is required.'
        raise ShapeError(msg)

    corners = np.zeros((2, vec.shape[0]+1), dtype=vec.dtype)
    corners[:] = get_bounds_vector_from_centroids(vec)

    return corners


def get_formatted_slice(slc, n_dims):

    def _format_(slc):
        if isinstance(slc, int):
            ret = slice(slc, slc + 1)
        elif isinstance(slc, slice):
            ret = slc
        elif isinstance(slc, np.ndarray):
            ret = slc
        else:
            if len(slc) == 1:
                ret = slice(slc[0])
            elif len(slc) > 1:
                ret = np.array(slc)
            else:
                raise (NotImplementedError(slc, n_dims))
        return ret

    if isinstance(slc, slice) and slc == slice(None):
        if n_dims == 1:
            ret = slc
        else:
            ret = [slice(None)] * n_dims
    elif n_dims == 1:
        ret = _format_(slc)
    elif n_dims > 1:
        try:
            assert (len(slc) == n_dims)
        except (TypeError, AssertionError):
            raise IndexError("Only {0}-d slicing allowed.".format(n_dims))
        ret = map(_format_, slc)
    else:
        raise (NotImplementedError((slc, n_dims)))

    return ret


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


def get_is_increasing(vec):
    """
    :param vec: A vector array.
    :type vec: :class:`numpy.ndarray`
    :returns: ``True`` if the array is increasing from index 0 to -1. ``False`` otherwise.
    :rtype: bool
    :raises: SingleElementError, ShapeError
    """

    if vec.shape == (1,):
        raise SingleElementError('Increasing can only be determined with a minimum of two elements.')
    if len(vec.shape) > 1:
        msg = 'Only vectors allowed.'
        raise ShapeError(msg)

    if vec[0] < vec[-1]:
        ret = True
    else:
        ret = False

    return ret


def get_iter(element, dtype=None):
    """
    :param element: The element comprising the base iterator. If the element is a ``basestring`` or :class:`numpy.ndarray`
     then the iterator will return the element and stop iteration.
    :type element: varying
    :param dtype: If not ``None``, use this argument as the argument to ``isinstance``. If ``element`` is an instance of
     ``dtype``, ``element`` will be placed in a list and passed to ``iter``.
    :type dtype: type or tuple
    """

    if dtype is not None:
        if isinstance(element, dtype):
            element = (element,)

    if isinstance(element, (basestring, np.ndarray)):
        it = iter([element])
    else:
        try:
            it = iter(element)
        except TypeError:
            it = iter([element])

    return it


def get_none_or_1d(target):
    if target is None:
        ret = None
    else:
        ret = np.atleast_1d(target)
    return ret


def get_none_or_2d(target):
    if target is None:
        ret = None
    else:
        ret = np.atleast_2d(target)
    return ret


def get_none_or_slice(target, slc):
    if target is None:
        ret = None
    else:
        ret = target[slc]
    return ret


def get_ocgis_corners_from_esmf_corners(ecorners):
    """
    :param ecorners: An array of ESMF corners.
    :type ecorners: :class:`numpy.ndarray`
    :returns: A masked array of OCGIS corners.
    :rtype: :class:`~numpy.ma.core.MaskedArray`
    """

    base_shape = [xx-1 for xx in ecorners.shape[1:]]
    grid_corners = np.zeros([2] + base_shape + [4], dtype=ecorners.dtype)
    slices = [(0, 0), (0, 1), (1, 1), (1, 0)]
    # collect the corners and insert into ocgis corners array
    for ii, jj in itertools.product(range(base_shape[0]), range(base_shape[1])):
        row_slice = slice(ii, ii+2)
        col_slice = slice(jj, jj+2)
        row_corners = ecorners[0][row_slice, col_slice]
        col_corners = ecorners[1][row_slice, col_slice]
        for kk, slc in enumerate(slices):
            grid_corners[:, ii, jj, kk] = row_corners[slc], col_corners[slc]
    grid_corners = np.ma.array(grid_corners, mask=False)
    return grid_corners


def get_reduced_slice(arr):
    arr_min, arr_max = arr.min(), arr.max()
    assert (arr_max - arr_min + 1 == arr.shape[0])
    ret = slice(arr_min, arr_max + 1)
    return ret


def get_sorted_uris_by_time_dimension(uris, variable=None):
    """
    Sort a sequence of NetCDF URIs by the maximum time extent in ascending order.

    :param uris: The sequence of NetCDF URIs to sort.
    :type uris: list[str]

    >>> uris = ['/path/to/file2.nc', 'path/to/file1.nc']

    :param str variable: The target variable for sorting. If ``None`` is provided, then the variable will be
     autodiscovered.
    :returns: A sequence of sorted URIs.
    :rtype: list[str]
    """

    from ocgis import RequestDataset

    to_sort = {}
    for uri in uris:
        rd = RequestDataset(uri=uri, variable=variable)
        to_sort[rd.get().temporal.extent_datetime[1]] = rd.uri
    sorted_keys = sorted(to_sort)
    ret = [to_sort[sk] for sk in sorted_keys]
    return ret


def get_trimmed_array_by_mask(arr,return_adjustments=False):
    '''
    Returns a slice of the masked array ``arr`` with masked rows and columns
    removed.

    :param arr: Two-dimensional array object.
    :type arr: :class:`numpy.ma.MaskedArray` or bool :class:`numpy.ndarray`
    :param bool return_adjustments: If ``True``, return a dictionary with
     values of index adjustments that may be added to a slice object.
    :raises NotImplementedError:
    :returns: :class:`numpy.ma.MaskedArray` or (:class:`numpy.ma.MaskedArray', {'row':slice(...),'col':slice(...)})
    '''
    try:
        _mask = arr.mask
    except AttributeError:
        ## likely a boolean array
        if arr.dtype == np.dtype(bool):
            _mask = arr
        else:
            raise(NotImplementedError('Array type is not implemented.'))
    ## row 0 to end
    start_row = 0
    for idx_row in range(arr.shape[0]):
        if _mask[idx_row,:].all():
            start_row += 1
        else:
            break

    ## row end to 0
    stop_row = 0
    idx_row_adjust = 1
    for __ in range(arr.shape[0]):
        if _mask[stop_row-idx_row_adjust,:].all():
            idx_row_adjust += 1
        else:
            idx_row_adjust -= 1
            break
    if idx_row_adjust == 0:
        stop_row = None
    else:
        stop_row = stop_row - idx_row_adjust

    ## col 0 to end
    start_col = 0
    for idx_col in range(arr.shape[1]):
        if _mask[:,idx_col].all():
            start_col += 1
        else:
            break

    ## col end to 0
    stop_col = 0
    idx_col_adjust = 1
    for __ in range(arr.shape[0]):
        if _mask[:,stop_col-idx_col_adjust,].all():
            idx_col_adjust += 1
        else:
            idx_col_adjust -= 1
            break
    if idx_col_adjust == 0:
        stop_col = None
    else:
        stop_col = stop_col - idx_col_adjust

    ret = arr[start_row:stop_row,start_col:stop_col]

    if return_adjustments:
        ret = (ret,{'row':slice(start_row,stop_row),'col':slice(start_col,stop_col)})

    return(ret)


def project_shapely_geometry(geom, from_sr, to_sr):
    if from_sr.IsSame(to_sr) == 1:
        ret = geom
    else:
        ogr_geom = CreateGeometryFromWkb(geom.wkb)
        ogr_geom.AssignSpatialReference(from_sr)
        ogr_geom.TransformTo(to_sr)
        ret = wkb_loads(ogr_geom.ExportToWkb())
    return ret


def set_name_attributes(name_mapping):
    """
    Set the name attributes on the keys of ``name_mapping``.
    :param dict name_mapping: The keys are objects with a name attribute to set to its value if the attribute is
     ``None``.
    """

    for target, name in name_mapping.iteritems():
        if target is not None and target.name is None:
            target.name = name


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

def write_geom_dict(dct, path=None, filename=None, epsg=4326, crs=None):
    """
    :param dct:
    :type dct: dict

    >>> dct = {1: Point(1, 2), 2: Point(3, 4)}

    :param path:
    :type path: str
    :param filename:
    :type filename: str
    """

    filename = filename or 'out'
    path = path or os.path.join(mkdtemp(), '{0}.shp'.format(filename))

    crs = crs or from_epsg(epsg)
    driver = 'ESRI Shapefile'
    schema = {'properties': {'UGID': 'int'}, 'geometry': dct.values()[0].geom_type}
    with fiona.open(path, 'w', driver=driver, crs=crs, schema=schema) as source:
        for k, v in dct.iteritems():
            rec = {'properties': {'UGID': k}, 'geometry': mapping(v)}
            source.write(rec)
    return path


def locate(pattern, root=os.curdir, followlinks=True):
    """
    Locate all files matching supplied filename pattern in and below supplied root directory.
    """

    for path, dirs, files in os.walk(os.path.abspath(root), followlinks=followlinks):
        for filename in filter(lambda x: x == pattern, files):
            yield os.path.join(path, filename)


def get_ordered_dicts_from_records_array(arr):
    """
    Convert a NumPy records array to an ordered dictionary.

    :param arr: The records array to convert with shape (m,).
    :type arr: :class:`numpy.core.multiarray.ndarray`
    :rtype: list[:class:`collections.OrderedDict`]
    """

    ret = []
    _names = arr.dtype.names
    for ii in range(arr.shape[0]):
        fill = OrderedDict()
        row = arr[ii]
        for name in _names:
            fill[name] = row[name]
        ret.append(fill)
    return ret


def iter_array(arr, use_mask=True, return_value=False):
    try:
        shp = arr.shape
    # assume array is not a numpy array
    except AttributeError:
        arr = np.array(arr, ndmin=1)
        shp = arr.shape
    iter_args = [range(0, ii) for ii in shp]
    if use_mask and not np.ma.isMaskedArray(arr):
        use_mask = False
    else:
        try:
            mask = arr.mask
            # if the mask is not being used, to skip some objects, set the arr to the underlying data value after
            # referencing the mask.
            if not use_mask:
                arr = arr.data
        # array is not masked
        except AttributeError:
            pass

    for ii in itertools.product(*iter_args):
        if use_mask:
            try:
                if mask[ii]:
                    continue
                else:
                    idx = ii
            # occurs with singleton dimension of masked array
            except IndexError:
                if mask:
                    continue
                else:
                    idx = ii
        else:
            idx = ii
        if return_value:
            ret = (idx, arr[ii])
        else:
            ret = idx
        yield ret

    
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