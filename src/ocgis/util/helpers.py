import datetime
import itertools
import os
import sys
import tempfile
from collections import OrderedDict
from copy import deepcopy
from pprint import pprint
from tempfile import mkdtemp

import fiona
import numpy as np
import six
from fiona.crs import from_epsg
from netCDF4 import netcdftime
from numpy.core.multiarray import ndarray
from numpy.ma import MaskedArray
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.geo import mapping
from shapely.geometry.polygon import Polygon
from shapely.wkb import loads as wkb_loads

from ocgis.constants import MPITag
from ocgis.exc import SingleElementError, ShapeError, AllElementsMaskedError


class ProgressBar(object):
    def __init__(self, title):
        sys.stdout.write(title + ": [" + "-" * 40 + "]" + chr(8) * 41)
        sys.stdout.flush()
        self.px = 0

    def progress(self, x):
        x = x * 40 // 100
        sys.stdout.write("#" * (x - self.px))
        sys.stdout.flush()
        self.px = x

    def endProgress(self):
        sys.stdout.write("#" * (40 - self.px))
        sys.stdout.write("]\n")
        sys.stdout.flush()


def add_shapefile_unique_identifier(in_path, out_path, name=None, template=None):
    """
    >>> add_shapefile_unique_identifier('/path/to/foo.shp', '/path/to/new_foo.shp')
    '/path/to/new_foo.shp'

    :param str in_path: Full path to the input shapefile.
    :param str out_path: Full path to the output shapefile.
    :param str name: The name of the unique identifer. If ``None``, defaults to
     :attr:`ocgis.constants.OCGIS_UNIQUE_GEOMETRY_IDENTIFIER`.
    :param str template: The integer attribute to copy as the unique identifier.
    :returns: Path to the copied shapefile with the addition of a unique integer attribute called ``name``.
    :rtype: str
    """

    from ocgis.util.shp_process import ShpProcess

    out_folder, key = os.path.split(out_path)
    sp = ShpProcess(in_path, out_folder)
    key = os.path.splitext(key)[0]
    sp.process(key=key, ugid=template, name=name)

    return out_path


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


def arange_from_bool_ndarray(arr, start=0):
    """
    Create a collapsed range from ``start`` to the sum of ``True`` elements in ``arr`` plus one.

    :param arr: The boolean array to use as a filter for the range creation.
    :type arr: :class:`numpy.ndarray`
    :param int start: The start index for the range array.
    :return: An integer range array.
    :rtype: :class:`numpy.ndarray`
    """

    npw = np.where(arr)[0]
    npw += start
    return npw


def arange_from_dimension(dim, start=0, dtype=None, dist=True):
    """
    Create a sequential integer range similar to ``numpy.arange``. Call is collective across the current
    :class:`~ocgis.OcgVM` if ``dist=True`` (the default).

    :param dim: The dimension to use for creating the range.
    :type dim: :class:`~ocgis.Dimension`
    :param int start: The starting value for the range.
    :param dtype: The data type for the output array.
    :param bool dist: If ``True``, create range as a distributed array with a collective VM call. If ``False``, create
     the array locally.
    :rtype: :class:`numpy.ndarray`
    """

    if dtype is None:
        from ocgis import env
        dtype = env.NP_INT

    local_size = len(dim)
    if dist:
        from ocgis import vm
        for rank in vm.ranks:
            dest_rank = rank + 1
            if dest_rank == vm.size:
                break
            else:
                if vm.rank == rank:
                    data = np.array([start + local_size], dtype=dtype)
                    buf = [data, vm.get_mpi_type(dtype)]
                    vm.comm.Send(buf, dest=dest_rank, tag=MPITag.ARANGE_FROM_DIMENSION)
                elif vm.rank == dest_rank:
                    data = np.zeros(1, dtype=dtype)
                    buf = [data, vm.get_mpi_type(dtype)]
                    vm.comm.Recv(buf, source=rank, tag=MPITag.ARANGE_FROM_DIMENSION)
                    start = data[0]
                else:
                    pass
        vm.barrier()

    ret = np.arange(start, start + local_size, dtype=dtype)

    return ret


def get_added_slice(slice1, slice2):
    '''
    :param slice slice1:
    :param slice slice2:
    :raises AssertionError:
    :returns slice:
    '''
    assert (slice1.step == None)
    assert (slice2.step == None)

    def _add_(a, b):
        a = a or 0
        b = b or 0
        return (a + b)

    start = _add_(slice1.start, slice2.start)
    stop = _add_(slice1.stop, slice2.stop)

    return (slice(start, stop))


def get_bbox_poly(minx, miny, maxx, maxy):
    rtup = (miny, maxy)
    ctup = (minx, maxx)
    return make_poly(rtup, ctup)


def get_by_key_list(src, keys):
    found = False
    ret = None
    for key in keys:
        if key in src:
            if found:
                raise ValueError('Key already found in source.')
            else:
                found = True
                ret = src[key]
    return ret


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


def get_by_sequence(dictionary, key_sequence, default=None):
    target = dictionary
    for key in get_iter(key_sequence):
        try:
            target = target[key]
        except KeyError:
            target = default
            break
    return target


def get_date_list(start, stop, days):
    ret = []
    delta = datetime.timedelta(days=days)
    check = start
    while check <= stop:
        ret.append(check)
        try:
            check += delta
        except TypeError:
            # Probably an older netcdftime object
            check = datetime.datetime(check.year, check.month, check.day, check.hour, check.second, check.microsecond)
            check += delta
            check = netcdftime.datetime(check.year, check.month, check.day, check.hour, check.second, check.microsecond)
    return ret


def get_default_or_apply(target, f, default=None):
    if target is None:
        ret = default
    else:
        ret = f(target)
    return ret


def create_exact_field_value(longitude, latitude):
    """
    Create an exact field from spherical coordinates. Expects units of degrees - function will convert to radians.
    """

    longitude = np.atleast_1d(longitude)
    latitude = np.atleast_1d(latitude)
    select = longitude < 0
    if np.any(select):
        longitude = deepcopy(longitude)
        longitude[select] += 360.
    longitude_radians = longitude * 0.0174533
    latitude_radians = latitude * 0.0174533
    exact = 2.0 + np.cos(latitude_radians) ** 2 + np.cos(2.0 * longitude_radians)
    return exact


def create_ocgis_corners_from_esmf_corners(ecorners):
    """
    :param ecorners: An array of ESMF corners.
    :type ecorners: :class:`numpy.ndarray`
    :return: :class:`numpy.ndarray`
    """
    assert ecorners.ndim == 2

    # ESMF corners have an extra row and column.
    base_shape = [xx - 1 for xx in ecorners.shape]
    grid_corners = np.zeros(base_shape + [4], dtype=ecorners.dtype)
    # Upper left, upper right, lower right, lower left - this is an ideal order. Actual order may differ later.
    slices = [(0, 0), (0, 1), (1, 1), (1, 0)]
    for ii, jj in itertools.product(list(range(base_shape[0])), list(range(base_shape[1]))):
        row_slice = slice(ii, ii + 2)
        col_slice = slice(jj, jj + 2)
        corners = ecorners[row_slice, col_slice]
        for kk, slc in enumerate(slices):
            grid_corners[ii, jj, kk] = corners[slc]
    return grid_corners


def create_zero_padded_integer(integer, nzeros):
    sint = str(integer)
    nstr_zeros = nzeros - len(sint)
    if nstr_zeros <= 0:
        ret = sint
    else:
        ret = '0' * nstr_zeros
        ret += sint
    return ret


def get_extrapolated_corners_esmf(arr):
    """
    :param arr: Array of centroids.
    :type arr: :class:`numpy.ndarray`
    :returns: A two-dimensional array of extrapolated corners with dimension ``(arr.shape[0]+1, arr.shape[1]+1)``.
    :rtype: :class:`numpy.ndarray`
    """

    assert not isinstance(arr, MaskedArray)

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
    corners = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=arr.dtype)

    # fill the interior of the array first with a 2x2 moving window. then do edges.
    for ii in range(arr.shape[0] - 1):
        for jj in range(arr.shape[1] - 1):
            window_values = arr[ii:ii + 2, jj:jj + 2]
            corners[ii + 1, jj + 1] = np.mean(window_values)

    # flag to determine if rows are increasing in value
    row_increasing = get_is_increasing(arr[:, 0])
    # flag to determine if columns are increasing in value
    col_increasing = get_is_increasing(arr[0, :])

    # the absolute difference of row and column elements
    row_diff = np.mean(np.abs(np.diff(arr[:, 0])))
    col_diff = np.mean(np.abs(np.diff(arr[0, :])))

    # fill the rows accounting for increasing flag
    for ii in range(1, corners.shape[0] - 1):
        if col_increasing:
            corners[ii, 0] = corners[ii, 1] - col_diff
            corners[ii, -1] = corners[ii, -2] + col_diff
        else:
            corners[ii, 0] = corners[ii, 1] + col_diff
            corners[ii, -1] = corners[ii, -2] - col_diff

    # fill the columns accounting for increasing flag
    for jj in range(1, corners.shape[1] - 1):
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

    corners = np.zeros((2, vec.shape[0] + 1), dtype=vec.dtype)
    corners[:] = get_bounds_vector_from_centroids(vec)

    return corners


def get_formatted_slice(slc, n_dims):
    def _format_singleton_(single_slc):
        # Avoid numpy typing by attempting to convert to an integer. Floats will be mangled, but whatever.
        try:
            do_int_conversion = True
            if hasattr(single_slc, 'dtype') and single_slc.dtype == bool:
                do_int_conversion = False
            if do_int_conversion:
                single_slc = int(single_slc)
        except:
            pass

        if isinstance(single_slc, int):
            ret = slice(single_slc, single_slc + 1)
        elif isinstance(single_slc, slice):
            ret = single_slc
        elif isinstance(single_slc, np.ndarray):
            ret = get_optimal_slice_from_array(single_slc)
        elif isinstance(single_slc, (list, tuple)):
            if len(single_slc) == 1 and isinstance(single_slc[0], slice):
                ret = single_slc[0]
            elif len(single_slc) == 1 and isinstance(single_slc[0], ndarray):
                ret = get_optimal_slice_from_array(single_slc[0])
            else:
                ret = get_optimal_slice_from_array(np.array(single_slc, ndmin=1))
        elif single_slc is None:
            ret = slice(None)
        else:
            raise NotImplementedError(single_slc, n_dims)
        return ret

    slice_none = slice(None)
    if isinstance(slc, slice) and slc == slice_none:
        if n_dims == 1:
            ret = slc
        else:
            ret = [slice_none] * n_dims
    elif slc is None and n_dims == 1:
        ret = slice_none
    elif n_dims == 1:
        ret = _format_singleton_(slc)
    elif n_dims > 1:
        try:
            assert len(slc) == n_dims
        except (TypeError, AssertionError):
            raise IndexError("Only {0}-d slicing allowed.".format(n_dims))
        ret = list(map(_format_singleton_, slc))
    else:
        raise NotImplementedError((slc, n_dims))

    if isinstance(ret, list):
        ret = tuple(ret)
    if not isinstance(ret, tuple):
        ret = (ret,)

    return ret


def get_is_date_between(lower, upper, month=None, year=None):
    """
    :param lower: The lower boundary time coordinate.
    :type lower: :class:`datetime.datetime`
    :param upper: The upper boundary time coordinate.
    :type upper: :class:`datetime.datetime`
    :param int month: The month to check.
    :param int year: The year to check.
    :returns: ``True`` if the check value occurs in the interval.
    :rtype: bool
    """

    if month is not None:
        attr = 'month'
        to_test = month
    else:
        attr = 'year'
        to_test = year

    part_lower, part_upper = getattr(lower, attr), getattr(upper, attr)
    if part_lower != part_upper:
        if part_lower > part_upper:
            # in the case of a year overlap, increment the upper into another year by adding 12 months
            part_upper += 12
        ret = np.logical_and(to_test >= part_lower, to_test < part_upper)
    else:
        ret = np.logical_and(to_test >= part_lower, to_test <= part_upper)

    return ret


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

    if isinstance(element, tuple(list(six.string_types) + [np.ndarray])):
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


def get_esmf_corners_from_ocgis_corners(ocorners, fill=None):
    """
    :param ocorners: Corners array with dimension (m, n, 4).
    :type ocorners: :class:`numpy.ma.core.MaskedArray`
    :returns: An ESMF corners array with dimension (n + 1, m + 1).
    :rtype: :class:`numpy.ndarray`
    """

    if fill is None:
        shp = np.flipud(ocorners.shape[0:2])
        fill = np.zeros([element + 1 for element in shp], dtype=ocorners.dtype)
    range_row = list(range(ocorners.shape[0]))
    range_col = list(range(ocorners.shape[1]))

    if isinstance(ocorners, MaskedArray):
        _corners = ocorners.data
    else:
        _corners = ocorners

    for ii, jj in itertools.product(range_row, range_col):
        ref = fill[jj:jj + 2, ii:ii + 2]
        ref[0, 0] = _corners[ii, jj, 0]
        ref[1, 0] = _corners[ii, jj, 1]
        ref[1, 1] = _corners[ii, jj, 2]
        ref[0, 1] = _corners[ii, jj, 3]
    return fill


def get_optimal_slice_from_array(arr, check_diff=True):
    if arr.dtype == bool:
        ret = arr
    else:
        if check_diff and np.any(np.diff(arr) > 1):
            ret = arr
        else:
            arr_min, arr_max = arr.min(), arr.max()
            ret = slice(arr_min, arr_max + 1)
    return ret


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
            fill_value = row[name]
            # A masked value of True in the records array indicates a NULL value in OGR files.
            if np.ma.is_masked(fill_value):
                fill_value = None
            fill[name] = fill_value
        ret.append(fill)
    return ret


def get_reduced_slice(arr):
    arr_min, arr_max = arr.min(), arr.max()
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


def get_swap_chain(actual, desired):
    """
    :param sequence actual: The current names.
    :param sequence desired: The desired names.
    :return: Tuple of tuples containing swap indices to achieve the desired names ordering.
    :rtype: tuple(tuple(), ...)
    """
    assert set(actual) == set(desired)
    swap_ops = []

    running_archetype = deepcopy(actual)
    for swap_src in range(len(actual)):
        if running_archetype[swap_src] != desired[swap_src]:
            swap_dst = running_archetype.index(desired[swap_src])
            the_swap = (swap_src, swap_dst)
            original_value = running_archetype[swap_src]
            running_archetype[the_swap[0]] = running_archetype[swap_dst]
            running_archetype[the_swap[1]] = original_value
            swap_ops.append((swap_src, swap_dst))

    return tuple(swap_ops)


def get_trimmed_array_by_mask(arr, return_adjustments=False):
    """"
    Returns a slice of the masked array ``arr`` with masked rows and columns removed.

    :param arr: An array.
    :type arr: :class:`numpy.ma.MaskedArray` or bool :class:`numpy.ndarray`
    :param bool return_adjustments: If ``True``, return a dictionary with values of index adjustments that may be added
     to a slice object.
    :raises NotImplementedError:
    :returns: :class:`numpy.ma.MaskedArray` or (:class:`numpy.ma.MaskedArray', {'row':slice(...),'col':slice(...)})
    """
    assert arr.ndim <= 2

    has_col = False
    try:
        _mask = arr.mask
    except AttributeError:
        # Likely a boolean array.
        if arr.dtype == np.dtype(bool):
            _mask = arr
        else:
            raise NotImplementedError('Array type is not implemented.')

    if _mask.all():
        raise AllElementsMaskedError

    # Row 0 to end.
    start_row = 0
    for idx_row in range(arr.shape[0]):
        if not _mask[idx_row, ...].all():
            start_row = idx_row
            break

    # Row end to 0.
    row_count = _mask.shape[0]
    stop_row = row_count
    for idx_row in range(row_count):
        row_to_test = row_count - (idx_row + 1)
        if not _mask[row_to_test, ...].all():
            stop_row = row_to_test + 1
            break

    if arr.ndim == 2:
        has_col = True

        # col 0 to end.
        start_col = 0
        for idx_col in range(arr.shape[1]):
            if not _mask[:, idx_col].all():
                start_col = idx_col
                break

        # col end to 0.
        col_count = _mask.shape[1]
        stop_col = col_count
        for idx_col in range(col_count):
            col_to_test = col_count - (idx_col + 1)
            if not _mask[:, col_to_test].all():
                stop_col = col_to_test + 1
                break

    slc = [slice(start_row, stop_row)]

    if has_col:
        slc.append(slice(start_col, stop_col))
    ret = arr.__getitem__(slc)

    if return_adjustments:
        ret = (ret, tuple(slc))

    return ret


def get_tuple(value):
    """
    :returns: A tuple constructed from ``value``. If ``value`` is a string or ``None``, a one-element tuple containing
     value will be returned.
    :rtype: tuple
    """

    if isinstance(value, six.string_types) or value is None:
        ret = (value,)
    else:
        ret = tuple(value)
    return ret


def is_auto_dtype(dtype):
    try:
        is_auto = dtype == 'auto'
    except TypeError:
        # NumPy does interesting things with data type comparisons.
        is_auto = False
    return is_auto


def is_crs_variable(obj):
    from ocgis.variable.crs import AbstractCRS
    return isinstance(obj, AbstractCRS)


def is_geometry(obj):
    return isinstance(obj, BaseGeometry)


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
    except TypeError:  # fails only when cls is type
        subs = cls.__subclasses__(cls)
    for sub in subs:
        if sub not in _seen:
            _seen.add(sub)
            yield sub
            for sub in itersubclasses(sub, _seen):
                yield sub


def iter_array(arr, use_mask=True, return_value=False, mask=None):
    try:
        shp = arr.shape
    # assume array is not a numpy array
    except AttributeError:
        arr = np.array(arr, ndmin=1)
        shp = arr.shape
    iter_args = [list(range(0, ii)) for ii in shp]
    if use_mask and not np.ma.isMaskedArray(arr) and mask is None:
        use_mask = False
    else:
        try:
            if mask is None:
                mask = arr.mask
            # if the mask is not being used, to skip some objects, set the arr to the underlying data value after
            # referencing the mask.
            if not use_mask and np.ma.isMaskedArray(arr):
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


def locate(pattern, root=os.curdir, followlinks=True):
    """
    Locate all files matching supplied filename pattern in and below supplied root directory.
    """

    for path, dirs, files in os.walk(os.path.abspath(root), followlinks=followlinks):
        for filename in [x for x in files if x == pattern]:
            yield os.path.join(path, filename)


def pprint_dict(target):
    def _convert_(input):
        ret = input
        if isinstance(ret, dict):
            ret = dict(input)
            for k, v in ret.items():
                ret[k] = _convert_(v)
        return ret

    to_print = _convert_(target)
    print('')
    pprint(to_print)


def project_shapely_geometry(geom, from_sr, to_sr):
    from ocgis.environment import ogr
    CreateGeometryFromWkb = ogr.CreateGeometryFromWkb

    if from_sr.IsSame(to_sr) == 1:
        ret = geom
    else:
        ogr_geom = CreateGeometryFromWkb(geom.wkb)
        ogr_geom.AssignSpatialReference(from_sr)
        ogr_geom.TransformTo(to_sr)
        ret = wkb_loads(ogr_geom.ExportToWkb())
    return ret


def reduce_multiply(sequence):
    ret = 1.
    for s in sequence:
        ret *= s
    return ret


def reindex_and_minimize_variables_global(idxvar, datavar, start_index=0):
    """
    Reindex ``idxvar`` to start-based index ``start_index`` and minimize elements in ``datavar`` based on its index
    presence in ``idxvar``.

    :param idxvar: N-dimensional, integer variable containing indices.
    :type idxvar: :class:`~ocgis.Variable`
    :param datavar: One-dimensional variable containing data for indices in ``idxvar``.
    :param int start_index: The start index to use for the first element in ``idxvar``.
    :return: A two-element tuple with the first element being the re-indexed ``idxvar`` and the second element being
     the minimized data indexed by the new ``idxvar``.
    :rtype: tuple
    """

    raise NotImplementedError('work in progress')


def create_index_array(src, dst, dtype=int, indices=None):
    """
    Index the values in ``src`` inside ``dst``. Returns an integer index array that will create the values in ``src``
    if used to slice ``dst``.
    
    :param src: Array containing source values to index.
    :type src: :class:`numpy.ndarray`
    :param dst: Array containing the value database to index into.
    :type dst: :class:`numpy.ndarray`
    :param dtype: Output data type for the index array.
    :type dtype: `NumPy integer type`
    :rtype: :class:`numpy.ndarray`
    """

    src = np.array(src)
    dst = np.array(dst)
    ret = np.zeros_like(src, dtype=dtype)

    for idx, src_value in enumerate(src.flat):
        select = src_value == dst
        idx_dst = np.where(select)[0][0]
        if indices is not None:
            a = indices[idx_dst]
        else:
            a = idx_dst
        ret[idx] = a

    return ret


def create_index_variable_global(name, src, dst, dtype=int, start=0):
    """
    Create an index variable in parallel.
    
    Similar to :meth:`~ocgis.util.helpers.create_index_array` except :class:`~ocgis.Variable` objects are used and 
    returned instead of :class:`numpy.ndarray` objects. This method will accept empty objects for both ``src`` and
    ``dst``. If ``src`` is empty on the calling rank, then ``None`` will be returned.
    
    :param str name: Name of the output global index variable.
    """

    raise NotImplementedError('work in progress')

    global_index = dst.create_ugid_global('global_index', start=start)

    for src_value in src.get_value().flat:
        tkk

    # barrier_print(global_index.get_value())

    # tkk

    index_array = create_index_array(src.get_value(), dst.get_value(), dtype=dtype, indices=global_index.get_value())
    ret = src.__class__(name=name, value=index_array, dimensions=src.dimensions)
    return ret


def set_name_attributes(name_mapping):
    """
    Set the name attributes on the keys of ``name_mapping``.
    :param dict name_mapping: The keys are objects with a name attribute to set to its value if the attribute is
     ``None``.
    """

    for target, name in name_mapping.items():
        if target is not None and target.name is None:
            target.name = name


def create_unique_global_array(arr):
    """
    Create a distributed NumPy array containing unique elements. If the rank has no unique items, an array with zero
    elements will be returned. This call is collective across the current VM.

    :param arr: Input array for unique operation.
    :type arr: :class:`numpy.ndarray`
    :rtype: :class:`numpy.ndarray`
    :raises: ValueError
    """

    from ocgis import vm

    if arr is None:
        raise ValueError('Input must be a NumPy array.')

    unique_local = np.unique(arr)
    vm.barrier()

    local_bounds = min(unique_local), max(unique_local)
    lb_global = vm.gather(local_bounds)
    lb_global = vm.bcast(lb_global)

    # Find the vm ranks the local rank cares about. It cares if unique values have overlapping unique bounds.
    overlaps = []
    for rank, lb in enumerate(lb_global):
        if rank == vm.rank:
            continue
        contains = []
        for lb2 in local_bounds:
            if lb[0] <= lb2 <= lb[1]:
                to_app = True
            else:
                to_app = False
            contains.append(to_app)
        if any(contains) or (local_bounds[0] <= lb[0] and local_bounds[1] >= lb[1]):
            overlaps.append(rank)

    # Send out the overlapping sources.
    tag_overlap = MPITag.OVERLAP_CHECK
    tag_select_send_size = MPITag.SELECT_SEND_SIZE
    vm.barrier()

    # NumPy and MPI types.
    np_type = unique_local.dtype
    mpi_type = vm.get_mpi_type(np_type)

    for o in overlaps:
        if vm.rank != o and vm.rank < o:
            dest_rank_bounds = lb_global[o]
            select_send = np.logical_and(unique_local >= dest_rank_bounds[0], unique_local <= dest_rank_bounds[1])
            u_src = unique_local[select_send]
            select_send_size = u_src.size
            _ = vm.comm.Isend([np.array([select_send_size], dtype=np_type), mpi_type], dest=o, tag=tag_select_send_size)
            _ = vm.comm.Isend([u_src, mpi_type], dest=o, tag=tag_overlap)

    # Receive and process conflicts to reduce the unique local values.
    if vm.rank != 0:
        for o in overlaps:
            if vm.rank != o and vm.rank > o:
                select_send_size = np.array([0], dtype=np_type)
                req_select_send_size = vm.comm.Irecv([select_send_size, mpi_type], source=o, tag=tag_select_send_size)
                req_select_send_size.wait()
                select_send_size = select_send_size[0]

                u_src = np.zeros(select_send_size.astype(int), dtype=np_type)
                req = vm.comm.Irecv([u_src, mpi_type], source=o, tag=tag_overlap)
                req.wait()

                utokeep = np.ones_like(unique_local, dtype=bool)
                for uidx, u in enumerate(unique_local.flat):
                    if u in u_src:
                        utokeep[uidx] = False
                unique_local = unique_local[utokeep]

    vm.barrier()
    return unique_local


def update_or_pass(target, key, value):
    """
    :param dict target: The target dictionary to update.
    :param key: The dictionary's key to update.
    :param value: The dictionary's value to use for the update if the key is not present in `target`.
    """

    if key not in target:
        target[key] = value


def validate_time_subset(time_range, time_region):
    """
    Ensure ``time_range`` and ``time_region`` overlap. If one of the values is ``None``, the function always returns
     ``True``. Function will return `False` if the two time range descriptions do not overlap.

    :param time_range: Sequence with two datetime elements.
    :type time_range: sequence
    :param time_region: Dictionary with two keys ``'month'`` and ``'year'`` each containing an integer sequence
     corresponding to the respective time parts. For example:

    >>> time_region = {'month':[1,2],'year':[2013]}

     If a 'month' or 'year' key is missing, the key will be added with a default of ``None``.
    :type time_region: dict
    :rtype: bool
    """

    def _between_(target, lower, upper):
        if target >= lower and target <= upper:
            ret = True
        else:
            ret = False
        return ret

    def _check_months_(targets, months):
        check = [target in months for target in targets]
        if all(check):
            ret = True
        else:
            ret = False
        return ret

    def _check_years_(targets, min_range_year, max_range_year):
        if all([_between_(year_bound, min_range_year, max_range_year) for year_bound in targets]):
            ret = True
        else:
            ret = False
        return ret

    # By default we return that it does not validate.
    ret = False
    # If any of the parameters are none, then it will validate True.
    if any([t is None for t in [time_range, time_region]]):
        ret = True
    else:
        # Ensure time region has the necessary keys.
        copy_time_region = deepcopy(time_region)
        for key in ['month', 'year']:
            if key not in copy_time_region:
                copy_time_region[key] = None
        # Pull basic date information from the time range.
        min_range_year, max_range_year = time_range[0].year, time_range[1].year
        delta = datetime.timedelta(days=29, hours=12)
        months = set()
        current = time_range[0]
        while current <= time_range[1]:
            current += delta
            months.update([current.month])
            if len(months) == 12:
                break
        # Construct boundaries from time region. first, the case of only months.
        if copy_time_region['month'] is not None and copy_time_region['year'] is None:
            month_bounds = min(copy_time_region['month']), max(copy_time_region['month'])
            ret = _check_months_(month_bounds, months)
        # Case of only years.
        elif copy_time_region['month'] is None and copy_time_region['year'] is not None:
            year_bounds = min(copy_time_region['year']), max(copy_time_region['year'])
            ret = _check_years_(year_bounds, min_range_year, max_range_year)
        # Case with both years and months.
        else:
            month_bounds = min(copy_time_region['month']), max(copy_time_region['month'])
            year_bounds = min(copy_time_region['year']), max(copy_time_region['year'])
            ret_months = _check_months_(month_bounds, months)
            ret_years = _check_years_(year_bounds, min_range_year, max_range_year)
            if all([ret_months, ret_years]):
                ret = True
    return ret


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
    schema = {'properties': {'UGID': 'int'}, 'geometry': list(dct.values())[0].geom_type}
    with fiona.open(path, 'w', driver=driver, crs=crs, schema=schema) as source:
        for k, v in dct.items():
            rec = {'properties': {'UGID': k}, 'geometry': mapping(v)}
            source.write(rec)
    return path


def make_poly(rtup, ctup):
    """
    rtup = (row min, row max)
    ctup = (col min, col max)
    """

    return Polygon(((ctup[0], rtup[0]),
                    (ctup[0], rtup[1]),
                    (ctup[1], rtup[1]),
                    (ctup[1], rtup[0])))


def get_temp_path(suffix='', name=None, nest=False, only_dir=False, wd=None, dir_prefix=None):
    """Return absolute path to a temporary file."""

    if dir_prefix is not None:
        if not dir_prefix.endswith('_'):
            dir_prefix = dir_prefix + '_'
    else:
        dir_prefix = ''

    def _get_wd_():
        if wd is None:
            return tempfile.gettempdir()
        else:
            return wd

    if nest:
        f = tempfile.NamedTemporaryFile()
        f.close()
        dir = os.path.join(_get_wd_(), dir_prefix + os.path.split(f.name)[-1])
        os.mkdir(dir)
    else:
        dir = _get_wd_()
    if only_dir:
        ret = dir
    else:
        if name is not None:
            ret = os.path.join(dir, name + suffix)
        else:
            f = tempfile.NamedTemporaryFile(suffix=suffix, dir=dir)
            f.close()
            ret = f.name
    return str(ret)


def get_group(ddict, keyseq, has_root=True, create=False, last=None):
    keyseq = deepcopy(keyseq)
    if last is None:
        last = {}

    if keyseq is None:
        keyseq = [None]
    elif isinstance(keyseq, six.string_types):
        keyseq = [keyseq]

    if keyseq[0] is not None:
        keyseq.insert(0, None)

    curr = ddict
    len_keyseq = len(keyseq)
    for ctr, key in enumerate(keyseq, start=1):
        if key is None:
            if has_root:
                curr = curr[None]
        else:
            # Allow dimension maps to not have groups. Return empty dictionaries for groups that do not exist in the
            # dimension map.
            # curr = curr.get('groups', {}).get(key, {})
            if create:
                curr = get_or_create_dict(curr, 'groups', {})
                if ctr == len_keyseq:
                    default = last
                else:
                    default = {}
                curr = get_or_create_dict(curr, key, default)
            else:
                curr = curr['groups'][key]
    return curr


def get_or_create_dict(dct, key, default):
    if key not in dct:
        dct[key] = default
    return dct[key]


def find_index(seqs, to_find):
    """
    Find the first matching index of a sequence of scalar values in ``to_find`` that match the first unique index
    in the sequence of arrays in ``seq``.

    :param seqs: A sequence of :class:`numpy.ndarray` objects with the same length as ``to_find``.
    :type seqs: sequence
    :param to_find: A sequence of scalar objects to search ``seqs`` for.
    :type to_find: sequence
    :return: The integer index value for ``to_find``'s location in ``seqs``.
    :rtype: int
    """
    seqs = [np.array(s) for s in seqs]

    selects = []
    for seq_idx, seq in enumerate(seqs):
        selects.append(seq == to_find[seq_idx])

    reduced = None
    for idx, curr_select in enumerate(selects):
        if reduced is None:
            reduced = curr_select
        else:
            reduced = np.logical_and(reduced, curr_select)

    try:
        found_idx = np.where(reduced)[0][0]
    except IndexError:
        found_idx = None

    return found_idx


def iter_exploded_geometries(geom):
    """Yield single-part Shapely geometry objects."""

    if isinstance(geom, BaseMultipartGeometry):
        itr = iter(geom)
    else:
        itr = [geom]
    for element1 in itr:
        if isinstance(element1, BaseMultipartGeometry):
            for element2 in iter_exploded_geometries(element1):
                yield element2
        else:
            yield element1
