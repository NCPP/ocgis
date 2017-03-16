from abc import ABCMeta, abstractmethod

import numpy as np
from shapely.geometry.linestring import LineString
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon

from ocgis.base import AbstractOcgisObject
from ocgis.util.helpers import make_poly


class AbstractWrapper(AbstractOcgisObject):
    """Base class for wrapping objects."""

    __metaclass__ = ABCMeta

    def __init__(self, center_axis=0.0, wrap_axis=180.0):
        """
        :param float center_axis: The longitude value for the center axis.
        :param float wrap_axis: The longitude value for the wrap axis.
        """

        self.center_axis = float(center_axis)
        self.wrap_axis = float(wrap_axis)

    @abstractmethod
    def wrap(self, *args, **kwargs):
        pass

    @abstractmethod
    def unwrap(self, *args, **kwargs):
        pass


class CoordinateArrayWrapper(AbstractWrapper):
    """Wrap and unwrap spherical coordinate arrays."""

    def __init__(self, *args, **kwargs):
        """
        .. note:: Accepts all parameters to :class:`~ocgis.util.spatial.wrap.AbstractWrapper`.

        Additional arguments:

        :keyword bool inplace: (``=True``) If ``True``, modify the array in-place. Otherwise, return a deepcopy of the
         array.
        """

        self.inplace = kwargs.pop('inplace', True)
        super(CoordinateArrayWrapper, self).__init__(*args, **kwargs)

    def wrap(self, arr):
        """
        :param arr: The longitude coordinate array to wrap. Array must have spherical coordinates. Only one- or
         two-dimensional arrays are allowed. The array must have coordinates increasing from low to high:

        >>> arr = np.array([1., 90., 180., 270., 360.])

        :type arr: :class:`~numpy.core.multiarray.ndarray`
        :returns: An array copy if not in-place.
        :rtype: :class:`~numpy.core.multiarray.ndarray`
        """
        if not self.inplace:
            arr = arr.copy()

        ret = np.atleast_2d(arr)
        select = ret > self.wrap_axis
        ret[select] -= np.asscalar(np.array(360.).astype(ret.dtype))

        return ret

    def unwrap(self, arr):
        if not self.inplace:
            arr = arr.copy()

        ret = np.atleast_2d(arr)
        select = ret < self.center_axis
        ret[select] += np.asscalar(np.array(360.).astype(ret.dtype))

        return ret


class GeometryWrapper(AbstractWrapper):
    """
    Wraps and unwraps geometry objects with a spherical coordinate system.
    """

    def __init__(self, *args, **kwargs):
        super(GeometryWrapper, self).__init__(*args, **kwargs)

        self.right_clip = make_poly((-90, 90), (self.wrap_axis, 360))
        self.left_clip = make_poly((-90, 90), (-self.wrap_axis, self.wrap_axis))
        self.clip1 = make_poly((-90, 90), (-self.wrap_axis, self.center_axis))
        self.clip2 = make_poly((-90, 90), (self.center_axis, self.wrap_axis))

    def unwrap(self, geom):
        """
        :param geom: The target geometry to unwrap (i.e. move to 0 to 360 spatial domain).
        :type geom: :class:`shapely.geometry.base.BaseGeometry`
        :rtype: :class:`shapely.geometry.base.BaseGeometry`
        """

        if type(geom) in [MultiPoint, Point]:
            return self._unwrap_point_(geom)

        assert type(geom) in [Polygon, MultiPolygon]

        # Return the geometry iterator.
        it = self._get_iter_(geom)
        # Loop through the polygons determining if any coordinates need to be shifted and flag accordingly.
        adjust = False
        for polygon in it:
            coords = np.array(polygon.exterior.coords)
            if np.any(coords[:, 0] < self.center_axis):
                adjust = True
                break

        # Wrap the polygon if requested.
        if adjust:
            # Loop through individual polygon components. doing this operation on multipolygons causes unexpected
            # behavior mostly due to the possibility of invalid geometries.
            processed_geometries = []
            for to_process in self._get_iter_(geom):
                try:
                    assert to_process.is_valid
                except AssertionError:
                    # Attempt a simple buffering trick to make the geometry valid.
                    to_process = to_process.buffer(0)
                    assert to_process.is_valid

                # Intersection with the two regions.
                left = to_process.intersection(self.clip1)
                right = to_process.intersection(self.clip2)

                # Pull out the right side polygons.
                right_polygons = [poly for poly in self._get_iter_(right)]

                # Adjust polygons falling the left window.
                if isinstance(left, Polygon):
                    left_polygons = [self._unwrap_shift_(left)]
                else:
                    left_polygons = []
                    for polygon in left:
                        left_polygons.append(self._unwrap_shift_(polygon))

                # Merge polygons into single unit.
                try:
                    processed = MultiPolygon(left_polygons + right_polygons)
                except TypeError:
                    left = filter(lambda x: type(x) != LineString, left_polygons)
                    right = filter(lambda x: type(x) != LineString, right_polygons)
                    processed = MultiPolygon(left + right)

                # Return a single polygon if the output multigeometry has only one component. otherwise, explode the
                # multigeometry.
                if len(processed) == 1:
                    processed = [processed[0]]
                else:
                    processed = [i for i in processed]

                # Hold for final adjustment.
                processed_geometries += processed

            # Convert to a multigeometry if there are more than one output.
            if len(processed_geometries) > 1:
                ret = MultiPolygon(processed_geometries)
            else:
                ret = processed_geometries[0]

        # If polygon does not need adjustment, just return it.
        else:
            ret = geom

        return ret

    def wrap(self, geom):
        """
        :param geom: The target geometry to adjust.
        :type geom: :class:`shapely.geometry.base.BaseGeometry`
        :rtype: :class:`shapely.geometry.base.BaseGeometry`
        """

        def _shift_(geom):
            try:
                coords = np.array(geom.exterior.coords)
                coords[:, 0] = coords[:, 0] - 360
                ret = Polygon(coords)
            # Likely a MultiPolygon.
            except AttributeError:
                polygons = np.empty(len(geom), dtype=object)
                for ii, polygon in enumerate(geom):
                    coords = np.array(polygon.exterior.coords)
                    coords[:, 0] = coords[:, 0] - 360
                    polygons[ii] = Polygon(coords)
                ret = MultiPolygon(list(polygons))
            return ret

        if isinstance(geom, (Polygon, MultiPolygon)):
            bounds = np.array(geom.bounds)
            # If the polygon column bounds are both greater than 180 degrees shift the coordinates of the entire
            # polygon.
            if np.all([bounds[0] > self.wrap_axis, bounds[2] > self.wrap_axis]):
                new_geom = _shift_(geom)
            # If a polygon crosses the 180 axis, then the polygon will need to be split with intersection and
            # recombined.
            elif bounds[1] <= self.wrap_axis < bounds[2]:
                left = [poly for poly in self._get_iter_(geom.intersection(self.left_clip))]
                right = [poly for poly in self._get_iter_(_shift_(geom.intersection(self.right_clip)))]
                try:
                    new_geom = MultiPolygon(left + right)
                except TypeError:
                    left = filter(lambda x: type(x) != LineString, left)
                    right = filter(lambda x: type(x) != LineString, right)
                    new_geom = MultiPolygon(left + right)
            # Otherwise, the polygon coordinates are not affected by wrapping and the geometry may be passed through.
            else:
                new_geom = geom
        # Likely a point type object.
        else:
            if isinstance(geom, Point):
                if geom.x > 180:
                    new_geom = Point(geom.x - 360, geom.y)
                else:
                    new_geom = geom
            # Likely a MultiPoint.
            elif isinstance(geom, MultiPoint):
                points = [None] * len(geom)
                for ii, point in enumerate(geom):
                    if point.x > 180:
                        new_point = Point(point.x - 360, point.y)
                    else:
                        new_point = point
                    points[ii] = new_point
                new_geom = MultiPoint(points)
            else:
                raise NotImplementedError(geom)

        return new_geom

    @staticmethod
    def _get_iter_(geom):
        """
        :param geom: The target geometry to adjust.
        :type geom: :class:`shapely.geometry.base.BaseGeometry`
        :rtype: sequence of :class:`shapely.geometry.base.BaseGeometry` objects
        """

        try:
            it = iter(geom)
        except TypeError:
            it = [geom]
        return it

    def _unwrap_point_(self, geom):
        """
        :param geom: The target geometry to adjust.
        :type geom: :class:`shapely.geometry.base.BaseGeometry`
        :rtype: :class:`shapely.geometry.point.Point`
        """

        if isinstance(geom, MultiPoint):
            pts = []
            for pt in geom:
                if pt.x < self.center_axis:
                    n = Point(pt.x + 360, pt.y)
                else:
                    n = pt
                pts.append(n)
            ret = MultiPoint(pts)
        else:
            if geom.x < self.center_axis:
                ret = Point(geom.x + 360, geom.y)
            else:
                ret = geom
        return ret

    @staticmethod
    def _unwrap_shift_(polygon):
        """
        :param polygon: The target geometry to adjust.
        :type polygon: :class:`shapely.geometry.polygon.Polygon`
        :rtype: :class:`shapely.geometry.polygon.Polygon`
        """

        coords = np.array(polygon.exterior.coords)
        coords[:, 0] = coords[:, 0] + 360
        return Polygon(coords)


def apply_wrapping_index_map(imap, to_remap):
    """
    Using the indices map ``imap`` returned from :meth:`~ocgis.util.spatial.wrap.CoordinateArrayWrapper.wrap` or
    :meth:`~ocgis.util.spatial.wrap.CoordinateArrayWrapper.unwrap`, remap the array ``to_remap`` in-place.

    :param imap: The input integer indices array.
    :type imap: :class:`~numpy.core.multiarray.ndarray`
    :param to_remap: The input array with same dimension as ``imap``. The array will be reordered in-place.
    :type to_remap: :class:`~numpy.core.multiarray.ndarray`
    """

    imap = imap.reshape(-1)
    to_remap = to_remap.reshape(-1)
    to_remap[:] = to_remap[imap]
