import numpy as np
from shapely.geometry.multipolygon import MultiPolygon
from shapely.geometry.polygon import Polygon
from shapely.geometry.point import Point
from shapely.geometry.linestring import LineString
from shapely.geometry.multipoint import MultiPoint

from ocgis.util.helpers import make_poly


class Wrapper(object):
    """
    Wraps and unwraps WGS84 geometry objects.

    :param axis: The longitude value for the center axis.
    :type axis: float
    """

    def __init__(self, axis=0.0):
        self.axis = float(axis)
        self.right_clip = make_poly((-90, 90), (180, 360))
        self.left_clip = make_poly((-90, 90), (-180, 180))
        self.clip1 = make_poly((-90, 90), (-180, axis))
        self.clip2 = make_poly((-90, 90), (axis, 180))

    def unwrap(self, geom):
        """
        :param geom: The target geometry to unwrap (i.e. move to 0 to 360 spatial domain).
        :type geom: :class:`shapely.geometry.base.BaseGeometry`
        :rtype: :class:`shapely.geometry.base.BaseGeometry`
        """

        if type(geom) in [MultiPoint, Point]:
            return self._unwrap_point_(geom)

        assert type(geom) in [Polygon, MultiPolygon]

        # return the geometry iterator
        it = self._get_iter_(geom)
        # loop through the polygons determining if any coordinates need to be shifted and flag accordingly.
        adjust = False
        for polygon in it:
            coords = np.array(polygon.exterior.coords)
            if np.any(coords[:, 0] < self.axis):
                adjust = True
                break

        # wrap the polygon if requested
        if adjust:
            # loop through individual polygon components. doing this operation on multipolygons causes unexpected
            # behavior mostly due to the possibility of invalid geometries
            processed_geometries = []
            for to_process in self._get_iter_(geom):
                try:
                    assert to_process.is_valid
                except AssertionError:
                    # Attempt a simple buffering trick to make the geometry valid.
                    to_process = to_process.buffer(0)
                    assert to_process.is_valid

                # intersection with the two regions
                left = to_process.intersection(self.clip1)
                right = to_process.intersection(self.clip2)

                # pull out the right side polygons
                right_polygons = [poly for poly in self._get_iter_(right)]

                # adjust polygons falling the left window
                if isinstance(left, Polygon):
                    left_polygons = [self._unwrap_shift_(left)]
                else:
                    left_polygons = []
                    for polygon in left:
                        left_polygons.append(self._unwrap_shift_(polygon))

                # merge polygons into single unit
                try:
                    processed = MultiPolygon(left_polygons + right_polygons)
                except TypeError:
                    left = filter(lambda x: type(x) != LineString, left_polygons)
                    right = filter(lambda x: type(x) != LineString, right_polygons)
                    processed = MultiPolygon(left + right)

                # return a single polygon if the output multigeometry has only one component. otherwise, explode the
                # multigeometry
                if len(processed) == 1:
                    processed = [processed[0]]
                else:
                    processed = [i for i in processed]

                # hold for final adjustment
                processed_geometries += processed

            # convert to a multigeometry if there are more than one output
            if len(processed_geometries) > 1:
                ret = MultiPolygon(processed_geometries)
            else:
                ret = processed_geometries[0]

        ## if polygon does not need adjustment, just return it.
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
            # likely a MultiPolygon
            except AttributeError:
                polygons = np.empty(len(geom), dtype=object)
                for ii, polygon in enumerate(geom):
                    coords = np.array(polygon.exterior.coords)
                    coords[:, 0] = coords[:, 0] - 360
                    polygons[ii] = Polygon(coords)
                ret = MultiPolygon(list(polygons))
            return ret

        try:
            bounds = np.array(geom.bounds)
            # if the polygon column bounds are both greater than 180 degrees
            # shift the coordinates of the entire polygon
            if np.all([bounds[0] > 180, bounds[2] > 180]):
                new_geom = _shift_(geom)
            # if a polygon crosses the 180 axis, then the polygon will need to be split with intersection and recombined.
            elif bounds[1] <= 180 and bounds[2] > 180:
                left = [poly for poly in self._get_iter_(geom.intersection(self.left_clip))]
                right = [poly for poly in self._get_iter_(_shift_(geom.intersection(self.right_clip)))]
                try:
                    new_geom = MultiPolygon(left + right)
                except TypeError:
                    left = filter(lambda x: type(x) != LineString, left)
                    right = filter(lambda x: type(x) != LineString, right)
                    new_geom = MultiPolygon(left + right)
            # otherwise, the polygon coordinates are not affected by wrapping and the geometry may be passed through.
            else:
                new_geom = geom
        # likely a point type object
        except (AttributeError, TypeError):
            try:
                if geom.x > 180:
                    new_geom = Point(geom.x - 360, geom.y)
                else:
                    new_geom = geom
            # likely a MultiPoint
            except AttributeError:
                points = [None] * len(geom)
                for ii, point in enumerate(geom):
                    if point.x > 180:
                        new_point = Point(point.x - 360, point.y)
                    else:
                        new_point = point
                    points[ii] = new_point
                new_geom = MultiPoint(points)

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
                if pt.x < self.axis:
                    n = Point(pt.x + 360, pt.y)
                else:
                    n = pt
                pts.append(n)
            ret = MultiPoint(pts)
        else:
            if geom.x < self.axis:
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
