from collections import deque, OrderedDict
import itertools
from copy import copy
import numpy as np

from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely.prepared import prep
from shapely.geometry.multipoint import MultiPoint
from shapely.geometry.multipolygon import MultiPolygon
from shapely import wkb
import fiona
from shapely.geometry.geo import mapping, shape

from ocgis.util.environment import ogr
import base
from ocgis.interface.base.crs import CFWGS84, CoordinateReferenceSystem, WGS84
from ocgis.util.helpers import iter_array, get_formatted_slice, get_reduced_slice, get_trimmed_array_by_mask, \
    get_added_slice, make_poly, set_name_attributes, get_extrapolated_corners_esmf, get_ocgis_corners_from_esmf_corners, \
    get_none_or_2d
from ocgis import constants, env
from ocgis.exc import EmptySubsetError, SpatialWrappingError, MultipleElementsFound, BoundsAlreadyAvailableError
from ocgis.util.ugrid.helpers import get_update_feature, write_to_netcdf_dataset


CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint


class GeomMapping(object):
    """Used to simulate a dictionary key look up for data stored in 2-d ndarrays."""

    def __init__(self, uid, value):
        self.uid = uid
        self.value = value

    def __getitem__(self, key):
        sel = self.uid == key
        return self.value[sel][0]


class SingleElementRetriever(object):
    """
    Simplifies access to a spatial dimension with a single element.

    :param sdim:
    :type sdim: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`
    """

    def __init__(self, sdim):
        try:
            assert (sdim.shape == (1, 1))
        except AssertionError:
            raise MultipleElementsFound(sdim)
        self.sdim = sdim

    @property
    def crs(self):
        return self.sdim.crs

    @property
    def geom(self):
        return self.sdim.abstraction_geometry.value[0, 0]

    @property
    def properties(self):
        return self.sdim.properties

    @property
    def uid(self):
        return self.sdim.uid[0, 0]


class SpatialDimension(base.AbstractUidDimension):
    """
    :param grid: :class:`ocgis.interface.base.dimension.spatial.SpatialGridDimension`
    :param crs: :class:`ocgis.crs.CoordinateReferenceSystem`
    :param abstraction: str
    :param geom: :class:`ocgis.interface.base.dimension.spatial.SpatialGeometryDimension`
    """

    _ndims = 2
    _attrs_slice = ('uid', 'grid', '_geom')

    def __init__(self, *args, **kwargs):
        self.grid = kwargs.pop('grid', None)
        self.crs = kwargs.pop('crs', None)
        self._geom = kwargs.pop('geom', None)

        # convert the input crs to CFWGS84 if they are equivalent
        if self.crs == CFWGS84():
            self.crs = CFWGS84()

        # remove row and col dimension keywords if they are present. we do not want to pass them to the superclass
        # constructor.
        row = kwargs.pop('row', None)
        col = kwargs.pop('col', None)

        # always provide a default name for iteration
        kwargs['name'] = kwargs.get('name') or 'spatial'
        kwargs['name_uid'] = kwargs.get('name_uid') or 'gid'

        # # attempt to build the geometry dimension
        point = kwargs.pop('point', None)
        polygon = kwargs.pop('polygon', None)
        geom_kwds = dict(point=point, polygon=polygon)
        if any([g is not None for g in geom_kwds.values()]):
            self._geom = SpatialGeometryDimension(**geom_kwds)

        # attempt to construct some core dimensions if they are not passed at initialization
        if self._grid is None and self._geom is None:
            self.grid = SpatialGridDimension(row=row, col=col)

        self._abstraction = kwargs.pop('abstraction', None)
        self.abstraction = self._abstraction

        super(SpatialDimension, self).__init__(*args, **kwargs)

    @property
    def abstraction(self):
        return self.geom.abstraction

    @abstraction.setter
    def abstraction(self, value):
        self._abstraction = value
        self.geom.abstraction = value

    @property
    def abstraction_geometry(self):
        return self.geom.get_highest_order_abstraction()

    @property
    def geom(self):
        if self._geom is None:
            if self.grid is None:
                msg = 'At least a grid is required to construct a geometry dimension.'
                raise ValueError(msg)
            else:
                self._geom = SpatialGeometryDimension(grid=self.grid, uid=self.grid.uid, abstraction=self._abstraction)
        return self._geom

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        if value is not None:
            assert (isinstance(value, SpatialGridDimension))
        self._grid = value

    @property
    def shape(self):
        if self.grid is None:
            ret = self.geom.shape
        else:
            ret = self.grid.shape
        return ret

    @property
    def single(self):
        return SingleElementRetriever(self)

    @property
    def weights(self):
        if self.geom is None:
            ret = self.grid.weights
        else:
            if self.geom.polygon is None:
                ret = self.geom.point.weights
            else:
                ret = self.geom.polygon.weights
        return ret

    @property
    def wrapped_state(self):
        try:
            ret = self.crs.get_wrapped_state(self)
        except AttributeError:
            ret = None
        return ret

    def assert_uniform_mask(self):
        """
        Check that the mask for the major spatial components are equivalent. This will only test loaded elements.

        :raises: AssertionError
        """

        to_compare = []
        if self._grid is not None:
            to_compare.append(self._grid.value[0].mask)
            to_compare.append(self._grid.value[1].mask)
        if self._geom is not None:
            if self._geom._point is not None:
                to_compare.append(self._geom._point.value.mask)
            if self._geom._polygon is not None:
                to_compare.append(self._geom._polygon.value.mask)
        to_compare.append(self.uid.mask)

        for arr1, arr2 in itertools.combinations(to_compare, 2):
            assert np.all(arr1 == arr2)

        # check the mask on corners
        if self._grid is not None and self._grid._corners is not None:
            corners_mask = self._grid._corners.mask
            for (ii, jj), mask_value in iter_array(to_compare[0], return_value=True):
                to_check = corners_mask[:, ii, jj, :]
                if mask_value:
                    assert to_check.all()
                else:
                    assert not to_check.any()

    @classmethod
    def from_records(cls, records, crs=None, uid=None):
        """
        Create a :class:`ocgis.interface.base.dimension.SpatialDimension` from Fiona-like records.

        :param records: A sequence of records returned from an Fiona file object.
        :type records: sequence
        :param crs: If ``None``, default to :attr:`~ocgis.env.DEFAULT_COORDSYS`.
        :type crs: dict or :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        :param str uid: If provided, use this attribute name as the unique identifier. Otherwise search for
         :attr:`env.DEFAULT_GEOM_UID` and, if not present, construct a 1-based identifier with this name.
        :returns: A spatial dimension object constructed from the records.
        :rtype: :class:`ocgis.interface.base.dimension.SpatialDimension`
        """

        if uid is None:
            uid = env.DEFAULT_GEOM_UID

        if not isinstance(crs, CoordinateReferenceSystem):
            # if there is no crs dictionary passed, assume WGS84
            crs = crs or env.DEFAULT_COORDSYS.value
            crs = CoordinateReferenceSystem(value=crs)

        # these are mappings used to construct the SpatialDimension
        mapping_geometry = {SpatialGeometryPolygonDimension: ('Polygon', 'MultiPolygon'),
                            SpatialGeometryPointDimension: ('Point', 'MultiPoint')}
        mapping_kwds = {SpatialGeometryPolygonDimension: 'polygon',
                        SpatialGeometryPointDimension: 'point'}

        # holds data types for the property structure array
        dtype = []
        # holds geometry objects
        deque_geoms = deque()
        # holds unique identifiers
        deque_uid = deque()

        build = True
        for ctr, record in enumerate(records, start=1):

            # get the geometry from a keyword present on the input dictionary or construct from the coordinates sequence
            try:
                current_geom = record['geom']
            except KeyError:
                current_geom = shape(record['geometry'])
            deque_geoms.append(current_geom)

            # this is to set up the properties array
            if build:
                build = False

                if uid in record['properties']:
                    has_uid = True
                else:
                    has_uid = False

                for k, v in record['properties'].iteritems():
                    the_type = type(v)
                    if the_type == unicode:
                        the_type = object
                    if isinstance(v, basestring):
                        the_type = object
                    dtype.append((str(k), the_type))
                properties = np.empty(0, dtype=dtype)
                property_order = record['properties'].keys()

            # the UGID may be present as a property. otherwise the enumeration counter is used for the identifier.
            if has_uid:
                to_append = int(record['properties'][uid])
            else:
                to_append = ctr
            deque_uid.append(to_append)

            # append to the properties array
            properties_new = np.empty(1, dtype=dtype)
            properties_new[0] = tuple([record['properties'][key] for key in property_order])
            properties = np.append(properties, properties_new)

        # fill the geometry array. to avoid having the geometry objects turned into coordinates, fill by index...
        geoms = np.empty((1, len(deque_geoms)), dtype=object)
        for idx in range(geoms.shape[1]):
            geoms[0, idx] = deque_geoms[idx]

        # convert the unique identifiers to an array
        uid_values = np.array(deque_uid).reshape(*geoms.shape)

        # this will choose the appropriate geometry dimension
        geom_type = geoms[0, 0].geom_type
        for k, v in mapping_geometry.iteritems():
            if geom_type in v:
                klass = k
                break

        # this constructs the geometry dimension
        dim_geom_type = klass(value=geoms)
        # arguments to geometry dimension
        kwds = {mapping_kwds[klass]: dim_geom_type}
        dim_geom = SpatialGeometryDimension(**kwds)

        sdim = SpatialDimension(geom=dim_geom, uid=uid_values, properties=properties, crs=crs,
                                abstraction=mapping_kwds[klass], name_uid=uid)

        return sdim

    def get_clip(self, polygon, return_indices=False, use_spatial_index=True, select_nearest=False):
        assert (type(polygon) in (Polygon, MultiPolygon))

        ret, slc = self.get_intersects(polygon, return_indices=True, use_spatial_index=use_spatial_index,
                                       select_nearest=select_nearest)

        # # clipping with points is okay...
        if ret.geom.polygon is not None:
            ref_value = ret.geom.polygon.value
        else:
            ref_value = ret.geom.point.value
        for (row_idx, col_idx), geom in iter_array(ref_value, return_value=True):
            ref_value[row_idx, col_idx] = geom.intersection(polygon)

        if return_indices:
            ret = (ret, slc)

        return (ret)

    def get_fiona_schema(self):
        """
        :returns: A :module:`fiona` schema dictionary.
        :rtype: dict
        """

        fproperties = OrderedDict()
        if self.properties is not None:
            from ocgis.conv.fiona_ import AbstractFionaConverter

            dtype = self.properties.dtype
            for idx, name in enumerate(dtype.names):
                fproperties[name] = AbstractFionaConverter.get_field_type(dtype[idx])
        schema = {'geometry': self.abstraction_geometry.geom_type,
                  'properties': fproperties}
        return schema

    def get_geom_iter(self, target=None, as_multipolygon=True):
        """
        :param str target: The target geometry. One of "point" or "polygon". If ``None``, return the highest order
         abstraction.
        :param bool as_multipolygon: If ``True``, convert all polygons to multipolygons.
        :returns: An iterator yielding a tuple: (int row index, int column index, Shapely geometry, int unique id)
        :rtype: tuple
        :raises: AttributeError
        """

        target = target or self.abstraction
        if target is None:
            value = self.geom.get_highest_order_abstraction().value
        else:
            try:
                value = getattr(self.geom, target).value
            except AttributeError:
                msg = 'The target abstraction "{0}" is not available.'.format(target)
                raise ValueError(msg)

        # no need to attempt and convert to MultiPolygon if we are working with point data.
        if as_multipolygon and target == 'point':
            as_multipolygon = False

        r_uid = self.uid
        for (row_idx, col_idx), geom in iter_array(value, return_value=True):
            if as_multipolygon:
                if isinstance(geom, Polygon):
                    geom = MultiPolygon([geom])
            uid = r_uid[row_idx, col_idx]
            yield (row_idx, col_idx, geom, uid)

    def get_intersects(self, polygon, return_indices=False, use_spatial_index=True, select_nearest=False):
        """
        :param polygon: The subset geometry objec to use for the intersects operation.
        :type polygon: :class:`shapely.geometry.polygon.Polygon` or :class:`shapely.geometry.multipolygon.MultiPolygon`
        :param bool return_indices: If ``True``, also return the slice objects used to slice the object.
        :param bool use_spatial_index: If ``True``, use an ``rtree`` spatial index.
        :param bool select_nearest: If ``True``, select the geometry nearest ``polygon`` using
         :meth:`shapely.geometry.base.BaseGeometry.distance`.
        :raises: ValueError, NotImplementedError
        :rtype: If ``return_indices`` is ``False``: :class:`ocgis.interface.base.dimension.spatial.SpatialDimension`.
         If ``return_indices`` is ``True``: (:class:`ocgis.interface.base.dimension.spatial.SpatialDimension`,
         (:class:`slice`, :class:`slice`))
        """
        assert not self.uid.mask.any()
        ret = copy(self)

        # based on the set spatial abstraction, decide if bounds should be used for subsetting the row and column
        # dimensions
        use_bounds = False if self.abstraction == 'point' else True

        if type(polygon) in (Point, MultiPoint):
            msg = 'Only Polygons and MultiPolygons are acceptable geometry types for intersects operations.'
            raise ValueError(msg)
        elif type(polygon) in (Polygon, MultiPolygon):
            # for a polygon subset, first the grid is subsetted by the bounds of the polygon object. the intersects
            # operations is then performed on the polygon/point representation as appropriate.
            if self.grid is None:
                if self.geom.polygon is not None:
                    target_geom = self.geom.polygon
                else:
                    target_geom = self.geom.point
                masked_geom = target_geom.get_intersects_masked(polygon, use_spatial_index=use_spatial_index)
                ret_slc = np.where(masked_geom.value.mask == False)
                ret = ret[ret_slc[0], ret_slc[1]]
            else:
                minx, miny, maxx, maxy = polygon.bounds
                # subset the grid by its bounding box
                ret.grid, slc = self.grid.get_subset_bbox(minx, miny, maxx, maxy, return_indices=True,
                                                          use_bounds=use_bounds)

                # slice the geometries if they are available
                if ret._geom is not None:
                    ret._geom = ret._geom[slc[0], slc[1]]

                # update the unique identifier to copy the grid uid
                ret.uid = ret.grid.uid
                assert not self.uid.mask.any()
                # attempt to mask the polygons if the abstraction is point or none
                if self.geom.polygon is not None and self.abstraction in ['polygon', None]:
                    ret._geom._polygon = ret.geom.polygon.get_intersects_masked(polygon,
                                                                                use_spatial_index=use_spatial_index)
                    grid_mask = ret.geom.polygon.value.mask
                else:
                    ret._geom._point = ret.geom.point.get_intersects_masked(polygon,
                                                                            use_spatial_index=use_spatial_index)
                    grid_mask = ret.geom.point.value.mask
                assert not self.uid.mask.any()
                ret.grid.value.unshare_mask()
                # transfer the geometry mask to the grid mask
                ret.grid.value.mask[:, :, :] = grid_mask.copy()
                # transfer the geometry mask to the grid uid mask
                ret.grid.uid.unshare_mask()
                ret.grid.uid.mask = grid_mask.copy()
                # also transfer the mask to corners
                if ret.grid._corners is not None:
                    ret.grid.corners.unshare_mask()
                    ref = ret.grid.corners.mask
                    for (ii, jj), mask_value in iter_array(grid_mask, return_value=True):
                        ref[:, ii, jj, :] = mask_value

                # barbed and circular geometries may result in rows and or columns being entirely masked. these rows and
                # columns should be trimmed.
                _, adjust = get_trimmed_array_by_mask(ret.get_mask(), return_adjustments=True)
                # use the adjustments to trim the returned data object
                ret = ret[adjust['row'], adjust['col']]

                # adjust the returned slices
                if return_indices and not select_nearest:
                    ret_slc = [None, None]
                    ret_slc[0] = get_added_slice(slc[0], adjust['row'])
                    ret_slc[1] = get_added_slice(slc[1], adjust['col'])

        else:
            raise NotImplementedError

        assert not self.uid.mask.any()

        if select_nearest:
            if self.geom.polygon is not None and self.abstraction in ['polygon', None]:
                target_geom = ret.geom.polygon.value
            else:
                target_geom = ret.geom.point.value
            distances = {}
            centroid = polygon.centroid
            for select_nearest_index, geom in iter_array(target_geom, return_value=True):
                distances[centroid.distance(geom)] = select_nearest_index
            select_nearest_index = distances[min(distances.keys())]
            ret = ret[select_nearest_index[0], select_nearest_index[1]]
            ret_slc = np.where(self.uid.data == ret.uid.data)

        if return_indices:
            ret = (ret, tuple(ret_slc))

        return ret

    def get_mask(self):
        """
        :returns: A deepcopy of a the boolean mask used on the spatial dimension.
        :rtype: :class:`numpy.ndarray`
        :raises: ValueError
        """

        if self.grid is None:
            if self.geom.point is None:
                ret = self.geom.polygon.value.mask
            else:
                ret = self.geom.point.value.mask
        else:
            ret = self.grid.value.mask[0, :, :]
        return ret.copy()

    def get_report(self):
        try:
            res = self.grid.resolution
            extent = self.grid.extent
        except AttributeError:
            if self.grid is None:
                res = 'NA (no grid present)'
                extent = 'NA (no grid present)'
            else:
                raise

        itype = self.geom.get_highest_order_abstraction().__class__.__name__
        if self.crs is None:
            projection = 'NA (no coordinate system)'
            sref = projection
        else:
            projection = self.crs.sr.ExportToProj4()
            sref = self.crs.__class__.__name__

        lines = ['Spatial Reference = {0}'.format(sref),
                 'Proj4 String = {0}'.format(projection),
                 'Extent = {0}'.format(extent),
                 'Geometry Interface = {0}'.format(itype),
                 'Resolution = {0}'.format(res),
                 'Count = {0}'.format(self.uid.reshape(-1).shape[0])]

        return lines

    def set_mask(self, mask):
        """
        :param mask: The spatial mask to apply to available representations.
        :type mask: boolean :class:`numpy.core.multiarray.ndarray`
        """

        if self._grid is not None:
            self._grid.value.mask[0, :, :] = mask.copy()
            self._grid.value.mask[1, :, :] = mask.copy()
            self._grid.uid.mask = mask.copy()
            if self._grid._corners is not None:
                ref_corners_mask = self._grid._corners.mask
                for idx_rc in range(2):
                    for (idx_row, idx_col), v in iter_array(mask, return_value=True):
                        ref_corners_mask[idx_rc, idx_row, idx_col, :] = v
        if self._geom is not None:
            if self._geom._point is not None:
                self._geom._point.value.mask[:] = mask.copy()
                self._geom._point.uid.mask = mask.copy()
            if self._geom._polygon is not None:
                self._geom._polygon.value.mask[:] = mask.copy()
                self._geom._polygon.uid.mask = mask.copy()
            self._geom.grid = self._grid

    def unwrap(self):
        try:
            self.crs.unwrap(self)
        except AttributeError:
            if self.crs is None or self.crs != WGS84():
                msg = 'Only WGS84 coordinate systems may be unwrapped.'
                raise (SpatialWrappingError(msg))

    def update_crs(self, to_crs):
        """
        Update the coordinate system in place.

        :param to_crs: The destination coordinate system.
        :type to_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        """

        assert self.crs is not None

        try:
            # if the crs values are the same, pass through
            if to_crs != self.crs:
                to_sr = to_crs.sr
                from_sr = self.crs.sr

                if self.grid is not None:
                    # update grid values
                    value_row = self.grid.value.data[0].reshape(-1)
                    value_col = self.grid.value.data[1].reshape(-1)
                    self._update_crs_with_geometry_collection_(to_sr, value_row, value_col)
                    self.grid.value.data[0] = value_row.reshape(*self.grid.shape)
                    self.grid.value.data[1] = value_col.reshape(*self.grid.shape)

                    if self.grid.corners is not None:
                        # update the corners
                        corner_row = self.grid.corners.data[0].reshape(-1)
                        corner_col = self.grid.corners.data[1].reshape(-1)
                        self._update_crs_with_geometry_collection_(to_sr, corner_row, corner_col)

                    self.grid.row = None
                    self.grid.col = None

                if self._geom is not None:
                    if self.geom._point is not None:
                        self.geom.point.update_crs(to_crs, self.crs)
                    if self.geom._polygon is not None:
                        self.geom.polygon.update_crs(to_crs, self.crs)

                self.crs = to_crs

        # likely a rotated pole coordinate system.
        except RuntimeError as e:
            try:
                _crs = self.crs
                """:type: ocgis.interface.base.crs.CFRotatedPole"""
                new_spatial = _crs.get_rotated_pole_transformation(self)
            # likely an inverse transformation if the destination crs is rotated pole.
            except AttributeError:
                try:
                    new_spatial = to_crs.get_rotated_pole_transformation(self, inverse=True)
                except AttributeError:
                    raise e
            self.__dict__ = new_spatial.__dict__
            self.crs = to_crs

    def wrap(self):
        try:
            self.crs.wrap(self)
        except AttributeError:
            if self.crs is None or self.crs != WGS84():
                msg = 'Only WGS84 coordinate systems may be wrapped.'
                raise (SpatialWrappingError(msg))

    def write_fiona(self, path, target='polygon', driver='ESRI Shapefile'):
        attr = getattr(self.geom, target)
        attr.write_fiona(path, self.crs.value, driver=driver)
        return path

    def _format_uid_(self, value):
        return np.atleast_2d(value)

    def _get_sliced_properties_(self, slc):
        if self.properties is not None:
            # # determine major axis
            major = self.shape.index(max(self.shape))
            return self.properties[slc[major]]
        else:
            return None

    def _get_uid_(self):
        if self._geom is not None:
            ret = self._geom.uid
        else:
            ret = self.grid.uid
        return ret

    def _update_crs_with_geometry_collection_(self, to_sr, value_row, value_col):
        """
        Update coordinate vectors in place to match the destination coordinate system.

        :param to_sr: The destination coordinate system.
        :type to_sr: :class:`osgeo.osr.SpatialReference`
        :param value_row: Vector of row or Y values.
        :type value_row: :class:`numpy.ndarray`
        :param value_col: Vector of column or X values.
        :type value_col: :class:`numpy.ndarray`
        """

        # build the geometry collection
        geomcol = Geometry(wkbGeometryCollection)
        for ii in range(value_row.shape[0]):
            point = Geometry(wkbPoint)
            point.AddPoint(value_col[ii], value_row[ii])
            geomcol.AddGeometry(point)
        geomcol.AssignSpatialReference(self.crs.sr)
        geomcol.TransformTo(to_sr)
        for ii, geom in enumerate(geomcol):
            value_col[ii] = geom.GetX()
            value_row[ii] = geom.GetY()


class SpatialGridDimension(base.AbstractUidValueDimension):
    _ndims = 2
    _attrs_slice = None
    _name_row = None


    def __init__(self, *args, **kwargs):
        self._corners = None

        self.row = kwargs.pop('row', None)
        self.col = kwargs.pop('col', None)

        self.corners = kwargs.pop('corners', None)

        kwargs['name'] = kwargs.get('name') or 'grid'

        self.name_row = kwargs.pop('name_row', constants.DEFAULT_NAME_ROW_COORDINATES)
        self.name_col = kwargs.pop('name_col', constants.DEFAULT_NAME_COL_COORDINATES)

        super(SpatialGridDimension, self).__init__(*args, **kwargs)

        self._validate_()

        # set names of row and column if available
        name_mapping = {self.row: 'yc', self.col: 'xc'}
        set_name_attributes(name_mapping)

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, 2)

        uid = self.uid[slc]

        if self._value is not None:
            value = self._value[:, slc[0], slc[1]]
        else:
            value = None

        if self.row is not None:
            row = self.row[slc[0]]
            col = self.col[slc[1]]
        else:
            row = None
            col = None

        ret = copy(self)

        if ret._corners is not None:
            ret._corners = ret._corners[:, slc[0], slc[1], :]

        ret.uid = uid
        ret._value = value
        ret.row = row
        ret.col = col

        return ret

    @property
    def corners(self):
        """
        2 x row x column x 4

        2 = y, x or row, column
        row
        column
        4 = ul, ur, lr, ll
        """

        if self._corners is None:
            if self.row is None or self.col is None:
                pass
            elif self.row.bounds is None or self.col.bounds is None:
                pass
            else:
                fill = np.zeros([2] + list(self.shape) + [4], dtype=self.row.value.dtype)
                col_bounds = self.col.bounds
                row_bounds = self.row.bounds
                for ii, jj in itertools.product(range(self.shape[0]), range(self.shape[1])):
                    fill_element = fill[:, ii, jj]
                    fill_element[:, 0] = row_bounds[ii, 0], col_bounds[jj, 0]
                    fill_element[:, 1] = row_bounds[ii, 0], col_bounds[jj, 1]
                    fill_element[:, 2] = row_bounds[ii, 1], col_bounds[jj, 1]
                    fill_element[:, 3] = row_bounds[ii, 1], col_bounds[jj, 0]

                mask_value = self.value.mask
                mask_fill = np.zeros(fill.shape, dtype=bool)
                for (ii, jj), m in iter_array(mask_value[0, :, :], return_value=True):
                    mask_fill[:, ii, jj, :] = m
                fill = np.ma.array(fill, mask=mask_fill)

                self._corners = fill

        return self._corners

    @corners.setter
    def corners(self, value):
        if value is not None:
            if not isinstance(value, np.ma.MaskedArray):
                value = np.ma.array(value, mask=False)
            assert value.ndim == 4
            assert value.shape[3] == 4
        self._corners = value

    @property
    def corners_esmf(self):
        fill = np.zeros([2] + [element + 1 for element in self.shape], dtype=self.value.dtype)
        range_row = range(self.shape[0])
        range_col = range(self.shape[1])
        _corners = self.corners
        for ii, jj in itertools.product(range_row, range_col):
            ref = fill[:, ii:ii + 2, jj:jj + 2]
            ref[:, 0, 0] = _corners[:, ii, jj, 0]
            ref[:, 0, 1] = _corners[:, ii, jj, 1]
            ref[:, 1, 1] = _corners[:, ii, jj, 2]
            ref[:, 1, 0] = _corners[:, ii, jj, 3]
        return fill

    @property
    def extent(self):
        if self.row is None:
            if self.corners is not None:
                minx = self.corners[1].min()
                miny = self.corners[0].min()
                maxx = self.corners[1].max()
                maxy = self.corners[0].max()
            else:
                minx = self.value[1, :, :].min()
                miny = self.value[0, :, :].min()
                maxx = self.value[1, :, :].max()
                maxy = self.value[0, :, :].max()
        else:
            if self.row.bounds is None:
                minx = self.col.value.min()
                miny = self.row.value.min()
                maxx = self.col.value.max()
                maxy = self.row.value.max()
            else:
                minx = self.col.bounds.min()
                miny = self.row.bounds.min()
                maxx = self.col.bounds.max()
                maxy = self.row.bounds.max()
        return minx, miny, maxx, maxy

    @property
    def extent_polygon(self):
        minx, miny, maxx, maxy = self.extent
        return make_poly([miny, maxy], [minx, maxx])

    @property
    def resolution(self):
        try:
            ret = np.mean([self.row.resolution, self.col.resolution])
        except AttributeError:
            resolution_limit = int(constants.RESOLUTION_LIMIT) / 2
            r_value = self.value[:, 0:resolution_limit, 0:resolution_limit]
            rows = np.mean(np.diff(r_value[0, :, :], axis=0))
            cols = np.mean(np.diff(r_value[1, :, :], axis=1))
            ret = np.mean([rows, cols])
        return ret

    @property
    def shape(self):
        try:
            ret = (len(self.row), len(self.col))
        # occurs if either of these are empty. get the shape from the grid value.
        except TypeError:
            ret = (self.uid.shape[0], self.uid.shape[1])
        return ret

    def get_subset_bbox(self, min_col, min_row, max_col, max_row, return_indices=False, closed=True,
                        use_bounds=True):
        assert (min_row <= max_row)
        assert (min_col <= max_col)

        if self.row is None:
            r_row = self.value[0, :, :]
            real_idx_row = np.arange(0, r_row.shape[0])
            r_col = self.value[1, :, :]
            real_idx_col = np.arange(0, r_col.shape[1])

            if closed:
                lower_row = r_row > min_row
                upper_row = r_row < max_row
                lower_col = r_col > min_col
                upper_col = r_col < max_col
            else:
                lower_row = r_row >= min_row
                upper_row = r_row <= max_row
                lower_col = r_col >= min_col
                upper_col = r_col <= max_col

            idx_row = np.logical_and(lower_row, upper_row)
            idx_col = np.logical_and(lower_col, upper_col)

            keep_row = np.any(idx_row, axis=1)
            keep_col = np.any(idx_col, axis=0)

            # # slice reduction may fail due to empty bounding box returns. catch
            # # these value errors and repurpose as subset errors.
            try:
                row_slc = get_reduced_slice(real_idx_row[keep_row])
            except ValueError:
                if real_idx_row[keep_row].shape[0] == 0:
                    raise (EmptySubsetError(origin='Y'))
                else:
                    raise
            try:
                col_slc = get_reduced_slice(real_idx_col[keep_col])
            except ValueError:
                if real_idx_col[keep_col].shape[0] == 0:
                    raise (EmptySubsetError(origin='X'))
                else:
                    raise

            new_mask = np.invert(np.logical_or(idx_row, idx_col)[row_slc, col_slc])

        else:
            new_row, row_indices = self.row.get_between(min_row, max_row, return_indices=True, closed=closed,
                                                        use_bounds=use_bounds)
            new_col, col_indices = self.col.get_between(min_col, max_col, return_indices=True, closed=closed,
                                                        use_bounds=use_bounds)
            row_slc = get_reduced_slice(row_indices)
            col_slc = get_reduced_slice(col_indices)

        ret = self[row_slc, col_slc]

        try:
            grid_mask = np.zeros((2, new_mask.shape[0], new_mask.shape[1]), dtype=bool)
            grid_mask[:, :, :] = new_mask
            ret._value = np.ma.array(ret._value, mask=grid_mask)
            ret.uid = np.ma.array(ret.uid, mask=new_mask)
        except UnboundLocalError:
            if self.row is not None:
                pass
            else:
                raise

        if return_indices:
            ret = (ret, (row_slc, col_slc))

        return ret

    def set_extrapolated_corners(self):
        """
        Extrapolate corners from grid centroids. If corners are already available, an exception will be raised.

        :raises: BoundsAlreadyAvailableError
        """

        if self.corners is not None:
            raise BoundsAlreadyAvailableError
        else:
            data = self.value.data
            corners_esmf = get_extrapolated_corners_esmf(data[0])
            corners_esmf.resize(*list([2] + list(corners_esmf.shape)))
            corners_esmf[1, :, :] = get_extrapolated_corners_esmf(data[1])
            corners = get_ocgis_corners_from_esmf_corners(corners_esmf)

        # update the corners mask if there are masked values
        if self.value.mask.any():
            idx_true = np.where(self.value.mask[0] == True)
            corners.mask[:, idx_true[0], idx_true[1], :] = True

        self.corners = corners

    def write_to_netcdf_dataset(self, dataset, **kwargs):
        """
        :param dataset:
        :type dataset: :class:`netCDF4.Dataset`
        """

        try:
            self.row.write_to_netcdf_dataset(dataset, **kwargs)
            self.col.write_to_netcdf_dataset(dataset, **kwargs)
        except AttributeError:
            # likely no row and column. write the grid value.
            name_yc = self.name_row
            name_xc = self.name_col
            dataset.createDimension(name_yc, size=self.shape[0])
            dataset.createDimension(name_xc, size=self.shape[1])
            value = self.value
            dimensions = (name_yc, name_xc)
            yc = dataset.createVariable(name_yc, value.dtype, dimensions=dimensions)
            yc[:] = value[0, :, :]
            yc.axis = 'Y'
            xc = dataset.createVariable(name_xc, value.dtype, dimensions=dimensions)
            xc[:] = value[1, :, :]
            xc.axis = 'X'

            if self.corners is not None:
                corners = self.corners
                ncorners = constants.DEFAULT_NAME_CORNERS_DIMENSION
                dataset.createDimension(ncorners, size=4)
                name_yc_corner = '{0}_corners'.format(name_yc)
                name_xc_corner = '{0}_corners'.format(name_xc)
                dimensions = (name_yc, name_xc, ncorners)
                for idx, name in zip([0, 1], [name_yc_corner, name_xc_corner]):
                    var = dataset.createVariable(name, corners.dtype, dimensions=dimensions)
                    var[:] = corners[idx]
                yc.corners = name_yc_corner
                xc.corners = name_xc_corner

    def _validate_(self):
        if self._value is None:
            if self.row is None or self.col is None:
                msg = 'Without a value, a row and column dimension are required.'
                raise ValueError(msg)

    def _format_private_value_(self, value):
        if value is None:
            ret = None
        else:
            assert len(value.shape) == 3
            assert value.shape[0] == 2
            assert isinstance(value, np.ma.MaskedArray)
            ret = value
        return ret

    def _get_uid_(self, shp=None):
        if shp is None:
            if self._value is None:
                shp = len(self.row), len(self.col)
            else:
                shp = self._value.shape[1], self._value.shape[2]
        ret = np.arange(1, (shp[0] * shp[1]) + 1, dtype=constants.NP_INT).reshape(shp)
        ret = np.ma.array(ret, mask=False)
        return ret

    def _get_value_(self):
        # assert types of row and column are equivalent
        if self.row.value.dtype != self.col.value.dtype:
            self.col._value = self.col._value.astype(self.row.value.dtype)
        # fill the centroids
        fill = np.empty((2, self.row.shape[0], self.col.shape[0]), dtype=self.row.value.dtype)
        fill = np.ma.array(fill, mask=False)
        col_coords, row_coords = np.meshgrid(self.col.value, self.row.value)
        fill[0, :, :] = row_coords
        fill[1, :, :] = col_coords
        return fill


class SpatialGeometryDimension(base.AbstractUidDimension):
    _ndims = 2
    _attrs_slice = ('uid', 'grid', '_point', '_polygon')

    def __init__(self, *args, **kwargs):
        self.grid = kwargs.pop('grid', None)
        self._point = kwargs.pop('point', None)
        self._polygon = kwargs.pop('polygon', None)
        self._abstraction = kwargs.pop('abstraction', None)

        kwargs['name'] = kwargs.get('name') or 'geometry'

        super(SpatialGeometryDimension, self).__init__(*args, **kwargs)

        if self.grid is None and self._point is None and self._polygon is None:
            msg = 'At minimum, a grid, point, or polygon dimension is required.'
            raise ValueError(msg)

    @property
    def abstraction(self):
        return self._abstraction

    @abstraction.setter
    def abstraction(self, value):
        options = ['point', 'polygon', None]
        if value not in options:
            raise ValueError('Must be one of: {0}'.format(options))
        # reset polygons if the point abstraction is set.
        if value == 'point':
            self._polygon = None
        self._abstraction = value

    @property
    def point(self):
        if self._point is None and self.grid is not None:
            self._point = SpatialGeometryPointDimension(grid=self.grid, uid=self.grid.uid)
        return self._point

    @property
    def polygon(self):
        if self._polygon is None:
            if self.abstraction in ['polygon', None]:
                if self.grid is not None:
                    try:
                        self._polygon = SpatialGeometryPolygonDimension(grid=self.grid, uid=self.grid.uid)
                    except ValueError:
                        none_bounds_row = self.grid.row is None or self.grid.row.bounds is None
                        none_bounds_col = self.grid.col is None or self.grid.col.bounds is None
                        if any([none_bounds_row, none_bounds_col]):
                            pass
                        else:
                            raise
        return self._polygon

    @property
    def shape(self):
        if self.point is None:
            ret = self.polygon.shape
        else:
            ret = self.point.shape
        return ret

    def get_highest_order_abstraction(self):
        """
        :returns: Return the highest order abstraction geometry with preference given by:
         1. Polygon
         2. Point
        :rtype: :class:`~ocgis.interface.base.dimension.spatial.SpatialGeometryDimension`
        """

        if self.abstraction == 'point':
            ret = self.point
        elif self.abstraction == 'polygon':
            ret = self.polygon
        else:
            if self.polygon is None:
                ret = self.point
            else:
                ret = self.polygon

        if ret is None:
            msg = 'No abstraction geometry found. Is "abstraction" compatible with the geometries available?'
            raise ValueError(msg)

        return ret

    def get_iter(self):
        raise NotImplementedError

    def _get_uid_(self):
        if self._point is not None:
            ret = self._point.uid
        elif self._polygon is not None:
            ret = self._polygon.uid
        else:
            ret = self.grid.uid
        return ret


class SpatialGeometryPointDimension(base.AbstractUidValueDimension):
    """
    :keyword str geom_type: (``=None``) If ``None``, default to :attrs:`ocgis.interface.base.dimension.spatial.SpatialGeometryPointDimension.__geom_type_default`.
     If ``'auto'``, automatically determine the geometry type from the value data.
    """

    _ndims = 2
    _attrs_slice = ('uid', '_value', 'grid')
    _geom_type_default = 'Point'

    def __init__(self, *args, **kwargs):
        self._geom_type = None

        self.grid = kwargs.pop('grid', None)
        self.geom_type = kwargs.pop('geom_type', None) or self._geom_type_default

        super(SpatialGeometryPointDimension, self).__init__(*args, **kwargs)

        if self.name is None:
            self.name = self.geom_type.lower()

    @property
    def geom_type(self):
        if self._geom_type == 'auto':
            for geom in self.value.data.flat:
                if geom.geom_type.startswith('Multi'):
                    break
            self._geom_type = geom.geom_type
        return self._geom_type

    @geom_type.setter
    def geom_type(self, value):
        self._geom_type = value

    @property
    def weights(self):
        ret = np.ones(self.value.shape, dtype=constants.NP_FLOAT)
        ret = np.ma.array(ret, mask=self.value.mask)
        return ret

    def get_intersects_masked(self, polygon, use_spatial_index=True):
        """
        :param polygon: The Shapely geometry to use for subsetting.
        :type polygon: :class:`shapely.geometry.Polygon' or :class:`shapely.geometry.MultiPolygon'
        :param bool use_spatial_index: If ``False``, do not use the :class:`rtree.index.Index` for spatial subsetting.
         If the geometric case is simple, it may marginally improve execution times to turn this off. However, turning
         this off for a complex case will negatively impact (significantly) spatial operation execution times.
        :raises: NotImplementedError, EmptySubsetError
        :returns: :class:`ocgis.interface.base.dimension.spatial.SpatialGeometryPointDimension`
        """

        # only polygons are acceptable for subsetting. if a point is required, buffer it.
        if type(polygon) not in (Polygon, MultiPolygon):
            raise NotImplementedError(type(polygon))

        # return a shallow copy of self
        ret = copy(self)
        # create the fill array and reference the mask. this is the output geometry value array.
        fill = np.ma.array(ret.value, mask=True)
        ref_fill_mask = fill.mask

        # this is the path if a spatial index is used.
        if use_spatial_index:
            # keep this as a local import as it is not a required dependency
            from ocgis.util.spatial.index import SpatialIndex
            # create the index object and reference import members
            si = SpatialIndex()
            _add = si.add
            _value = self.value
            # add the geometries to the index
            for (ii, jj), id_value in iter_array(self.uid, return_value=True):
                _add(id_value, _value[ii, jj])
            # this mapping simulates a dictionary for the item look-ups from two-dimensional arrays
            geom_mapping = GeomMapping(self.uid, self.value)
            _uid = ret.uid
            # return the identifiers of the objects intersecting the target geometry and update the mask accordingly
            for intersect_id in si.iter_intersects(polygon, geom_mapping, keep_touches=False):
                sel = _uid == intersect_id
                ref_fill_mask[sel] = False
        # this is the slower simpler case
        else:
            # prepare the polygon for faster spatial operations
            prepared = prep(polygon)
            # we are not keeping touches at this point. remember the mask is an inverse.
            for (ii, jj), geom in iter_array(self.value, return_value=True):
                bool_value = False
                if prepared.intersects(geom):
                    if polygon.touches(geom):
                        bool_value = True
                else:
                    bool_value = True
                ref_fill_mask[ii, jj] = bool_value

        # if everything is masked, this is an empty subset
        if ref_fill_mask.all():
            raise EmptySubsetError(self.name)

        # set the returned value to the fill array
        ret._value = fill
        # also update the unique identifier array
        ret.uid = np.ma.array(ret.uid, mask=fill.mask.copy())

        return ret

    def update_crs(self, to_crs, from_crs):
        """
        :type to_crs: :class:`ocgis.crs.CoordinateReferenceSystem`
        :type from_crs: :class:`ocgis.crs.CoordinateReferenceSystem`
        """

        # be sure and project masked geometries to maintain underlying geometries for masked values.
        r_value = self.value.data
        r_loads = wkb.loads
        to_sr = to_crs.sr
        from_sr = from_crs.sr
        for (idx_row, idx_col), geom in iter_array(r_value, return_value=True, use_mask=False):
            ogr_geom = CreateGeometryFromWkb(geom.wkb)
            ogr_geom.AssignSpatialReference(from_sr)
            ogr_geom.TransformTo(to_sr)
            r_value[idx_row, idx_col] = r_loads(ogr_geom.ExportToWkb())

    def write_fiona(self, path, crs, driver='ESRI Shapefile'):
        schema = {'geometry': self.geom_type,
                  'properties': {'UGID': 'int'}}
        ref_prep = self._write_fiona_prep_geom_
        ref_uid = self.uid

        with fiona.open(path, 'w', driver=driver, crs=crs, schema=schema) as f:
            for (ii, jj), geom in iter_array(self.value, return_value=True):
                geom = ref_prep(geom)
                uid = int(ref_uid[ii, jj])
                feature = {'properties': {'UGID': uid}, 'geometry': mapping(geom)}
                f.write(feature)

        return path

    @staticmethod
    def _write_fiona_prep_geom_(geom):
        return geom

    def _format_private_value_(self, value):
        if value is not None:
            try:
                assert (len(value.shape) == 2)
                ret = value
            except (AssertionError, AttributeError):
                msg = 'Geometry values must come in as 2-d NumPy arrays to avoid array interface modifications by shapely.'
                raise ValueError(msg)
        else:
            ret = None
        ret = self._get_none_or_array_(ret, masked=True)
        return ret

    def _format_slice_state_(self, state, slc):
        state._value = get_none_or_2d(state._value)
        return state

    def _get_geometry_fill_(self, shape=None):
        if shape is None:
            shape = (self.grid.shape[0], self.grid.shape[1])
            mask = self.grid.value[0].mask
        else:
            mask = False
        fill = np.ma.array(np.zeros(shape), mask=mask, dtype=object)

        return fill

    def _get_value_(self):
        # we are interested in creating geometries for all the underlying coordinates regardless if the data is masked
        ref_grid = self.grid.value.data

        fill = self._get_geometry_fill_()
        r_data = fill.data
        for idx_row, idx_col in iter_array(ref_grid[0], use_mask=False):
            y = ref_grid[0, idx_row, idx_col]
            x = ref_grid[1, idx_row, idx_col]
            pt = Point(x, y)
            r_data[idx_row, idx_col] = pt
        return fill


class SpatialGeometryPolygonDimension(SpatialGeometryPointDimension):
    _geom_type_default = 'MultiPolygon'

    def __init__(self, *args, **kwargs):
        kwargs['name'] = kwargs.get('name') or 'polygon'
        super(SpatialGeometryPolygonDimension, self).__init__(*args, **kwargs)

        if self._value is None:
            # we can construct from a grid dimension having bounds
            if self.grid is None:
                msg = 'A grid dimension is required for constructing a polygon dimension without a value.'
                raise ValueError(msg)
            else:
                # corners may also be used to construct polygons. if they are not immediately available, check for
                # bounds are on the row and column.
                none_bounds_row = self.grid.row is None or self.grid.row.bounds is None
                none_bounds_col = self.grid.col is None or self.grid.col.bounds is None
                should_raise = True
                if any([none_bounds_row, none_bounds_col]):
                    if self.grid.corners is not None:
                        should_raise = False
                else:
                    should_raise = False
                if should_raise:
                    msg = 'Row/column bounds or grid corners are required to construct polygons.'
                    raise ValueError(msg)

    @property
    def area(self):
        r_value = self.value
        fill = np.ones(r_value.shape, dtype=constants.NP_FLOAT)
        fill = np.ma.array(fill, mask=r_value.mask)
        for (ii, jj), geom in iter_array(r_value, return_value=True):
            fill[ii, jj] = geom.area
        return fill

    @property
    def weights(self):
        return self.area / self.area.max()

    def write_to_netcdf_dataset_ugrid(self, dataset):
        """
        Write a UGRID formatted netCDF4 file following conventions: https://github.com/ugrid-conventions/ugrid-conventions/tree/v0.9.0

        :param dataset: An open netCDF4 dataset object.
        :type dataset: :class:`netCDF4.Dataset`
        """

        def _iter_features_():
            for ctr, geom in enumerate(self.value.compressed()):
                yld = {'geometry': {'type': geom.geom_type, 'coordinates': [np.array(geom.exterior.coords).tolist()]}}
                yld = get_update_feature(ctr, yld)
                yield yld

        write_to_netcdf_dataset(dataset, list(_iter_features_()))

    def _get_value_(self):
        fill = self._get_geometry_fill_()
        r_data = fill.data
        try:
            ref_row_bounds = self.grid.row.bounds
            ref_col_bounds = self.grid.col.bounds
            for idx_row, idx_col in itertools.product(range(ref_row_bounds.shape[0]), range(ref_col_bounds.shape[0])):
                row_min, row_max = ref_row_bounds[idx_row, :].min(), ref_row_bounds[idx_row, :].max()
                col_min, col_max = ref_col_bounds[idx_col, :].min(), ref_col_bounds[idx_col, :].max()
                r_data[idx_row, idx_col] = Polygon(
                    [(col_min, row_min), (col_min, row_max), (col_max, row_max), (col_max, row_min)])
        # the grid dimension may not have row/col or row/col bounds
        except AttributeError:
            # we want geometries for everything even if masked
            corners = self.grid.corners.data
            range_row = range(self.grid.shape[0])
            range_col = range(self.grid.shape[1])
            for row, col in itertools.product(range_row, range_col):
                current_corner = corners[:, row, col]
                coords = np.hstack((current_corner[1, :].reshape(-1, 1),
                                    current_corner[0, :].reshape(-1, 1)))
                polygon = Polygon(coords)
                r_data[row, col] = polygon
        return fill
