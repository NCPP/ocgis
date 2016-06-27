import itertools
import os
from collections import OrderedDict
from copy import deepcopy, copy

import fiona
import numpy as np
from fiona.crs import from_epsg
from shapely import wkt, wkb
from shapely.geometry import shape, mapping, Polygon, MultiPoint
from shapely.geometry.geo import box
from shapely.geometry.point import Point

from ocgis import constants, RequestDataset
from ocgis.constants import WrappedState
from ocgis.exc import EmptySubsetError, SpatialWrappingError, MultipleElementsFound, BoundsAlreadyAvailableError
from ocgis.interface.base.crs import CoordinateReferenceSystem, WGS84, CFWGS84, CFRotatedPole, \
    Spherical
from ocgis.interface.base.dimension.base import AbstractUidValueDimension
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialDimension, SpatialGeometryDimension, \
    SpatialGeometryPolygonDimension, SpatialGridDimension, SpatialGeometryPointDimension, SingleElementRetriever
from ocgis.test import strings
from ocgis.test.base import TestBase, attr
from ocgis.util.helpers import iter_array, make_poly
from ocgis.util.itester import itr_products_keywords
from ocgis.util.ugrid.convert import mesh2_nc_to_fiona


class AbstractTestSpatialDimension(TestBase):
    def assertGeometriesAlmostEquals(self, a, b):

        def _almost_equals_(a, b):
            return a.almost_equals(b)

        vfunc = np.vectorize(_almost_equals_, otypes=[bool])
        to_test = vfunc(a.data, b.data)
        self.assertTrue(to_test.all())
        self.assertNumpyAll(a.mask, b.mask)

    def get_col(self, bounds=True):
        value = [-100., -99., -98., -97.]
        if bounds:
            bounds = [[v - 0.5, v + 0.5] for v in value]
        else:
            bounds = None
        row = VectorDimension(value=value, bounds=bounds, name='col')
        return (row)

    def get_row(self, bounds=True):
        value = [40., 39., 38.]
        if bounds:
            bounds = [[v + 0.5, v - 0.5] for v in value]
        else:
            bounds = None
        row = VectorDimension(value=value, bounds=bounds, name='row')
        return (row)

    def get_sdim(self, bounds=True, crs=None, name=None):
        row = self.get_row(bounds=bounds)
        col = self.get_col(bounds=bounds)
        sdim = SpatialDimension(row=row, col=col, crs=crs, name=name)
        return sdim

    @property
    def grid_value_regular(self):
        grid_value_regular = [[[40.0, 40.0, 40.0, 40.0], [39.0, 39.0, 39.0, 39.0], [38.0, 38.0, 38.0, 38.0]],
                              [[-100.0, -99.0, -98.0, -97.0], [-100.0, -99.0, -98.0, -97.0],
                               [-100.0, -99.0, -98.0, -97.0]]]
        grid_value_regular = np.ma.array(grid_value_regular, mask=False)
        return grid_value_regular

    @property
    def grid_corners_regular(self):
        grid_corners_regular = [
            [[[40.5, 40.5, 39.5, 39.5], [40.5, 40.5, 39.5, 39.5], [40.5, 40.5, 39.5, 39.5], [40.5, 40.5, 39.5, 39.5]],
             [[39.5, 39.5, 38.5, 38.5], [39.5, 39.5, 38.5, 38.5], [39.5, 39.5, 38.5, 38.5], [39.5, 39.5, 38.5, 38.5]],
             [[38.5, 38.5, 37.5, 37.5], [38.5, 38.5, 37.5, 37.5], [38.5, 38.5, 37.5, 37.5], [38.5, 38.5, 37.5, 37.5]]],
            [[[-100.5, -99.5, -99.5, -100.5], [-99.5, -98.5, -98.5, -99.5], [-98.5, -97.5, -97.5, -98.5],
              [-97.5, -96.5, -96.5, -97.5]],
             [[-100.5, -99.5, -99.5, -100.5], [-99.5, -98.5, -98.5, -99.5], [-98.5, -97.5, -97.5, -98.5],
              [-97.5, -96.5, -96.5, -97.5]],
             [[-100.5, -99.5, -99.5, -100.5], [-99.5, -98.5, -98.5, -99.5], [-98.5, -97.5, -97.5, -98.5],
              [-97.5, -96.5, -96.5, -97.5]]]]
        grid_corners_regular = np.ma.array(grid_corners_regular, mask=False)
        return grid_corners_regular

    def get_shapely_from_wkt_array(self, wkts):
        ret = np.array(wkts)
        vfunc = np.vectorize(wkt.loads, otypes=[object])
        ret = vfunc(ret)
        ret = np.ma.array(ret, mask=False)
        return ret

    @property
    def point_value(self):
        pts = [['POINT (-100 40)', 'POINT (-99 40)', 'POINT (-98 40)', 'POINT (-97 40)'],
               ['POINT (-100 39)', 'POINT (-99 39)', 'POINT (-98 39)', 'POINT (-97 39)'],
               ['POINT (-100 38)', 'POINT (-99 38)', 'POINT (-98 38)', 'POINT (-97 38)']]
        ret = self.get_shapely_from_wkt_array(pts)
        return ret

    @property
    def polygon_value(self):
        polys = [['POLYGON ((-100.5 39.5, -100.5 40.5, -99.5 40.5, -99.5 39.5, -100.5 39.5))',
                  'POLYGON ((-99.5 39.5, -99.5 40.5, -98.5 40.5, -98.5 39.5, -99.5 39.5))',
                  'POLYGON ((-98.5 39.5, -98.5 40.5, -97.5 40.5, -97.5 39.5, -98.5 39.5))',
                  'POLYGON ((-97.5 39.5, -97.5 40.5, -96.5 40.5, -96.5 39.5, -97.5 39.5))'],
                 ['POLYGON ((-100.5 38.5, -100.5 39.5, -99.5 39.5, -99.5 38.5, -100.5 38.5))',
                  'POLYGON ((-99.5 38.5, -99.5 39.5, -98.5 39.5, -98.5 38.5, -99.5 38.5))',
                  'POLYGON ((-98.5 38.5, -98.5 39.5, -97.5 39.5, -97.5 38.5, -98.5 38.5))',
                  'POLYGON ((-97.5 38.5, -97.5 39.5, -96.5 39.5, -96.5 38.5, -97.5 38.5))'],
                 ['POLYGON ((-100.5 37.5, -100.5 38.5, -99.5 38.5, -99.5 37.5, -100.5 37.5))',
                  'POLYGON ((-99.5 37.5, -99.5 38.5, -98.5 38.5, -98.5 37.5, -99.5 37.5))',
                  'POLYGON ((-98.5 37.5, -98.5 38.5, -97.5 38.5, -97.5 37.5, -98.5 37.5))',
                  'POLYGON ((-97.5 37.5, -97.5 38.5, -96.5 38.5, -96.5 37.5, -97.5 37.5))']]
        return self.get_shapely_from_wkt_array(polys)

    @property
    def polygon_value_alternate_ordering(self):
        polys = [['POLYGON ((-100.5 40.5, -99.5 40.5, -99.5 39.5, -100.5 39.5, -100.5 40.5))',
                  'POLYGON ((-99.5 40.5, -98.5 40.5, -98.5 39.5, -99.5 39.5, -99.5 40.5))',
                  'POLYGON ((-98.5 40.5, -97.5 40.5, -97.5 39.5, -98.5 39.5, -98.5 40.5))',
                  'POLYGON ((-97.5 40.5, -96.5 40.5, -96.5 39.5, -97.5 39.5, -97.5 40.5))'],
                 ['POLYGON ((-100.5 39.5, -99.5 39.5, -99.5 38.5, -100.5 38.5, -100.5 39.5))',
                  'POLYGON ((-99.5 39.5, -98.5 39.5, -98.5 38.5, -99.5 38.5, -99.5 39.5))',
                  'POLYGON ((-98.5 39.5, -97.5 39.5, -97.5 38.5, -98.5 38.5, -98.5 39.5))',
                  'POLYGON ((-97.5 39.5, -96.5 39.5, -96.5 38.5, -97.5 38.5, -97.5 39.5))'],
                 ['POLYGON ((-100.5 38.5, -99.5 38.5, -99.5 37.5, -100.5 37.5, -100.5 38.5))',
                  'POLYGON ((-99.5 38.5, -98.5 38.5, -98.5 37.5, -99.5 37.5, -99.5 38.5))',
                  'POLYGON ((-98.5 38.5, -97.5 38.5, -97.5 37.5, -98.5 37.5, -98.5 38.5))',
                  'POLYGON ((-97.5 38.5, -96.5 38.5, -96.5 37.5, -97.5 37.5, -97.5 38.5))']]
        return self.get_shapely_from_wkt_array(polys)

    @property
    def uid_value(self):
        return np.ma.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], mask=False, dtype=np.int32)

    def write_sdim(self):
        sdim = self.get_sdim(bounds=True)
        crs = from_epsg(4326)
        schema = {'geometry': 'Polygon', 'properties': {'UID': 'int:8'}}
        with fiona.open('/tmp/test.shp', 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as sink:
            for ii, poly in enumerate(sdim.geom.polygon.value.flat):
                row = {'geometry': mapping(poly),
                       'properties': {'UID': int(sdim.geom.uid.flatten()[ii])}}
                sink.write(row)


class TestSingleElementRetriever(AbstractTestSpatialDimension):
    def test_init(self):
        sdim = self.get_sdim()
        with self.assertRaises(MultipleElementsFound):
            SingleElementRetriever(sdim)
        sub = sdim[1, 2]
        single = SingleElementRetriever(sub)
        self.assertIsNone(single.properties)
        self.assertEqual(single.uid, 7)
        self.assertIsInstance(single.geom, Polygon)
        sub.abstraction = 'point'
        self.assertIsInstance(single.geom, Point)
        self.assertIsNone(single.crs)


class TestSpatialDimension(AbstractTestSpatialDimension):
    def get_records(self):
        path = os.path.join(self.path_bin, 'shp', 'state_boundaries', 'state_boundaries.shp')
        with fiona.open(path, 'r') as source:
            records = list(source)
            meta = source.meta

        return {'records': records, 'meta': meta}

    def get_spatial_dimension_from_records(self):
        record_dict = self.get_records()
        return SpatialDimension.from_records(record_dict['records'], crs=record_dict['meta']['crs'])

    def test_init(self):
        sdim = self.get_sdim(bounds=True)
        self.assertEqual(sdim.name, 'spatial')
        self.assertEqual(sdim.name_uid, 'gid')
        self.assertIsNone(sdim.abstraction)
        self.assertNumpyAll(sdim.grid.value, self.grid_value_regular)

        def _almost_equals_(a, b):
            return a.almost_equals(b)

        vfunc = np.vectorize(_almost_equals_, otypes=[bool])
        to_test = vfunc(sdim.geom.point.value.data, self.point_value.data)
        self.assertTrue(to_test.all())
        self.assertFalse(sdim.geom.point.value.mask.any())
        to_test = vfunc(sdim.geom.polygon.value.data, self.polygon_value.data)
        self.assertTrue(to_test.all())
        self.assertFalse(sdim.geom.polygon.value.mask.any())

        sdim = self.get_sdim(name='foobuar')
        self.assertEqual(sdim.name, 'foobuar')

    def test_abstraction(self):
        sdim = self.get_sdim()
        self.assertIsNone(sdim.abstraction)
        self.assertEqual(sdim.abstraction, sdim._abstraction)
        self.assertIsInstance(sdim.abstraction_geometry, SpatialGeometryPointDimension)

        sdim = self.get_sdim(bounds=True)
        self.assertIsInstance(sdim.geom.polygon, SpatialGeometryPolygonDimension)
        sdim.abstraction = 'point'
        self.assertEqual(sdim.geom.abstraction, 'point')
        self.assertEqual(sdim.abstraction, sdim._abstraction)
        self.assertIsInstance(sdim.abstraction_geometry, SpatialGeometryPointDimension)

    def test_abstraction_geometry(self):
        sdim = self.get_sdim(bounds=True)
        self.assertIsInstance(sdim.abstraction_geometry, SpatialGeometryPolygonDimension)

    def test_get_report(self):
        keywords = dict(crs=[WGS84(), None],
                        with_grid=[True, False])
        for k in self.iter_product_keywords(keywords):
            sdim = self.get_sdim(crs=k.crs)
            if not k.with_grid:
                sdim.grid = None
            actual = sdim.get_report()
            self.assertEqual(len(actual), 6)

    def test_init_combinations(self):
        """
        - points only
        - polygons only
        - grid only
        - row and column only
        - grid bounds only

        intersects, clip, update_crs, union,
        """

        def iter_grid():
            add_row_col_bounds = [True, False]
            for arcb in add_row_col_bounds:
                keywords = dict(row=[None, self.get_row(bounds=arcb)],
                                col=[None, self.get_col(bounds=arcb)],
                                value=[None, self.grid_value_regular],
                                corners=[None, self.grid_corners_regular])
                for k in itr_products_keywords(keywords):
                    try:
                        grid = SpatialGridDimension(**k)
                        self.assertNumpyAll(grid.uid, self.uid_value)
                    except ValueError:
                        if k['value'] is None and (k['row'] is None or k['col'] is None):
                            continue
                        else:
                            raise
                    self.assertNumpyAll(grid.value, self.grid_value_regular)
                    if k['corners'] is not None:
                        self.assertNumpyAll(grid.corners, self.grid_corners_regular)
                    yield dict(grid=grid,
                               row=k['row'],
                               col=k['col'])

        def iter_geom():
            keywords = dict(grid=[None, True],
                            point=[None, SpatialGeometryPointDimension(value=self.point_value)],
                            polygon=[None, SpatialGeometryPolygonDimension(value=self.polygon_value)])
            for k in itr_products_keywords(keywords):
                if k['grid'] is None:
                    grid_iterator = [{}]
                else:
                    grid_iterator = iter_grid()
                for grid_dict in grid_iterator:
                    if grid_dict == {}:
                        k['grid'] = None
                    else:
                        k['grid'] = grid_dict['grid']
                    try:
                        geom = SpatialGeometryDimension(**k)
                        self.assertNumpyAll(geom.uid, self.uid_value)
                        self.assertIsNotNone(geom.get_highest_order_abstraction())
                    except ValueError:
                        if all([v is None for v in k.values()]):
                            continue
                        raise
                    try:
                        self.assertGeometriesAlmostEquals(geom.point.value, self.point_value)
                        self.assertNumpyAll(geom.point.uid, self.uid_value)
                    except AttributeError:
                        if k['grid'] is None and k['point'] is None:
                            continue
                        raise
                    if k['polygon'] is not None or k['grid'] is not None:
                        if geom.polygon is not None:
                            try:
                                self.assertGeometriesAlmostEquals(geom.polygon.value, self.polygon_value)
                            except AssertionError:
                                # coordinates may be ordered differently
                                self.assertGeometriesAlmostEquals(geom.polygon.value,
                                                                  self.polygon_value_alternate_ordering)
                            self.assertNumpyAll(geom.polygon.uid, self.uid_value)
                        else:
                            if k['polygon'] is None and k['grid'].corners is None:
                                if k['grid'].row is None or k['grid'].col is None:
                                    continue
                            if geom.grid.corners is None:
                                if geom.grid.row.bounds is None or geom.grid.col.bounds is None:
                                    continue
                            raise

                    yield (dict(geom=geom,
                                grid=grid_dict.get('grid'),
                                row=grid_dict.get('row'),
                                col=grid_dict.get('col'),
                                polygon=geom.polygon,
                                point=geom.point))

        def geom_iterator():
            for k in iter_geom():
                yield k
                k['geom'] = None
                yield k

        for k in geom_iterator():
            sdim = SpatialDimension(**k)
            self.assertGeometriesAlmostEquals(sdim.geom.point.value, self.point_value)
            try:
                try:
                    self.assertGeometriesAlmostEquals(sdim.geom.polygon.value, self.polygon_value)
                except AssertionError:
                    # polygons may have a different ordering
                    self.assertGeometriesAlmostEquals(sdim.geom.polygon.value, self.polygon_value_alternate_ordering)
            except AttributeError:
                if sdim.geom.polygon is None and sdim.grid is None:
                    continue
                raise

            try:
                self.assertNumpyAll(sdim.grid.value, self.grid_value_regular)
            except AttributeError:
                if k['grid'] is None:
                    pass
                else:
                    raise

            try:
                if sdim.grid.corners is not None:
                    self.assertNumpyAll(sdim.grid.corners, self.grid_corners_regular)
                else:
                    if k['row'] is None or k['col'] is None:
                        pass
                    else:
                        if k['row'].bounds is None or k['col'].bounds is None:
                            pass
                        else:
                            raise
            except AttributeError:
                if k['grid'] is None:
                    pass
                else:
                    raise

    def test_abstraction_behavior(self):
        """Test abstraction limits what elements are loaded and returned."""

        row = VectorDimension(value=[2, 4])
        col = VectorDimension(value=[4, 6])
        for element in [row, col]:
            element.set_extrapolated_bounds()
        grid = SpatialGridDimension(row=row, col=col)

        sdim = SpatialDimension(grid=grid, abstraction='point')
        self.assertIsNone(sdim.geom.polygon)

    def test_set_mask(self):

        kwds = dict(with_grid=[True, False],
                    with_grid_corners=[True, False],
                    with_geom=[True, False],
                    with_point=[True, False],
                    with_polygon=[True, False])

        for ctr, k in enumerate(itr_products_keywords(kwds, as_namedtuple=True)):
            sdim = self.get_sdim()
            sdim.grid
            sdim.grid.corners
            sdim.geom.polygon
            sdim.geom.point

            if not k.with_grid:
                sdim._grid = None
            if k.with_grid and not k.with_grid_corners:
                sdim.grid._corners = None
            if not k.with_geom:
                sdim._geom = None
            if k.with_geom:
                if not k.with_point:
                    sdim.geom._point = None
                if not k.with_polygon:
                    sdim.geom._polygon = None
                if not k.with_grid:
                    sdim.geom.grid = None

            np.random.seed(1)
            mask = np.random.randint(0, 2, (3, 4))
            sdim.set_mask(mask)

            if not k.with_grid:
                self.assertIsNone(sdim._grid)
            if k.with_grid and not k.with_grid_corners:
                self.assertIsNone(sdim.grid._corners)
            if not k.with_geom:
                self.assertIsNone(sdim._geom)
            if k.with_geom:
                if not k.with_point:
                    self.assertIsNone(sdim.geom._point)
                if not k.with_polygon:
                    self.assertIsNone(sdim.geom._polygon)

            if sdim.grid is not None:
                for ii, jj in iter_array(mask):
                    if mask[ii, jj]:
                        self.assertTrue(sdim.grid.corners.mask[:, ii, jj, :].all())
                    else:
                        self.assertFalse(sdim.grid.corners.mask[:, ii, jj, :].any())

            actual = mask.astype(bool)
            if k.with_grid:
                self.assertNumpyAll(sdim.grid.value.mask[0], actual)
                self.assertNumpyAll(sdim.grid.value.mask[1], actual)
            if k.with_geom:
                if k.with_point or k.with_grid:
                    self.assertNumpyAll(sdim.geom.point.value.mask, actual)
                if k.with_polygon or k.with_grid:
                    self.assertNumpyAll(sdim.geom.polygon.value.mask, actual)
            if k.with_grid or k.with_geom:
                try:
                    self.assertNumpyAll(sdim.get_mask(), actual)
                except AttributeError:
                    # there is actually nothing on this sdim, so the mask may not be retrieved
                    if not k.with_grid and not k.with_point and not k.with_polygon:
                        continue
                    else:
                        raise

    def test_overloaded_crs(self):
        """Test CFWGS84 coordinate system is always used if the input CRS is equivalent."""

        sdim = self.get_sdim(crs=CoordinateReferenceSystem(epsg=4326))
        self.assertIsInstance(sdim.crs, CFWGS84)
        sdim = self.get_sdim(crs=CoordinateReferenceSystem(epsg=2346))
        self.assertEqual(sdim.crs, CoordinateReferenceSystem(epsg=2346))

    def test_from_records(self):
        """Test creating SpatialDimension directly from Fiona records."""

        record_dict = self.get_records()
        schema = {'geometry': None,
                  'properties': OrderedDict([(u'UGID', 'int'), (u'STATE_FIPS', u'str:2'), (u'ID', 'float'),
                                             (u'STATE_NAME', u'str'), (u'STATE_ABBR', u'str:2')])}
        keywords = dict(crs=[record_dict['meta']['crs'], None, CFWGS84()],
                        abstraction=['polygon', 'point'],
                        add_geom=[True, False],
                        schema=[None, schema])
        for k in self.iter_product_keywords(keywords):
            record_dict = deepcopy(record_dict)
            records = deepcopy(record_dict['records'])

            if k.add_geom:
                for record in records:
                    record['geom'] = shape(record['geometry'])
                    if k.abstraction == 'point':
                        record['geom'] = record['geom'].centroid
            else:
                if k.abstraction == 'point':
                    for record in records:
                        geom = shape(record['geometry']).centroid
                        record['geometry'] = mapping(geom)
                self.assertTrue('geom' not in records[10])

            sdim = SpatialDimension.from_records(records, crs=k.crs, schema=schema)

            self.assertIsInstance(sdim, SpatialDimension)
            self.assertEqual(sdim.shape, (1, 51))
            self.assertEqual(sdim.properties.shape, (51,))
            if k.abstraction == 'polygon':
                self.assertIsInstance(sdim.geom.get_highest_order_abstraction(),
                                      SpatialGeometryPolygonDimension)
            else:
                self.assertIsInstance(sdim.geom.get_highest_order_abstraction(), SpatialGeometryPointDimension)
            self.assertEqual(sdim.properties[0]['UGID'], sdim.uid[0, 0])
            self.assertEqual(sdim.properties.dtype.names,
                             ('UGID', 'STATE_FIPS', 'ID', 'STATE_NAME', 'STATE_ABBR'))
            self.assertEqual(sdim.crs, CFWGS84())
            self.assertDictEqual(sdim.meta, {})
            if k.abstraction == 'polygon':
                self.assertNumpyAllClose(np.array(sdim.geom.polygon.value[0, 23].bounds),
                                         np.array((-114.04727330260259, 36.991746361915986, -109.04320629794219,
                                                   42.00230036658243)))
            else:
                _point = sdim.geom.point.value[0, 23]
                xy = _point.x, _point.y
                self.assertNumpyAllClose(np.array(xy), np.array((-111.67605350692477, 39.322512402249)))
            self.assertNumpyAll(sdim.uid, np.ma.array(range(1, 52)).reshape(1, 51))
            self.assertEqual(sdim.abstraction, k.abstraction)

            for prop in sdim.properties[0]:
                self.assertNotEqual(prop, None)

    def test_from_records_proper_uid(self):
        """Test records without 'UGID' property."""

        record_dict = self.get_records()
        for record in record_dict['records']:
            record['properties'].pop('UGID')
        self.assertFalse('UGID' in record_dict['records'][23]['properties'])

        sdim = SpatialDimension.from_records(record_dict['records'], crs=record_dict['meta']['crs'])
        self.assertNumpyAll(sdim.uid, np.ma.array(range(1, 52)).reshape(1, 51))

    def test_from_records_using_nondefault_identifier(self):
        """Test passing the non-default unique identifier."""

        record_dict = self.get_records()
        for record in record_dict['records']:
            record['properties'].pop('UGID')
            record['properties']['ID'] += 5.0
        sdim = SpatialDimension.from_records(record_dict['records'], crs=record_dict['meta']['crs'], uid='ID')
        self.assertEqual(sdim.uid[0, 3], 9)
        self.assertEqual(sdim.name_uid, 'ID')

    def test_get_intersects_select_nearest(self):
        pt = Point(-99, 39)
        return_indices = [True, False]
        use_spatial_index = [True, False]
        select_nearest = [True, False]
        abstraction = ['polygon', 'point']
        bounds = [True, False]
        for args in itertools.product(return_indices, use_spatial_index, select_nearest, abstraction, bounds):
            ri, usi, sn, a, b = args
            sdim = self.get_sdim(bounds=b)
            sdim.abstraction = a
            ret = sdim.get_intersects(pt.buffer(1), select_nearest=sn, return_indices=ri)
            ## return_indice will return a tuple if True
            if ri:
                ret, ret_slc = ret
            if sn:
                ## select_nearest=True always returns a single element
                self.assertEqual(ret.shape, (1, 1))
                try:
                    self.assertTrue(ret.geom.polygon.value[0, 0].centroid.almost_equals(pt))
                # ## polygons will not be present if the abstraction is point or there are no bounds on the created
                # ## spatial dimension object
                except AttributeError:
                    if a == 'point' or b is False:
                        self.assertTrue(ret.geom.point.value[0, 0].almost_equals(pt))
                    else:
                        raise
            ## if we are not returning the nearest geometry...
            else:
                ## bounds with a spatial abstraction of polygon will have this dimension
                if b and a == 'polygon':
                    self.assertEqual(ret.shape, (3, 3))
                ## with points, there is only intersecting geometry
                else:
                    self.assertEqual(ret.shape, (1, 1))

    def test_get_clip(self):
        sdim = self.get_sdim(bounds=True)
        poly = make_poly((37.75, 38.25), (-100.25, -99.75))

        for b in [True, False]:
            ret = sdim.get_clip(poly, use_spatial_index=b)

            self.assertEqual(ret.uid, np.array([[9]]))
            self.assertTrue(poly.almost_equals(ret.geom.polygon.value[0, 0]))

            self.assertEqual(ret.geom.point.value.shape, ret.geom.polygon.shape)
            ref_pt = ret.geom.point.value[0, 0]
            ref_poly = ret.geom.polygon.value[0, 0]
            self.assertTrue(ref_poly.intersects(ref_pt))

    def test_get_fiona_schema(self):
        sdim = self.get_sdim(crs=Spherical())
        schema = sdim.get_fiona_schema()
        self.assertEqual(schema, {'geometry': 'MultiPolygon', 'properties': OrderedDict()})

        properties = np.zeros(2, dtype={'names': ['a', 'b'], 'formats': [np.int32, np.float64]})
        sdim.properties = properties
        schema = sdim.get_fiona_schema()
        self.assertEqual(schema,
                         {'geometry': 'MultiPolygon', 'properties': OrderedDict([('a', 'int'), ('b', 'float')])})

    def test_get_geom_iter(self):
        sdim = self.get_sdim(bounds=True)
        tt = list(sdim.get_geom_iter())
        ttt = list(tt[4])
        ttt[2] = ttt[2].bounds
        self.assertEqual(ttt, [1, 0, (-100.5, 38.5, -99.5, 39.5), 5])

        sdim = self.get_sdim(bounds=False)
        tt = list(sdim.get_geom_iter(target='point'))
        ttt = list(tt[4])
        ttt[2] = [ttt[2].x, ttt[2].y]
        self.assertEqual(ttt, [1, 0, [-100.0, 39.0], 5])

        sdim = self.get_sdim(bounds=False)
        self.assertIsNone(sdim.abstraction)
        # this abstraction is not available
        with self.assertRaises(ValueError):
            list(sdim.get_geom_iter(target='polygon'))

    def test_get_intersects_no_grid(self):
        """
        Test an intersects operation without a grid.
        """

        poly = strings.S2
        poly = wkt.loads(poly)
        path = os.path.join(self.path_bin, 'shp', 'state_boundaries', 'state_boundaries.shp')
        field = RequestDataset(path).get()
        sdim = field.spatial
        """:type sdim: :class:`ocgis.SpatialDimension`"""
        actual_ret_slc = [[0, 0, 0, 0, 0, 0, 0, 0, 0], [5, 10, 13, 21, 26, 29, 32, 35, 39]]
        actual_ret_slc = [np.array(xx) for xx in actual_ret_slc]

        keywords = dict(ctr=range(2), return_indices=[True, False], select_nearest=[False, True])

        for k in self.iter_product_keywords(keywords):
            ret = sdim.get_intersects(poly, return_indices=k.return_indices, select_nearest=k.select_nearest)

            if k.return_indices:
                ret, ret_slc = ret
                if k.select_nearest:
                    self.assertEqual(ret_slc, (np.array([0]), np.array([26])))
                else:
                    for idx in range(2):
                        self.assertNumpyAll(ret_slc[idx], actual_ret_slc[idx])

            if k.select_nearest:
                actual_shape = (1, 1)
            else:
                actual_shape = (1, 9)

            for element in [ret, ret.geom, ret.geom.polygon, ret.geom.polygon.value, ret.uid, ret.geom.uid,
                            ret.geom.polygon.uid]:
                self.assertEqual(element.shape, actual_shape)
            ret.assert_uniform_mask()

            if k.select_nearest:
                actual = strings.S1
                self.assertTrue(ret.geom.polygon.value[0, 0].almost_equals(wkt.loads(actual)))
            else:
                actual = strings.S3
                actual = [wkb.loads(xx) for xx in actual]
                for geom in ret.geom.polygon.value.flat:
                    for idx, xx in enumerate(actual):
                        if geom.almost_equals(xx):
                            actual.pop(idx)
                            break
                self.assertEqual(actual, [])

        # test with no point dimension
        value = sdim.geom.polygon.value.copy()
        for ii, jj in iter_array(value):
            value[ii, jj] = value[ii, jj].centroid
        sdim.geom._point = SpatialGeometryPointDimension(value=value)
        sdim.geom._polygon = None
        self.assertIsNone(sdim.geom.polygon)
        self.assertIsNotNone(sdim.geom.point)
        sub = sdim.get_intersects(poly)
        self.assertEqual(sub.shape, (1, 3))

    def test_get_intersects_point_abstraction(self):
        """Test with a point abstraction checking that masks are appropriately updated."""

        sdim = self.get_sdim(crs=WGS84())
        self.assertFalse(sdim.uid.mask.any())
        sdim.assert_uniform_mask()
        sdim.abstraction = 'point'
        poly = wkt.loads(
            'POLYGON((-98.26574367088608142 40.19952531645570559,-98.71764240506330168 39.54825949367089066,-99.26257911392406186 39.16281645569620906,-99.43536392405064817 38.64446202531645724,-98.78409810126584034 38.33876582278481493,-98.23916139240508016 37.71408227848101546,-97.77397151898735217 37.67420886075949937,-97.62776898734178133 38.15268987341772799,-98.39865506329114453 38.52484177215190186,-98.23916139240508016 39.33560126582278826,-97.73409810126582897 39.58813291139241386,-97.52143987341773368 40.27927215189873777,-97.52143987341773368 40.27927215189873777,-98.26574367088608142 40.19952531645570559))')
        actual_mask = np.array([[True, False], [False, True], [True, False]])
        for use_spatial_index in [True]:  # , False]:
            ret = sdim.get_intersects(poly, use_spatial_index=use_spatial_index)
            self.assertIsInstance(ret, SpatialDimension)
            self.assertNumpyAll(actual_mask, ret.get_mask())
            self.assertNumpyAll(actual_mask, ret.uid.mask)
            ret.assert_uniform_mask()
            self.assertFalse(sdim.uid.mask.any())
            self.assertFalse(sdim.grid.value.mask.any())
            self.assertFalse(sdim.grid.corners.mask.any())
            sdim.assert_uniform_mask()

    def test_get_intersects_polygon_small(self):
        for b in [True, False]:
            sdim = self.get_sdim(bounds=b)
            poly = make_poly((37.75, 38.25), (-100.25, -99.75))
            for u in [True, False]:
                ret = sdim.get_intersects(poly, use_spatial_index=u)
                to_test = np.ma.array([[[38.]], [[-100.]]], mask=False)
                self.assertNumpyAll(ret.grid.value, to_test)
                self.assertNumpyAll(ret.uid, np.ma.array([[9]], dtype=np.int32))
                self.assertEqual(ret.shape, (1, 1))
                to_test = ret.geom.point.value.compressed()[0]
                self.assertTrue(to_test.almost_equals(Point(-100, 38)))
                if b is False:
                    self.assertIsNone(ret.geom.polygon)
                else:
                    to_test = ret.geom.polygon.value.compressed()[0].bounds
                    self.assertEqual((-100.5, 37.5, -99.5, 38.5), to_test)

    def test_get_intersects_polygon_no_point_overlap(self):
        for b in [True, False]:
            sdim = self.get_sdim(bounds=b)
            poly = make_poly((39.25, 39.75), (-97.75, -97.25))
            for u in [True, False]:
                if b is False:
                    with self.assertRaises(EmptySubsetError):
                        sdim.get_intersects(poly, use_spatial_index=u)
                else:
                    ret = sdim.get_intersects(poly, use_spatial_index=u)
                    self.assertEqual(ret.shape, (2, 2))

    def test_get_intersects_polygon_all(self):
        for b in [True, False]:
            sdim = self.get_sdim(bounds=b)
            poly = make_poly((37, 41), (-101, -96))
            for u in [True, False]:
                ret = sdim.get_intersects(poly, use_spatial_index=u)
                self.assertNumpyAll(sdim.grid.value, ret.grid.value)
                self.assertNumpyAll(sdim.grid.value.mask[0, :, :], sdim.geom.point.value.mask)
                self.assertEqual(ret.shape, (3, 4))

    def test_get_intersects_polygon_empty(self):
        for b in [True, False]:
            sdim = self.get_sdim(bounds=b)
            poly = make_poly((1000, 1001), (-1000, -1001))
            with self.assertRaises(EmptySubsetError):
                sdim.get_intersects(poly)

    def test_state_boundaries_weights(self):
        """Test weights are correctly constructed from arbitrary geometries."""

        sdim = self.get_spatial_dimension_from_records()
        ref = sdim.weights
        self.assertEqual(ref[0, 50], 1.0)
        self.assertAlmostEqual(sdim.weights.mean(), 0.07744121084026262)

    def test_geom_mask_by_polygon(self):
        sdim = self.get_spatial_dimension_from_records()
        spdim = sdim.geom.polygon
        ref = spdim.value.mask
        self.assertEqual(ref.shape, (1, 51))
        self.assertFalse(ref.any())
        select = sdim.properties['STATE_ABBR'] == 'NE'
        subset_polygon = sdim[:, select].geom.polygon.value[0, 0]

        for b in [True, False]:
            msked = spdim.get_intersects_masked(subset_polygon, use_spatial_index=b)

            self.assertEqual(msked.value.mask.sum(), 50)
            self.assertTrue(msked.value.compressed()[0].almost_equals(subset_polygon))

            with self.assertRaises(NotImplementedError):
                msked = spdim.get_intersects_masked(subset_polygon.centroid)
                self.assertTrue(msked.value.compressed()[0].almost_equals(subset_polygon))

            with self.assertRaises(EmptySubsetError):
                spdim.get_intersects_masked(Point(1000, 1000).buffer(1))

    def test_geom_mask_by_polygon_equivalent_without_spatial_index(self):
        sdim = self.get_spatial_dimension_from_records()
        spdim = sdim.geom.polygon
        ref = spdim.value.mask
        self.assertEqual(ref.shape, (1, 51))
        self.assertFalse(ref.any())
        select = sdim.properties['STATE_ABBR'] == 'NE'
        subset_polygon = sdim[:, select].geom.polygon.value[0, 0]

        msked_spatial_index = spdim.get_intersects_masked(subset_polygon, use_spatial_index=True)
        msked_without_spatial_index = spdim.get_intersects_masked(subset_polygon, use_spatial_index=False)
        self.assertNumpyAll(msked_spatial_index.value, msked_without_spatial_index.value)
        self.assertNumpyAll(msked_spatial_index.value.mask, msked_without_spatial_index.value.mask)

    def test_update_crs_geom_combinations(self):
        """Test CRS updating without a grid and different point/polygon combinations."""

        keywords = dict(with_point=[True, False],
                        with_polygon=[True, False])

        to_crs = CoordinateReferenceSystem(epsg=2163)
        from_crs = WGS84()

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            sdim = self.get_sdim(crs=from_crs)
            sdim.geom.polygon.value
            sdim.geom.point.value
            sdim.grid = None
            sdim.geom.grid = None

            if not k.with_point:
                sdim.geom._point = None
                self.assertIsNone(sdim.geom.point)

            if not k.with_polygon:
                sdim.geom._polygon = None
                self.assertIsNone(sdim.geom.polygon)

            sdim.update_crs(to_crs)

            if k.with_polygon:
                self.assertEqual(sdim.geom.polygon.value[2, 2].bounds,
                                 (130734.585229303, -832179.0855220362, 220974.77455120225, -719113.1357226598))
            if k.with_point:
                self.assertEqual(sdim.geom.point.value[2, 2].bounds,
                                 (175552.29305101855, -775779.6191590576, 175552.29305101855, -775779.6191590576))

    def test_update_crs_grid_combinations(self):
        """Test CRS is updated as expected with different types of grids."""

        keywords = dict(with_grid=[True, False],
                        with_corners=[True, False],
                        with_mask=[False, True])

        to_crs = CoordinateReferenceSystem(epsg=2163)
        from_crs = WGS84()
        for k in itr_products_keywords(keywords, as_namedtuple=True):
            sdim = self.get_sdim(crs=from_crs)

            if k.with_mask:
                sdim.grid.value.mask[:, 0, 1] = True
                sdim.grid.value.mask[:, 1, 2] = True

            original_grid_value = deepcopy(sdim.grid.value)
            original_grid_corners = deepcopy(sdim.grid.corners)

            if not k.with_grid:
                sdim.geom.point.value
                sdim.geom.polygon.value
                sdim.grid = None
                self.assertIsNone(sdim.grid)

            if k.with_grid and not k.with_corners:
                sdim.grid.row
                sdim.grid.row = None
                sdim.grid.col
                sdim.grid.col = None
                sdim.grid.corners = None

            sdim.update_crs(to_crs)
            self.assertEqual(sdim.crs, to_crs)

            if k.with_grid:
                self.assertAlmostEqual(sdim.grid.value.data.mean(), -267630.25728117273)
                if sdim.grid.corners is None:
                    self.assertFalse(k.with_corners)
                else:
                    self.assertAlmostEqual(sdim.grid.corners.data.mean(), -267565.33741344721)
                self.assertIsNone(sdim._geom._point)
                self.assertIsNone(sdim._geom._polygon)
                self.assertIsNone(sdim.grid.row)
                self.assertIsNone(sdim.grid.col)

            try:
                self.assertEqual(sdim.geom.polygon.value[2, 2].bounds,
                                 (130734.585229303, -832179.0855220362, 220974.77455120225, -719113.1357226598))
            except AttributeError:
                self.assertFalse(k.with_corners)
            self.assertEqual(sdim.geom.point.value[2, 2].bounds,
                             (175552.29305101855, -775779.6191590576, 175552.29305101855, -775779.6191590576))

            sdim.update_crs(from_crs)

            if k.with_grid:
                self.assertNumpyAllClose(sdim.grid.value, original_grid_value)
                if sdim.grid.corners is None:
                    self.assertFalse(k.with_corners)
                else:
                    self.assertNumpyAllClose(sdim.grid.corners, original_grid_corners)

    def test_update_crs_general_error(self):
        """Test general OGR errors are appropriately raised if it is not a rotated pole transformation."""

        sdim = self.get_spatial_dimension_from_records()
        to_crs = CoordinateReferenceSystem(epsg=2136)
        with self.assertRaises(RuntimeError):
            sdim.update_crs(to_crs)

    @attr('data')
    def test_update_crs_rotated_pole(self):
        """Test moving between rotated pole and WGS84."""

        rd = self.test_data.get_rd('rotated_pole_ichec')
        field = rd.get()
        """:type: ocgis.interface.base.field.Field"""
        self.assertIsInstance(field.spatial.crs, CFRotatedPole)
        original_spatial = deepcopy(field.spatial)
        original_crs = copy(field.spatial.crs)
        field.spatial.update_crs(CFWGS84())
        # Test source indices are copied to the target grid object.
        self.assertIsNotNone(field.spatial.grid._src_idx)
        self.assertNumpyNotAllClose(original_spatial.grid.value, field.spatial.grid.value)
        field.spatial.update_crs(original_crs)
        self.assertNumpyAllClose(original_spatial.grid.value, field.spatial.grid.value)
        self.assertIsInstance(field.spatial.crs, CFRotatedPole)

    def test_grid_value(self):
        for b in [True, False]:
            row = self.get_row(bounds=b)
            col = self.get_col(bounds=b)
            sdim = SpatialDimension(row=row, col=col)
            col_test, row_test = np.meshgrid(col.value, row.value)
            self.assertNumpyAll(sdim.grid.value[0].data, row_test)
            self.assertNumpyAll(sdim.grid.value[1].data, col_test)
            self.assertFalse(sdim.grid.value.mask.any())

    def test_grid_slice_all(self):
        sdim = self.get_sdim(bounds=True)
        slc = sdim[:]
        self.assertNumpyAll(sdim.grid.value, slc.grid.value)

    def test_grid_slice_1d(self):
        sdim = self.get_sdim(bounds=True)
        sdim_slc = sdim.grid[0, :]
        self.assertEqual(sdim_slc.value.shape, (2, 1, 4))
        self.assertNumpyAll(sdim_slc.value,
                            np.ma.array([[[40, 40, 40, 40]], [[-100, -99, -98, -97]]], mask=False, dtype=float))
        self.assertEqual(sdim_slc.row.value[0], 40)
        self.assertNumpyAll(sdim_slc.col.value, np.array([-100, -99, -98, -97], dtype=float))

    def test_grid_slice_2d(self):
        sdim = self.get_sdim(bounds=True)
        sdim_slc = sdim.grid[0, 1]
        self.assertNumpyAll(sdim_slc.value, np.ma.array([[[40]], [[-99]]], mask=False, dtype=float))
        self.assertNumpyAll(sdim_slc.row.bounds, np.array([[40.5, 39.5]]))
        self.assertEqual(sdim_slc.col.value[0], -99)

    def test_grid_slice_2d_range(self):
        sdim = self.get_sdim(bounds=True)
        sdim_slc = sdim.grid[1:3, 0:3]
        self.assertNumpyAll(sdim_slc.value,
                            np.ma.array([[[39, 39, 39], [38, 38, 38]], [[-100, -99, -98], [-100, -99, -98]]],
                                        mask=False, dtype=float))
        self.assertNumpyAll(sdim_slc.row.value, np.array([39, 38], dtype=float))

    def test_geom_point(self):
        sdim = self.get_sdim(bounds=True)
        with self.assertRaises(AttributeError):
            sdim.geom.value
        pt = sdim.geom.point.value
        fill = np.ma.array(np.zeros((2, 3, 4)), mask=False)
        for idx_row, idx_col in iter_array(pt):
            fill[0, idx_row, idx_col] = pt[idx_row, idx_col].y
            fill[1, idx_row, idx_col] = pt[idx_row, idx_col].x
        self.assertNumpyAll(fill, sdim.grid.value)

    def test_geom_polygon_no_bounds(self):
        sdim = self.get_sdim(bounds=False)
        self.assertIsNone(sdim.geom.polygon)

    def test_geom_polygon_bounds(self):
        sdim = self.get_sdim(bounds=True)
        poly = sdim.geom.polygon.value
        fill = np.ma.array(np.zeros((2, 3, 4)), mask=False)
        for idx_row, idx_col in iter_array(poly):
            fill[0, idx_row, idx_col] = poly[idx_row, idx_col].centroid.y
            fill[1, idx_row, idx_col] = poly[idx_row, idx_col].centroid.x
        self.assertNumpyAll(fill, sdim.grid.value)

    def test_grid_shape(self):
        sdim = self.get_sdim()
        shp = sdim.grid.shape
        self.assertEqual(shp, (3, 4))

    def test_empty(self):
        with self.assertRaises(ValueError):
            SpatialDimension()

    def test_geoms_only(self):
        geoms = []
        path = os.path.join(self.path_bin, 'shp', 'state_boundaries', 'state_boundaries.shp')
        with fiona.open(path, 'r') as source:
            for row in source:
                geoms.append(shape(row['geometry']))
        geoms = np.atleast_2d(geoms)
        poly_dim = SpatialGeometryPolygonDimension(value=geoms)
        sg_dim = SpatialGeometryDimension(polygon=poly_dim)
        sdim = SpatialDimension(geom=sg_dim)
        self.assertEqual(sdim.shape, (1, 51))

    def test_slicing(self):
        """Test variations of slicing."""

        sdim = self.get_sdim(bounds=True)
        self.assertIsNone(sdim._geom._point)
        self.assertIsNone(sdim._geom._polygon)
        self.assertEqual(sdim.shape, (3, 4))
        self.assertEqual(sdim.geom.point.shape, (3, 4))
        self.assertEqual(sdim.geom.polygon.shape, (3, 4))
        self.assertEqual(sdim.grid.shape, (3, 4))
        with self.assertRaises(IndexError):
            sdim[0]
        sdim_slc = sdim[0, 1]
        self.assertEqual(sdim_slc.shape, (1, 1))
        self.assertEqual(sdim_slc.uid, np.array([[2]], dtype=np.int32))
        self.assertNumpyAll(sdim_slc.grid.value, np.ma.array([[[40.]], [[-99.]]], mask=False))
        self.assertNotEqual(sdim_slc, None)
        to_test = sdim_slc.geom.point.value[0, 0].y, sdim_slc.geom.point.value[0, 0].x
        self.assertEqual((40.0, -99.0), to_test)
        to_test = sdim_slc.geom.polygon.value[0, 0].centroid.y, sdim_slc.geom.polygon.value[0, 0].centroid.x
        self.assertEqual((40.0, -99.0), to_test)

        refs = [sdim_slc.geom.point.value, sdim_slc.geom.polygon.value]
        for ref in refs:
            self.assertIsInstance(ref, np.ma.MaskedArray)

        sdim_all = sdim[:, :]
        self.assertNumpyAll(sdim_all.grid.value, sdim.grid.value)

    def test_slicing_1d_none(self):
        sdim = self.get_sdim(bounds=True)
        sdim_slc = sdim[1, :]
        self.assertEqual(sdim_slc.shape, (1, 4))

    def test_point_as_value(self):
        pt = Point(100.0, 10.0)
        pt2 = Point(200.0, 20.0)
        with self.assertRaises(ValueError):
            SpatialGeometryPointDimension(value=Point(100.0, 10.0))
        with self.assertRaises(ValueError):
            SpatialGeometryPointDimension(value=[pt, pt])

        pts = np.array([[pt, pt2]], dtype=object)
        g = SpatialGeometryPointDimension(value=pts)
        self.assertEqual(g.value.mask.any(), False)
        self.assertNumpyAll(g.uid, np.ma.array([[1, 2]], dtype=np.int32))

        sgdim = SpatialGeometryDimension(point=g)
        sdim = SpatialDimension(geom=sgdim)
        self.assertEqual(sdim.shape, (1, 2))
        self.assertNumpyAll(sdim.uid, np.ma.array([[1, 2]], dtype=np.int32))
        sdim_slc = sdim[:, 1]
        self.assertEqual(sdim_slc.shape, (1, 1))
        self.assertTrue(sdim_slc.geom.point.value[0, 0].almost_equals(pt2))

    def test_grid_get_subset_bbox(self):
        for b in [True, False]:
            sdim = self.get_sdim(bounds=b)
            bg = sdim.grid.get_subset_bbox(-99, 39, -98, 39, closed=False)
            self.assertEqual(bg._value, None)
            self.assertEqual(bg.uid.shape, (1, 2))
            self.assertNumpyAll(bg.uid, np.ma.array([[6, 7]], dtype=np.int32))
            with self.assertRaises(EmptySubsetError):
                sdim.grid.get_subset_bbox(1000, 1000, 1001, 10001)

            bg2 = sdim.grid.get_subset_bbox(-99999, 1, 1, 1000)
            self.assertNumpyAll(bg2.value, sdim.grid.value)

    def test_weights(self):
        for b in [True, False]:
            sdim = self.get_sdim(bounds=b)
            ref = sdim.weights
            self.assertEqual(ref.mean(), 1.0)
            self.assertFalse(ref.mask.any())

    def test_singletons(self):
        row = VectorDimension(value=10, name='row')
        col = VectorDimension(value=100, name='col')
        grid = SpatialGridDimension(row=row, col=col, name='grid')
        self.assertNumpyAll(grid.value, np.ma.array([[[10]], [[100]]], mask=False))
        sdim = SpatialDimension(grid=grid)
        to_test = sdim.geom.point.value[0, 0].y, sdim.geom.point.value[0, 0].x
        self.assertEqual((10.0, 100.0), (to_test))

    def test_unwrap(self):
        """Test unwrapping a SpatialDimension."""

        def assertUnwrapped(arr):
            select = arr < 0
            self.assertFalse(select.any())

        sdim = self.get_sdim()
        with self.assertRaises(SpatialWrappingError):
            sdim.unwrap()
        sdim.crs = WGS84()
        self.assertEqual(sdim.wrapped_state, WrappedState.WRAPPED)
        sdim.unwrap()

        assertUnwrapped(sdim.grid.value)
        assertUnwrapped(sdim.grid.corners)
        assertUnwrapped(sdim.grid.col.value)
        assertUnwrapped(sdim.grid.col.bounds)

        assertUnwrapped(sdim.geom.grid.value)
        assertUnwrapped(sdim.geom.grid.corners)
        assertUnwrapped(sdim.geom.grid.col.value)
        assertUnwrapped(sdim.geom.grid.col.bounds)

        ref = sdim.grid.corners.data[:, 2, 2, :]
        bounds_from_corner = np.min(ref[1]), np.min(ref[0]), np.max(ref[1]), np.max(ref[0])
        self.assertNumpyAll(np.array(sdim.geom.polygon.value[2, 2].bounds), np.array(bounds_from_corner))
        self.assertEqual(sdim.geom.polygon.value[2, 2].bounds, (261.5, 37.5, 262.5, 38.5))
        self.assertNumpyAll(np.array(sdim.geom.point.value[2, 2]), np.array([262., 38.]))
        self.assertEqual(sdim.wrapped_state, WrappedState.UNWRAPPED)

    def test_wrap(self):
        """Test wrapping a SpatialDimension"""

        def assertWrapped(arr):
            select = arr >= constants.MERIDIAN_180TH
            self.assertFalse(select.any())

        sdim = self.get_sdim(crs=WGS84())
        original = deepcopy(sdim)
        sdim.unwrap()
        sdim.crs = None
        with self.assertRaises(SpatialWrappingError):
            sdim.wrap()
        sdim.crs = WGS84()
        sdim.wrap()

        assertWrapped(sdim.grid.value)
        assertWrapped(sdim.grid.corners)
        assertWrapped(sdim.grid.col.value)
        assertWrapped(sdim.grid.col.bounds)

        assertWrapped(sdim.geom.grid.value)
        assertWrapped(sdim.geom.grid.corners)
        assertWrapped(sdim.geom.grid.col.value)
        assertWrapped(sdim.geom.grid.col.bounds)

        self.assertGeometriesAlmostEquals(sdim.geom.polygon.value, original.geom.polygon.value)
        self.assertGeometriesAlmostEquals(sdim.geom.point.value, original.geom.point.value)
        self.assertNumpyAll(sdim.grid.value, original.grid.value)
        self.assertNumpyAll(sdim.grid.corners, original.grid.corners)

    def test_wrap_unwrap_non_wgs84(self):
        """Test wrapping and unwrapping with a different coordinate system."""

        sdim = self.get_sdim(crs=CoordinateReferenceSystem(epsg=2346))
        for method in ['wrap', 'unwrap']:
            with self.assertRaises(SpatialWrappingError):
                getattr(sdim, method)()

    def test_wrapped_state(self):
        sdim = self.get_sdim()
        self.assertIsNone(sdim.wrapped_state)

        sdim = self.get_sdim(crs=CFWGS84())
        self.assertEqual(sdim.wrapped_state, WrappedState.WRAPPED)


class TestSpatialGeometryDimension(TestBase):
    def get(self, **kwargs):
        with_bounds = kwargs.pop('with_bounds', True)

        row = VectorDimension(value=[2., 4.])
        col = VectorDimension(value=[4., 6.])
        if with_bounds:
            for element in [row, col]:
                element.set_extrapolated_bounds()
        grid = SpatialGridDimension(row=row, col=col)
        kwargs['grid'] = grid
        gdim = SpatialGeometryDimension(**kwargs)

        return gdim

    def test_init(self):
        with self.assertRaises(ValueError):
            SpatialGeometryDimension()

        gdim = self.get()
        self.assertIsNone(gdim.abstraction)
        self.assertIsInstance(gdim.point, SpatialGeometryPointDimension)
        self.assertIsInstance(gdim.polygon, SpatialGeometryPolygonDimension)

        gdim = self.get(abstraction='point')
        self.assertEqual(gdim.abstraction, 'point')
        self.assertIsNone(gdim.polygon)
        self.assertIsInstance(gdim.point, SpatialGeometryPointDimension)

        gdim = self.get()
        gdim2 = SpatialGeometryDimension(point=gdim.point)
        self.assertIsNone(gdim2.abstraction)

        self.assertEqual(gdim.name, 'geometry')
        self.assertEqual(gdim.point.name, 'point')
        self.assertEqual(gdim.polygon.name, 'polygon')

    def test_abstraction(self):
        gdim = self.get()
        with self.assertRaises(ValueError):
            gdim.abstraction = 'foo'
        self.assertIsInstance(gdim.polygon, SpatialGeometryPolygonDimension)
        gdim.abstraction = 'point'
        self.assertIsNone(gdim.polygon)

    def test_polygon(self):
        gdim = self.get()
        self.assertIsNone(gdim.abstraction)
        self.assertIsInstance(gdim.polygon, SpatialGeometryPolygonDimension)

        gdim = self.get()
        gdim.abstraction = 'polygon'
        self.assertIsInstance(gdim.polygon, SpatialGeometryPolygonDimension)

        gdim = self.get(with_bounds=False)
        self.assertIsNone(gdim.grid.row.bounds)
        self.assertIsNone(gdim.polygon)

    def test_get_highest_order_abstraction(self):
        gdim = self.get()
        self.assertIsNone(gdim.abstraction)
        self.assertIsInstance(gdim.get_highest_order_abstraction(), SpatialGeometryPolygonDimension)

        gdim = self.get(abstraction='point')
        self.assertIsInstance(gdim.get_highest_order_abstraction(), SpatialGeometryPointDimension)

        gdim = self.get()
        gdim.point
        gdim.grid = None
        self.assertIsNone(gdim.polygon)
        self.assertIsInstance(gdim.get_highest_order_abstraction(), SpatialGeometryPointDimension)

        gdim = self.get()
        gdim2 = SpatialGeometryDimension(point=gdim.point, abstraction='polygon')
        with self.assertRaises(ValueError):
            gdim2.get_highest_order_abstraction()


class TestSpatialGeometryPointDimension(AbstractTestSpatialDimension):
    def test_init(self):
        row = VectorDimension(value=[5])
        col = VectorDimension(value=[7])
        grid = SpatialGridDimension(row=row, col=col)
        sgpd = SpatialGeometryPointDimension(grid=grid, geom_type='Wrong')
        self.assertEqual(sgpd.name, 'wrong')
        self.assertEqual(sgpd.geom_type, 'Wrong')

        value = np.array([[Point(1, 2)]], dtype=object)
        sgpd = SpatialGeometryPointDimension(value=value)
        self.assertEqual(sgpd.name, 'point')

    def test_geom_type(self):
        row = VectorDimension(value=[5])
        col = VectorDimension(value=[7])
        grid = SpatialGridDimension(row=row, col=col)
        sgpd = SpatialGeometryPointDimension(grid=grid)
        self.assertEqual(sgpd.geom_type, 'Point')

        mp = MultiPoint([Point(1, 2), Point(3, 4)])
        value = np.array([[None, None]])
        value[0, 1] = Point(3, 4)
        value[0, 0] = mp
        sgpd = SpatialGeometryPointDimension(value=value)
        self.assertEqual(sgpd.geom_type, 'Point')
        sgpd = SpatialGeometryPointDimension(value=value, geom_type=None)
        self.assertEqual(sgpd.geom_type, 'Point')

        sgpd = SpatialGeometryPointDimension(value=value, geom_type='auto')
        self.assertEqual(sgpd.geom_type, 'MultiPoint')

    def test_get_intersects_masked(self):
        sdim = self.get_sdim(crs=WGS84())
        self.assertIsNotNone(sdim.grid)
        sdim.assert_uniform_mask()
        poly = wkt.loads(
            'POLYGON((-98.26574367088608142 40.19952531645570559,-98.71764240506330168 39.54825949367089066,-99.26257911392406186 39.16281645569620906,-99.43536392405064817 38.64446202531645724,-98.78409810126584034 38.33876582278481493,-98.23916139240508016 37.71408227848101546,-97.77397151898735217 37.67420886075949937,-97.62776898734178133 38.15268987341772799,-98.39865506329114453 38.52484177215190186,-98.23916139240508016 39.33560126582278826,-97.73409810126582897 39.58813291139241386,-97.52143987341773368 40.27927215189873777,-97.52143987341773368 40.27927215189873777,-98.26574367088608142 40.19952531645570559))')
        actual_mask = np.array([[True, True, False, True], [True, False, True, True], [True, True, False, True]])
        actual_uid_data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.int32)
        for use_spatial_index in [True, False]:
            ret = sdim.geom.point.get_intersects_masked(poly, use_spatial_index=use_spatial_index)
            self.assertNumpyAll(actual_mask, ret.value.mask)
            self.assertNumpyAll(actual_mask, ret.uid.mask)
            self.assertNumpyAll(actual_uid_data, ret.uid.data)
            self.assertFalse(sdim.uid.mask.any())
            for element in ret.value.data.flat:
                self.assertIsInstance(element, Point)
            self.assertIsNotNone(sdim.grid)

        # Test pre-masked values in geometry are okay for intersects operation.
        value = [Point(1, 1), Point(2, 2), Point(3, 3)]
        value = np.ma.array(value, mask=[False, True, False], dtype=object).reshape(-1, 1)
        s = SpatialGeometryPointDimension(value=value)
        b = box(0, 0, 5, 5)
        res = s.get_intersects_masked(b)
        self.assertNumpyAll(res.value.mask, value.mask)


class TestSpatialGeometryPolygonDimension(AbstractTestSpatialDimension):
    def test_init(self):
        with self.assertRaises(ValueError):
            SpatialGeometryPolygonDimension()

        row = VectorDimension(value=[2, 3])
        col = VectorDimension(value=[4, 5])
        grid = SpatialGridDimension(row=row, col=col)
        self.assertIsNone(grid.corners)
        with self.assertRaises(ValueError):
            SpatialGeometryPolygonDimension(grid=grid)

        value = grid.value
        grid = SpatialGridDimension(value=value)
        with self.assertRaises(ValueError):
            SpatialGeometryPolygonDimension(grid=grid)

        row = VectorDimension(value=[2, 3])
        row.set_extrapolated_bounds()
        col = VectorDimension(value=[4, 5])
        col.set_extrapolated_bounds()
        grid = SpatialGridDimension(row=row, col=col)
        gd = SpatialGeometryPolygonDimension(grid=grid)
        self.assertEqual(gd.name, 'polygon')
        self.assertIsInstance(gd, SpatialGeometryPointDimension)
        self.assertEqual(gd.geom_type, 'MultiPolygon')

    def test_get_value(self):
        # the ordering of vertices when creating from corners is slightly different

        keywords = dict(with_grid_row_col_bounds=[True, False],
                        with_grid_mask=[True, False])
        for k in itr_products_keywords(keywords, as_namedtuple=True):
            sdim = self.get_sdim()
            if k.with_grid_mask:
                sdim.grid.value.mask[:, 1, 1] = True
            sdim.grid.corners
            self.assertIsNone(sdim.geom.polygon._value)
            if not k.with_grid_row_col_bounds:
                sdim.grid.row.bounds = None
                sdim.grid.col.bounds = None
                actual = self.polygon_value_alternate_ordering
            else:
                actual = self.polygon_value
            if k.with_grid_mask:
                actual.mask[1, 1] = True
            poly = sdim.geom.polygon.value
            self.assertGeometriesAlmostEquals(poly, actual)

    def test_write_to_netcdf_dataset_ugrid(self):
        ugrid_polygons = [
            'POLYGON((-1.5019011406844105 0.18377693282636276,-1.25475285171102646 0.02534854245880869,-1.35614702154626099 -0.28517110266159684,-1.68567807351077303 -0.50697084917617241,-1.99619771863117879 -0.41191381495564006,-2.08491761723700897 -0.24714828897338403,-1.9264892268694549 -0.03802281368821281,-1.88212927756653992 0.13307984790874539,-1.5019011406844105 0.18377693282636276))',
            'POLYGON((-2.25602027883396694 0.63371356147021585,-1.76172370088719887 0.51330798479087481,-1.88212927756653992 0.13307984790874539,-1.9264892268694549 -0.03802281368821281,-2.30671736375158432 0.01901140684410674,-2.51584283903675532 0.27249683143219272,-2.52217997465145771 0.48795944233206612,-2.25602027883396694 0.63371356147021585))',
            'POLYGON((-1.55893536121673004 0.86818757921419554,-1.03929024081115307 0.65906210392902409,-1.07097591888466415 0.46261089987325743,-1.5019011406844105 0.18377693282636276,-1.88212927756653992 0.13307984790874539,-1.76172370088719887 0.51330798479087481,-1.55893536121673004 0.86818757921419554))',
            'POLYGON((-2.13561470215462634 0.87452471482889749,-1.83143219264892276 0.98225602027883419,-1.83143219264892276 0.98225602027883419,-1.55893536121673004 0.86818757921419554,-1.58428390367553851 0.66539923954372648,-1.76172370088719887 0.51330798479087481,-2.12294043092522156 0.44993662864385309,-2.25602027883396694 0.63371356147021585,-2.13561470215462634 0.87452471482889749))'
        ]

        polygons = [wkt.loads(xx) for xx in ugrid_polygons]
        polygons = np.atleast_2d(np.array(polygons))
        spoly = SpatialGeometryPolygonDimension(value=polygons)

        path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(path, 'w') as ds:
            spoly.write_to_netcdf_dataset_ugrid(ds)
        with self.nc_scope(path) as ds:
            self.assertEqual(len(ds.dimensions['nMesh2_face']), 4)

        shp_path = os.path.join(self.current_dir_output, 'ugrid.shp')
        mesh2_nc_to_fiona(path, shp_path)
        with fiona.open(shp_path) as source:
            for record in source:
                geom = shape(record['geometry'])
                check = [geom.almost_equals(xx) for xx in polygons.flat]
                self.assertEqual(sum(check), 1)

        # test with a mask
        spoly.value.mask[0, 1] = True
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(path, 'w') as ds:
            spoly.write_to_netcdf_dataset_ugrid(ds)
        with self.nc_scope(path) as ds:
            self.assertEqual(len(ds.dimensions['nMesh2_face']), 3)


class TestSpatialGridDimension(AbstractTestSpatialDimension):
    def assertGridCorners(self, grid):
        """
        :type grid: :class:`ocgis.interface.base.dimension.spatial.SpatialGridDimension`
        """

        assert (grid.corners is not None)

        def _get_is_ascending_(arr):
            """
            Return ``True`` if the array is ascending from index 0 to -1.

            :type arr: :class:`numpy.core.multiarray.ndarray`
            :rtype: bool
            """

            assert (arr.ndim == 1)
            if arr[0] < arr[-1]:
                ret = True
            else:
                ret = False

            return ret

        for ii, jj in itertools.product(range(grid.shape[0]), range(grid.shape[1])):
            pt = Point(grid.value.data[1, ii, jj], grid.value.data[0, ii, jj])
            poly_corners = grid.corners.data[:, ii, jj]
            rtup = (poly_corners[0, :].min(), poly_corners[0, :].max())
            ctup = (poly_corners[1, :].min(), poly_corners[1, :].max())
            poly = make_poly(rtup, ctup)
            self.assertTrue(poly.contains(pt))

        for (ii, jj), m in iter_array(grid.value.mask[0, :, :], return_value=True):
            if m:
                self.assertTrue(grid.corners.mask[:, ii, jj].all())
            else:
                self.assertFalse(grid.corners.mask[:, ii, jj].any())

        if grid.row is not None or grid.col is not None:
            self.assertEqual(_get_is_ascending_(grid.row.value), _get_is_ascending_(grid.corners.data[0, :, 0][:, 0]))
            self.assertEqual(_get_is_ascending_(grid.col.value), _get_is_ascending_(grid.corners.data[1, 0, :][:, 0]))

    def iter_grid_combinations_for_corners(self):
        """Yield grid combinations without corners."""

        keywords = dict(with_row_column=[True, False],
                        with_row_column_bounds=[True, False],
                        vectorized=[True, False],
                        with_masked=[True, False],
                        reverse_row=[True, False],
                        reverse_col=[True, False])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            sdim = self.get_sdim(crs=WGS84())
            self.assertIsNone(sdim.grid._corners)
            if k.reverse_row:
                sdim.grid.row._value = np.flipud(sdim.grid.row.value)
                sdim.grid.row.bounds = np.fliplr(np.flipud(sdim.grid.row.bounds))
            if k.reverse_col:
                sdim.grid.col._value = np.flipud(sdim.grid.col.value)
                sdim.grid.col.bounds = np.fliplr(np.flipud(sdim.grid.col.bounds))
            sdim.grid.value
            if k.with_masked:
                sdim.grid.value.mask[:, 0, 1] = True
                sdim.grid.value.mask[:, -1, -1] = True
            if not k.vectorized:
                to_crs = CoordinateReferenceSystem(epsg=2163)
                self.assertIsNone(sdim.grid._corners)
                sdim.update_crs(to_crs)
                self.assertIsNotNone(sdim.grid._corners)
            else:
                if not k.with_row_column:
                    sdim.grid.row = None
                    sdim.grid.col = None
                else:
                    if not k.with_row_column_bounds:
                        sdim.grid.row.value
                        sdim.grid.row.bounds = None
                        sdim.grid.col.value
                        sdim.grid.col.bounds = None
            self.assertIsNotNone(sdim.grid.value)

            if k.with_masked:
                actual_mask = np.array(
                    [[[False, True, False, False], [False, False, False, False], [False, False, False, True]],
                     [[False, True, False, False], [False, False, False, False], [False, False, False, True]]])
                self.assertNumpyAll(actual_mask, sdim.grid.value.mask)

            yield sdim.grid

    def test_init(self):
        self.assertEqual(SpatialGridDimension.__bases__, (AbstractUidValueDimension,))

        with self.assertRaises(ValueError):
            SpatialGridDimension()
        row = VectorDimension(value=[5])
        col = VectorDimension(value=[6])
        grid = SpatialGridDimension(row=row, col=col)
        self.assertEqual(grid.name, 'grid')
        self.assertEqual(grid.row.name, 'yc')
        self.assertEqual(grid.col.name, 'xc')
        self.assertEqual(grid.name_row, constants.DEFAULT_NAME_ROW_COORDINATES)
        self.assertEqual(grid.name_col, constants.DEFAULT_NAME_COL_COORDINATES)

        grid = SpatialGridDimension(row=row, col=col, name_row='foo', name_col='whatever')
        self.assertEqual(grid.name_row, 'foo')
        self.assertEqual(grid.name_col, 'whatever')

    def test_assert_uniform_mask(self):
        """Test masks are uniform across major spatial components."""

        sdim = self.get_sdim()
        sdim.assert_uniform_mask()

        sdim.uid.mask[1, 1] = True
        with self.assertRaises(AssertionError):
            sdim.assert_uniform_mask()
        sdim.uid.mask[1, 1] = False

        sdim.assert_uniform_mask()
        sdim.grid.value.mask[0, 2, 2] = True
        with self.assertRaises(AssertionError):
            sdim.assert_uniform_mask()
        sdim.grid.value.mask[0, 2, 2] = False

        sdim.assert_uniform_mask()
        sdim.geom.point.value.mask[2, 2] = True
        with self.assertRaises(AssertionError):
            sdim.assert_uniform_mask()
        sdim.geom.point.value.mask[2, 2] = False

        sdim.assert_uniform_mask()
        sdim.geom.polygon.value.mask[2, 2] = True
        with self.assertRaises(AssertionError):
            sdim.assert_uniform_mask()
        sdim.geom.polygon.value.mask[2, 2] = False

        sdim.grid.corners.mask[0, 2, 1, 3] = True
        with self.assertRaises(AssertionError):
            sdim.assert_uniform_mask()
        sdim.grid.corners.mask[0, 2, 1, 3] = False
        self.assertIsNotNone(sdim.grid.corners)
        sdim.assert_uniform_mask()

    def test_corners(self):
        for grid in self.iter_grid_combinations_for_corners():
            try:
                self.assertGridCorners(grid)
            except AssertionError:
                if grid.row is None or grid.row.bounds is None:
                    continue
                else:
                    raise

    def test_corners_as_parameter(self):
        """Test passing bounds during initialization."""

        grid = SpatialGridDimension(value=self.grid_value_regular, corners=self.grid_corners_regular)
        sub = grid[1, 2]
        self.assertEqual(sub.corners.shape, (2, 1, 1, 4))
        actual = np.ma.array([[[[39.5, 39.5, 38.5, 38.5]]], [[[-98.5, -97.5, -97.5, -98.5]]]], mask=False)
        self.assertNumpyAll(sub.corners, actual)

    def test_corners_esmf(self):
        sdim = self.get_sdim()
        actual = np.array([[[40.5, 40.5, 40.5, 40.5, 40.5], [39.5, 39.5, 39.5, 39.5, 39.5],
                            [38.5, 38.5, 38.5, 38.5, 38.5], [37.5, 37.5, 37.5, 37.5, 37.5]],
                           [[-100.5, -99.5, -98.5, -97.5, -96.5], [-100.5, -99.5, -98.5, -97.5, -96.5],
                            [-100.5, -99.5, -98.5, -97.5, -96.5], [-100.5, -99.5, -98.5, -97.5, -96.5]]],
                          dtype=sdim.grid.value.dtype)
        self.assertNumpyAll(actual, sdim.grid.corners_esmf)

    def test_extent_and_extent_polygon(self):
        for grid in self.iter_grid_combinations_for_corners():
            extent = grid.extent
            self.assertEqual(len(extent), 4)
            self.assertTrue(extent[0] < extent[2])
            self.assertTrue(extent[1] < extent[3])
            self.assertEqual(extent, grid.extent_polygon.bounds)

    def test_load_from_source_grid_slicing(self):
        row = VectorDimension(src_idx=[10, 20, 30, 40], name='row', request_dataset='foo')
        self.assertEqual(row.name, 'row')
        col = VectorDimension(src_idx=[100, 200, 300], name='col', request_dataset='foo')
        grid = SpatialGridDimension(row=row, col=col, name='grid')
        self.assertEqual(grid.shape, (4, 3))
        grid_slc = grid[1, 2]
        self.assertEqual(grid_slc.shape, (1, 1))
        with self.assertRaises(NotImplementedError):
            grid_slc.value
        with self.assertRaises(NotImplementedError):
            grid_slc.row.bounds
        self.assertNumpyAll(grid_slc.row._src_idx, np.array([20]))
        self.assertNumpyAll(grid_slc.col._src_idx, np.array([300]))
        self.assertEqual(grid_slc.row.name, 'row')
        self.assertEqual(grid_slc.uid, np.array([[6]], dtype=np.int32))

    def test_set_extrapolated_corners(self):
        sdim = self.get_sdim(bounds=False)
        self.assertIsNone(sdim.grid.corners)
        sdim.grid.set_extrapolated_corners()
        sdim2 = self.get_sdim()
        self.assertNumpyAll(sdim2.grid.corners, sdim.grid.corners)

        # test with a mask
        np.random.seed(1)
        sdim = self.get_sdim(bounds=False)
        mask = np.random.randint(0, 2, size=sdim.shape).astype(bool)
        sdim.set_mask(mask)
        self.assertIsNone(sdim.grid.corners)
        sdim.grid.set_extrapolated_corners()
        self.assertTrue(sdim.grid.corners.mask.any())
        for (ii, jj), val in iter_array(mask, return_value=True):
            ref = sdim.grid.corners[:, ii, jj, :]
            if val:
                self.assertTrue(ref.mask.all())
            else:
                self.assertFalse(ref.mask.any())

        # test with corners already available
        sdim = self.get_sdim()
        with self.assertRaises(BoundsAlreadyAvailableError):
            sdim.grid.set_extrapolated_corners()

    def test_singletons(self):
        row = VectorDimension(value=10, name='row')
        col = VectorDimension(value=100, name='col')
        grid = SpatialGridDimension(row=row, col=col, name='grid')
        self.assertNumpyAll(grid.value, np.ma.array([[[10]], [[100]]], mask=False))

    def test_validate(self):
        with self.assertRaises(ValueError):
            SpatialGridDimension()

    def test_without_row_and_column(self):
        row = np.arange(39, 42.5, 0.5)
        col = np.arange(-104, -95, 0.5)
        x, y = np.meshgrid(col, row)
        value = np.zeros([2] + list(x.shape))
        value = np.ma.array(value, mask=False)
        value[0, :, :] = y
        value[1, :, :] = x
        minx, miny, maxx, maxy = x.min(), y.min(), x.max(), y.max()
        grid = SpatialGridDimension(value=value)
        sub = grid.get_subset_bbox(minx, miny, maxx, maxy, closed=False)
        self.assertNumpyAll(sub.value, value)

    def test_write_to_netcdf_dataset(self):
        path = os.path.join(self.current_dir_output, 'foo.nc')

        kwds = dict(with_rc=[True, False],
                    with_corners=[False, True])

        for k in self.iter_product_keywords(kwds):
            row = VectorDimension(value=[4., 5.])
            col = VectorDimension(value=[6., 7.])
            grid = SpatialGridDimension(row=row, col=col)

            if k.with_corners:
                row.set_extrapolated_bounds()
                col.set_extrapolated_bounds()
                grid.corners

            if not k.with_rc:
                grid.value
                grid.row = None
                grid.col = None

            with self.nc_scope(path, mode='w') as ds:
                grid.write_netcdf(ds)
            with self.nc_scope(path) as ds:
                if k.with_rc:
                    self.assertNumpyAll(ds.variables[grid.row.name][:], row.value)
                    self.assertNumpyAll(ds.variables[grid.col.name][:], col.value)
                else:
                    yc = ds.variables[constants.DEFAULT_NAME_ROW_COORDINATES]
                    xc = ds.variables[constants.DEFAULT_NAME_COL_COORDINATES]
                    self.assertNumpyAll(yc[:], grid.value[0].data)
                    self.assertNumpyAll(xc[:], grid.value[1].data)
                    self.assertEqual(yc.axis, 'Y')
                    self.assertEqual(xc.axis, 'X')
                if k.with_corners and not k.with_rc:
                    name_yc_corners, name_xc_corners = ['{0}_corners'.format(xx) for xx in
                                                        [constants.DEFAULT_NAME_ROW_COORDINATES,
                                                         constants.DEFAULT_NAME_COL_COORDINATES]]
                    for idx, name in zip([0, 1], [name_yc_corners, name_xc_corners]):
                        var = ds.variables[name]
                        self.assertNumpyAll(var[:], grid.corners[idx].data)
                    self.assertEqual(ds.variables[constants.DEFAULT_NAME_ROW_COORDINATES].corners, name_yc_corners)
                    self.assertEqual(ds.variables[constants.DEFAULT_NAME_COL_COORDINATES].corners, name_xc_corners)

        # test with names for the rows and columns and no row/col objects
        row = VectorDimension(value=[4., 5.])
        col = VectorDimension(value=[6., 7.])
        grid = SpatialGridDimension(row=row, col=col, name_row='imrow', name_col='im_col')
        grid.value
        grid.row = None
        grid.col = None
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            grid.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            self.assertEqual(ds.variables.keys(), ['imrow', 'im_col'])
