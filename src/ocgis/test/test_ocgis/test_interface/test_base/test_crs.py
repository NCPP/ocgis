import os
from copy import deepcopy

import netCDF4 as nc
import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.geometry.multipolygon import MultiPolygon

import ocgis
from ocgis import constants
from ocgis.constants import WrappedState, WrapAction
from ocgis.exc import SpatialWrappingError
from ocgis.interface.base.crs import CoordinateReferenceSystem, WGS84, \
    CFAlbersEqualArea, CFLambertConformal, CFRotatedPole, CFWGS84, Spherical, WrappableCoordinateReferenceSystem, \
    CFCoordinateReferenceSystem
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialGridDimension, \
    SpatialDimension
from ocgis.test.base import TestBase, nc_scope
from ocgis.test.base import attr
from ocgis.util.helpers import make_poly
from ocgis.util.itester import itr_products_keywords


class TestCoordinateReferenceSystem(TestBase):
    def test_init(self):
        keywords = dict(
            value=[None, {'no_defs': True, 'ellps': 'WGS84', 'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0'}],
            epsg=[None, 4326],
            proj4=[None, '+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs '])

        prev_crs = None
        for k in itr_products_keywords(keywords):
            try:
                crs = CoordinateReferenceSystem(**k)
            except ValueError:
                if all([ii is None for ii in k.values()]):
                    continue
                else:
                    raise
            self.assertEqual(crs.proj4, '+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs ')
            try:
                self.assertDictEqual(crs.value,
                                     {'no_defs': True, 'ellps': 'WGS84', 'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0'})
            except AssertionError:
                self.assertDictEqual(crs.value,
                                     {'no_defs': True, 'datum': 'WGS84', 'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0'})
            if prev_crs is not None:
                self.assertEqual(crs, prev_crs)
            prev_crs = deepcopy(crs)

        # test with a name parameter
        crs = CoordinateReferenceSystem(epsg=4326)
        self.assertEqual(crs.name, constants.DEFAULT_COORDINATE_SYSTEM_NAME)
        crs = CoordinateReferenceSystem(epsg=4326, name='foo')
        self.assertEqual(crs.name, 'foo')

        # test using the init parameter
        value = {'init': 'epsg:4326'}
        self.assertEqual(CoordinateReferenceSystem(value=value), WGS84())

    def test_ne(self):
        crs1 = CoordinateReferenceSystem(epsg=4326)
        crs2 = CoordinateReferenceSystem(epsg=2136)

        self.assertNotEqual(crs1, crs2)
        self.assertNotEqual(crs2, crs1)
        self.assertNotEqual(crs1, None)
        self.assertNotEqual(None, crs1)

        # try nonetype and string
        self.assertNotEqual(None, crs1)
        self.assertNotEqual('input', crs1)

    def test_write_to_rootgrp(self):
        crs = CoordinateReferenceSystem(epsg=4326, name='hello_world')
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with nc_scope(path, 'w') as ds:
            variable = crs.write_to_rootgrp(ds)
            self.assertIsInstance(variable, nc.Variable)
            with self.assertRaises(AttributeError):
                variable.proj4


class TestCFCoordinateReferenceSystem(TestBase):
    def test_init(self):
        c = CFLambertConformal(false_easting=0, false_northing=0, latitude_of_projection_origin=38,
                               standard_parallel=[30., 60.], longitude_of_central_meridian=-77., units='km')
        self.assertEqual(c.name, c.grid_mapping_name)


class TestWrappableCoordinateSystem(TestBase):
    create_dir = False

    def test_get_wrap_action(self):
        _w = WrappableCoordinateReferenceSystem
        possible = [WrappedState.WRAPPED, WrappedState.UNWRAPPED, WrappedState.UNKNOWN, 'foo']
        keywords = dict(state_src=possible,
                        state_dst=possible)
        for k in itr_products_keywords(keywords, as_namedtuple=True):
            try:
                ret = _w.get_wrap_action(k.state_src, k.state_dst)
            except ValueError:
                self.assertTrue(k.state_src == 'foo' or k.state_dst == 'foo')
                continue
            if k.state_dst == WrappedState.UNKNOWN:
                self.assertIsNone(ret)
            elif k.state_src == WrappedState.UNWRAPPED and k.state_dst == WrappedState.WRAPPED:
                self.assertEqual(ret, WrapAction.WRAP)
            elif k.state_src == WrappedState.WRAPPED and k.state_dst == WrappedState.UNWRAPPED:
                self.assertEqual(ret, WrapAction.UNWRAP)
            else:
                self.assertIsNone(ret)

    def test_get_wrapped_state(self):
        refv = WrappableCoordinateReferenceSystem
        refm = refv.get_wrapped_state

        # Test grid wrapping ###########################################################################################

        row = VectorDimension(value=[50, 60])

        col = VectorDimension(value=[0, 90, 180])
        grid = SpatialGridDimension(row=row, col=col)
        sdim = SpatialDimension(grid=grid)
        self.assertEqual(refm(sdim), WrappedState.UNKNOWN)

        col = VectorDimension(value=[-170, 0, 30])
        grid = SpatialGridDimension(row=row, col=col)
        sdim = SpatialDimension(grid=grid)
        self.assertEqual(refm(sdim), WrappedState.WRAPPED)

        col = VectorDimension(value=[0, 90, 180, 270])
        grid = SpatialGridDimension(row=row, col=col)
        sdim = SpatialDimension(grid=grid)
        self.assertEqual(refm(sdim), WrappedState.UNWRAPPED)

        # Test geometry wrapping #######################################################################################

        for with_polygon in [True, False]:
            row = VectorDimension(value=[50, 60])
            col = VectorDimension(value=[155, 165, 175])
            if with_polygon:
                row.set_extrapolated_bounds()
                col.set_extrapolated_bounds()
            grid = SpatialGridDimension(row=row, col=col)
            sdim = SpatialDimension(grid=grid)
            sdim.grid = None
            self.assertEqual(refm(sdim), WrappedState.UNKNOWN)

            row = VectorDimension(value=[50, 60])
            col = VectorDimension(value=[160, 170, 180])
            if with_polygon:
                row.set_extrapolated_bounds()
                col.set_extrapolated_bounds()
            grid = SpatialGridDimension(row=row, col=col)
            sdim = SpatialDimension(grid=grid)
            sdim.grid = None
            if with_polygon:
                actual = WrappedState.UNWRAPPED
            else:
                actual = WrappedState.UNKNOWN
            self.assertEqual(refm(sdim), actual)

            row = VectorDimension(value=[50, 60])
            col = VectorDimension(value=[-160, -150, -140])
            if with_polygon:
                row.set_extrapolated_bounds()
                col.set_extrapolated_bounds()
            grid = SpatialGridDimension(row=row, col=col)
            sdim = SpatialDimension(grid=grid)
            sdim.grid = None
            self.assertEqual(refm(sdim), WrappedState.WRAPPED)

    def test_get_wrapped_state_from_array(self):

        def _run_(arr, actual_wrapped_state):
            ret = WrappableCoordinateReferenceSystem._get_wrapped_state_from_array_(arr)
            self.assertEqual(ret, actual_wrapped_state)

        arr = np.array([-170])
        _run_(arr, WrappedState.WRAPPED)

        arr = np.array([270])
        _run_(arr, WrappedState.UNWRAPPED)

        arr = np.array([30])
        _run_(arr, WrappedState.UNKNOWN)

        arr = np.array([-180, 0, 30])
        _run_(arr, WrappedState.WRAPPED)

        arr = np.array([0])
        _run_(arr, WrappedState.UNKNOWN)

        arr = np.array([0, 30, 50])
        _run_(arr, WrappedState.UNKNOWN)

        arr = np.array([0, 30, 50, 181])
        _run_(arr, WrappedState.UNWRAPPED)

        arr = np.array([0, 30, 50, 180])
        _run_(arr, WrappedState.UNKNOWN)

        arr = np.array([-180])
        _run_(arr, WrappedState.WRAPPED)

        arr = np.array([-180, 0, 50])
        _run_(arr, WrappedState.WRAPPED)

    def test_get_wrapped_state_from_geometry(self):
        geoms = [Point(-130, 40),
                 MultiPoint([Point(-130, 40), Point(30, 50)]),
                 make_poly((30, 40), (-130, -120)),
                 MultiPolygon([make_poly((30, 40), (-130, -120)), make_poly((30, 40), (130, 160))])]

        for geom in geoms:
            ret = WrappableCoordinateReferenceSystem._get_wrapped_state_from_geometry_(geom)
            self.assertEqual(ret, WrappedState.WRAPPED)

        pt = Point(270, 50)
        ret = WrappableCoordinateReferenceSystem._get_wrapped_state_from_geometry_(pt)
        self.assertEqual(ret, WrappedState.UNWRAPPED)


class TestSpherical(TestBase):
    def test_init(self):
        crs = Spherical()
        self.assertDictEqual(crs.value, {'a': 6370997, 'no_defs': True, 'b': 6370997, 'proj': 'longlat',
                                         'towgs84': '0,0,0,0,0,0,0'})

        crs = Spherical(semi_major_axis=6370998.1)
        self.assertDictEqual(crs.value, {'a': 6370998.1, 'no_defs': True, 'b': 6370998.1, 'proj': 'longlat',
                                         'towgs84': '0,0,0,0,0,0,0'})
        self.assertEqual(crs.name, 'latitude_longitude')

    def test_place_prime_meridian_array(self):
        arr = np.array([123, 180, 200, 180], dtype=float)
        ret = Spherical._place_prime_meridian_array_(arr)
        self.assertNumpyAll(ret, np.array([False, True, False, True]))
        self.assertNumpyAll(arr, np.array([123., constants.MERIDIAN_180TH, 200., constants.MERIDIAN_180TH]))

    @attr('data')
    def test_wrap_unwrap_with_mask(self):
        """Test wrapped and unwrapped geometries with a mask ensuring that masked values are wrapped and unwrapped."""

        rd = self.test_data.get_rd('cancm4_tas')

        ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23], spatial_wrapping='wrap')
        ret = ops.execute()
        sdim = ret[23]['tas'].spatial
        actual = np.array([[[34.88252349825162, 34.88252349825162, 34.88252349825162],
                            [37.67309213352349, 37.67309213352349, 37.67309213352349],
                            [40.4636506825932, 40.4636506825932, 40.4636506825932],
                            [43.254197169829105, 43.254197169829105, 43.254197169829105]],
                           [[-120.9375, -118.125, -115.3125], [-120.9375, -118.125, -115.3125],
                            [-120.9375, -118.125, -115.3125], [-120.9375, -118.125, -115.3125]]],
                          dtype=sdim.grid.value.dtype)
        self.assertNumpyAll(actual, sdim.grid.value.data)

        Spherical().unwrap(sdim)
        actual = np.array([[[34.88252349825162, 34.88252349825162, 34.88252349825162],
                            [37.67309213352349, 37.67309213352349, 37.67309213352349],
                            [40.4636506825932, 40.4636506825932, 40.4636506825932],
                            [43.254197169829105, 43.254197169829105, 43.254197169829105]],
                           [[239.0625, 241.875, 244.6875], [239.0625, 241.875, 244.6875], [239.0625, 241.875, 244.6875],
                            [239.0625, 241.875, 244.6875]]], dtype=sdim.grid.value.dtype)
        self.assertNumpyAll(actual, sdim.grid.value.data)

    def test_wrap_normal(self):
        """Test exception thrown when attempting to wrap already wrapped coordinate values."""

        row = VectorDimension(value=40., bounds=[38., 42.])
        col = VectorDimension(value=0., bounds=[-1., 1.])
        grid = SpatialGridDimension(row=row, col=col)
        self.assertEqual(grid.resolution, 3.0)
        sdim = SpatialDimension(grid=grid, crs=Spherical())
        self.assertEqual(sdim.wrapped_state, WrappedState.UNKNOWN)
        with self.assertRaises(SpatialWrappingError):
            sdim.crs.wrap(sdim)

    def test_wrap_360(self):
        """Test wrapping."""

        row = VectorDimension(value=40., bounds=[38., 42.])
        col = VectorDimension(value=181.5, bounds=[181., 182.])
        grid = SpatialGridDimension(row=row, col=col)
        self.assertEqual(grid.value[1, 0, 0], 181.5)
        sdim = SpatialDimension(grid=grid, crs=WGS84())
        orig_sdim = deepcopy(sdim)
        orig_grid = deepcopy(sdim.grid)
        sdim.crs.wrap(sdim)
        self.assertNumpyAll(np.array(sdim.geom.point.value[0, 0]), np.array([-178.5, 40.]))
        self.assertEqual(sdim.geom.polygon.value[0, 0].bounds, (-179.0, 38.0, -178.0, 42.0))
        self.assertNumpyNotAll(orig_grid.value, sdim.grid.value)
        sdim.crs.unwrap(sdim)
        to_test = ([sdim.grid.value, orig_sdim.grid.value], [sdim.grid.corners, orig_sdim.grid.corners])
        for tt in to_test:
            self.assertNumpyAll(*tt)

    def test_wrap_360_prime_meridian(self):
        """Test wrapping with bounds interacting with the prime meridian."""

        def _get_sdim_(value, bounds):
            row1 = VectorDimension(value=40., bounds=[38., 42.])
            try:
                bounds = map(float, bounds)
            except TypeError:
                bounds = [map(float, b) for b in bounds]
            try:
                value = float(value)
            except TypeError:
                value = [float(v) for v in value]
            col1 = VectorDimension(value=value, bounds=bounds)
            grid1 = SpatialGridDimension(row=row1, col=col1)
            sdim1 = SpatialDimension(grid=grid1, crs=Spherical())
            return deepcopy(sdim1), sdim1

        # bounds values at the prime meridian of 180.
        orig, sdim = _get_sdim_(178, [176, 180.])
        self.assertEqual(sdim.wrapped_state, WrappedState.UNKNOWN)

        # bounds values on the other side of the prime meridian
        orig, sdim = _get_sdim_(182, [180, 184])
        sdim.wrap()
        self.assertNumpyAll(sdim.grid.col.bounds, np.array([[180., -176.]]))
        self.assertNumpyAll(sdim.grid.row.bounds, np.array([[38., 42.]]))
        self.assertNumpyAll(sdim.grid.corners, np.ma.array([[[[38.0, 38.0, 42.0, 42.0]]],
                                                            [[[180.0, -176.0, -176.0, 180.0]]]]))
        self.assertEqual(sdim.geom.polygon.value[0, 0][0].bounds, (-180.0, 38.0, -176.0, 42.0))
        self.assertNumpyAll(np.array(sdim.geom.point.value[0, 0]), np.array([-178., 40.]))

        # centroid directly on prime meridian
        orig, sdim = _get_sdim_(180, [178, 182])
        self.assertEqual(sdim.wrapped_state, WrappedState.UNKNOWN)
        with self.assertRaises(SpatialWrappingError):
            sdim.wrap()

        # no row/column bounds but with corners
        orig, sdim = _get_sdim_([182, 186], [[180, 184], [184, 188]])
        sdim.grid.corners
        sdim.grid.row.bounds
        sdim.grid.row.bounds = None
        sdim.grid.col.bounds
        sdim.grid.col.bounds = None
        sdim.wrap()
        self.assertIsNone(sdim.grid.corners)

        # unwrap a wrapped spatial dimension making sure the unwrapped multipolygon bounds are the same as the wrapped
        # polygon bounds.
        row = VectorDimension(value=40, bounds=[38, 42])
        col = VectorDimension(value=185, bounds=[184, 186])
        grid = SpatialGridDimension(row=row, col=col)
        sdim = SpatialDimension(grid=grid, crs=Spherical())
        orig_sdim = deepcopy(sdim)
        self.assertEqual(orig_sdim.wrapped_state, WrappedState.UNWRAPPED)
        sdim.crs.wrap(sdim)
        self.assertEqual(sdim.wrapped_state, WrappedState.WRAPPED)
        sdim.crs.unwrap(sdim)
        self.assertEqual(orig_sdim.geom.polygon.value[0, 0].bounds, sdim.geom.polygon.value[0, 0].bounds)


class TestWGS84(TestBase):
    def test_init(self):
        self.assertEqual(WGS84(), CoordinateReferenceSystem(epsg=4326))
        self.assertIsInstance(WGS84(), WrappableCoordinateReferenceSystem)
        self.assertNotIsInstance(WGS84(), Spherical)
        self.assertEqual(WGS84().name, 'latitude_longitude')


class TestCFWGS84(TestBase):
    def test_init(self):
        crs = CFWGS84()
        self.assertEqual(crs.map_parameters_values, {})
        self.assertIsInstance(crs, WGS84)
        self.assertIsInstance(crs, CFCoordinateReferenceSystem)


class TestCFAlbersEqualArea(TestBase):
    def test_constructor(self):
        crs = CFAlbersEqualArea(standard_parallel=[29.5, 45.5], longitude_of_central_meridian=-96,
                                latitude_of_projection_origin=37.5, false_easting=0,
                                false_northing=0)
        self.assertEqual(crs.value, {'lon_0': -96, 'ellps': 'WGS84', 'y_0': 0, 'no_defs': True, 'proj': 'aea', 'x_0': 0,
                                     'units': 'm', 'lat_2': 45.5, 'lat_1': 29.5, 'lat_0': 37.5})

    def test_empty(self):
        with self.assertRaises(KeyError):
            CFAlbersEqualArea()

    def test_bad_parms(self):
        with self.assertRaises(KeyError):
            CFAlbersEqualArea(standard_parallel=[29.5, 45.5], longitude_of_central_meridian=-96,
                              latitude_of_projection_origin=37.5, false_easting=0,
                              false_nothing=0)


class TestCFLambertConformalConic(TestBase):
    @property
    def archetype_minimal_metadata(self):
        min_meta = {'variables': {'pr': {'name': 'pr',
                                         'attrs': {'grid_mapping': 'Lambert_Conformal'}
                                         },
                                  'Lambert_Conformal': {'name': 'Lambert_Conformal',
                                                        'attrs': {'false_easting': 3325000.0,
                                                                  'standard_parallel': [30., 60.],
                                                                  'false_northing': 2700000.0,
                                                                  'grid_mapping_name': 'lambert_conformal_conic',
                                                                  'latitude_of_projection_origin': 47.5,
                                                                  'longitude_of_central_meridian': -97.0},
                                                        },
                                  'xc': {'name': 'xc',
                                         'attrs': {'units': 'm',
                                                   'standard_name': 'projection_x_coordinate',
                                                   'axis': 'X'}
                                         },
                                  'yc': {'name': 'yc',
                                         'attrs': {'units': 'm',
                                                   'standard_name': 'projection_y_coordinate',
                                                   'axis': 'Y'}
                                         }
                                  }
                    }
        return min_meta

    def test_load_from_metadata(self):
        crs = CFLambertConformal.load_from_metadata('pr', self.archetype_minimal_metadata)
        self.assertEqual(crs.name, 'Lambert_Conformal')
        self.assertEqual(crs.value, {'lon_0': -97, 'ellps': 'WGS84', 'y_0': 2700000, 'no_defs': True, 'proj': 'lcc',
                                     'x_0': 3325000, 'units': 'm', 'lat_2': 60, 'lat_1': 30, 'lat_0': 47.5})
        self.assertIsInstance(crs, CFLambertConformal)
        self.assertEqual(['xc', 'yc'], [crs.projection_x_coordinate, crs.projection_y_coordinate])
        self.assertEqual([30., 60.], crs.map_parameters_values.pop('standard_parallel'))
        self.assertEqual(crs.map_parameters_values,
                         {'latitude_of_projection_origin': 47.5, 'longitude_of_central_meridian': -97.0,
                          'false_easting': 3325000.0, 'false_northing': 2700000.0, 'units': 'm'})

    def test_load_from_metadata_no_falses(self):
        """Test without false easting and false northing in attributes."""

        meta = self.archetype_minimal_metadata
        to_pop = ['false_easting', 'false_northing']
        for t in to_pop:
            meta['variables']['Lambert_Conformal']['attrs'].pop(t)

        crs = CFLambertConformal.load_from_metadata('pr', meta)
        self.assertIsInstance(crs, CFLambertConformal)

        for proj_parm in ['x_0', 'y_0']:
            self.assertEqual(crs.value[proj_parm], 0)

    def test_write_to_rootgrp(self):
        meta = self.archetype_minimal_metadata
        crs = CFLambertConformal.load_from_metadata('pr', meta)
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with nc_scope(path, 'w') as ds:
            variable = crs.write_to_rootgrp(ds)
            self.assertEqual(variable.grid_mapping_name, crs.grid_mapping_name)
            for k, v in crs.map_parameters_values.iteritems():
                variable_v = variable.__dict__[k]
                try:
                    self.assertEqual(variable_v, v)
                except ValueError:
                    # Some values come back as NumPy arrays.
                    self.assertEqual(variable_v.tolist(), v)

        with nc_scope(path) as ds:
            meta2 = {'variables': {'Lambert_Conformal': {'attrs': dict(ds.variables['Lambert_Conformal'].__dict__),
                                                         'name': 'Lambert_Conformal'}}}
        meta['variables']['Lambert_Conformal'] = meta2['variables']['Lambert_Conformal']
        crs2 = CFLambertConformal.load_from_metadata('pr', meta)
        self.assertEqual(crs, crs2)

        path2 = os.path.join(self.current_dir_output, 'foo2.nc')
        with nc_scope(path2, 'w') as ds:
            crs2.write_to_rootgrp(ds)


class TestCFRotatedPole(TestBase):
    def test_init(self):
        rp = CFRotatedPole(grid_north_pole_latitude=39.25, grid_north_pole_longitude=-162.0)
        self.assertDictEqual(rp.value,
                             {'lonc': 0, 'ellps': 'WGS84', 'y_0': 0, 'no_defs': True, 'proj': 'omerc', 'x_0': 0,
                              'units': 'm', 'alpha': 0, 'k': 1, 'gamma': 0, 'lat_0': 0})

    @attr('data')
    def test_equal(self):
        rd = self.test_data.get_rd('rotated_pole_ichec')
        rd2 = deepcopy(rd)
        self.assertEqual(rd.get().spatial.crs, rd2.get().spatial.crs)

    @attr('data')
    def test_get_rotated_pole_transformation(self):
        """Test SpatialDimension objects are appropriately transformed."""

        rd = self.test_data.get_rd('rotated_pole_ichec')
        field = rd.get()
        field = field[:, 10:20, :, 40:55, 55:65]
        spatial = field.spatial
        self.assertIsNotNone(spatial._grid)

        # modify the mask to ensure it is appropriately updated and copied during the transformations
        spatial.grid.value.mask[:, 5, 6] = True
        spatial.grid.uid.mask[5, 6] = True
        spatial.assert_uniform_mask()

        self.assertIsNone(spatial._geom._polygon)
        self.assertIsNone(spatial._geom._point)
        spatial.geom
        self.assertIsNotNone(spatial._geom.point)
        new_spatial = field.spatial.crs.get_rotated_pole_transformation(spatial)
        original_crs = deepcopy(field.spatial.crs)
        self.assertIsInstance(new_spatial.crs, CFWGS84)
        self.assertIsNone(new_spatial._geom)
        new_spatial.geom
        self.assertIsNotNone(new_spatial._geom)

        self.assertNumpyNotAllClose(spatial.grid.value, new_spatial.grid.value)

        field_copy = deepcopy(field)
        self.assertIsNone(field_copy.variables['tas']._value)
        field_copy.spatial = new_spatial
        value = field_copy.variables['tas'].value
        self.assertIsNotNone(field_copy.variables['tas']._value)
        self.assertIsNone(field.variables['tas']._value)

        self.assertNumpyAll(field.variables['tas'].value, field_copy.variables['tas'].value)

        inverse_spatial = original_crs.get_rotated_pole_transformation(new_spatial, inverse=True)
        for attr in ['row', 'col']:
            target = getattr(inverse_spatial.grid, attr)
            target_actual = getattr(spatial.grid, attr)
            self.assertDictEqual(target.attrs, target_actual.attrs)
        inverse_spatial.assert_uniform_mask()

        self.assertNumpyAll(inverse_spatial.uid, spatial.uid)
        self.assertNumpyAllClose(inverse_spatial.grid.row.value, spatial.grid.row.value)
        self.assertNumpyAllClose(inverse_spatial.grid.col.value, spatial.grid.col.value)
        self.assertDictEqual(spatial.grid.row.meta, inverse_spatial.grid.row.meta)
        self.assertEqual(spatial.grid.row.name, inverse_spatial.grid.row.name)
        self.assertDictEqual(spatial.grid.col.meta, inverse_spatial.grid.col.meta)
        self.assertEqual(spatial.grid.col.name, inverse_spatial.grid.col.name)

    @attr('data')
    def test_in_operations(self):
        rd = self.test_data.get_rd('rotated_pole_ichec')
        rd2 = deepcopy(rd)
        rd2.alias = 'tas2'
        # these projections are equivalent so it is okay to write them to a common output file
        ops = ocgis.OcgOperations(dataset=[rd, rd2], output_format='csv', snippet=True)
        ops.execute()

    @attr('data')
    def test_load_from_metadata(self):
        rd = self.test_data.get_rd('rotated_pole_ichec')
        self.assertIsInstance(rd.get().spatial.crs, CFRotatedPole)

        # Test without the grid_mapping attribute attached to a variable.
        meta = rd.source_metadata.copy()
        meta['variables']['tas']['attrs'].pop('grid_mapping')
        res = CFRotatedPole.load_from_metadata('tas', meta)
        self.assertIsInstance(res, CFRotatedPole)

    @attr('data')
    def test_write_to_rootgrp(self):
        rd = self.test_data.get_rd('narccap_rotated_pole')
        path = os.path.join(self.current_dir_output, 'foo.nc')

        with nc_scope(path, 'w') as ds:
            variable = rd.crs.write_to_rootgrp(ds)
            self.assertIsInstance(variable, nc.Variable)
            self.assertEqual(variable.proj4, '')
            self.assertEqual(variable.proj4_transform, rd.crs._trans_proj)
