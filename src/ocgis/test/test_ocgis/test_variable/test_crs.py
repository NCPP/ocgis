import os
from copy import deepcopy
from unittest import SkipTest
import sys

import netCDF4 as nc
import numpy as np
from shapely.geometry import Point, MultiPoint
from shapely.geometry.multipolygon import MultiPolygon

import ocgis
from ocgis import constants, vm
from ocgis.collection.field import Field
from ocgis.constants import WrapAction, WrappedState, ConversionFactor, OcgisUnits
from ocgis.exc import CRSNotEquivalenError
from ocgis.spatial.grid import Grid
from ocgis.test.base import TestBase, nc_scope, create_gridxy_global
from ocgis.test.base import attr
from ocgis.util.helpers import make_poly
from ocgis.util.itester import itr_products_keywords
from ocgis.variable.base import Variable
from ocgis.variable.crs import CoordinateReferenceSystem, CFAlbersEqualArea, CFLambertConformal, \
    CFRotatedPole, WGS84, Spherical, CFSpherical, Tripole, Cartesian
from ocgis.vmachine.mpi import OcgDist, MPI_RANK, variable_scatter, MPI_COMM


class TestCoordinateReferenceSystem(TestBase):
    def test_init(self):
        keywords = dict(
            value=[None, {'no_defs': True, 'ellps': 'WGS84', 'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0'}],
            epsg=[None, 4326],
            proj4=[None, '+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs '])

        prev_crs = None
        for k in itr_products_keywords(keywords):
            if k['epsg'] is not None and k['value'] is None and k['proj4'] is None:
                epsg_only = True
                prev_crs = None
            else:
                epsg_only = False

            try:
                crs = CoordinateReferenceSystem(**k)
            except ValueError:
                if all([ii is None for ii in list(k.values())]):
                    continue
                else:
                    raise

            self.assertEqual(crs.proj4, '+proj=longlat +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +no_defs ')
            try:
                self.assertDictEqual(crs.value,
                                     {'no_defs': True, 'ellps': 'WGS84', 'proj': 'longlat',
                                      'towgs84': '0,0,0,0,0,0,0'})
            except AssertionError:
                self.assertDictEqual(crs.value,
                                     {'no_defs': True, 'datum': 'WGS84', 'proj': 'longlat',
                                      'towgs84': '0,0,0,0,0,0,0'})
            if prev_crs is not None:
                self.assertEqual(crs, prev_crs)
            if not epsg_only:
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

    def test_get_wrap_action(self):
        c = CoordinateReferenceSystem
        possible = [WrappedState.WRAPPED, WrappedState.UNWRAPPED, WrappedState.UNKNOWN]
        keywords = dict(state_src=possible,
                        state_dst=possible)
        for k in itr_products_keywords(keywords, as_namedtuple=True):
            try:
                ret = c.get_wrap_action(k.state_src, k.state_dst)
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

    @attr('mpi')
    def test_get_wrapped_state(self):
        if sys.version_info.major == 3 and sys.version_info.minor == 5:
            raise SkipTest('undefined behavior with Python 3.5')

        ompi = OcgDist()
        ompi.create_dimension('x', 5, dist=True)
        ompi.create_dimension('y', 1)
        ompi.update_dimension_bounds()

        values = [{'value': [-179, -90, 0, 90, 180], 'desired': WrappedState.WRAPPED},
                  {'value': [0, 90, 180, 270, 360], 'desired': WrappedState.UNWRAPPED},
                  {'value': [1, 2, 3, 4, 5], 'desired': WrappedState.UNKNOWN}]
        kwds = {'values': values, 'crs': [Spherical(), None]}

        for k in self.iter_product_keywords(kwds):
            ompi = deepcopy(ompi)
            if MPI_RANK == 0:
                vx = Variable(name='x', value=k.values['value'], dimensions='x')
                vy = Variable(name='y', value=[0], dimensions='y')
            else:
                vx, vy = [None] * 2
            vx = variable_scatter(vx, ompi)
            vy = variable_scatter(vy, ompi)

            grid = Grid(vx, vy)
            field = Field(grid=grid, crs=k.crs)

            with vm.scoped_by_emptyable('wrap', field):
                if not vm.is_null:
                    wrapped_state = field.wrapped_state
                else:
                    wrapped_state = None

            if not field.is_empty:
                if k.crs is None:
                    self.assertIsNone(wrapped_state)
                else:
                    self.assertIsNotNone(wrapped_state)

            if k.crs is None or field.is_empty:
                self.assertIsNone(wrapped_state)
            else:
                self.assertEqual(wrapped_state, k.values['desired'])

    def test_get_wrapped_state_from_array(self):

        def _run_(arr, actual_wrapped_state):
            ret = CoordinateReferenceSystem._get_wrapped_state_from_array_(arr)
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
        c = CoordinateReferenceSystem

        for geom in geoms:
            ret = c._get_wrapped_state_from_geometry_(geom)
            self.assertEqual(ret, WrappedState.WRAPPED)

        pt = Point(270, 50)
        ret = c._get_wrapped_state_from_geometry_(pt)
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
        # sdim = ret[23]['tas'].spatial
        field = ret.get_element(container_ugid=23)
        actual = np.array([[[34.88252349825162, 34.88252349825162, 34.88252349825162],
                            [37.67309213352349, 37.67309213352349, 37.67309213352349],
                            [40.4636506825932, 40.4636506825932, 40.4636506825932]],
                           [[-120.9375, -118.125, -115.3125], [-120.9375, -118.125, -115.3125],
                            [-120.9375, -118.125, -115.3125]]],
                          dtype=field.grid.archetype.dtype)
        self.assertNumpyAll(actual, field.grid.get_value_stacked())

        field.grid.unwrap()
        actual = np.array([[[34.88252349825162, 34.88252349825162, 34.88252349825162],
                            [37.67309213352349, 37.67309213352349, 37.67309213352349],
                            [40.4636506825932, 40.4636506825932, 40.4636506825932]],
                           [[239.0625, 241.875, 244.6875], [239.0625, 241.875, 244.6875],
                            [239.0625, 241.875, 244.6875]]], dtype=field.grid.archetype.dtype)
        self.assertNumpyAll(actual, field.grid.get_value_stacked())


class TestWGS84(TestBase):
    def test_init(self):
        self.assertTrue(WGS84().is_geographic)
        self.assertNotIsInstance(WGS84(), Spherical)
        self.assertIn('towgs84', WGS84().value)


class TestCFAlbersEqualArea(TestBase):
    def test_init(self):
        crs = CFAlbersEqualArea(standard_parallel=[29.5, 45.5], longitude_of_central_meridian=-96,
                                latitude_of_projection_origin=37.5, false_easting=0,
                                false_northing=0)
        self.assertEqual(crs.value, {'lon_0': -96, 'ellps': 'WGS84', 'y_0': 0, 'no_defs': True, 'proj': 'aea',
                                     'x_0': 0, 'units': 'm', 'lat_2': 45.5, 'lat_1': 29.5, 'lat_0': 37.5})

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
            for k, v in crs.map_parameters_values.items():
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
        self.assertDictEqual(rp.value, {'lonc': 0, 'ellps': 'WGS84', 'y_0': 0, 'no_defs': True, 'proj': 'omerc',
                                        'x_0': 0, 'units': 'm', 'alpha': 0, 'k': 1, 'gamma': 0, 'lat_0': 0})

    @attr('data')
    def test_equal(self):
        rd = self.test_data.get_rd('rotated_pole_ichec')
        rd2 = deepcopy(rd)
        self.assertEqual(rd.get().crs, rd2.get().crs)

    @attr('data')
    def test_get_rotated_pole_transformation(self):
        """Test SpatialDimension objects are appropriately transformed."""

        rd = self.test_data.get_rd('rotated_pole_ichec')
        field = rd.get()

        original_coordinate_values = field.grid.get_value_stacked().copy()
        field.update_crs(CFSpherical())
        self.assertEqual(field.crs, CFSpherical())

        field.update_crs(rd.get().crs)
        back_to_original_coordinate_values = field.grid.get_value_stacked().copy()
        self.assertNumpyAllClose(original_coordinate_values, back_to_original_coordinate_values)

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
        self.assertIsInstance(rd.get().crs, CFRotatedPole)

        # Test without the grid_mapping attribute attached to a variable.
        meta = rd.metadata.copy()
        meta['variables']['tas']['attrs'].pop('grid_mapping')
        res = CFRotatedPole.load_from_metadata('tas', meta, strict=False)
        self.assertIsInstance(res, CFRotatedPole)

    @attr('data')
    def test_write_to_rootgrp(self):
        rd = self.test_data.get_rd('narccap_rotated_pole')
        path = os.path.join(self.current_dir_output, 'foo.nc')

        with nc_scope(path, 'w') as ds:
            variable = rd.crs.write_to_rootgrp(ds, with_proj4=True)
            self.assertIsInstance(variable, nc.Variable)
            self.assertEqual(variable.proj4, '')
            self.assertEqual(variable.proj4_transform, rd.crs._trans_proj)


class TestTripole(TestBase):
    def test_init(self):
        Tripole()

    def test_transform_coordinates(self):
        desired_min_maxes = [[-0.9330127018922193, 0.93301270189221941], [-0.93301270189221941, 0.93301270189221941],
                             [-0.96592582628906831, 0.96592582628906831]]

        keywords = {'wrapped': [
            False,
            True
        ],
            'angular_units': [
                OcgisUnits.DEGREES,
                OcgisUnits.RADIANS
            ],
            'other_crs': [
                Cartesian(),
                WGS84()
            ]
        }

        for k in self.iter_product_keywords(keywords):
            spherical = Spherical(angular_units=k.angular_units)
            tp = Tripole(spherical=spherical)

            grid = create_gridxy_global(resolution=30.0, wrapped=k.wrapped)

            if not k.wrapped:
                x_value = grid.x.get_value()
                select = x_value > 180.
                x_value[select] -= 360.

            grid.expand()

            x = grid.x.get_value()
            y = grid.y.get_value()

            if k.angular_units == OcgisUnits.RADIANS:
                x *= ConversionFactor.DEG_TO_RAD
                y *= ConversionFactor.DEG_TO_RAD

            z = np.ones(x.shape, dtype=x.dtype)

            desired = (x, y, z)
            try:
                as_cart = tp.transform_coordinates(k.other_crs, x, y, z)
            except CRSNotEquivalenError:
                self.assertNotEqual(k.other_crs, Cartesian())
                continue
            x_cart, y_cart, z_cart = as_cart

            for idx, ii in enumerate(as_cart):
                actual_min_max = [ii.min(), ii.max()]
                self.assertNumpyAllClose(np.array(actual_min_max), np.array(desired_min_maxes[idx]))

            actual = tp.transform_coordinates(k.other_crs, x_cart, y_cart, z_cart, inverse=True)

            for a, d in zip(actual, desired):
                are = np.abs(a - d)
                self.assertLessEqual(are.max(), 1e-6)
