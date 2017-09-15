import numpy as np

from ocgis import Variable, Dimension
from ocgis.constants import DMK, GridAbstraction
from ocgis.driver.dimension_map import DimensionMap
from ocgis.driver.nc import DriverNetcdfCF
from ocgis.driver.nc_ugrid import DriverNetcdfUGRID
from ocgis.exc import DimensionMapError
from ocgis.spatial.grid import create_grid_mask_variable
from ocgis.test.base import TestBase


class TestDimensionMap(TestBase):
    def test(self):
        dmap = DimensionMap()

        _ = dmap.get_variable('x')
        _ = dmap.get_dimension('x')
        _ = dmap.get_attrs('x')
        _ = dmap.get_bounds('x')

        dmap.set_variable('y', 'latitude', dimension='ache', bounds='lat_bounds')

        _ = dmap.get_crs()

        dmap.set_crs('latitude_longitude')

        desired = {'crs': {'variable': 'latitude_longitude'},
                   'x': {'attrs': {}, 'bounds': None, 'dimension': [], 'variable': None},
                   'y': {'attrs': {},
                         'bounds': 'lat_bounds',
                         'dimension': ['ache'],
                         'variable': 'latitude'}}

        self.assertDictEqual(dmap._storage, desired)

    def test_import(self):
        from ocgis import DimensionMap
        assert DimensionMap

    def test_system_grouped_dimension_map(self):
        initial_data = {DMK.X: {DMK.VARIABLE: 'longitude'},
                        DMK.GROUPS: {'nested': {DMK.Y: {DMK.VARIABLE: 'latitude'},
                                                DMK.GROUPS: {
                                                    'nested_in_nested': {DMK.CRS: {DMK.VARIABLE: 'coord_sys'}}}}}}

        dmap = DimensionMap.from_dict(initial_data)
        nested = dmap.get_group('nested')
        self.assertEqual(nested.get_variable(DMK.Y), 'latitude')
        self.assertIsNone(dmap.get_variable(DMK.Y))
        self.assertEqual(dmap.get_group([None, 'nested', 'nested_in_nested']).get_crs(), 'coord_sys')

    def test_from_old_style_dimension_map(self):
        old_style = {'Y': {'variable': u'lat', 'bounds': u'lat_bnds', 'dimension': u'lat', 'pos': 1},
                     'X': {'variable': u'lon', 'bounds': u'lon_bnds', 'dimension': u'lon', 'pos': 2},
                     'Z': {'variable': u'height', 'bounds': None, 'dimension': None, 'pos': None},
                     'T': {'variable': u'time', 'bounds': u'time_bnds', 'dimension': u'time', 'pos': 0}}
        new_style = DimensionMap.from_old_style_dimension_map(old_style)

        desired = {'level': {'attrs': {},
                             'bounds': None,
                             'dimension': [],
                             'variable': u'height'},
                   'time': {'attrs': {'axis': 'T'},
                            'bounds': u'time_bnds',
                            'dimension': [u'time'],
                            'variable': u'time'},
                   'x': {'attrs': {},
                         'bounds': u'lon_bnds',
                         'dimension': [u'lon'],
                         'variable': u'lon'},
                   'y': {'attrs': {},
                         'bounds': u'lat_bnds',
                         'dimension': [u'lat'],
                         'variable': u'lat'}}
        self.assertEqual(new_style.as_dict(), desired)

    def test_get_driver(self):
        dmap = DimensionMap()
        self.assertEqual(dmap.get_driver(), DriverNetcdfCF.key)

    def test_get_group(self):
        dmap = DimensionMap()
        dmap.set_crs('who')
        actual = dmap.get_group(None)
        self.assertEqual(dmap.as_dict(), actual.as_dict())

        dmap = DimensionMap()
        dmap.set_crs('what')
        with self.assertRaises(DimensionMapError):
            dmap.get_group(['level_1'])

        level_1 = DimensionMap()
        level_1.set_variable(DMK.LEVEL, 'level_1')
        dmap.set_group('level_1', level_1)
        desired = {'crs': {'variable': 'what'},
                   'groups': {'level_1': {'level': {'attrs': {},
                                                    'bounds': None,
                                                    'dimension': [],
                                                    'variable': 'level_1'}}}}
        actual = dmap.as_dict()
        self.assertEqual(actual, desired)

        level_1_1 = DimensionMap()
        level_1_1.set_variable(DMK.X, 'longitude')
        level_1.set_group('level_1_1', level_1_1)
        desired = {'crs': {'variable': 'what'},
                   'groups': {'level_1': {'groups': {'level_1_1': {'x': {'attrs': {},
                                                                         'bounds': None,
                                                                         'dimension': [],
                                                                         'variable': 'longitude'}}},
                                          'level': {'attrs': {},
                                                    'bounds': None,
                                                    'dimension': [],
                                                    'variable': 'level_1'}}}}
        actual = dmap.as_dict()
        self.assertEqual(actual, desired)

    def test_get_topology(self):
        dmap = DimensionMap()
        self.assertEqual(dmap.get_topology(GridAbstraction.POINT), GridAbstraction.AUTO)

        dmap_poly = dmap.get_topology(GridAbstraction.POLYGON, create=True)
        self.assertEqual(dmap_poly, DimensionMap())

        dmap_poly.set_variable(DMK.X, 'polyx')
        dmap_poly.set_variable(DMK.ELEMENT_NODE_CONNECTIVITY, 'element_connectivity')

        dmap_point = dmap.get_topology(GridAbstraction.POINT, create=True)
        dmap_point.set_variable(DMK.Y, 'point_y')
        self.assertEqual(dmap.get_topology(GridAbstraction.POINT).get_variable(DMK.Y), 'point_y')

    def test_set_bounds(self):
        dmap = DimensionMap()
        dmap.set_variable(DMK.X, 'lon', bounds='lon_bounds')
        actual = dmap.get_bounds(DMK.X)
        self.assertEqual(actual, 'lon_bounds')
        dmap.set_bounds(DMK.X, None)
        actual = dmap.get_bounds(DMK.X)
        self.assertIsNone(actual)

    def test_set_driver(self):
        dmap = DimensionMap()
        dmap.set_driver(DriverNetcdfUGRID)
        self.assertEqual(dmap.get_driver(), DriverNetcdfUGRID.key)

    def test_set_spatial_mask(self):
        dmap = DimensionMap()
        dims = Dimension('x', 3), Dimension('y', 7)
        mask_var = create_grid_mask_variable('a_mask', None, dims)
        self.assertFalse(np.any(mask_var.get_mask()))
        dmap.set_spatial_mask(mask_var)
        self.assertEqual(dmap.get_spatial_mask(), mask_var.name)

        with self.assertRaises(DimensionMapError):
            dmap.set_variable(DMK.SPATIAL_MASK, mask_var)

        # Test custom variables may be used.
        dmap = DimensionMap()
        dims = Dimension('x', 3), Dimension('y', 7)
        mask_var = create_grid_mask_variable('a_mask', None, dims)
        attrs = {'please keep me': 'no overwriting'}
        dmap.set_spatial_mask(mask_var, attrs=attrs)
        attrs = dmap.get_attrs(DMK.SPATIAL_MASK)
        self.assertIn('please keep me', attrs)

    def test_set_variable(self):
        var = Variable(name='test', value=[1, 2], dimensions='two')
        dmap = DimensionMap()
        dmap.set_variable(DMK.X, var, dimension='not_two')
        actual = dmap.get_dimension(DMK.X)
        desired = ['not_two']
        self.assertEqual(actual, desired)

        # Test setting with a variable and bounds.
        var = Variable(name='center', value=[1, 2, 3], dtype=float, dimensions='one')
        var.set_extrapolated_bounds('bounds_data', 'bounds')
        dmap = DimensionMap()
        dmap.set_variable(DMK.Y, var)
        self.assertEqual(dmap.get_bounds(DMK.Y), var.bounds.name)
        new_var = Variable(name='new_center', value=[1, 2, 3], dtype=float, dimensions='one')
        dmap.set_variable(DMK.Y, new_var)
        self.assertIsNone(dmap.get_bounds(DMK.Y))

        # Test a dimension position argument is needed if the variable has more than one dimension.
        var = Variable(name='two_dims', value=np.zeros((4, 5)), dimensions=['one', 'two'])
        dmap = DimensionMap()
        with self.assertRaises(DimensionMapError):
            dmap.set_variable(DMK.X, var)
        dmap.set_variable(DMK.X, var, pos=1)
        self.assertEqual(dmap.get_dimension(DMK.X), ['two'])

        # Test a scalar dimension.
        var = Variable(name='scalar_dimension', dimensions=[])
        dmap = DimensionMap()
        dmap.set_variable(DMK.LEVEL, var)
        self.assertEqual(dmap.get_variable(DMK.LEVEL), 'scalar_dimension')
        self.assertIsNone(dmap.get_bounds(DMK.LEVEL))
        self.assertEqual(dmap.get_dimension(DMK.LEVEL), [])

        # Test dimensionless variable.
        var = Variable(name='two_dims', value=np.zeros((4, 5)), dimensions=['one', 'two'])
        dmap = DimensionMap()
        dmap.set_variable(DMK.X, var, dimensionless=True)
        self.assertEqual(dmap.get_dimension(DMK.X), [])
