from ocgis import Variable
from ocgis.constants import DMK
from ocgis.driver.dimension_map import DimensionMap
from ocgis.exc import DimensionMapError
from ocgis.test.base import TestBase


class TestDimensionMap(TestBase):
    def test(self):
        dmap = DimensionMap()

        _ = dmap.get_variable('x')
        _ = dmap.get_dimensions('x')
        _ = dmap.get_attrs('x')
        _ = dmap.get_bounds('x')

        dmap.set_variable('y', 'latitude', dimensions='ache', bounds='lat_bounds')

        _ = dmap.get_crs()

        dmap.set_crs('latitude_longitude')

        desired = {'crs': {'variable': 'latitude_longitude'},
                   'x': {'attrs': {}, 'bounds': None, 'dimensions': [], 'variable': None},
                   'y': {'attrs': {'axis': 'Y'},
                         'bounds': 'lat_bounds',
                         'dimensions': ['ache'],
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
                   'groups': {'level_1': {'level': {'attrs': {'axis': 'Z'},
                                                    'bounds': None,
                                                    'dimensions': [],
                                                    'variable': 'level_1'}}}}
        actual = dmap.as_dict()
        self.assertEqual(actual, desired)

        level_1_1 = DimensionMap()
        level_1_1.set_variable(DMK.X, 'longitude')
        level_1.set_group('level_1_1', level_1_1)
        desired = {'crs': {'variable': 'what'},
                   'groups': {'level_1': {'groups': {'level_1_1': {'x': {'attrs': {'axis': 'X'},
                                                                         'bounds': None,
                                                                         'dimensions': [],
                                                                         'variable': 'longitude'}}},
                                          'level': {'attrs': {'axis': 'Z'},
                                                    'bounds': None,
                                                    'dimensions': [],
                                                    'variable': 'level_1'}}}}
        actual = dmap.as_dict()
        self.assertEqual(actual, desired)

    def test_set_bounds(self):
        dmap = DimensionMap()
        dmap.set_variable(DMK.X, 'lon', bounds='lon_bounds')
        actual = dmap.get_bounds(DMK.X)
        self.assertEqual(actual, 'lon_bounds')
        dmap.set_bounds(DMK.X, None)
        actual = dmap.get_bounds(DMK.X)
        self.assertIsNone(actual)

    def test_set_variable(self):
        var = Variable(name='test', value=[1, 2], dimensions='two')
        dmap = DimensionMap()
        dmap.set_variable(DMK.X, var, dimensions='not_two')
        actual = dmap.get_dimensions(DMK.X)
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
