import datetime

import numpy as np
from ocgis import RequestDataset, Variable
from ocgis.collection.field import Field
from ocgis.constants import TagName, MiscName, DimensionMapKey, DriverKey
from ocgis.driver.nc import DriverNetcdf, DriverNetcdfCF
from ocgis.driver.request.core import get_autodiscovered_driver, get_is_none
from ocgis.exc import RequestValidationError, NoDataVariablesFound
from ocgis.test.base import TestBase, attr
from ocgis.test.test_simple.make_test_data import SimpleNc
from ocgis.test.test_simple.test_simple import TestSimpleBase
from ocgis.variable.crs import CoordinateReferenceSystem, Tripole


class Test(TestBase):
    def test_get_autodiscovered_driver(self):
        # Test the priority driver is chosen for netcdf.
        path = self.get_temporary_file_path('foo.nc')
        driver = get_autodiscovered_driver(path)
        self.assertEqual(driver, DriverNetcdfCF)

    def test_get_is_none_false(self):
        possible_false = ['a', ['a', 'b'], ['b', None]]
        for p in possible_false:
            self.assertFalse(get_is_none(p))

    def test_get_is_none_true(self):
        possible_true = [None, [None, None]]
        for p in possible_true:
            self.assertTrue(get_is_none(p))


class TestRequestDataset(TestSimpleBase):
    base_value = np.array([[1.0, 1.0, 2.0, 2.0],
                           [1.0, 1.0, 2.0, 2.0],
                           [3.0, 3.0, 4.0, 4.0],
                           [3.0, 3.0, 4.0, 4.0]])
    nc_factory = SimpleNc
    fn = 'test_simple_spatial_01.nc'

    def get_request_dataset_netcdf(self, **kwargs):
        path = self.get_temporary_file_path('rd_netcdf.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('a', 5)
            ds.createDimension('lvl', 6)

            var_a = ds.createVariable('a', int, ('a',))
            var_a[:] = [1, 2, 3, 4, 5]
            var_a.units = 'original_units'

            var_b = ds.createVariable('b', float)
            var_b.something = 'an_attribute'

            var_time = ds.createVariable('tt', float, ('a',))
            var_time[:] = [10, 20, 30, 40, 50]
            var_time.calendar = '360_day'
            var_time.units = 'days since 2000-1-1'

            var_level = ds.createVariable('the_level', float, ('lvl',))
            var_level[:] = [9, 10, 11, 12, 13, 14]
            var_level.units = 'meters'

        kwargs['uri'] = path
        return RequestDataset(**kwargs)

    def test_init(self):
        # Test passing an open dataset object.
        rd = self.get_request_dataset_netcdf()
        path = rd.uri
        with self.nc_scope(path) as ds:
            for _ in range(1):
                rd2 = RequestDataset(opened=ds, driver=DriverNetcdf)
                field = rd2.get()
                self.assertIsInstance(field, Field)

        # Test unique identifier.
        rd = self.get_request_dataset_netcdf(uid=43)
        self.assertEqual(rd.uid, 43)
        field = rd.get()
        self.assertEqual(field.uid, 43)

    def test_init_field_name(self):
        """Test setting the field name."""

        desired = 'wtf_its_a_field'
        rd = self.get_request_dataset_netcdf(field_name=desired)
        self.assertEqual(rd.field_name, desired)
        field = rd.get()
        self.assertEqual(field.name, desired)
        self.assertIsNone(field.source_name)
        field.load()

    def test_init_metadata_only(self):
        metadata = {'variables': {'foo': {}}}
        rd = RequestDataset(metadata=metadata)
        self.assertEqual(rd.driver.key, DriverKey.NETCDF_CF)
        self.assertIsNone(rd.uri)
        self.assertEqual(rd.metadata, metadata)
        field = rd.create_field()
        self.assertIn('foo', field.keys())

    def test_system_predicate(self):
        """Test creating a request dataset with a predicate."""

        path = self.get_temporary_file_path('foo.nc')
        field = self.get_field()
        to_exclude = Variable(name='exclude')
        field.add_variable(to_exclude)
        field.write(path)

        rd = RequestDataset(uri=path, predicate=lambda x: not x.startswith('exclude'))
        self.assertNotIn('exclude', rd.metadata['variables'])
        actual = rd.get()
        self.assertNotIn('exclude', actual)

        # Test predicate affects data variable identification.
        path = self.get_temporary_file_path('foo.nc')
        rd = RequestDataset(uri=path, predicate=lambda x: x != 'foo')
        with self.assertRaises(NoDataVariablesFound):
            assert rd.variable

    @attr('cfunits')
    def test_conform_units_to(self):
        rd = self.get_request_dataset_netcdf(variable='a', units='celsius', conform_units_to='fahrenheit')
        self.assertEqual(rd.conform_units_to, 'fahrenheit')
        field = rd.get()

        self.assertAlmostEqual(field['a'].get_value().mean(), 36.79999999999999)
        self.assertEqual(field['a'].units, 'fahrenheit')

        # Test modifying the metadata to conform arbitrary variables.
        m = rd.metadata
        m['variables']['the_level']['conform_units_to'] = 'kilometers'
        rd = self.get_request_dataset_netcdf(metadata=m)
        actual = rd.get()['the_level'].get_value().mean()
        desired = 0.011500000000000002
        self.assertAlmostEqual(actual, desired)

        # Test units are evaluated for equivalence.
        with self.assertRaises(RequestValidationError):
            self.get_request_dataset_netcdf(variable='a', units='celsius', conform_units_to='meters')

    def test_crs(self):
        rd = self.get_request_dataset_netcdf(crs=None)
        self.assertIsNone(rd.crs)
        self.assertTrue(rd._has_assigned_coordinate_system)
        self.assertIsNone(rd.get().crs)

        crs = CoordinateReferenceSystem(epsg=2136)
        rd = self.get_request_dataset_netcdf(crs=crs)
        self.assertEqual(rd.crs, crs)

    def test_crs_with_dimension_map(self):
        """Test CRS overloading in the presence of a dimension map."""

        field = self.get_field()
        path = self.get_temporary_file_path('foo.nc')
        field.write(path)

        dmap = {DimensionMapKey.X: {DimensionMapKey.VARIABLE: 'col'},
                DimensionMapKey.Y: {DimensionMapKey.VARIABLE: 'row'}}
        rd = RequestDataset(path, dimension_map=dmap, crs=Tripole())
        field = rd.get()
        self.assertIsInstance(field.crs, Tripole)

    def test_dimension_map(self):
        # Test variable dimension names are automatically set if not provided by a dimension map.
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('the_time')
            tvar = ds.createVariable('time', float, dimensions=['the_time'])
            tvar.axis = 'T'
            tvar[0:3] = [1, 2, 3]
        dmap = {'time': {'variable': 'time'}}
        rd = RequestDataset(path, dimension_map=dmap)
        self.assertEqual(rd.dimension_map.get_dimension(DimensionMapKey.TIME), ['the_time'])

    def test_field_name(self):
        for name in [None, 'morning']:
            rd = self.get_request_dataset_netcdf(field_name=name)
            field = rd.get()
            if name is None:
                desired = MiscName.DEFAULT_FIELD_NAME
            else:
                desired = name
            self.assertEqual(field.name, desired)

    def test_get(self):
        format_time = False
        grid_abstraction = 'pool'
        dimension_map = {'time': {'variable': 'tt'}}
        rd = self.get_request_dataset_netcdf(dimension_map=dimension_map, format_time=format_time,
                                             grid_abstraction=grid_abstraction)
        field = rd.get()
        self.assertEqual(field.grid_abstraction, grid_abstraction)
        self.assertEqual(field.time.format_time, False)

    def test_level_range(self):
        dimension_map = {'level': {'variable': 'the_level'}}
        rd = self.get_request_dataset_netcdf(dimension_map=dimension_map, level_range=[10.5, 13.5])

        field = rd.get()

        self.assertEqual(len(field.dimensions['lvl']), 3)
        self.assertIsInstance(field, Field)

    def test_metadata(self):
        # Test overloaded metadata is held on the request dataset but the original remains the same.
        rd = self.get_request_dataset_netcdf()
        rd.metadata['variables']['a']['attrs']['units'] = 'overloaded_units'
        rd.metadata['variables']['a']['dtype'] = float
        rd.metadata['variables']['a']['fill_value'] = 1000.
        rd.metadata['variables']['a']['fill_value'] = 1000.
        rd.metadata['variables']['tt']['attrs']['calendar'] = 'blah'

        self.assertNotEqual(rd.metadata, rd.driver.metadata_raw)
        field = rd.get()
        self.assertEqual(field['a'].attrs['units'], 'overloaded_units')
        self.assertEqual(field['a'].dtype, float)
        self.assertEqual(field['a'].fill_value, 1000.)
        self.assertEqual(field['tt'].attrs['calendar'], 'blah')

    def test_rename_variable(self):
        rd = self.get_request_dataset(rename_variable='unfoo')
        self.assertEqual(rd.variable, self.var)
        self.assertEqual(rd.rename_variable, 'unfoo')
        field = rd.get()
        self.assertNotIn(self.var, field)
        self.assertIn('unfoo', field)
        self.assertEqual(field['unfoo'].source_name, self.var)
        self.assertIn('unfoo', field._tags[TagName.DATA_VARIABLES])

    @attr('cfunits')
    def test_t_conform_units_to(self):
        t_conform_units_to = 'hours since 2000-1-1'
        rd = self.get_request_dataset_netcdf(dimension_map={'time': {'variable': 'tt'}},
                                             t_conform_units_to=t_conform_units_to)
        field = rd.get()
        self.assertEqual(field.time.get_value().mean(), 720.0)
        self.assertEqual(field.time.units, t_conform_units_to)

    def test_time_range(self):
        time_range = [datetime.datetime(2000, 1, 20), datetime.datetime(2000, 2, 12)]
        dimension_map = {'time': {'variable': 'tt'}}
        rd = self.get_request_dataset_netcdf(dimension_map=dimension_map, time_range=time_range)

        field = rd.get()

        self.assertEqual(len(field.dimensions['a']), 3)
        self.assertIsInstance(field, Field)
        self.assertIsNone(field['a']._value)

    def test_time_region(self):
        time_region = {'month': [1]}
        dimension_map = {'time': {'variable': 'tt'}}
        rd = self.get_request_dataset_netcdf(dimension_map=dimension_map, time_region=time_region)

        field = rd.get()

        self.assertEqual(len(field.dimensions['a']), 2)
        self.assertIsInstance(field, Field)
        self.assertIsNone(field['a']._value)

    def test_time_subset_func(self):

        def tsf(dts, bounds=None):
            ret = []
            for idx, dt in enumerate(dts):
                if dt.day == 11:
                    ret.append(idx)
            return ret

        dimension_map = {'time': {'variable': 'tt'}}
        rd = self.get_request_dataset_netcdf(dimension_map=dimension_map, time_subset_func=tsf)

        field = rd.get()

        self.assertEqual(len(field.dimensions['a']), 2)
        self.assertIsInstance(field, Field)
        self.assertIsNone(field['a']._value)

    def test_units(self):
        variable = ['a', 'b']
        units = ['kelvin', 'celsius']
        rd = self.get_request_dataset_netcdf(variable=variable, units=units)
        self.assertEqual(rd.units, ('kelvin', 'celsius'))
        field = rd.get()

        for v, u in zip(variable, units):
            self.assertEqual(field[v].units, u)

        # Test with no variables.
        rd = self.get_request_dataset_netcdf()
        with self.assertRaises(NoDataVariablesFound):
            assert rd.units

        # Test with a single variable. Multi-character variable names were not appropriately iterated across.
        rd = self.get_request_dataset_netcdf(variable='the_level', units='hectopascals')
        field = rd.get()
        self.assertEqual(field['the_level'].units, 'hectopascals')
