import datetime
import os
import pickle
import shutil
from collections import OrderedDict
from datetime import datetime as dt

import numpy as np
from types import FunctionType

import ocgis
from ocgis import RequestDataset
from ocgis import env
from ocgis.collection.field import OcgField
from ocgis.constants import TagNames, MiscNames
from ocgis.driver.nc import DriverNetcdf, DriverNetcdfCF
from ocgis.driver.request.core import get_autodiscovered_driver, get_is_none
from ocgis.driver.vector import DriverVector
from ocgis.exc import DefinitionValidationError, VariableNotFoundError, RequestValidationError, \
    NoDataVariablesFound
from ocgis.spatial.geom_cabinet import GeomCabinet
from ocgis.test.base import TestBase, nc_scope, attr
from ocgis.test.test_simple.make_test_data import SimpleNc
from ocgis.test.test_simple.test_simple import TestSimpleBase
from ocgis.variable.crs import CoordinateReferenceSystem, CFWGS84


# tdk: clean-up file


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
                self.assertIsInstance(field, OcgField)

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

    def test_conform_units_to(self):
        rd = self.get_request_dataset_netcdf(variable='a', units='celsius', conform_units_to='fahrenheit')
        self.assertEqual(rd.conform_units_to, 'fahrenheit')
        field = rd.get()

        self.assertAlmostEqual(field['a'].value.mean(), 36.79999999999999)
        self.assertEqual(field['a'].units, 'fahrenheit')

        # Test modifying the metadata to conform arbitrary variables.
        m = rd.metadata
        m['variables']['the_level']['conform_units_to'] = 'kilometers'
        rd = self.get_request_dataset_netcdf(metadata=m)
        actual = rd.get()['the_level'].value.mean()
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
        self.assertEqual(rd.dimension_map['time']['names'], ['the_time'])

    def test_field_name(self):
        for name in [None, 'morning']:
            rd = self.get_request_dataset_netcdf(field_name=name)
            field = rd.get()
            if name is None:
                desired = MiscNames.DEFAULT_FIELD_NAME
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
        self.assertIsInstance(field, OcgField)

    def test_metadata(self):
        # Test overloaded metadata is held on the request dataset but the original remains the same.
        rd = self.get_request_dataset_netcdf()
        rd.metadata['variables']['a']['attributes']['units'] = 'overloaded_units'
        rd.metadata['variables']['a']['dtype'] = float
        rd.metadata['variables']['a']['fill_value'] = 1000.
        rd.metadata['variables']['a']['fill_value'] = 1000.
        rd.metadata['variables']['tt']['attributes']['calendar'] = 'blah'

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
        self.assertIn('unfoo', field._tags[TagNames.DATA_VARIABLES])

    def test_t_conform_units_to(self):
        t_conform_units_to = 'hours since 2000-1-1'
        rd = self.get_request_dataset_netcdf(dimension_map={'time': {'variable': 'tt'}},
                                             t_conform_units_to=t_conform_units_to)
        field = rd.get()
        self.assertEqual(field.time.value.mean(), 720.0)
        self.assertEqual(field.time.units, t_conform_units_to)

    def test_time_range(self):
        time_range = [datetime.datetime(2000, 1, 20), datetime.datetime(2000, 2, 12)]
        dimension_map = {'time': {'variable': 'tt'}}
        rd = self.get_request_dataset_netcdf(dimension_map=dimension_map, time_range=time_range)

        field = rd.get()

        self.assertEqual(len(field.dimensions['a']), 3)
        self.assertIsInstance(field, OcgField)
        self.assertIsNone(field['a']._value)

    def test_time_region(self):
        time_region = {'month': [1]}
        dimension_map = {'time': {'variable': 'tt'}}
        rd = self.get_request_dataset_netcdf(dimension_map=dimension_map, time_region=time_region)

        field = rd.get()

        self.assertEqual(len(field.dimensions['a']), 2)
        self.assertIsInstance(field, OcgField)
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
        self.assertIsInstance(field, OcgField)
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


# tdk: migrate to TestRequestDataset or remove
class OldTestRequestDataset(TestBase):
    def setUp(self):
        TestBase.setUp(self)
        rd = self.test_data.get_rd('cancm4_rhs')
        self.uri = rd.uri
        self.variable = rd.variable

    def get_multiple_variable_request_dataset_dictionary(self):
        rd_orig = self.test_data.get_rd('cancm4_tas')
        dest_uri = os.path.join(self.current_dir_output, os.path.split(rd_orig.uri)[1])
        shutil.copy2(rd_orig.uri, dest_uri)
        with nc_scope(dest_uri, 'a') as ds:
            var = ds.variables['tas']
            outvar = ds.createVariable(var._name + 'max', var.dtype, var.dimensions)
            outvar[:] = var[:] + 3
            outvar.setncatts(var.__dict__)
        with nc_scope(dest_uri) as ds:
            self.assertTrue(set(['tas', 'tasmax']).issubset(set(ds.variables.keys())))
        return {'uri': dest_uri, 'variable': ['tas', 'tasmax']}

    @attr('data')
    def test_init(self):
        rd = RequestDataset(uri=self.uri)
        self.assertTrue(rd.regrid_source)
        self.assertFalse(rd.regrid_destination)

        self.assertFalse(rd._has_assigned_coordinate_system)
        # If a coordinate system was assigned, this flag should become true.
        rd = RequestDataset(uri=self.uri, crs=CFWGS84())
        self.assertTrue(rd._has_assigned_coordinate_system)

        desired = 'days since 1949-1-1'
        rd = RequestDataset(uri=self.uri, t_conform_units_to=desired)
        actual = rd.metadata['variables']['time']['conform_units_to']
        self.assertEqual(str(actual), desired)

        # Test variable only loaded on request.
        path = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=path)
        self.assertEqual(rd._variable, None)
        self.assertEqual(rd.variable, 'tas')

        # Test variable appropriately formatted.
        rd = RequestDataset(uri=path, variable='tas')
        self.assertEqual(rd._variable, ('tas',))

    @attr('data')
    def test_init_driver(self):
        uri = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=uri, driver='vector')
        self.assertIsNotNone(rd.variable)
        self.assertIsInstance(rd.get(), OcgField)

        uri_nc = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri_nc)
        self.assertIsInstance(rd.driver, DriverNetcdf)

    @attr('data')
    def test_init_variable(self):
        rd = RequestDataset(uri=self.uri, variable='rhs')
        self.assertEqual(rd.variable, 'rhs')
        self.assertEqual(rd._variable, ('rhs',))

    @attr('data')
    def test_init_variable_not_found(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd_bad = RequestDataset(uri=rd.uri, variable='crap')
        with self.assertRaises(VariableNotFoundError):
            rd_bad.get()

    @attr('data')
    def test_init_variable_with_rename(self):
        rd = RequestDataset(uri=self.uri, variable='tas', rename_variable='tas_foo')
        self.assertEqual(rd.rename_variable, 'tas_foo')

    @attr('data')
    def test_init_multiple_variables_with_rename(self):
        rd = RequestDataset(uri=self.uri, variable=['tas', 'tasmax'], rename_variable=['tas_foo', 'tas_what'])
        self.assertEqual(rd.rename_variable, ('tas_foo', 'tas_what'))

    @attr('data')
    def test_rename_variable(self):
        path = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=path)
        self.assertIsNone(rd._rename_variable)
        self.assertEqual(rd.rename_variable, 'tas')
        rd.alias = 'temperature'
        self.assertEqual(rd.alias, 'temperature')

    @attr('data')
    def test_conform_units_to(self):
        rd = RequestDataset(uri=self.uri)
        self.assertIsNone(rd.conform_units_to)
        rd = RequestDataset(uri=self.uri, conform_units_to=None)
        self.assertIsNone(rd.conform_units_to)

        # Test for univariate.
        poss = ['K', ['K']]
        uri = self.test_data.get_rd('cancm4_tas').uri
        for p in poss:
            rd = RequestDataset(uri=uri, conform_units_to=p)
            self.assertEqual(rd.conform_units_to, 'K')

    @attr('data')
    def test_crs_overload(self):
        kwds = {'crs': CoordinateReferenceSystem(epsg=4362)}
        rd = self.test_data.get_rd('cancm4_tas', kwds=kwds)
        field = rd.get()
        self.assertDictEqual(kwds['crs'].value, field.crs.value)

    @attr('data')
    def test_drivers(self):
        self.assertIsInstance(RequestDataset._Drivers, OrderedDict)

    @attr('data')
    def test_env_dir_data(self):
        """Test setting the data directory to a single directory."""

        env.DIR_DATA = ocgis.env.DIR_TEST_DATA
        rd = self.test_data.get_rd('cancm4_rhs')
        target = os.path.join(env.DIR_DATA, 'nc', 'CanCM4', 'rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc')
        self.assertEqual(rd.uri, target)

        # test none and not finding the data
        env.DIR_DATA = None
        with self.assertRaises(ValueError):
            RequestDataset('does_not_exists.nc', variable='foo')

        # set data directory and not find it.
        env.DIR_DATA = os.path.join(ocgis.env.DIR_TEST_DATA, 'CCSM4')
        with self.assertRaises(ValueError):
            RequestDataset('rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc', variable='rhs')

    @attr('data')
    def test_get_autodiscovered_driver(self):
        uri_shp = '/path/to/shapefile.shp'
        uri_nc = '/path/to/netcdf/file/foo.nc'
        uri_nc_opendap = 'http://cida.usgs.gov/thredds/dodsC/maurer/maurer_brekke_w_meta.ncml'

        driver = get_autodiscovered_driver(uri_shp)
        self.assertEqual(driver, DriverVector)

        for poss in [uri_nc, [uri_nc, uri_nc], uri_nc_opendap, [uri_nc_opendap, uri_nc_opendap]]:
            driver = get_autodiscovered_driver(poss)
            self.assertEqual(driver, DriverNetcdfCF)

        with self.assertRaises(RequestValidationError):
            get_autodiscovered_driver('something/meaninglyess.foobuar')
        with self.assertRaises(RequestValidationError):
            get_autodiscovered_driver('something/meaninglyess')

    @attr('data')
    def test_get_field_nonequivalent_units_in_source_data(self):
        new_path = self.test_data.copy_file('cancm4_tas', self.current_dir_output)

        # Put non-equivalent units on the source data and attempt to conform.
        with nc_scope(new_path, 'a') as ds:
            ds.variables['tas'].units = 'coulomb'
        with self.assertRaises(RequestValidationError):
            RequestDataset(uri=new_path, variable='tas', conform_units_to='celsius')

        # Remove units altogether.
        with nc_scope(new_path, 'a') as ds:
            ds.variables['tas'].delncattr('units')
        with self.assertRaises(RequestValidationError):
            RequestDataset(uri=new_path, variable='tas', conform_units_to='celsius')

    @attr('data')
    def test_inspect(self):
        rd = RequestDataset(self.uri, self.variable)
        with self.print_scope() as ps:
            rd.inspect()
        self.assertTrue(len(ps.storage) >= 1)

        uri = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=uri, driver='vector')
        with self.print_scope() as ps:
            rd.inspect()
        self.assertTrue(len(ps.storage) >= 1)

        # test with a request dataset having no dimensioned variables
        path = self.get_temporary_file_path('bad.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('foo')
            var = ds.createVariable('foovar', int, dimensions=('foo',))
            var.a_name = 'a name'
        rd = RequestDataset(uri=path)
        with self.print_scope() as ps:
            rd.inspect()
        self.assertTrue(len(ps.storage) >= 1)

    @attr('data')
    def test_len(self):
        path = self.get_netcdf_path_no_dimensioned_variables()
        rd = RequestDataset(uri=path)
        self.assertEqual(len(rd), 0)

        rd = self.test_data.get_rd('cancm4_tas')
        self.assertEqual(len(rd), 1)

    @attr('data')
    def test_level_subset_without_level(self):
        lr = [1, 2]
        rd = self.test_data.get_rd('cancm4_tas')
        rd.level_range = lr
        with self.assertRaises(AttributeError):
            rd.get()

    @attr('data')
    def test_name(self):
        path = GeomCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=path, driver='vector')
        self.assertIsNotNone(rd.field_name)

        rd = RequestDataset(uri=path, driver='vector', field_name='states')
        self.assertEqual(rd.field_name, 'states')
        field = rd.get()
        self.assertEqual(field.name, 'states')

    @attr('data')
    def test_nonsense_units(self):
        with self.assertRaises(RequestValidationError):
            self.test_data.get_rd('cancm4_tas', kwds={'units': 'nonsense', 'conform_units_to': 'celsius'})

    @attr('data')
    def test_pickle(self):
        rd = RequestDataset(uri=self.uri, variable=self.variable)
        rd_path = os.path.join(ocgis.env.DIR_OUTPUT, 'rd.pkl')
        with open(rd_path, 'w') as f:
            pickle.dump(rd, f)
        with open(rd_path, 'r') as f:
            pickle.load(f)

    @attr('data')
    def test_source_index_matches_constant_value(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        self.assertEqual(field.temporal.dimensions[0]._src_idx.dtype, np.int32)

    @attr('data')
    def test_time_subset_func(self):
        rd = self.test_data.get_rd('cancm4_tas')
        self.assertIsNone(rd.time_subset_func)

        rd.time_subset_func = lambda x, y: [1, 2, 3]
        self.assertIsInstance(rd.time_subset_func, FunctionType)

    @attr('data')
    def test_uri_cannot_be_set(self):
        rd = self.test_data.get_rd('cancm4_tas')
        other_uri = self.test_data.get_uri('cancm4_rhs')
        with self.assertRaises(AttributeError):
            rd.uri = other_uri

    @attr('data')
    def test_variable(self):
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=uri)
        self.assertEqual(rd._variable, None)
        self.assertEqual(rd.variable, 'tas')

        # test with no dimensioned variables
        path = self.get_netcdf_path_no_dimensioned_variables()
        rd = RequestDataset(uri=path)
        with self.assertRaises(RequestValidationError):
            rd.variable

    @attr('data')
    def test_with_bad_units_attempting_conform(self):
        # Pass bad units to the init and an attempt a conform. values from the source dataset are not used for overload.
        with self.assertRaises(RequestValidationError):
            self.test_data.get_rd('cancm4_tas', kwds={'conform_units_to': 'celsius', 'units': 'coulomb'})

    @attr('data')
    def test_with_bad_units_passing_to_field(self):
        rd = self.test_data.get_rd('cancm4_tas', kwds={'units': 'celsius'})
        field = rd.get()
        self.assertEqual(field['tas'].units, 'celsius')

    @attr('data')
    def test_with_units(self):
        units = 'celsius'
        rd = self.test_data.get_rd('cancm4_tas', kwds={'units': units})
        self.assertEqual(rd.units, 'celsius')

    @attr('data')
    def test_without_units_attempting_conform(self):
        # This will work because the units read from the metadata are equivalent.
        self.test_data.get_rd('cancm4_tas', kwds={'conform_units_to': 'celsius'})
        # This will not work because the units are not equivalent.
        with self.assertRaises(RequestValidationError):
            self.test_data.get_rd('cancm4_tas', kwds={'conform_units_to': 'coulomb'})

    @attr('data')
    def test_time_range(self):
        tr = [dt(2000, 1, 1), dt(2000, 12, 31)]
        rd = RequestDataset(self.uri, self.variable, time_range=tr)
        self.assertEqual(rd.time_range, tuple(tr))

    @attr('data')
    def test_level_range(self):
        lr = [1, 1]
        rd = RequestDataset(self.uri, self.variable, level_range=lr)
        self.assertEqual(rd.level_range, tuple([1, 1]))

    @attr('data')
    def test_multiple_uris(self):
        rd = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        self.assertEqual(len(rd.uri), 2)

        with self.print_scope() as ps:
            rd.inspect()
        self.assertTrue(len(ps.storage) >= 1)

    @attr('data')
    def test_time_region(self):
        tr1 = {'month': [6], 'year': [2001]}
        rd = RequestDataset(uri=self.uri, variable=self.variable, time_region=tr1)
        self.assertEqual(rd.time_region, tr1)

        tr2 = {'bad': 15}
        with self.assertRaises(DefinitionValidationError):
            RequestDataset(uri=self.uri, variable=self.variable, time_region=tr2)
