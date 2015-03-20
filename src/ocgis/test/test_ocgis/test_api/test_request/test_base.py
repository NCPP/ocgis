from collections import OrderedDict
from copy import deepcopy
import itertools
import os
import pickle
import shutil
import numpy as np
from datetime import datetime as dt
import datetime

from cfunits.cfunits import Units

from ocgis.api.request.driver.nc import DriverNetcdf
from ocgis.api.request.driver.vector import DriverVector
from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.interface.base.field import Field
from ocgis.exc import DefinitionValidationError, NoUnitsError, VariableNotFoundError, RequestValidationError
from ocgis.api.request.base import RequestDataset, RequestDatasetCollection, get_tuple, get_is_none
import ocgis
from ocgis import env, constants
from ocgis.interface.base.crs import CoordinateReferenceSystem, CFWGS84
from ocgis.test.base import TestBase, nc_scope, attr
from ocgis.api.operations import OcgOperations
from ocgis.util.helpers import get_iter
from ocgis.util.itester import itr_products_keywords


class Test(TestBase):

    def test_get_is_none_false(self):
        possible_false = ['a', ['a', 'b'], ['b', None]]
        for p in possible_false:
            self.assertFalse(get_is_none(p))

    def test_get_is_none_true(self):
        possible_true = [None, [None, None]]
        for p in possible_true:
            self.assertTrue(get_is_none(p))


class TestRequestDataset(TestBase):

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

    def test_init(self):
        rd = RequestDataset(uri=self.uri)
        self.assertTrue(rd.regrid_source)
        self.assertFalse(rd.regrid_destination)

        self.assertFalse(rd._has_assigned_coordinate_system)
        # if a coordinate system was assigned, this flag should become true
        rd = RequestDataset(uri=self.uri, crs=CFWGS84())
        self.assertTrue(rd._has_assigned_coordinate_system)

        rd = RequestDataset(uri=self.uri, t_conform_units_to='days since 1949-1-1')
        self.assertEqual(rd.t_conform_units_to, 'days since 1949-1-1')

        # test variable only loaded on request
        path = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=path)
        self.assertEqual(rd._variable, None)
        self.assertEqual(rd.variable, 'tas')

        # test variable appropriately formatted
        rd = RequestDataset(uri=path, variable='tas')
        self.assertEqual(rd._variable, ('tas',))

    def test_init_combinations(self):
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

        keywords = dict(
            name=[None, 'foo'],
            uri=[None, dest_uri],
            variable=[None, 'tas', ['tas', 'tasmax'], 'crap'],
            alias=[None, 'tas', ['tas', 'tasmax'], ['tas_alias', 'tasmax_alias']],
            units=[None, [None, None], ['celsius', 'fahrenheit'], 'crap', [None, 'kelvin'], ['crap', 'crappy']],
            conform_units_to=[None, [None, None], ['celsius', 'fahrenheit'], 'crap', [None, 'kelvin'],
                              ['crap', 'crappy'], [None, 'coulomb'], ['coulomb', 'coulomb']])

        def itr_row(key, sequence):
            for element in sequence:
                yield ({key: element})

        def itr_products_keywords(keywords):
            iterators = [itr_row(ki, vi) for ki, vi in keywords.iteritems()]
            for dictionaries in itertools.product(*iterators):
                yld = {}
                for dictionary in dictionaries:
                    yld.update(dictionary)
                yield yld

        for k in itr_products_keywords(keywords):
            try:
                rd = RequestDataset(**k)
                self.assertEqual(rd._source_metadata, None)
                self.assertEqual(len(get_tuple(rd.variable)), len(get_tuple(rd.units)))
                if k['name'] is None:
                    self.assertEqual(rd.name, '_'.join(get_tuple(rd.alias)))
                else:
                    self.assertEqual(rd.name, 'foo')
                for v in rd._variable:
                    try:
                        self.assertTrue(v in rd.source_metadata['variables'].keys())
                    except VariableNotFoundError:
                        if 'crap' in rd._variable:
                            self.assertEqual(rd._source_metadata, None)
                            break
                if k['units'] is None and len(rd._variable) == 1:
                    self.assertEqual(rd.units, None)
                    self.assertEqual(rd._units, None)

                try:
                    field = rd.get()
                    self.assertEqual(field.name, rd.name)
                    self.assertEqual(set(field.variables.keys()), set(get_tuple(rd.alias)))
                except VariableNotFoundError:
                    if 'crap' in rd._variable:
                        continue
                    else:
                        raise
                except RequestValidationError:
                    if 'coulomb' in get_tuple(k['conform_units_to']):
                        continue
                    else:
                        raise
            except RequestValidationError as e:
                # uris cannot be None
                if k['uri'] is None:
                    pass
                # variables cannot be None
                elif k['variable'] is None:
                    pass
                # 'crap' is not a real variable name
                elif k['conform_units_to'] is not None and (k['conform_units_to'] == 'crap' or \
                                                                        'crap' in k['conform_units_to']):
                    pass
                # conform_units_to must match units element-wise
                elif k['conform_units_to'] is not None and k['variable'] is not None and \
                                len(k['conform_units_to']) != len(k['variable']):
                    pass
                # aliases must occur for each variable
                elif len(get_tuple(k['alias'])) != len(get_tuple(k['variable'])):
                    pass
                # units must occur for each variable
                elif len(get_tuple(k['units'])) != len(get_tuple(k['variable'])):
                    pass
                # bad unit definition
                # 'crap' is not a real variable name
                elif k['units'] is not None and (k['units'] == 'crap' or \
                                                             'crap' in k['units']):
                    pass
                # alway need a uri and variable
                elif k['uri'] is None:
                    pass
                else:
                    raise
            except:
                raise

    def test_init_driver(self):
        uri = ShpCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=uri, driver='vector')
        self.assertIsNotNone(rd.variable)
        self.assertIsInstance(rd.get(), Field)

        uri_nc = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri_nc)
        self.assertIsInstance(rd.driver, DriverNetcdf)

        rd = RequestDataset(uri_nc, driver='vector')
        with self.assertRaises(ValueError):
            assert rd.variable

    def test_init_multiple_variable(self):
        rd = RequestDataset(uri=self.uri, variable=['tas', 'tasmax'])
        self.assertEqual(rd.variable, ('tas', 'tasmax'))
        self.assertEqual(rd._variable, ('tas', 'tasmax'))
        self.assertEqual(rd.alias, ('tas', 'tasmax'))

    def test_init_variable(self):
        rd = RequestDataset(uri=self.uri, variable='tas')
        self.assertEqual(rd.variable, 'tas')
        self.assertEqual(rd._variable, ('tas',))
        self.assertEqual(rd.alias, 'tas')
        self.assertIsNone(rd._alias, None)

    def test_init_variable_not_found(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd_bad = RequestDataset(uri=rd.uri, variable='crap')
        with self.assertRaises(VariableNotFoundError):
            rd_bad.get()

    def test_init_variable_with_alias(self):
        rd = RequestDataset(uri=self.uri, variable='tas', alias='tas_foo')
        self.assertEqual(rd.alias, 'tas_foo')

    def test_init_multiple_variables_with_alias(self):
        rd = RequestDataset(uri=self.uri, variable=['tas', 'tasmax'], alias=['tas_foo', 'tas_what'])
        self.assertEqual(rd.alias, ('tas_foo', 'tas_what'))

    def test_init_multiple_variables_with_alias_wrong_count(self):
        with self.assertRaises(RequestValidationError):
            RequestDataset(uri=self.uri, variable=['tas', 'tasmax'], alias='tas_what')

    def test_alias(self):
        path = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=path)
        self.assertIsNone(rd._alias)
        self.assertEqual(rd.alias, 'tas')
        rd.alias = 'temperature'
        self.assertEqual(rd.alias, 'temperature')

    def test_alias_change_after_init_one_variable(self):
        rd = self.test_data.get_rd('cancm4_tas')
        self.assertEqual(rd.name, 'tas')
        rd.alias = 'foo'
        self.assertEqual(rd.name, 'foo')

    def test_alias_change_after_init_two_variables(self):
        kwds = self.get_multiple_variable_request_dataset_dictionary()
        rd = RequestDataset(**kwds)
        self.assertEqual(rd.name, 'tas_tasmax')
        with self.assertRaises(RequestValidationError):
            rd.alias = 'foo'
        rd.alias = ['foo', 'foo2']
        self.assertEqual(rd.alias, ('foo', 'foo2'))
        self.assertEqual(rd.name, 'foo_foo2')

        with self.assertRaises(RequestValidationError):
            rd.units = 'crap'
        with self.assertRaises(RequestValidationError):
            rd.units = ('crap', 'crap')
        rd.units = ['celsius', 'celsius']
        self.assertEqual(rd.units, ('celsius', 'celsius'))

    def test_conform_units_to(self):
        rd = RequestDataset(uri=self.uri)
        self.assertIsNone(rd.conform_units_to)
        rd = RequestDataset(uri=self.uri, conform_units_to=None)
        self.assertIsNone(rd.conform_units_to)

        # these are exceptions
        problems = ['K', 'not_real']
        for prob in problems:
            with self.assertRaises(RequestValidationError):
                RequestDataset(uri=self.uri, variable=['one', 'two'], conform_units_to=prob)

        # test for univariate
        poss = ['K', ['K']]
        for p in poss:
            rd = RequestDataset(uri=self.uri, conform_units_to=p)
            self.assertEqual(rd.conform_units_to, 'K')

        # test for multivariate
        target = ['K', 'celsius']
        poss = [target]
        for p in poss:
            rd = RequestDataset(uri=self.uri, variable=['one', 'two'], conform_units_to=p)
            self.assertEqual(rd.conform_units_to, tuple(target))

    def test_crs_overload(self):
        kwds = {'crs': CoordinateReferenceSystem(epsg=4362)}
        rd = self.test_data.get_rd('cancm4_tas', kwds=kwds)
        field = rd.get()
        self.assertDictEqual(kwds['crs'].value, field.spatial.crs.value)

    def test_dimension_map(self):
        rd = self.test_data.get_rd('cancm4_tas')
        self.assertIsNone(rd._dimension_map)
        with self.assertRaises(AttributeError):
            rd.dimension_map = {}

        # test for deepcopy
        kwds = {'dimension_map': {'X': 'lon', 'Y': 'lat', 'T': 'time'}}
        rd = self.test_data.get_rd('cancm4_tas', kwds=kwds)
        self.assertEqual(rd.dimension_map, kwds['dimension_map'])
        kwds['dimension_map']['Y'] = 'foo'
        self.assertNotEqual(rd.dimension_map, kwds['dimension_map'])

    def test_drivers(self):
        # always test for netcdf first
        self.assertIsInstance(RequestDataset._Drivers, OrderedDict)
        self.assertEqual(RequestDataset._Drivers.values()[0], DriverNetcdf)

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

    def test_get_autodiscovered_driver(self):
        uri_shp = '/path/to/shapefile.shp'
        uri_nc = '/path/to/netcdf/file/foo.nc'
        uri_nc_opendap = 'http://cida.usgs.gov/thredds/dodsC/maurer/maurer_brekke_w_meta.ncml'

        driver = RequestDataset._get_autodiscovered_driver_(uri_shp)
        self.assertEqual(driver, DriverVector)

        for poss in [uri_nc, [uri_nc, uri_nc], uri_nc_opendap, [uri_nc_opendap, uri_nc_opendap]]:
            driver = RequestDataset._get_autodiscovered_driver_(poss)
            self.assertEqual(driver, DriverNetcdf)

        with self.assertRaises(RequestValidationError):
            RequestDataset._get_autodiscovered_driver_('something/meaninglyess.foobuar')
        with self.assertRaises(RequestValidationError):
            RequestDataset._get_autodiscovered_driver_('something/meaninglyess')

    def test_get_field_nonequivalent_units_in_source_data(self):
        new_path = self.test_data.copy_file('cancm4_tas', self.current_dir_output)

        # put non-equivalent units on the source data and attempto to conform
        with nc_scope(new_path, 'a') as ds:
            ds.variables['tas'].units = 'coulomb'
        rd = RequestDataset(uri=new_path, variable='tas', conform_units_to='celsius')
        with self.assertRaises(RequestValidationError):
            rd.get()

        # remove units altogether
        with nc_scope(new_path, 'a') as ds:
            ds.variables['tas'].delncattr('units')
        rd = RequestDataset(uri=new_path, variable='tas', conform_units_to='celsius')
        with self.assertRaises(NoUnitsError):
            rd.get()

    def test_get_field_with_overloaded_units(self):
        rd = self.test_data.get_rd('cancm4_tas', kwds={'conform_units_to': 'celsius'})
        preload = [False, True]
        for pre in preload:
            field = rd.get()
            # conform units argument needs to be attached to a field variable
            self.assertEqual(field.variables['tas']._conform_units_to, Units('celsius'))
            sub = field.get_time_region({'year': [2009], 'month': [5]})
            if pre:
                # if we wanted to load the data prior to subset then do so and manually perform the units conversion
                to_test = Units.conform(sub.variables['tas'].value, sub.variables['tas'].cfunits, Units('celsius'))
            # assert the conform attribute makes it though the subset
            self.assertEqual(sub.variables['tas']._conform_units_to, Units('celsius'))
            value = sub.variables['tas'].value
            self.assertAlmostEqual(np.ma.mean(value), 5.921925206338206)
            self.assertAlmostEqual(np.ma.median(value), 10.745431900024414)
            if pre:
                # assert the manually converted array matches the loaded value
                self.assertNumpyAll(to_test, value)

    def test_inspect(self):
        rd = RequestDataset(self.uri, self.variable)
        with self.print_scope() as ps:
            rd.inspect()
        self.assertTrue(len(ps.storage) >= 1)

        uri = ShpCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=uri, driver='vector')
        with self.print_scope() as ps:
            rd.inspect()
        self.assertTrue(len(ps.storage) >= 1)

        # test with a request dataset having no dimensioned variables
        path = self.get_temporary_file_path('bad.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('foo')
            var = ds.createVariable('foovar', int, dimensions=('foo',))
            var.name = 'a name'
        rd = RequestDataset(uri=path)
        with self.print_scope() as ps:
            rd.inspect()
        self.assertTrue(len(ps.storage) >= 1)

    def test_inspect_as_dct(self):
        variables = [
            self.variable,
            None,
            'foo',
            'time'
        ]

        for variable in variables:
            try:
                rd = RequestDataset(self.uri, variable)
                ret = rd.inspect_as_dct()
            except VariableNotFoundError:
                if variable == 'foo':
                    continue
                else:
                    raise
            except ValueError:
                if variable == 'time':
                    continue
                else:
                    raise
            ref = ret['derived']

            self.assertEqual(ref['End Date'], '2021-01-01 00:00:00')
            self.assertEqual(ref.keys(),
                             ['Name', 'Count', 'Has Bounds', 'Data Type', 'Start Date', 'End Date', 'Calendar', 'Units',
                              'Resolution (Days)', 'Spatial Reference', 'Proj4 String', 'Extent', 'Geometry Interface',
                              'Resolution'])

    def test_len(self):
        path = self.get_netcdf_path_no_dimensioned_variables()
        rd = RequestDataset(uri=path)
        self.assertEqual(len(rd), 0)

        rd = self.test_data.get_rd('cancm4_tas')
        self.assertEqual(len(rd), 1)

    def test_level_subset_without_level(self):
        lr = [1, 2]
        rd = self.test_data.get_rd('cancm4_tas')
        rd.level_range = lr
        with self.assertRaises(ValueError):
            rd.get()

    def test_name(self):
        path = ShpCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(uri=path, driver='vector')
        self.assertIsNotNone(rd.name)

        rd = RequestDataset(uri=path, driver='vector', name='states')
        self.assertEqual(rd.name, 'states')
        field = rd.get()
        self.assertEqual(field.name, 'states')

    def test_nonsense_units(self):
        with self.assertRaises(RequestValidationError):
            self.test_data.get_rd('cancm4_tas', kwds={'units': 'nonsense', 'conform_units_to': 'celsius'})

    def test_pickle(self):
        rd = RequestDataset(uri=self.uri, variable=self.variable)
        rd_path = os.path.join(ocgis.env.DIR_OUTPUT, 'rd.pkl')
        with open(rd_path, 'w') as f:
            pickle.dump(rd, f)
        with open(rd_path, 'r') as f:
            rd2 = pickle.load(f)
        self.assertTrue(rd == rd2)

    def test_source_dictionary_is_deepcopied(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        self.assertEqual(rd.source_metadata, field.meta)
        # the source metadata dictionary should be deepcopied prior to passing to a request dataset
        rd.source_metadata['dim_map'] = None
        self.assertNotEqual(rd.source_metadata, field.meta)

    def test_source_index_matches_constant_value(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        self.assertEqual(field.temporal._src_idx.dtype, constants.NP_INT)

    def test_str(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ss = str(rd)
        self.assertTrue(ss.startswith('RequestDataset'))
        self.assertTrue('crs' in ss)
        self.assertGreater(len(ss), 400)

    def test_units(self):
        rd = self.test_data.get_rd('cancm4_tas')
        self.assertEqual(rd.units, None)
        rd.units = 'K'
        self.assertEqual(rd.units, 'K')
        with self.assertRaises(RequestValidationError):
            rd.units = 'bad'
        path = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=path, units='celsius')
        self.assertEqual(rd.units, 'celsius')

    def test_uri_cannot_be_set(self):
        rd = self.test_data.get_rd('cancm4_tas')
        other_uri = self.test_data.get_uri('cancm4_rhs')
        with self.assertRaises(AttributeError):
            rd.uri = other_uri

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

    def test_with_bad_units_attempting_conform(self):
        # pass bad units to the init and an attempt a conform. values from the source dataset are not used for overload.
        rd = self.test_data.get_rd('cancm4_tas', kwds={'conform_units_to': 'celsius', 'units': 'coulomb'})
        with self.assertRaises(RequestValidationError):
            rd.get()

    def test_with_bad_units_passing_to_field(self):
        rd = self.test_data.get_rd('cancm4_tas', kwds={'units': 'celsius'})
        field = rd.get()
        self.assertEqual(field.variables['tas'].units, 'celsius')

    def test_with_units(self):
        units = 'celsius'
        rd = self.test_data.get_rd('cancm4_tas', kwds={'units': units})
        self.assertEqual(rd.units, 'celsius')

    def test_without_units_attempting_conform(self):
        # this will work because the units read from the metadata are equivalent
        self.test_data.get_rd('cancm4_tas', kwds={'conform_units_to': 'celsius'})
        # this will not work because the units are not equivalent
        rd = self.test_data.get_rd('cancm4_tas', kwds={'conform_units_to': 'coulomb'})
        with self.assertRaises(RequestValidationError):
            rd.get()

    def test_with_alias(self):
        rd = RequestDataset(self.uri, self.variable, alias='an_alias')
        self.assertEqual(rd.alias, 'an_alias')
        rd = RequestDataset(self.uri, self.variable, alias=None)
        self.assertEqual(rd.alias, self.variable)

    def test_time_range(self):
        tr = [dt(2000, 1, 1), dt(2000, 12, 31)]
        rd = RequestDataset(self.uri, self.variable, time_range=tr)
        self.assertEqual(rd.time_range, tuple(tr))

    def test_level_range(self):
        lr = [1, 1]
        rd = RequestDataset(self.uri, self.variable, level_range=lr)
        self.assertEqual(rd.level_range, tuple([1, 1]))

    def test_multiple_uris(self):
        rd = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        self.assertEqual(len(rd.uri), 2)

        with self.print_scope() as ps:
            rd.inspect()
        self.assertTrue(len(ps.storage) >= 1)

    def test_time_region(self):
        tr1 = {'month': [6], 'year': [2001]}
        rd = RequestDataset(uri=self.uri, variable=self.variable, time_region=tr1)
        self.assertEqual(rd.time_region, tr1)

        tr2 = {'bad': 15}
        with self.assertRaises(DefinitionValidationError):
            RequestDataset(uri=self.uri, variable=self.variable, time_region=tr2)


class TestRequestDatasetCollection(TestBase):

    def iter_keywords(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_rhs')

        keywords = dict(target=[None, rd1, [rd1], [rd1, rd2], {'uri': rd1.uri, 'variable': rd1.variable}, rd1.get(),
                                [rd1.get(), rd2.get()], [rd1, rd2.get()]])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            yield k

    def test(self):
        env.DIR_DATA = ocgis.env.DIR_TEST_DATA

        daymet = self.test_data.get_rd('daymet_tmax')
        tas = self.test_data.get_rd('cancm4_tas')

        uris = [daymet.uri,
                tas.uri]
        variables = ['foo1', 'foo2']
        rdc = RequestDatasetCollection()
        for uri, variable in zip(uris, variables):
            rd = RequestDataset(uri, variable)
            rdc.update(rd)
        self.assertEqual([1, 2], [rd.did for rd in rdc.values()])

        variables = ['foo1', 'foo1']
        rdc = RequestDatasetCollection()
        for ii, (uri, variable) in enumerate(zip(uris, variables)):
            rd = RequestDataset(uri, variable)
            if ii == 1:
                with self.assertRaises(KeyError):
                    rdc.update(rd)
            else:
                rdc.update(rd)

        aliases = ['a1', 'a2']
        for uri, variable, alias in zip(uris, variables, aliases):
            rd = RequestDataset(uri, variable, alias=alias)
            rdc.update(rd)
        for row in rdc.values():
            self.assertIsInstance(row, RequestDataset)
        self.assertIsInstance(rdc.first(), RequestDataset)
        self.assertIsInstance(rdc['a2'], RequestDataset)

    def test_init(self):
        for k in self.iter_keywords():
            rdc = RequestDatasetCollection(target=k.target)
            if k.target is not None:
                self.assertEqual(len(rdc), len(list(get_iter(k.target, dtype=(dict, RequestDataset, Field)))))
                self.assertTrue(len(rdc) >= 1)
            else:
                self.assertEqual(len(rdc), 0)

    def test_get_meta_rows(self):
        for k in self.iter_keywords():
            rdc = RequestDatasetCollection(target=k.target)
            rows = rdc._get_meta_rows_()
            self.assertTrue(len(rows) >= 1)

    def test_get_unique_id(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        rd_did = deepcopy(rd)
        rd_did.did = 1
        field_uid = deepcopy(field)
        field_uid.uid = 1

        for element in [rd, field, rd_did, field_uid]:
            uid = RequestDatasetCollection._get_unique_id_(element)
            try:
                self.assertEqual(uid, 1)
            except AssertionError:
                try:
                    self.assertIsNone(element.did)
                except AttributeError:
                    self.assertIsNone(element.uid)

    def test_iter_request_datasets(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field.name = 'foo'
        rdc = RequestDatasetCollection(target=[rd, field])
        tt = list(rdc.iter_request_datasets())
        self.assertEqual(len(tt), 1)
        self.assertEqual(len(rdc), 2)
        self.assertIsInstance(tt[0], RequestDataset)

    def test_name_attribute_used_for_keys(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd.name = 'hi_there'
        rdc = RequestDatasetCollection(target=[rd])
        self.assertEqual(rdc.keys(), ['hi_there'])

    def test_set_unique_id(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()

        for element in [rd, field]:
            RequestDatasetCollection._set_unique_id_(element, 5)
            uid = RequestDatasetCollection._get_unique_id_(element)
            self.assertEqual(uid, 5)

    def test_str(self):
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_rhs')
        rdc = RequestDatasetCollection(target=[rd1, rd2])
        ss = str(rdc)
        self.assertTrue(ss.startswith('RequestDatasetCollection'))
        self.assertGreater(len(ss), 900)

    def test_update(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd.did = 10
        field = rd.get()
        self.assertEqual(field.uid, 10)
        field.uid = 20

        rdc = RequestDatasetCollection()
        rdc.update(rd)
        # name is already in collection and should yield a key error
        with self.assertRaises(KeyError):
            rdc.update(field)
        field.name = 'tas2'
        rdc.update(field)

        # add another object and check the increment
        field2 = deepcopy(field)
        field2.name = 'hanzel'
        field2.uid = None
        rdc.update(field2)
        self.assertEqual(field2.uid, 21)

    def test_with_overloads(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        # loaded calendar should match file metadata
        self.assertEqual(field.temporal.calendar, '365_day')
        # the overloaded calendar in the request dataset should still be None
        self.assertEqual(rd.t_calendar, None)

        dataset = [{'time_region': None,
                    'uri': [rd.uri],
                    'alias': u'tas',
                    't_units': u'days since 1940-01-01 00:00:00',
                    'variable': u'tas',
                    't_calendar': u'will_not_work'}]
        rdc = RequestDatasetCollection(dataset)
        rd2 = RequestDataset(**dataset[0])
        # the overloaded calendar should be passed to the request dataset
        self.assertEqual(rd2.t_calendar, 'will_not_work')
        self.assertEqual(rdc.first().t_calendar, 'will_not_work')
        # when this bad calendar value is used it should raise an exception
        with self.assertRaises(ValueError):
            rdc.first().get().temporal.value_datetime

        dataset = [{'time_region': None,
                    'uri': [rd.uri],
                    'alias': u'tas',
                    't_units': u'days since 1940-01-01 00:00:00',
                    'variable': u'tas'}]
        rdc = RequestDatasetCollection(dataset)
        # ensure the overloaded units are properly passed
        self.assertEqual(rdc.first().get().temporal.units, 'days since 1940-01-01 00:00:00')
        # the calendar was not overloaded and the value should be read from the metadata
        self.assertEqual(rdc.first().get().temporal.calendar, '365_day')

    @attr('slow')
    def test_with_overloads_real_data(self):
        # copy the test file as the calendar attribute will be modified
        rd = self.test_data.get_rd('cancm4_tas')
        filename = os.path.split(rd.uri)[1]
        dest = os.path.join(self.current_dir_output, filename)
        shutil.copy2(rd.uri, dest)
        # modify the calendar attribute
        with nc_scope(dest, 'a') as ds:
            self.assertEqual(ds.variables['time'].calendar, '365_day')
            ds.variables['time'].calendar = '365_days'
        # assert the calendar is in fact changed on the source file
        with nc_scope(dest, 'r') as ds:
            self.assertEqual(ds.variables['time'].calendar, '365_days')
        rd2 = RequestDataset(uri=dest, variable='tas')
        field = rd2.get()
        # the bad calendar will raise a value error when the datetimes are converted.
        with self.assertRaises(ValueError):
            field.temporal.value_datetime
        # overload the calendar and confirm the datetime values are the same as the datetime values from the original
        # good file
        rd3 = RequestDataset(uri=dest, variable='tas', t_calendar='365_day')
        field = rd3.get()
        self.assertNumpyAll(field.temporal.value_datetime, rd.get().temporal.value_datetime)

        # pass as a dataset collection to operations and confirm the data may be written to a flat file. dates are
        # converted in the process.
        time_range = (datetime.datetime(2001, 1, 1, 0, 0), datetime.datetime(2011, 1, 1, 0, 0))
        dataset = [{'time_region': None,
                    'uri': dest,
                    'time_range': time_range,
                    'alias': u'tas',
                    't_units': u'days since 1850-1-1',
                    'variable': u'tas',
                    't_calendar': u'365_day'}]
        rdc = RequestDatasetCollection(dataset)
        ops = OcgOperations(dataset=rdc, geom='state_boundaries', select_ugid=[25],
                            output_format=constants.OUTPUT_FORMAT_SHAPEFILE)
        ops.execute()