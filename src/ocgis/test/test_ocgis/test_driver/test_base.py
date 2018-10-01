from copy import deepcopy

import numpy as np
from mock import mock

from ocgis import OcgOperations, SourcedVariable
from ocgis.driver.base import AbstractDriver, driver_scope
from ocgis.driver.nc import DriverNetcdf
from ocgis.driver.request.core import RequestDataset
from ocgis.exc import DefinitionValidationError
from ocgis.test.base import TestBase
from ocgis.test.base import attr


class FakeAbstractDriver(AbstractDriver):
    output_formats = ['shp']
    key = 'fake_driver'

    @property
    def extensions(self):
        raise NotImplementedError

    def get_variable_value(self, variable):
        raise NotImplementedError

    def _get_metadata_main_(self):
        raise NotImplementedError

    def _init_variable_from_source_main_(self, variable_object, variable_metadata):
        pass

    def _write_variable_collection_main_(self, *args, **kwargs):
        pass


class FixtureAbstractDriver(object):
    def fixture_abstract_driver(self):
        rd = mock.create_autospec(RequestDataset, spec_set=True)

        group2 = {'variables': {'alone': {'dtype': np.int8},
                                'friendly': {}}}
        group3 = {'variables': {'food': {'dimensions': ('c',)}},
                  'dimensions': {'c': {'size': 5}}}

        rd.metadata = {'dimensions': {'foo': {'name': 'foo', 'size': 10},
                                      'foobar': {'name': 'foobar', 'size': 11},
                                      'boundless': {'name': 'boundless', 'size': 9, 'isunlimited': True}},
                       'variables': {'tester': {'name': 'tester', 'dimensions': ('foo', 'foobar'), 'dtype': 'int'}},
                       'groups': {'group2': group2,
                                  'group3': group3}}

        driver = FakeAbstractDriver(rd)

        rd.driver = driver
        return driver


class Test(FixtureAbstractDriver):
    def test_driver_scope(self):
        ocgis_driver = self.fixture_abstract_driver()
        ocgis_driver.rd.opened = None
        ocgis_driver.open = mock.Mock(return_value='opened dataset')
        ocgis_driver.close = mock.Mock(return_value='closed dataset')
        desired = {'my': 'driver arg'}
        ocgis_driver.rd.driver_kwargs = desired
        ocgis_driver.inquire_opened_state = mock.Mock(return_value=False)
        with driver_scope(ocgis_driver) as _:
            ocgis_driver.open.assert_called_once_with(my='driver arg', mode='r', rd=ocgis_driver.rd,
                                                      uri=ocgis_driver.rd.uri)


class TestAbstractDriver(TestBase, FixtureAbstractDriver):
    @attr('data')
    def test_eq(self):
        rd = self.test_data.get_rd('cancm4_tas')
        d = DriverNetcdf(rd)
        d2 = deepcopy(d)
        self.assertEqual(d, deepcopy(d))

        d2.key = 'bad'
        self.assertNotEqual(d, d2)

    def test_create_dimensions(self):
        # Test renamed dimension is found.
        driver = self.fixture_abstract_driver()
        dst_metadata = deepcopy(driver.rd.metadata)
        dst_metadata['dimensions']['foobar']['name'] = 'renamed'

        actual = driver.create_dimensions(dst_metadata)
        self.assertEqual(actual['foobar'].size, 11)
        self.assertEqual(actual['foobar'].name, 'renamed')

        self.assertTrue(actual['boundless'].is_unlimited)
        self.assertEqual(actual['boundless'].size_current, 9)
        self.assertIsNone(actual['boundless'].size)

    def test_create_raw_field(self):
        driver = self.fixture_abstract_driver()
        actual = driver.create_raw_field()
        self.assertEqual(actual.groups['group2']['alone'].dtype, np.int8)
        self.assertEqual(actual.groups['group2'].dimensions, {})

    def test_create_variables(self):
        # Test variable uses renamed dimension.
        driver = self.fixture_abstract_driver()
        dst_metadata = deepcopy(driver.rd.metadata)
        dst_metadata['dimensions']['foobar']['name'] = 'renamed'
        dst_metadata['variables']['tester']['name'] = 'imavar'

        dist = driver.create_dist(dst_metadata)
        driver._dist = dist
        actual = driver.create_variables(dst_metadata)
        self.assertEqual(actual['tester'].dimensions[1].name, 'renamed')
        self.assertEqual(actual['tester'].source_name, 'tester')

    def test_create_varlike(self):
        var = AbstractDriver.create_varlike(name='foo')
        self.assertEqual(var.name, 'foo')
        self.assertIsInstance(var, SourcedVariable)

    @attr('data')
    def test_inspect(self):
        rd = self.test_data.get_rd('cancm4_tas')
        driver = DriverNetcdf(rd)
        with self.print_scope() as ps:
            driver.inspect()
        self.assertTrue(len(ps.storage) > 1)

    @attr('data')
    def test_validate_ops(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = OcgOperations(dataset=rd)

        with self.assertRaises(DefinitionValidationError):
            FakeAbstractDriver.validate_ops(ops)

        prev = FakeAbstractDriver.output_formats
        FakeAbstractDriver.output_formats = 'all'
        try:
            FakeAbstractDriver.validate_ops(ops)
        finally:
            FakeAbstractDriver.output_formats = prev
