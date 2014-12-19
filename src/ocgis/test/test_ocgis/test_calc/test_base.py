from copy import deepcopy

from cfunits.cfunits import Units
import numpy as np

from ocgis.interface.base.variable import VariableCollection, DerivedVariable
from ocgis.test.base import TestBase
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
from ocgis.calc.base import AbstractUnivariateFunction, AbstractUnivariateSetFunction, AbstractFunction, \
    AbstractMultivariateFunction
from ocgis import constants, OcgOperations, FunctionRegistry
from ocgis.exc import UnitsValidationError, DefinitionValidationError


class FooNeedsUnits(AbstractUnivariateFunction):
    description = 'calculation with units'
    dtype = constants.np_float
    key = 'fnu'
    required_units = ['K', 'kelvin']
    standard_name = 'foo_needs_units'
    long_name = 'Foo Needs Units'
    
    def calculate(self, values):
        return values
            
            
class FooNeedsUnitsSet(AbstractUnivariateSetFunction):
    description = 'calculation with units'
    dtype = constants.np_float
    key = 'fnu'
    required_units = ['K', 'kelvin']
    standard_name = ''
    long_name = ''
    
    def calculate(self,values):
        return np.ma.mean(values,axis=0)


class FooSampleSize(FooNeedsUnitsSet):
    standard_name = 'the_standard'
    long_name = 'the_standard_long_name'


class TestAbstractFunction(AbstractTestField):

    def test_add_to_collection(self):
        kwds = dict(calc_sample_size=[False, True])

        for k in self.iter_product_keywords(kwds):
            field = self.get_field(with_value=True)
            tgd = field.temporal.get_grouping(['month'])
            fb = FooSampleSize(field=field, calc_sample_size=k.calc_sample_size, tgd=tgd)
            res = fb.execute()
            variable = res.first()
            self.assertIsInstance(res, VariableCollection)
            self.assertIsInstance(variable, DerivedVariable)
            attrs = {'standard_name': fb.standard_name, 'long_name': fb.long_name}
            self.assertDictEqual(attrs, variable.attrs)

            if k.calc_sample_size:
                alias = 'n_{0}'.format(variable.alias)
                ss = res[alias]
                attrs = {'standard_name': constants.default_sample_size_standard_name,
                         'long_name': constants.default_sample_size_long_name}
                self.assertDictEqual(ss.attrs, attrs)

    def test_add_to_collection_parents(self):
        """Test adding parents to the output derived variable."""

        field = self.get_field(with_value=True)
        ff = FooNeedsUnits(field=field)
        res = ff.execute()
        self.assertIsNone(res['fnu'].parents)

        ff = FooNeedsUnits(field=field, add_parents=True)
        res = ff.execute()
        var = res['fnu']
        self.assertIsInstance(var.parents, VariableCollection)

    def test_execute_meta_attrs(self):
        """Test overloaded metadata attributes are appropriately applied."""

        for has_meta in [True, False]:
            field = self.get_field(with_value=True)
            if has_meta:
                field.meta = {'attrs': 'already has something'}
            else:
                field.meta = {}
            for oload in [True, False]:
                if oload:
                    meta_attrs = {'something_new': 'is about to happen', 'standard_name': 'never!'}
                else:
                    meta_attrs = {'something_new': 'is about to happen'}
                fb = FooNeedsUnits(field=field, meta_attrs=meta_attrs)
                ret = fb.execute()
                if oload:
                    actual = {'long_name': 'Foo Needs Units', 'standard_name': 'never!',
                              'something_new': 'is about to happen'}
                else:
                    actual = {'long_name': 'Foo Needs Units', 'standard_name': 'foo_needs_units',
                              'something_new': 'is about to happen'}
                self.assertDictEqual(ret['fnu'].attrs, actual)
                if oload:
                    self.assertDictEqual(meta_attrs, {'something_new': 'is about to happen', 'standard_name': 'never!'})
                else:
                    self.assertDictEqual(meta_attrs, {'something_new': 'is about to happen'})


class FakeAbstractMultivariateFunction(AbstractMultivariateFunction):
    description = ''
    dtype = int
    key = 'fmv'
    long_name = 'long'
    standard_name = 'short'
    required_variables = ['tas', 'pr']

    def calculate(self, *args, **kwargs):
        pass


class TestAbstractMultivariateFunction(TestBase):

    def test_init(self):
        self.assertEqual(AbstractMultivariateFunction.__bases__, (AbstractFunction,))

        FakeAbstractMultivariateFunction()

    def test_validate(self):
        FunctionRegistry.append(FakeAbstractMultivariateFunction)
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd1.alias = 'tas2'
        rd2 = deepcopy(rd1)
        rd2.alias = 'pr2'

        # test non-string keyword arguments will not raise an exception
        calc = [{'func': 'fmv', 'name': 'fmv', 'kwds': {'tas': 'tas2', 'pr': 'pr2', 'random': {}}}]
        OcgOperations(dataset=[rd1, rd2], calc=calc)

        # test with an alias map missing
        calc = [{'func': 'fmv', 'name': 'fmv', 'kwds': {'pr': 'pr2', 'random': {}}}]
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=[rd1, rd2], calc=calc)

        # test with the wrong alias mapped
        calc = [{'func': 'fmv', 'name': 'fmv', 'kwds': {'tas': 'tas2', 'pr': 'pr3', 'random': {}}}]
        with self.assertRaises(DefinitionValidationError):
            OcgOperations(dataset=[rd1, rd2], calc=calc)


class TestAbstractUnivariateFunction(AbstractTestField):
    
    def test_validate_units(self):
        field = self.get_field(with_value=True)
        fnu = FooNeedsUnits(field=field)
        ret = fnu.execute()
        self.assertNumpyAll(field.variables['tmax'].value.astype(FooNeedsUnits.dtype),
                            ret['fnu'].value)
        
    def test_validate_units_bad_units(self):
        field = self.get_field(with_value=True)
        field.variables['tmax'].units = 'celsius'
        self.assertEqual(field.variables['tmax'].cfunits,Units('celsius'))
        fnu = FooNeedsUnits(field=field)
        with self.assertRaises(UnitsValidationError):
            fnu.execute()
            
            
class TestAbstractUnivariateSetFunction(AbstractTestField):
    
    def test_validate_units(self):
        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        fnu = FooNeedsUnitsSet(field=field,tgd=tgd)
        fnu.execute()
        
    def test_validate_units_bad_units(self):
        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        field.variables['tmax'].units = 'celsius'
        self.assertEqual(field.variables['tmax'].cfunits,Units('celsius'))
        fnu = FooNeedsUnitsSet(field=field,tgd=tgd)
        with self.assertRaises(UnitsValidationError):
            fnu.execute()
