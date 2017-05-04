from copy import deepcopy

import numpy as np

from ocgis import constants, OcgOperations, FunctionRegistry
from ocgis import env
from ocgis.api.parms.definition_helpers import MetadataAttributes
from ocgis.calc.base import AbstractUnivariateFunction, AbstractUnivariateSetFunction, AbstractFunction, \
    AbstractMultivariateFunction, AbstractParameterizedFunction
from ocgis.exc import UnitsValidationError, DefinitionValidationError
from ocgis.interface.base.variable import VariableCollection, DerivedVariable
from ocgis.test.base import TestBase
from ocgis.test.base import attr
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
from ocgis.util.units import get_units_object


class FooNeedsUnits(AbstractUnivariateFunction):
    description = 'calculation with units'
    key = 'fnu'
    required_units = ['K', 'kelvin']
    standard_name = 'foo_needs_units'
    long_name = 'Foo Needs Units'

    def calculate(self, values):
        return values


class FooNeedsUnitsSet(AbstractUnivariateSetFunction):
    description = 'calculation with units'
    dtype_default = 'int'
    key = 'fnu'
    required_units = ['K', 'kelvin']
    standard_name = ''
    long_name = ''

    def calculate(self, values):
        return np.ma.mean(values, axis=0)


class FooSampleSize(FooNeedsUnitsSet):
    standard_name = 'the_standard'
    long_name = 'the_standard_long_name'


class TestAbstractFunction(AbstractTestField):
    def test_init(self):
        f = FooNeedsUnits()
        self.assertEqual(f.dtype, env.NP_FLOAT)

        # Test overloading the datatype.
        f = FooNeedsUnits(dtype=np.int16)
        self.assertEqual(f.dtype, np.int16)

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
                attrs = {'standard_name': constants.DEFAULT_SAMPLE_SIZE_STANDARD_NAME,
                         'long_name': constants.DEFAULT_SAMPLE_SIZE_LONG_NAME}
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
                    meta_attrs = MetadataAttributes({'something_new': 'is about to happen', 'standard_name': 'never!'})
                else:
                    meta_attrs = MetadataAttributes({'something_new': 'is about to happen'})
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
                    self.assertDictEqual(meta_attrs.value['variable'],
                                         {'something_new': 'is about to happen', 'standard_name': 'never!'})
                else:
                    self.assertDictEqual(meta_attrs.value['variable'], {'something_new': 'is about to happen'})

        # test attributes are applied to the field object
        field = self.get_field(with_value=True)
        meta_attrs = MetadataAttributes({'field': {'hoover': 'dam'}})
        fb = FooNeedsUnits(field=field, meta_attrs=meta_attrs)
        fb.execute()
        self.assertDictEqual(fb.field.attrs, {'hoover': 'dam'})


class FakeAbstractMultivariateFunction(AbstractMultivariateFunction):
    description = ''
    dtype_default = 'int'
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

    @attr('data')
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


class FooAbstractParameterizedFunction(AbstractParameterizedFunction):
    key = 'foo_pf'
    long_name = 'foo_pee_eff'
    standard_name = 'fpf'
    parms_definition = {'argA': int, 'argB': float}
    description = None

    def calculate(self, values, **kwargs):
        raise NotImplementedError

    def _execute_(self):
        raise NotImplementedError


class FooAbstractParameterizedFunctionRequiredParameters(FooAbstractParameterizedFunction):
    parms_required = ('argB',)


class FooAbstractParameterizedFunctionRequiredParametersMultivariate(AbstractMultivariateFunction,
                                                                     FooAbstractParameterizedFunctionRequiredParameters):
    required_variables = ['tas', 'pr']


class TestAbstractParameterizedFunction(AbstractTestField):
    def test_init(self):
        ff = FooAbstractParameterizedFunction()
        self.assertIsInstance(ff, AbstractParameterizedFunction)
        self.assertIsNone(ff.parms_required)

    def test_validate_definition(self):
        definition = {'func': 'foo_pf', 'name': 'food'}

        # Keywords are required.
        with self.assertRaises(DefinitionValidationError):
            FooAbstractParameterizedFunction.validate_definition(definition)

        # These are the wrong keyword arguments.
        definition = {'func': 'foo_pf', 'name': 'food', 'kwds': {'argC': 'never'}}
        with self.assertRaises(DefinitionValidationError):
            FooAbstractParameterizedFunction.validate_definition(definition)

        # One parameter passed.
        definition = {'func': 'foo_pf', 'name': 'food', 'kwds': {'argA': 5}}
        FooAbstractParameterizedFunction.validate_definition(definition)

        # This function class has some required parameters.
        definition = {'func': 'foo_pf', 'name': 'food', 'kwds': {'argA': 5}}
        with self.assertRaises(DefinitionValidationError):
            FooAbstractParameterizedFunctionRequiredParameters.validate_definition(definition)

        # Test with required variables present.
        definition = {'func': 'foo_pf', 'name': 'food', 'kwds': {'argB': 5, 'tas': None}}
        FooAbstractParameterizedFunctionRequiredParametersMultivariate.validate_definition(definition)


class TestAbstractUnivariateFunction(AbstractTestField):
    def test_validate_units(self):
        field = self.get_field(with_value=True)
        fnu = FooNeedsUnits(field=field)
        ret = fnu.execute()
        self.assertNumpyAll(field.variables['tmax'].value.astype(FooNeedsUnits.get_dtype()),
                            ret['fnu'].value)

    def test_validate_units_bad_units(self):
        field = self.get_field(with_value=True)
        field.variables['tmax'].units = 'celsius'
        self.assertEqual(field.variables['tmax'].cfunits, get_units_object('celsius'))
        fnu = FooNeedsUnits(field=field)
        with self.assertRaises(UnitsValidationError):
            fnu.execute()


class TestAbstractUnivariateSetFunction(AbstractTestField):
    def test_validate_units(self):
        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        fnu = FooNeedsUnitsSet(field=field, tgd=tgd)
        fnu.execute()

    def test_validate_units_bad_units(self):
        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        field.variables['tmax'].units = 'celsius'
        self.assertEqual(field.variables['tmax'].cfunits, get_units_object('celsius'))
        fnu = FooNeedsUnitsSet(field=field, tgd=tgd)
        with self.assertRaises(UnitsValidationError):
            fnu.execute()
