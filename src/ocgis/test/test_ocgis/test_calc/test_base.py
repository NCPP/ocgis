from copy import deepcopy

import numpy as np

from ocgis import OcgOperations, FunctionRegistry
from ocgis import env
from ocgis.calc.base import AbstractUnivariateFunction, AbstractUnivariateSetFunction, AbstractFunction, \
    AbstractMultivariateFunction, AbstractParameterizedFunction, AbstractFieldFunction
from ocgis.collection.field import OcgField
from ocgis.driver.request import RequestDataset
from ocgis.exc import UnitsValidationError, DefinitionValidationError
from ocgis.ops.parms.definition_helpers import MetadataAttributes
from ocgis.test.base import TestBase, AbstractTestField
from ocgis.test.base import attr
from ocgis.util.units import get_units_object
from ocgis.variable.base import Variable


class MockNeedsUnits(AbstractUnivariateFunction):
    description = 'calculation with units'
    key = 'fnu'
    required_units = ['K', 'kelvin']
    standard_name = 'foo_needs_units'
    long_name = 'Foo Needs Units'

    def calculate(self, values):
        return values


class MockNeedsUnitsSet(AbstractUnivariateSetFunction):
    description = 'calculation with units'
    dtype_default = 'int'
    key = 'fnu'
    required_units = ['K', 'kelvin']
    standard_name = ''
    long_name = ''

    def calculate(self, values):
        return np.ma.mean(values, axis=0)


class MockSampleSize(MockNeedsUnitsSet):
    standard_name = 'the_standard'
    long_name = 'the_standard_long_name'


class TestAbstractFunction(AbstractTestField):
    def test_init(self):
        f = MockNeedsUnits()
        self.assertEqual(f.get_default_dtype(), env.NP_FLOAT)

        # Test overloading the datatype.
        f = MockNeedsUnits(dtype=np.int16)
        self.assertEqual(f.dtype, np.int16)

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
                fb = MockNeedsUnits(field=field, meta_attrs=meta_attrs)
                ret = fb.execute()
                if oload:
                    actual = {'long_name': 'Foo Needs Units', 'standard_name': 'never!',
                              'something_new': 'is about to happen', 'units': 'kelvin'}
                else:
                    actual = {'long_name': 'Foo Needs Units', 'standard_name': 'foo_needs_units',
                              'something_new': 'is about to happen', 'units': 'kelvin'}
                self.assertDictEqual(ret['fnu'].attrs, actual)
                if oload:
                    self.assertDictEqual(meta_attrs.value['variable'],
                                         {'something_new': 'is about to happen', 'standard_name': 'never!'})
                else:
                    self.assertDictEqual(meta_attrs.value['variable'], {'something_new': 'is about to happen'})

        # test attributes are applied to the field object
        field = self.get_field(with_value=True)
        meta_attrs = MetadataAttributes({'field': {'hoover': 'dam'}})
        fb = MockNeedsUnits(field=field, meta_attrs=meta_attrs)
        fb.execute()
        self.assertDictEqual(fb.field.attrs, {'hoover': 'dam'})


class MockFieldFunction(AbstractFieldFunction):
    key = 'mff'
    long_name = 'expand the abbreviations'
    standard_name = 'mock_field_function'
    description = 'Used for testing a field function'

    def calculate(self):
        squared_value = self.field['data'].get_value() ** 2
        squared = self.get_fill_variable(self.field['data'], self.alias, self.field['data'].dimensions, dtype=float,
                                         variable_value=squared_value)
        self.vc.add_variable(squared)

        # Field functions modify their associated fields.
        self.field.pop('data')


class TestAbstractFieldFunction(TestBase):
    desired_value = [16.0, 25.0, 36.0]

    @property
    def field_for_test(self):
        data = Variable(name='data', value=[4, 5, 6], dimensions='three')
        field = OcgField(variables=data)
        return field

    def setUp(self):
        super(TestAbstractFieldFunction, self).setUp()
        FunctionRegistry.append(MockFieldFunction)

    def tearDown(self):
        super(TestAbstractFieldFunction, self).tearDown()
        FunctionRegistry.reg.pop(0)

    def test_execute(self):
        ff = MockFieldFunction(field=self.field_for_test)
        res = ff.execute()
        self.assertEqual(res[MockFieldFunction.key].get_value().tolist(), self.desired_value)

    def test_system_through_operations(self):
        # tdk: test only one field function allowed
        ops = OcgOperations(dataset=self.field_for_test, calc=[{'func': 'mff', 'name': 'my_mff'}])
        ret = ops.execute()

        actual_field = ret.get_element()
        actual_variable = actual_field['my_mff']
        self.assertEqual(actual_variable.attrs['long_name'], MockFieldFunction.long_name)
        self.assertEqual(actual_variable.get_value().tolist(), self.desired_value)
        self.assertNotIn('data', actual_field.keys())

        # Test writing output to netCDF.
        ops = OcgOperations(dataset=self.field_for_test, calc=[{'func': 'mff', 'name': 'my_mff'}], output_format='nc')
        ret = ops.execute()
        actual_field = RequestDataset(ret).get()
        self.assertEqual(actual_field['my_mff'].get_value().tolist(), self.desired_value)


class MockMultiParamFunction(AbstractFieldFunction, AbstractMultivariateFunction, AbstractParameterizedFunction):
    key = 'mock_mpf'
    long_name = 'expand the abbreviations again'
    standard_name = 'mock_multi_param_function'
    description = 'Used for testing a multivariate, parameterized field function'
    parms_definition = {'the_exponent': int, 'offset': float}
    required_variables = ('lhs', 'rhs')

    def calculate(self, lhs=None, rhs=None, the_exponent=None, offset=None):
        # Access the variable object by name from the calculation field.
        lhs = self.field[lhs]
        rhs = self.field[rhs]

        # Dimensions similar to netCDF dimensions are available on the variables.
        assert len(lhs.dimensions) > 0
        # The get_value() call returns a numpy array. Mask is retrieved by get_mask(). You can get a masked array
        # by using get_masked_value(). These return references.
        value = (rhs.get_value() - lhs.get_value()) ** the_exponent + offset
        # Recommended that this method is used to create the output variables. Adds appropriate calculations attributes,
        # extra record information for tabular output, etc. At the very least, it is import to reuse the dimensions
        # appropriately as they contain global/local bounds for parallel IO. You can pass a masked array to
        # "variable_value".
        variable = self.get_fill_variable(lhs, self.alias, lhs.dimensions, variable_value=value)
        # Add the output variable to calculations variable collection. This is what is returned by the execute() call.
        self.vc.add_variable(variable)


class TestMockMultiParamFunction(TestBase):
    desired_value = [30.5, 37.5, 57.5]

    @property
    def field_for_test(self):
        field = OcgField(variables=[self.variable_lhs_for_test, self.variable_rhs_for_test])
        return field

    @property
    def fields_for_ops_test(self):
        field1 = OcgField(variables=self.variable_lhs_for_test)
        field2 = OcgField(variables=self.variable_rhs_for_test)
        return [field1, field2]

    @property
    def parms_for_test(self):
        return {'lhs': 'left', 'rhs': 'right', 'the_exponent': 2, 'offset': 21.5}

    @property
    def variable_lhs_for_test(self):
        return Variable(name='left', value=[4, 5, 6], dimensions='three', dtype=float)

    @property
    def variable_rhs_for_test(self):
        return Variable(name='right', value=[7, 9, 12], dimensions='three', dtype=float)

    def setUp(self):
        super(TestMockMultiParamFunction, self).setUp()
        FunctionRegistry.append(MockMultiParamFunction)

    def tearDown(self):
        super(TestMockMultiParamFunction, self).tearDown()
        FunctionRegistry.reg.pop(0)

    def test_execute(self):
        ff = MockMultiParamFunction(field=self.field_for_test, parms=self.parms_for_test)
        res = ff.execute()
        self.assertEqual(res[MockMultiParamFunction.key].get_value().tolist(), self.desired_value)

    def test_system_through_operations(self):
        calc = [{'func': MockMultiParamFunction.key, 'name': 'my_mvp', 'kwds': self.parms_for_test}]
        ops = OcgOperations(dataset=self.fields_for_ops_test, calc=calc)
        ret = ops.execute()

        actual_variable = ret.get_element(variable_name='my_mvp')
        self.assertEqual(actual_variable.get_value().tolist(), self.desired_value)

        ops = OcgOperations(dataset=self.fields_for_ops_test, calc=calc, output_format='nc')
        ret = ops.execute()
        actual = RequestDataset(ret).get()['my_mvp']
        self.assertEqual(actual.get_value().tolist(), self.desired_value)


class MockAbstractMultivariateFunction(AbstractMultivariateFunction):
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

        MockAbstractMultivariateFunction()

    @attr('data')
    def test_validate(self):
        FunctionRegistry.append(MockAbstractMultivariateFunction)
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd1._rename_variable = 'tas2'
        rd2 = deepcopy(rd1)
        rd2._rename_variable = 'pr2'

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


class MockAbstractParameterizedFunction(AbstractParameterizedFunction):
    key = 'foo_pf'
    long_name = 'foo_pee_eff'
    standard_name = 'fpf'
    parms_definition = {'argA': int, 'argB': float}
    description = None

    def calculate(self, values, **kwargs):
        raise NotImplementedError

    def _execute_(self):
        raise NotImplementedError


class MockAbstractParameterizedFunctionRequiredParameters(MockAbstractParameterizedFunction):
    parms_required = ('argB',)


class MockAbstractParameterizedFunctionRequiredParametersMultivariate(AbstractMultivariateFunction,
                                                                      MockAbstractParameterizedFunctionRequiredParameters):
    required_variables = ['tas', 'pr']


class TestAbstractParameterizedFunction(AbstractTestField):
    def test_init(self):
        ff = MockAbstractParameterizedFunction()
        self.assertIsInstance(ff, AbstractParameterizedFunction)
        self.assertIsNone(ff.parms_required)

    def test_validate_definition(self):
        definition = {'func': 'foo_pf', 'name': 'food'}

        # Keywords are required.
        with self.assertRaises(DefinitionValidationError):
            MockAbstractParameterizedFunction.validate_definition(definition)

        # These are the wrong keyword arguments.
        definition = {'func': 'foo_pf', 'name': 'food', 'kwds': {'argC': 'never'}}
        with self.assertRaises(DefinitionValidationError):
            MockAbstractParameterizedFunction.validate_definition(definition)

        # One parameter passed.
        definition = {'func': 'foo_pf', 'name': 'food', 'kwds': {'argA': 5}}
        MockAbstractParameterizedFunction.validate_definition(definition)

        # This function class has some required parameters.
        definition = {'func': 'foo_pf', 'name': 'food', 'kwds': {'argA': 5}}
        with self.assertRaises(DefinitionValidationError):
            MockAbstractParameterizedFunctionRequiredParameters.validate_definition(definition)

        # Test with required variables present.
        definition = {'func': 'foo_pf', 'name': 'food', 'kwds': {'argB': 5, 'tas': None}}
        MockAbstractParameterizedFunctionRequiredParametersMultivariate.validate_definition(definition)


class TestAbstractUnivariateFunction(AbstractTestField):
    def test_validate_units(self):
        field = self.get_field(with_value=True)
        fnu = MockNeedsUnits(field=field)
        ret = fnu.execute()
        self.assertNumpyAll(field['tmax'].value.astype(MockNeedsUnits.get_default_dtype()),
                            ret['fnu'].value)

    def test_validate_units_bad_units(self):
        field = self.get_field(with_value=True)
        field['tmax'].units = 'celsius'
        self.assertEqual(field['tmax'].cfunits, get_units_object('celsius'))
        fnu = MockNeedsUnits(field=field)
        with self.assertRaises(UnitsValidationError):
            fnu.execute()


class TestAbstractUnivariateSetFunction(AbstractTestField):
    def test_validate_units(self):
        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        fnu = MockNeedsUnitsSet(field=field, tgd=tgd)
        fnu.execute()

    def test_validate_units_bad_units(self):
        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        field['tmax'].units = 'celsius'
        self.assertEqual(field['tmax'].cfunits, get_units_object('celsius'))
        fnu = MockNeedsUnitsSet(field=field, tgd=tgd)
        with self.assertRaises(UnitsValidationError):
            fnu.execute()
