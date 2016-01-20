import numpy as np

from ocgis import env
from ocgis.calc.eval_function import EvalFunction
from ocgis.test.base import TestBase
from ocgis.test.base import attr


class TestEvalFunction(TestBase):
    def test_init(self):
        expr = 'es=6.1078*exp(17.08085*(tas-273.16)/(234.175+(tas-273.16)))'
        ef = EvalFunction(expr=expr)
        self.assertEqual(ef.expr, expr)

    @attr('data')
    def test_calculation_file_only_one_variable(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field = field[:, 0:10, :, :, :]
        expr = 'es=6.1078*exp(17.08085*(tas-273.16)/(234.175+(tas-273.16)))'
        ef = EvalFunction(expr=expr, field=field, file_only=True)
        ret = ef.execute()
        self.assertEqual(ret['es']._value, None)
        self.assertEqual(ret['es'].dtype, env.NP_FLOAT)
        self.assertEqual(ret['es'].fill_value, np.ma.array([1], dtype=env.NP_FLOAT).fill_value)

    @attr('data')
    def test_calculation_file_only_two_variables(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tasmax_2001')
        field = rd.get()
        field2 = rd2.get()
        field.variables.add_variable(field2.variables['tasmax'], assign_new_uid=True)
        field = field[:, 0:10, :, :, :]
        expr = 'foo=log(1000*(tasmax-tas))/3'
        ef = EvalFunction(expr=expr, field=field, file_only=True)
        ret = ef.execute()
        self.assertEqual(ret['foo']._value, None)

    @attr('data')
    def test_calculation_one_variable_exp_and_log(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field = field[:, 0:10, :, :, :]
        expr = 'es=6.1078*exp(log(17.08085)*(tas-273.16)/(234.175+(tas-273.16)))'
        ef = EvalFunction(expr=expr, field=field)
        ret = ef.execute()
        var = field.variables['tas']
        actual_value = 6.1078 * np.exp(np.log(17.08085) * (var.value - 273.16) / (234.175 + (var.value - 273.16)))
        self.assertNumpyAll(ret['es'].value, actual_value)

    @attr('data')
    def test_calculation_one_variable_exp_only(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field = field[:, 0:10, :, :, :]
        expr = 'es=6.1078*exp(17.08085*(tas-273.16)/(234.175+(tas-273.16)))'
        ef = EvalFunction(expr=expr, field=field, add_parents=True)
        ret = ef.execute()
        self.assertEqual(ret.keys(), ['es'])
        self.assertEqual(ret['es'].units, None)
        self.assertEqual(ret['es'].alias, 'es')
        self.assertEqual(ret['es'].name, 'es')
        self.assertEqual(ret['es'].parents.keys(), ['tas'])

        var = field.variables['tas']
        actual_value = 6.1078 * np.exp(17.08085 * (var.value - 273.16) / (234.175 + (var.value - 273.16)))
        self.assertNumpyAll(ret['es'].value, actual_value)

    @attr('data')
    def test_calculation_two_variables_exp_only(self):
        rd = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tasmax_2001')
        field = rd.get()
        field2 = rd2.get()
        field.variables.add_variable(field2.variables['tasmax'], assign_new_uid=True)
        field = field[:, 0:10, :, :, :]
        expr = 'foo=log(1000*(tasmax-tas))/3'
        ef = EvalFunction(expr=expr, field=field, add_parents=True)
        ret = ef.execute()
        self.assertEqual(ret.keys(), ['foo'])
        self.assertEqual(set(ret['foo'].parents.keys()), {'tas', 'tasmax'})

        tas = field.variables['tas']
        tasmax = field.variables['tasmax']
        actual_value = np.log(1000 * (tasmax.value - tas.value)) / 3
        self.assertNumpyAll(ret['foo'].value, actual_value)

    def test_get_eval_string(self):
        expr = 'es=6.1078*exp(log(17.08085)*(tas-273.16)/(234.175+(tas-273.16)))'
        to_eval, out_variable_name = EvalFunction._get_eval_string_(expr, {'tas': 'var.value'})
        self.assertEqual(to_eval, '6.1078*np.exp(np.log(17.08085)*(var.value-273.16)/(234.175+(var.value-273.16)))')
        self.assertEqual(out_variable_name, 'es')

        expr = 'tas=tas-4'
        to_eval, out_variable_name = EvalFunction._get_eval_string_(expr, {'tas': 'var.value'})
        self.assertEqual(to_eval, 'var.value-4')
        self.assertEqual(out_variable_name, 'tas')

        expr = 'tas=4-tas'
        to_eval, out_variable_name = EvalFunction._get_eval_string_(expr, {'tas': 'var.value'})
        self.assertEqual(to_eval, '4-var.value')
        self.assertEqual(out_variable_name, 'tas')

        expr = 'tas=tas'
        to_eval, out_variable_name = EvalFunction._get_eval_string_(expr, {'tas': 'var.value'})
        self.assertEqual(to_eval, 'var.value')
        self.assertEqual(out_variable_name, 'tas')

        expr = 'tas=tas-tas-tas'
        to_eval, out_variable_name = EvalFunction._get_eval_string_(expr, {'tas': 'var.value'})
        self.assertEqual(to_eval, 'var.value-var.value-var.value')
        self.assertEqual(out_variable_name, 'tas')

        expr = 'tas=tas-tas-tas-tas'
        to_eval, out_variable_name = EvalFunction._get_eval_string_(expr, {'tas': 'var.value'})
        self.assertEqual(to_eval, 'var.value-var.value-var.value-var.value')
        self.assertEqual(out_variable_name, 'tas')

        expr = 'tas_2=tas_1-tas_1-tas_1-tas_1'
        to_eval, out_variable_name = EvalFunction._get_eval_string_(expr, {'tas_1': 'var.value'})
        self.assertEqual(to_eval, 'var.value-var.value-var.value-var.value')
        self.assertEqual(out_variable_name, 'tas_2')

        expr = 'tas=tas-tas-tas-tas-tasmax-tas'
        to_eval, out_variable_name = EvalFunction._get_eval_string_(expr, {'tas': 'var.value', 'tasmax': 'var2.value'})
        self.assertEqual(to_eval, 'var.value-var.value-var.value-var.value-var2.value-var.value')
        self.assertEqual(out_variable_name, 'tas')

    def test_get_eval_string_bad_string(self):
        # this string has no equals sign
        expr = 'es6.1078*exp(log(17.08085)*(tas-273.16)/(234.175+(tas-273.16)))'
        with self.assertRaises(ValueError):
            EvalFunction._get_eval_string_(expr, {'tas': 'var.value'})

        # this string has a numpy function "foo" that is not enabled (and does not exist)
        expr = 'es=6.1078*exp(foo(17.08085)*(tas-273.16)/(234.175+(tas-273.16)))'
        with self.assertRaises(ValueError):
            EvalFunction._get_eval_string_(expr, {'tas': 'var.value'})

    def test_get_eval_string_power(self):
        """Test the power ufunc is appropriately parsed from the eval string."""

        expr = 'es=power(foo, 4)'
        map_vars = {'foo': '_exec_foo.value'}
        parsed, output_variable = EvalFunction._get_eval_string_(expr, map_vars)
        self.assertEqual(output_variable, 'es')
        self.assertEqual(parsed, 'np.power(_exec_foo.value, 4)')

    def test_get_eval_string_two_variables(self):
        map_vars = {'tasmax': 'tasmax.value', 'tas': 'tas.value'}
        expr = 'foo=log(1000*(tasmax-tas))/3'
        to_eval, out_variable_name = EvalFunction._get_eval_string_(expr, map_vars)
        self.assertEqual(to_eval, 'np.log(1000*(tasmax.value-tas.value))/3')
        self.assertEqual(out_variable_name, 'foo')

    def test_is_multivariate(self):
        expr = 'tas2=tas+2'
        self.assertFalse(EvalFunction.is_multivariate(expr))

        expr = 'tas2=log(tas+2)'
        self.assertFalse(EvalFunction.is_multivariate(expr))

        expr = 'tas4=tas+exp(tasmax)'
        self.assertTrue(EvalFunction.is_multivariate(expr))
