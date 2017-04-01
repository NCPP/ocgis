import numpy as np

import ocgis
from ocgis import OcgOperations
from ocgis import env
from ocgis.calc.library.math import NaturalLogarithm, Divide, Sum, Convolve1D
from ocgis.calc.library.thresholds import Threshold
from ocgis.collection.field import OcgField
from ocgis.exc import SampleSizeNotImplemented
from ocgis.ops.parms.definition import Calc
from ocgis.spatial.grid import GridXY
from ocgis.test.base import attr, AbstractTestField
from ocgis.variable.base import Variable
from ocgis.variable.temporal import TemporalVariable


class TestNaturalLogarithm(AbstractTestField):
    def test_execute(self):
        field = self.get_field(with_value=True, month_count=2)
        ln = NaturalLogarithm(field=field)
        ret = ln.execute()
        self.assertEqual(ret['ln'].get_value().shape, (2, 60, 2, 3, 4))
        self.assertNumpyAllClose(ret['ln'].get_value(), np.log(field['tmax'].get_value()))

        ln = NaturalLogarithm(field=field, calc_sample_size=True)
        ret = ln.execute()
        self.assertNotIn('n_ln', list(ret.keys()))

    def test_execute_no_units(self):
        """Test there are no units associated with a natual log."""

        field = self.get_field(with_value=True, month_count=2)
        ln = NaturalLogarithm(field=field, alias='ln')
        dvc = ln.execute()
        self.assertEqual(dvc['ln'].units, None)

    def test_execute_spatial_aggregation(self):
        field = self.get_field(with_value=True, month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        ln = NaturalLogarithm(field=field, tgd=tgd, calc_sample_size=True, spatial_aggregation=True)
        ret = ln.execute()
        self.assertEqual(ret['ln'].get_value().shape, (2, 2, 2, 3, 4))
        self.assertEqual(ret['n_ln'].get_value().mean(), 30.0)

    def test_execute_temporally_grouped(self):
        field = self.get_field(with_value=True, month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        ln = NaturalLogarithm(field=field, tgd=tgd)
        ret = ln.execute()
        self.assertEqual(ret['ln'].get_value().shape, (2, 2, 2, 3, 4))

        to_test = np.log(field['tmax'].get_value())
        to_test = np.ma.mean(to_test[0, tgd.dgroups[0], 0, :, :], axis=0)
        to_test2 = ret['ln'].get_value()[0, 0, 0, :, :]
        self.assertNumpyAllClose(to_test, to_test2)

        ln = NaturalLogarithm(field=field, tgd=tgd, calc_sample_size=True)
        ret = ln.execute()
        self.assertEqual(ret['ln'].get_value().shape, (2, 2, 2, 3, 4))
        self.assertEqual(ret['n_ln'].get_value().mean(), 30.0)


class TestDivide(AbstractTestField):
    def test_execute(self):
        field = self.get_field(with_value=True, month_count=2)
        field.add_variable(
            Variable(value=field['tmax'].get_value() + 5, name='tmin', dimensions=field['tmax'].dimensions))
        dv = Divide(field=field, parms={'arr1': 'tmax', 'arr2': 'tmin'})
        ret = dv.execute()
        self.assertNumpyAllClose(ret['divide'].get_value(), field['tmax'].get_value() / field['tmin'].get_value())

        with self.assertRaises(SampleSizeNotImplemented):
            Divide(field=field, parms={'arr1': 'tmax', 'arr2': 'tmin'}, calc_sample_size=True)

    def test_execute_temporal_grouping(self):
        field = self.get_field(with_value=True, month_count=2)
        field.add_variable(
            Variable(value=field['tmax'].get_value() + 5, name='tmin', dimensions=field['tmax'].dimensions))
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        dv = Divide(field=field, parms={'arr1': 'tmax', 'arr2': 'tmin'}, tgd=tgd)
        ret = dv.execute()
        self.assertEqual(ret['divide'].get_value().shape, (2, 2, 2, 3, 4))
        self.assertNumpyAllClose(ret['divide'].masked_value[0, 1, 1, :, 2],
                                 np.ma.array([0.0833001563436, 0.0940192653632, 0.0916398919876],
                                             mask=False, fill_value=1e20))


class TestThreshold(AbstractTestField):
    def test_execute(self):
        field = self.get_field(with_value=True, month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        dv = Threshold(field=field, parms={'threshold': 0.5, 'operation': 'gte'}, tgd=tgd)
        ret = dv.execute()
        self.assertEqual(ret['threshold'].get_value().shape, (2, 2, 2, 3, 4))
        self.assertNumpyAllClose(ret['threshold'].masked_value[1, 1, 1, 0, :],
                                 np.ma.array([13, 16, 15, 12], mask=False))


class TestSum(AbstractTestField):
    def test_system_registry(self):
        """Test sum function is appropriately registered."""

        c = Calc([{'func': 'sum', 'name': 'sum'}])
        self.assertEqual(c.value[0]['ref'], Sum)

    def test_system_through_operations(self):
        """Test calculation through operations."""

        row = Variable(name='y', value=[1, 2, 3, 4], dimensions='y')
        col = Variable(name='x', value=[10, 11, 12], dimensions='x')
        grid = GridXY(col, row)
        time = TemporalVariable(name='time', value=[1, 2], dimensions='time')
        data = Variable(name='data', dimensions=[time.dimensions[0]] + list(grid.dimensions))
        data.get_value()[0, :] = 1
        data.get_value()[1, :] = 2
        field = OcgField(grid=grid, time=time, is_data=data)

        calc = [{'func': 'sum', 'name': 'sum'}]
        ops = OcgOperations(dataset=field, calc=calc, calc_grouping='day', calc_raw=True, aggregate=True)
        ret = ops.execute()
        actual = ret.get_element(variable_name='sum').get_masked_value().flatten()
        self.assertNumpyAll(actual, np.ma.array([12.0, 24.0]))

    def test_calculate(self):
        """Test calculate for the sum function."""

        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        sum = Sum(field=field, tgd=tgd)
        np.random.seed(1)
        values = np.random.rand(2, 2, 2)
        values = np.ma.array(values, mask=False)
        to_test = sum.calculate(values)
        self.assertNumpyAll(to_test, np.ma.sum(values, axis=0))


class TestConvolve1D(AbstractTestField):
    def get_convolve1d_field(self, slice_stop=3):
        field = self.get_field(month_count=1, with_value=True)
        # field = field[:, 0:slice_stop, :, :, :]
        field = field.get_field_slice({'time': slice(0, slice_stop)})
        field['tmax'].get_value()[:] = 1
        mask = field['tmax'].get_mask(create=True)
        # field['tmax'].value.mask[:, :, :, 1, 1] = True
        mask[:, :, :, 1, 1] = True
        field['tmax'].set_mask(mask)
        return field

    def test_system_registry(self):
        Calc([{'func': 'convolve_1d', 'name': 'convolve'}])

    def test_execute_same(self):
        """Test convolution with the 'same' mode (numpy default)."""

        field = self.get_convolve1d_field()
        parms = {'v': np.array([1, 1, 1])}
        cd = Convolve1D(field=field, parms=parms)
        self.assertDictEqual(cd._format_parms_(parms), parms)
        vc = cd.execute()
        self.assertNumpyAll(vc['convolve_1d'].get_mask(), field['tmax'].get_mask())
        self.assertEqual(vc['convolve_1d'].masked_value.fill_value, field['tmax'].masked_value.fill_value)
        actual = np.ma.array([[[[[2.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
                                [[2.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]],
                               [[[3.0, 3.0, 3.0, 3.0], [3.0, 0.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]],
                                [[3.0, 3.0, 3.0, 3.0], [3.0, 0.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]]],
                               [[[2.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
                                [[2.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]]], [
                                  [[[2.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
                                   [[2.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]],
                                  [[[3.0, 3.0, 3.0, 3.0], [3.0, 0.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]],
                                   [[3.0, 3.0, 3.0, 3.0], [3.0, 0.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]]],
                                  [[[2.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
                                   [[2.0, 2.0, 2.0, 2.0], [2.0, 0.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]]]]],
                             dtype=env.NP_FLOAT)
        self.assertAlmostEqual(actual.sum(), vc['convolve_1d'].get_value().sum())

    def test_execute_valid(self):
        """Test convolution with the 'valid' mode."""

        field = self.get_convolve1d_field(slice_stop=4)
        parms = {'v': np.array([1, 1, 1]), 'mode': 'valid'}
        cd = Convolve1D(field=field, parms=parms)
        self.assertDictEqual(cd._format_parms_(parms), parms)
        vc = cd.execute()
        self.assertEqual(264.0, vc['convolve_1d'].get_value().sum())
        # Non-valid regions are entirely masked with 'valid' mode.
        self.assertTrue(vc['convolve_1d'].get_mask()[:, -2:, :, :, :].all())

    @attr('data')
    def test_execute_valid_through_operations(self):
        """Test executing a "valid" convolution mode through operations ensuring the data is appropriately truncated."""

        rd = self.test_data.get_rd('cancm4_tas')
        calc = [{'func': 'convolve_1d', 'name': 'convolve', 'kwds': {'v': np.array([1, 1, 1, 1, 1]), 'mode': 'valid'}}]
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, slice=[None, [0, 365], None, [0, 10], [0, 10]])
        ret = ops.execute()
        actual = ret.get_element().data_variables[0].shape
        self.assertEqual(actual, (365, 10, 10))
        actual = ret.get_element(variable_name='convolve').get_masked_value().mean()
        self.assertAlmostEqual(actual, 1200.4075346260388)
