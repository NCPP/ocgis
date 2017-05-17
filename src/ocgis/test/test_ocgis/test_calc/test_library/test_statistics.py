import numpy as np

import ocgis
from ocgis.calc.library.statistics import Mean, FrequencyPercentile, MovingWindow, DailyPercentile
from ocgis.collection.field import Field
from ocgis.constants import OutputFormatName
from ocgis.exc import DefinitionValidationError
from ocgis.ops.parms.definition import Calc
from ocgis.test.base import attr, AbstractTestField
from ocgis.test.base import nc_scope
from ocgis.util.itester import itr_products_keywords
from ocgis.util.large_array import compute
from ocgis.util.units import get_units_object
from ocgis.variable.base import Variable


class TestDailyPercentile(AbstractTestField):
    @attr('data', 'slow')
    def test_system_compute(self):
        rd = self.test_data.get_rd('cancm4_tas')
        kwds = {'percentile': 90, 'window_width': 5}
        calc = [{'func': 'daily_perc', 'name': 'dp', 'kwds': kwds}]
        ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23], calc=calc,
                                  output_format='nc', time_region={'year': [2002, 2003]})
        ret = compute(ops, 2, verbose=False)

        rd = ocgis.RequestDataset(uri=ret)
        actual_field = rd.get()
        self.assertEqual(actual_field.data_variables[0].shape, (365, 3, 3))

    @attr('data', 'slow')
    def test_system_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        kwds = {'percentile': 90, 'window_width': 5}
        calc = [{'func': 'daily_perc', 'name': 'dp', 'kwds': kwds}]
        for output_format in [OutputFormatName.OCGIS, 'nc']:
            ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23], calc=calc,
                                      output_format=output_format, time_region={'year': [2002, 2003]})
            self.assertIsNone(ops.calc_grouping)
            ret = ops.execute()
            if output_format == OutputFormatName.OCGIS:
                actual = ret.get_element(container_ugid=23, variable_name='dp').get_mask().sum()
                self.assertEqual(actual, 730)

    def test_execute(self):
        field = self.get_field(with_value=True, month_count=2)
        field = field.get_field_slice({'realization': 0, 'level': 0})
        parms = {'percentile': 90, 'window_width': 5}
        dp = DailyPercentile(field=field, parms=parms)
        vc = dp.execute()

        self.assertAlmostEqual(vc['daily_perc'].get_value().mean(), 0.76756388346354165)

    @attr('data')
    def test_get_daily_percentile_from_request_dataset(self):
        rd = self.test_data.get_rd('cancm4_tas')
        kwds = {'percentile': 90, 'window_width': 5}
        calc = [{'func': 'daily_perc', 'name': 'dp', 'kwds': kwds}]
        ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23], calc=calc,
                                  output_format='nc', time_region={'year': [2002, 2003]})
        ret = ops.execute()

        new_rd = ocgis.RequestDataset(ret)
        for alias in [None, 'dp']:
            dp = DailyPercentile.get_daily_percentile_from_request_dataset(new_rd, alias=alias)
            self.assertEqual(len(list(dp.keys())), 365)
            self.assertAlmostEqual(dp[(4, 15)].mean(), 281.68076869419644)


class TestMovingWindow(AbstractTestField):
    def test_calculate(self):
        ma = MovingWindow()
        np.random.seed(1)
        values = np.ma.array(np.random.rand(10), dtype=float).reshape(1, -1, 1, 1, 1)
        k = 5
        ret = ma.calculate(values, k=k, mode='same', operation='mean')

        self.assertEqual(ret.shape, values.shape)
        desired = [
            [[[[0.37915362432069233]]], [[[0.3599483613984792]]], [[[0.317309867282206]]], [[[0.2523731852954507]]],
             [[[0.14556032888255327]]], [[[0.21464959932769387]]], [[[0.23353657964745986]]], [[[0.31194874828470864]]],
             [[[0.3668512866636864]]], [[[0.4270483117590249]]]]]
        desired = np.array(desired)
        self.assertNumpyAllClose(ret, desired)
        ret = ret.squeeze()
        values = values.squeeze()
        self.assertEqual(ret[4], np.mean(values[2:7]))

    def test_execute(self):
        field = self.get_field(month_count=1, with_value=True)
        field = field.get_field_slice({'time': slice(0, 4)})
        field['tmax'].get_value()[:] = 1

        mask = field['tmax'].get_mask(create=True)
        mask[:, :, :, 1, 1] = True
        field['tmax'].set_mask(mask)

        for mode in ['same', 'valid']:
            for operation in ('mean', 'min', 'max', 'median', 'var', 'std'):
                parms = {'k': 3, 'mode': mode, 'operation': operation}
                ma = MovingWindow(field=field, parms=parms)
                vc = ma.execute()
                if mode == 'same':
                    self.assertEqual(vc['moving_window'].get_value().shape, field['tmax'].shape)
                else:
                    actual_mask = vc['moving_window'].get_mask()
                    self.assertTrue(np.all(actual_mask[:, 0, :, :, :]))
                    self.assertTrue(np.all(actual_mask[:, -1, :, :, :]))
                    self.assertEqual(vc['moving_window'].get_value().shape, (2, 4, 2, 3, 4))

    @attr('data')
    def test_execute_valid_through_operations(self):
        """Test executing a "valid" convolution mode through operations ensuring the data is appropriately truncated."""

        rd = self.test_data.get_rd('cancm4_tas')
        calc = [{'func': 'moving_window', 'name': 'ma', 'kwds': {'k': 5, 'mode': 'valid', 'operation': 'mean'}}]
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, slice=[None, [0, 365], None, [0, 10], [0, 10]])
        ret = ops.execute()
        actual = ret.get_element().data_variables[0].shape
        self.assertEqual(actual, (365, 10, 10))
        actual = ret.get_element()['ma'].get_masked_value().mean()
        self.assertAlmostEqual(actual, 240.08149584487535)

    def test_registry(self):
        Calc([{'func': 'moving_window', 'name': 'ma'}])

    def test_iter_kernel_values_same(self):
        """Test returning kernel values with the 'same' mode."""

        values = np.arange(2, 11).reshape(-1, 1, 1)
        k = 5
        mode = 'same'
        itr = MovingWindow._iter_kernel_values_(values, k, mode=mode)
        to_test = list(itr)

        desired = [[0, [[[2]], [[3]], [[4]]]], [1, [[[2]], [[3]], [[4]], [[5]]]],
                   [2, [[[2]], [[3]], [[4]], [[5]], [[6]]]], [3, [[[3]], [[4]], [[5]], [[6]], [[7]]]],
                   [4, [[[4]], [[5]], [[6]], [[7]], [[8]]]], [5, [[[5]], [[6]], [[7]], [[8]], [[9]]]],
                   [6, [[[6]], [[7]], [[8]], [[9]], [[10]]]], [7, [[[7]], [[8]], [[9]], [[10]]]],
                   [8, [[[8]], [[9]], [[10]]]]]
        for idx in range(len(to_test)):
            self.assertEqual(to_test[idx][1].ndim, 3)
            self.assertEqual(to_test[idx][0], desired[idx][0])
            self.assertEqual(to_test[idx][1].tolist(), desired[idx][1])

    def test_iter_kernel_values_valid(self):
        """Test returning kernel values with the 'valid' mode."""

        values = np.arange(2, 11).reshape(-1, 1, 1)
        k = 5
        mode = 'valid'
        itr = MovingWindow._iter_kernel_values_(values, k, mode=mode)
        to_test = list(itr)

        desired = [[2, [[[2]], [[3]], [[4]], [[5]], [[6]]]], [3, [[[3]], [[4]], [[5]], [[6]], [[7]]]],
                   [4, [[[4]], [[5]], [[6]], [[7]], [[8]]]], [5, [[[5]], [[6]], [[7]], [[8]], [[9]]]],
                   [6, [[[6]], [[7]], [[8]], [[9]], [[10]]]]]

        for idx in range(len(to_test)):
            self.assertEqual(to_test[idx][1].ndim, 3)
            self.assertEqual(to_test[idx][0], desired[idx][0])
            self.assertEqual(to_test[idx][1].tolist(), desired[idx][1])

    def test_iter_kernel_values_asserts(self):
        """Test assert statements."""

        k = [1, 2, 3, 4]
        values = [np.array([[2, 3], [4, 5]]), np.arange(0, 13).reshape(-1, 1, 1)]
        mode = ['same', 'valid', 'foo']
        for kwds in itr_products_keywords({'k': k, 'values': values, 'mode': mode}, as_namedtuple=True):
            try:
                list(MovingWindow._iter_kernel_values_(kwds.values, kwds.k))
            except AssertionError:
                if kwds.k == 3:
                    if kwds.values.shape == (2, 2):
                        continue
                    else:
                        raise
                else:
                    continue
            except NotImplementedError:
                if kwds.mode == 'foo':
                    continue
                else:
                    raise

    @attr('data')
    def test_validate(self):
        rd = self.test_data.get_rd('cancm4_tas')
        calc = [{'func': 'moving_window', 'name': 'TGx5day', 'kwds': {'k': 5, 'operation': 'max', 'mode': 'same'}}]
        with self.assertRaises(DefinitionValidationError):
            ocgis.OcgOperations(dataset=rd, calc_grouping=['year'], calc=calc)


class TestFrequencyPercentile(AbstractTestField):
    def test_execute(self):
        field = self.get_field(with_value=True, month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        fp = FrequencyPercentile(field=field, tgd=tgd, parms={'percentile': 99})
        ret = fp.execute()
        self.assertNumpyAllClose(ret['freq_perc'].get_masked_value()[0, 1, 1, 0, :],
                                 np.ma.array(data=[0.92864656, 0.98615474, 0.95269281, 0.98542988],
                                             mask=False, fill_value=1e+20))


class TestMean(AbstractTestField):
    @attr('data')
    def test_system_file_only_through_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}], calc_grouping=['month'],
                                  geom='state_boundaries', select_ugid=[27], file_only=True, output_format='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            var = ds.variables['mean']
            # All data should be masked since this is file only.
            self.assertTrue(var[:].mask.all())

    @attr('data')
    def test_system_output_datatype(self):
        """Test the output data type is the same as the input data type of the variable."""

        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}], calc_grouping=['month'],
                                  geom='state_boundaries', select_ugid=[27])
        ret = ops.execute()
        with nc_scope(rd.uri) as ds:
            var_dtype = ds.variables['tas'].dtype
        actual = ret.get_element(variable_name='mean').dtype
        self.assertEqual(actual, var_dtype)

    def test_execute(self):
        field = self.get_field(with_value=True, month_count=2)
        self.assertIsInstance(field, Field)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field, tgd=tgd, alias='my_mean', dtype=np.float64)
        dvc = mu.execute()
        dv = dvc['my_mean']
        self.assertEqual(dv.name, 'my_mean')
        self.assertEqual(dv.get_value().shape, (2, 2, 2, 3, 4))
        self.assertNumpyAll(np.mean(field['tmax'].get_value()[1, tgd.dgroups[1], 0, :, :], axis=0),
                            dv.get_value()[1, 1, 0, :, :])

    @attr('data')
    def test_execute_file_only(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field = field.get_field_slice({'time': slice(10, 20), 'y': slice(20, 30), 'x': slice(40, 50)})
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        # Value should not be loaded at this point.
        self.assertEqual(field['tas']._value, None)
        mu = Mean(field=field, tgd=tgd, alias='my_mean', file_only=True)
        ret = mu.execute()
        # Value should still not be loaded.
        self.assertIsNone(field['tas']._value)
        # No value should be calculated for the calculation.
        self.assertIsNone(ret['my_mean']._value)

    def test_execute_sample_size(self):
        field = self.get_field(with_value=True, month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field, tgd=tgd, alias='my_mean', calc_sample_size=True, dtype=np.float64)
        dvc = mu.execute()
        dv = dvc['my_mean']
        self.assertEqual(dv.name, 'my_mean')
        self.assertEqual(dv.get_value().shape, (2, 2, 2, 3, 4))
        self.assertNumpyAll(np.mean(field['tmax'].get_value()[1, tgd.dgroups[1], 0, :, :], axis=0),
                            dv.get_value()[1, 1, 0, :, :])

        ret = dvc['n_my_mean']
        self.assertNumpyAll(ret.get_masked_value()[0, 0, 0],
                            np.ma.array(data=[[31, 31, 31, 31], [31, 31, 31, 31], [31, 31, 31, 31]],
                                        mask=[[False, False, False, False], [False, False, False, False],
                                              [False, False, False, False]],
                                        fill_value=999999,
                                        dtype=ret.dtype))

        mu = Mean(field=field, tgd=tgd, alias='my_mean', calc_sample_size=False)
        dvc = mu.execute()
        self.assertNotIn('n_my_mean', list(dvc.keys()))

    def test_execute_two_variables(self):
        """Test running a field with two variables through the mean calculation."""

        field = self.get_field(with_value=True, month_count=2)
        field.add_variable(
            Variable(value=field['tmax'].get_value() + 5, name='tmin', dimensions=field['tmax'].dimensions),
            is_data=True)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field, tgd=tgd, alias='my_mean', dtype=np.float64)
        ret = mu.execute()
        self.assertEqual(len(ret), 2)
        self.assertAlmostEqual(5.0,
                               abs(ret['my_mean_tmax'].get_value().mean() - ret['my_mean_tmin'].get_value().mean()))

    def test_execute_two_variables_sample_size(self):
        field = self.get_field(with_value=True, month_count=2)
        field.add_variable(
            Variable(value=field['tmax'].get_value() + 5, name='tmin', dimensions=field['tmax'].dimensions),
            is_data=True)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field, tgd=tgd, alias='my_mean', dtype=np.float64, calc_sample_size=True)
        ret = mu.execute()
        self.assertEqual(len(ret), 4)
        self.assertAlmostEqual(5.0,
                               abs(ret['my_mean_tmax'].get_value().mean() - ret['my_mean_tmin'].get_value().mean()))
        self.assertEqual({'my_mean_tmax', 'n_my_mean_tmax', 'my_mean_tmin', 'n_my_mean_tmin'},
                         set(ret.keys()))

    @attr('cfunits')
    def test_execute_units_are_maintained(self):
        field = self.get_field(with_value=True, month_count=2)
        units_kelvin = get_units_object('kelvin')
        self.assertEqual(field['tmax'].cfunits, units_kelvin)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field, tgd=tgd, alias='my_mean', calc_sample_size=False, dtype=np.float64)
        dvc = mu.execute()
        self.assertEqual(dvc['my_mean'].cfunits, units_kelvin)
