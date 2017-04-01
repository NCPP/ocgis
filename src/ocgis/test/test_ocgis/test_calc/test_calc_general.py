import datetime
import itertools
from datetime import datetime as dt

import numpy as np

import ocgis
from ocgis import constants
from ocgis.ops.core import OcgOperations
from ocgis.test.base import TestBase
from ocgis.test.base import attr


class AbstractCalcBase(TestBase):
    @staticmethod
    def get_reshaped(arr):
        ret = arr.reshape(arr.shape[0], 1, 1)
        ret = np.ma.array(ret, mask=False)
        assert len(ret.shape) == 3
        return ret

    def run_standard_operations(self, calc, capture=False, output_format=None):
        _aggregate = [False, True]
        _calc_grouping = [['month'], ['month', 'year'], 'all']
        _output_format = output_format or [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_CSV_SHAPEFILE,
                                           constants.OUTPUT_FORMAT_NETCDF]
        captured = []
        for ii, tup in enumerate(itertools.product(_aggregate, _calc_grouping, _output_format)):
            aggregate, calc_grouping, output_format = tup
            if aggregate is True and output_format == constants.OUTPUT_FORMAT_NETCDF:
                continue
            rd = self.test_data.get_rd('cancm4_tas', kwds={'time_region': {'year': [2001, 2002]}})
            try:
                ops = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[25], calc=calc,
                                    calc_grouping=calc_grouping, output_format=output_format, aggregate=aggregate,
                                    prefix=('standard_ops_' + str(ii)))
                ret = ops.execute()
                if output_format == constants.OUTPUT_FORMAT_NUMPY:
                    refv = ret.get_element(variable_name=calc[0]['name'], container_ugid=25)
                    ref = refv.get_value()
                    if aggregate:
                        space_shape = [1]
                    else:
                        space_shape = [4, 4]
                    if calc_grouping == ['month']:
                        shp1 = [12]
                    elif calc_grouping == 'all':
                        raise NotImplementedError('calc_grouping all')
                    else:
                        shp1 = [24]
                    test_shape = shp1 + space_shape
                    self.assertEqual(ref.shape, tuple(test_shape))
                    if not aggregate:
                        # Ensure the geometry mask is appropriately updated by the function.
                        self.assertTrue(refv.get_mask()[0, 0, 0])
            except ValueError:
                raise
            except AssertionError:
                raise
            except Exception as e:
                if capture:
                    parms = dict(aggregate=aggregate, calc_grouping=calc_grouping, output_format=output_format)
                    captured.append({'exception': e, 'parms': parms})
                else:
                    raise
        return captured


class Test(AbstractCalcBase):
    @attr('data')
    def test_system_date_groups_all(self):
        calc = [{'func': 'mean', 'name': 'mean'}]
        rd = self.test_data.get_rd('cancm4_tasmax_2011')
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping='all', geom='state_boundaries', select_ugid=[25])
        ret_calc = ops.execute()

        ops = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[25])
        ret_no_calc = ops.execute()

        field = ret_calc.get_element(container_ugid=25, field_name='tasmax')
        variable = field['mean']
        parents = ret_no_calc.get_element(container_ugid=25, field_name='tasmax')
        self.assertEqual(parents['tasmax'].shape, (3650, 4, 4))
        self.assertEqual(variable.shape, (1, 4, 4))
        desired_value = parents['tasmax'].get_masked_value()
        lhs = np.ma.mean(desired_value, axis=0).reshape(1, 4, 4).astype(desired_value.dtype)
        # NumPy does not update the fill value type in "astype". Set this manually.
        lhs.fill_value = desired_value.fill_value
        rhs = variable.get_masked_value()
        self.assertNumpyAll(lhs, rhs)

    @attr('data')
    def test_system_time_region(self):
        kwds = {'time_region': {'year': [2011]}}
        rd = self.test_data.get_rd('cancm4_tasmax_2011', kwds=kwds)
        calc = [{'func': 'mean', 'name': 'mean'}]
        calc_grouping = ['year', 'month']

        ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping,
                                  geom='state_boundaries', select_ugid=[25])
        ret = ops.execute()

        tgroup = ret.get_element(container_ugid=25, field_name='tasmax').time.date_parts
        self.assertEqual(set([2011]), set(tgroup['year']))
        self.assertEqual(tgroup['month'][-1], 12)

        kwds = {'time_region': {'year': [2011, 2013], 'month': [8]}}
        rd = self.test_data.get_rd('cancm4_tasmax_2011', kwds=kwds)
        calc = [{'func': 'threshold', 'name': 'threshold', 'kwds': {'threshold': 0.0, 'operation': 'gte'}}]
        calc_grouping = ['month']
        aggregate = True
        calc_raw = True
        geom = 'us_counties'
        select_ugid = [2762]

        ops = ocgis.OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, aggregate=aggregate,
                                  calc_raw=calc_raw, geom=geom, select_ugid=select_ugid, output_format='numpy')
        ret = ops.execute()
        threshold = ret.get_element(container_ugid=2762, field_name='tasmax', variable_name='threshold').get_value()
        self.assertEqual(threshold.flatten()[0], 62)

    @attr('data')
    def test_system_computational_nc_output(self):
        """Test writing a computation to netCDF."""

        rd = self.test_data.get_rd('cancm4_tasmax_2011',
                                   kwds={'time_range': [datetime.datetime(2011, 1, 1),
                                                        datetime.datetime(2011, 12, 31)]})
        calc = [{'func': 'mean', 'name': 'tasmax_mean'}]
        calc_grouping = ['month', 'year']

        ops = ocgis.OcgOperations(rd, calc=calc, calc_grouping=calc_grouping, output_format='nc')
        ret = ops.execute()

        with self.nc_scope(ret) as ds:
            ref = ds.variables['time']
            self.assertEqual(ref.climatology, 'climatology_bounds')
            self.assertEqual(len(ref[:]), 12)
            ref = ds.variables['climatology_bounds']
            self.assertEqual(ref[:].shape[0], 12)

        ops = ocgis.OcgOperations(dataset={'uri': ret, 'variable': calc[0]['name']},
                                  output_format='nc', prefix='subset_climatology')
        ret = ops.execute()

        with self.nc_scope(ret) as ds:
            ref = ds.variables['time'][:]
            self.assertEqual(len(ref), 12)
            self.assertEqual(set(ds.variables['tasmax_mean'].ncattrs()),
                             {'units', 'long_name', 'standard_name', 'grid_mapping'})

    @attr('data')
    def test_system_date_groups(self):
        calc = [{'func': 'mean', 'name': 'mean'}]
        rd = self.test_data.get_rd('cancm4_tasmax_2011')

        calc_grouping = ['month']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret.get_element(container_ugid=25).time
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == np.array([dt(2011, month, 16) for month in range(1, 13)])))

        calc_grouping = ['year']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret.get_element(container_ugid=25).time
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == [dt(year, 7, 1) for year in range(2011, 2021)]))

        calc_grouping = ['month', 'year']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret.get_element(container_ugid=25).time
        rdt = ref.value_datetime
        self.assertTrue(
            np.all(rdt == [dt(year, month, 16) for year, month in
                           itertools.product(list(range(2011, 2021)), list(range(1, 13)))]))

        calc_grouping = ['day']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret.get_element(container_ugid=25).time
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == [dt(2011, 1, day, 12) for day in range(1, 32)]))

        calc_grouping = ['month', 'day']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret.get_element(container_ugid=25).time
        rdt = ref.value_datetime
        self.assertEqual(rdt[0], dt(2011, 1, 1, 12))
        self.assertEqual(rdt[12], dt(2011, 1, 13, 12))

        calc_grouping = ['year', 'day']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret.get_element(container_ugid=25).time
        rdt = ref.value_datetime
        self.assertEqual(rdt[0], dt(2011, constants.CALC_YEAR_CENTROID_MONTH, 1, 12))

        rd = self.test_data.get_rd('cancm4_tasmax_2011', kwds={'time_region': {'month': [1], 'year': [2011]}})
        field = rd.get()
        calc_grouping = ['month', 'day', 'year']
        ops = OcgOperations(dataset=rd, calc=calc, calc_grouping=calc_grouping, geom='state_boundaries',
                            select_ugid=[25])
        ret = ops.execute()
        ref = ret.get_element(container_ugid=25).time
        rdt = ref.value_datetime
        self.assertTrue(np.all(rdt == ref.value_datetime))
        self.assertTrue(np.all(ref.bounds.value_datetime == field.time.bounds.value_datetime))
