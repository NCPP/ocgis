import itertools
import datetime

import netCDF4 as nc
import numpy as np

from ocgis.api.operations import OcgOperations
from ocgis.api.request.base import RequestDataset
from ocgis.test.base import TestBase, nc_scope, attr
from ocgis.calc.library.index.dynamic_kernel_percentile import DynamicDailyKernelPercentileThreshold


class TestDynamicDailyKernelPercentileThreshold(TestBase):
    def get_percentile_reference(self):
        years = [2001, 2002, 2003]
        days = [3, 4, 5, 6, 7]

        dates = []
        for year, day in itertools.product(years, days):
            dates.append(datetime.datetime(year, 6, day, 12))

        ds = nc.Dataset(self.test_data.get_uri('cancm4_tas'))
        try:
            calendar = ds.variables['time'].calendar
            units = ds.variables['time'].units
            ncdates = nc.num2date(ds.variables['time'][:], units, calendar=calendar)
            indices = []
            for ii, ndate in enumerate(ncdates):
                if ndate in dates:
                    indices.append(ii)
            tas = ds.variables['tas'][indices, :, :]
            ret = np.percentile(tas, 10, axis=0)
        finally:
            ds.close()

        return ret

    def test_constructor(self):
        DynamicDailyKernelPercentileThreshold()

    def test_calculate(self):
        # daily data for three years is wanted for the test. subset a CMIP5 decadal simulation to use for input into the 
        # computation.
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field = field.get_between('temporal', datetime.datetime(2001, 1, 1), datetime.datetime(2003, 12, 31, 23, 59))
        # the calculation will be for months and years. set the temporal grouping.
        temporal_group = field.temporal.get_grouping(['month', 'year'])
        # create calculation object
        percentile = 10
        width = 5
        operation = 'lt'
        kwds = dict(percentile=percentile, width=width, operation=operation)
        value = field.variables['tas'].value
        dkp = DynamicDailyKernelPercentileThreshold(tgd=temporal_group,
                                                    parms=kwds, field=field, alias='tg10p')
        dperc = dkp.get_daily_percentile(value, field.temporal.value_datetime, percentile, width)
        to_test = dperc[6, 5]
        ref = self.get_percentile_reference()
        self.assertNumpyAll(to_test, ref)

        ret = dkp.execute()
        self.assertEqual(ret['tg10p'].value.shape, (1, 36, 1, 64, 128))
        self.assertAlmostEqual(ret['tg10p'].value.mean(), 3.6267225477430554)

    @attr('slow')
    def test_operations(self):
        uri = self.test_data.get_uri('cancm4_tas')
        rd = RequestDataset(uri=uri, variable='tas')
        calc_grouping = ['month', 'year']
        calc = [{'func': 'dynamic_kernel_percentile_threshold', 'name': 'tg10p',
                 'kwds': {'percentile': 10, 'width': 5, 'operation': 'lt'}}]
        ops = OcgOperations(dataset=rd, calc_grouping=calc_grouping, calc=calc, output_format='nc')
        ret = ops.execute()

        with nc_scope(ret) as ds:
            ref = ds.variables['tg10p'][:]
            self.assertAlmostEqual(ref.mean(), 2.9778006)

    @attr('slow')
    def test_operations_two_steps(self):
        # get the request dataset to use as the basis for the percentiles
        uri = self.test_data.get_uri('cancm4_tas')
        variable = 'tas'
        rd = RequestDataset(uri=uri, variable=variable)
        # this is the underly OCGIS dataset object
        nc_basis = rd.get()

        # NOTE: if you want to subset the basis by time, this step is necessary
        # nc_basis = nc_basis.get_between('temporal',datetime.datetime(2001,1,1),datetime.datetime(2003,12,31,23,59))

        # these are the values to use when calculating the percentile basis. it may be good to wrap this in a function
        # to have memory freed after the percentile structure array is computed.
        all_values = nc_basis.variables[variable].value
        # these are the datetime objects used for window creation
        temporal = nc_basis.temporal.value_datetime
        # additional parameters for calculating the basis
        percentile = 10
        width = 5
        # get the structure array
        from ocgis.calc.library.index.dynamic_kernel_percentile import DynamicDailyKernelPercentileThreshold

        daily_percentile = DynamicDailyKernelPercentileThreshold.get_daily_percentile(all_values, temporal, percentile,
                                                                                      width)

        # perform the calculation using the precomputed basis. in this case, the basis and target datasets are the same,
        # so the RequestDataset is reused.
        calc_grouping = ['month', 'year']
        kwds = {'percentile': percentile, 'width': width, 'operation': 'lt', 'daily_percentile': daily_percentile}
        calc = [{'func': 'dynamic_kernel_percentile_threshold', 'name': 'tg10p', 'kwds': kwds}]
        ops = OcgOperations(dataset=rd, calc_grouping=calc_grouping, calc=calc,
                            output_format='nc')
        ret = ops.execute()

        # if we want to return the values as a three-dimenional numpy array the method below will do this. note the
        # interface arrangement for the next release will alter this slightly.
        ops = OcgOperations(dataset=rd, calc_grouping=calc_grouping, calc=calc,
                            output_format='numpy')
        arrs = ops.execute()
        # reference the returned numpy data. the first key is the geometry identifier. 1 in this case as this is the
        # default for no selection geometry. the second key is the request dataset alias and the third is the
        # calculation name. the variable name is appended to the end of the calculation to maintain a unique identifier.
        tg10p = arrs[1]['tas'].variables['tg10p'].value
        # if we want the date information for the temporal groups date attributes
        date_parts = arrs[1]['tas'].temporal.date_parts
        assert (date_parts.shape[0] == tg10p.shape[1])
        # these are the representative datetime objects
        # rep_dt = arrs[1]['tas'].temporal.value_datetime
        # and these are the lower and upper time bounds on the date groups
        # bin_bounds = arrs[1]['tas'].temporal.bounds_datetime

        # confirm we have values for each month and year (12*10)
        ret_ds = nc.Dataset(ret)
        try:
            self.assertEqual(ret_ds.variables['tg10p'].shape, (120, 64, 128))
        finally:
            ret_ds.close()
