import datetime
import itertools
import os
from collections import deque, OrderedDict

import numpy as np
from netCDF4 import date2num, num2date, netcdftime

from ocgis import RequestDataset
from ocgis import constants
from ocgis.constants import HeaderName, KeywordArgument, DimensionMapKey
from ocgis.driver.dimension_map import DimensionMap
from ocgis.exc import CannotFormatTimeError, IncompleteSeasonError
from ocgis.ops.parms.definition import CalcGrouping
from ocgis.test.base import attr, AbstractTestInterface
from ocgis.test.test_simple.test_dependencies import create_mftime_nc_files
from ocgis.util.helpers import get_date_list
from ocgis.util.units import get_units_object, get_are_units_equal, get_are_units_equivalent
from ocgis.variable.temporal import get_datetime_conversion_state, get_datetime_from_months_time_units, \
    get_datetime_from_template_time_units, get_difference_in_months, get_is_interannual, get_num_from_months_time_units, \
    get_origin_datetime_from_months_units, get_sorted_seasons, TemporalVariable, iter_boolean_groups_from_time_regions, \
    TemporalGroupVariable, get_time_regions, get_datetime_or_netcdftime
from ocgis.variable.temporal import get_datetime_or_netcdftime as dt


class AbstractTestTemporal(AbstractTestInterface):
    @property
    def value_template_units(self):
        return np.array([19710101.9375, 19710102.9375, 19710103.9375, 19710104.9375, 19710105.9375])

    @property
    def value_template_units_no_decimal(self):
        return np.array([20000101., 20000102., 20000103., 20000104., 20000105., 20000106.])


class Test(AbstractTestTemporal):
    def test_get_datetime_conversion_state(self):
        archetypes = [45.5, datetime.datetime(2000, 1, 1), netcdftime.datetime(2000, 4, 5)]
        for archetype in archetypes:
            res = get_datetime_conversion_state(archetype)
            try:
                self.assertFalse(res)
            except AssertionError:
                self.assertEqual(type(archetype), float)

    def test_get_datetime_from_months_time_units(self):
        units = "months since 1978-12"
        vec = list(range(0, 36))
        datetimes = get_datetime_from_months_time_units(vec, units)
        test_datetimes = [datetime.datetime(1978, 12, 16, 0, 0), datetime.datetime(1979, 1, 16, 0, 0),
                          datetime.datetime(1979, 2, 16, 0, 0), datetime.datetime(1979, 3, 16, 0, 0),
                          datetime.datetime(1979, 4, 16, 0, 0), datetime.datetime(1979, 5, 16, 0, 0),
                          datetime.datetime(1979, 6, 16, 0, 0), datetime.datetime(1979, 7, 16, 0, 0),
                          datetime.datetime(1979, 8, 16, 0, 0), datetime.datetime(1979, 9, 16, 0, 0),
                          datetime.datetime(1979, 10, 16, 0, 0), datetime.datetime(1979, 11, 16, 0, 0),
                          datetime.datetime(1979, 12, 16, 0, 0), datetime.datetime(1980, 1, 16, 0, 0),
                          datetime.datetime(1980, 2, 16, 0, 0), datetime.datetime(1980, 3, 16, 0, 0),
                          datetime.datetime(1980, 4, 16, 0, 0), datetime.datetime(1980, 5, 16, 0, 0),
                          datetime.datetime(1980, 6, 16, 0, 0), datetime.datetime(1980, 7, 16, 0, 0),
                          datetime.datetime(1980, 8, 16, 0, 0), datetime.datetime(1980, 9, 16, 0, 0),
                          datetime.datetime(1980, 10, 16, 0, 0), datetime.datetime(1980, 11, 16, 0, 0),
                          datetime.datetime(1980, 12, 16, 0, 0), datetime.datetime(1981, 1, 16, 0, 0),
                          datetime.datetime(1981, 2, 16, 0, 0), datetime.datetime(1981, 3, 16, 0, 0),
                          datetime.datetime(1981, 4, 16, 0, 0), datetime.datetime(1981, 5, 16, 0, 0),
                          datetime.datetime(1981, 6, 16, 0, 0), datetime.datetime(1981, 7, 16, 0, 0),
                          datetime.datetime(1981, 8, 16, 0, 0), datetime.datetime(1981, 9, 16, 0, 0),
                          datetime.datetime(1981, 10, 16, 0, 0), datetime.datetime(1981, 11, 16, 0, 0)]
        self.assertNumpyAll(datetimes, np.array(test_datetimes))

    def test_get_datetime_from_template_time_units(self):
        ret = get_datetime_from_template_time_units(self.value_template_units)
        self.assertEqual(ret.shape, self.value_template_units.shape)
        self.assertEqual(ret[2], datetime.datetime(1971, 1, 3, 22, 30))

        ret = get_datetime_from_template_time_units(self.value_template_units_no_decimal)
        self.assertEqual(ret.shape, self.value_template_units_no_decimal.shape)
        self.assertEqual(ret[2], datetime.datetime(2000, 1, 3))

    def test_get_difference_in_months(self):
        distance = get_difference_in_months(datetime.datetime(1978, 12, 1), datetime.datetime(1979, 3, 1))
        self.assertEqual(distance, 3)
        distance = get_difference_in_months(datetime.datetime(1978, 12, 1), datetime.datetime(1978, 7, 1))
        self.assertEqual(distance, -5)
        distance = get_difference_in_months(datetime.datetime(1978, 12, 1), datetime.datetime(1978, 12, 1))
        self.assertEqual(distance, 0)

    def test_get_is_interannual(self):
        self.assertTrue(get_is_interannual([11, 12, 1]))
        self.assertFalse(get_is_interannual([10, 11, 12]))

    def test_get_num_from_months_time_units_1d_array(self):
        units = "months since 1978-12"
        vec = list(range(0, 36))
        datetimes = get_datetime_from_months_time_units(vec, units)
        num = get_num_from_months_time_units(datetimes, units, dtype=np.int32)
        self.assertNumpyAll(num, np.array(vec, dtype=np.int32))
        self.assertEqual(num.dtype, np.int32)

    def test_get_origin_datetime_from_months_units(self):
        units = "months since 1978-12"
        self.assertEqual(get_origin_datetime_from_months_units(units), datetime.datetime(1978, 12, 1))
        units = "months since 1979-1-1 0"
        self.assertEqual(get_origin_datetime_from_months_units(units), datetime.datetime(1979, 1, 1))

    def test_get_sorted_seasons(self):
        calc_grouping = [[9, 10, 11], [12, 1, 2], [6, 7, 8]]
        methods = ['max', 'min']

        for method in methods:
            for perm in itertools.permutations(calc_grouping, r=3):
                ret = get_sorted_seasons(perm, method=method)
                if method == 'max':
                    self.assertEqual(ret, [[6, 7, 8], [9, 10, 11], [12, 1, 2]])
                else:
                    self.assertEqual(ret, [[12, 1, 2], [6, 7, 8], [9, 10, 11]])

    def test_iter_boolean_groups_from_time_regions(self):
        time_regions = [[{'month': [12], 'year': [1900]}, {'month': [2, 1], 'year': [1901]}]]
        yield_subset = True
        raise_if_incomplete = False

        start = datetime.datetime(1900, 1, 1)
        end = datetime.datetime(1902, 12, 31)
        value = self.get_time_series(start, end)
        tvar = TemporalVariable(value=value, units=constants.DEFAULT_TEMPORAL_UNITS,
                                dimensions=constants.DimensionName.TEMPORAL)

        itr = iter_boolean_groups_from_time_regions(time_regions, tvar, yield_subset=yield_subset,
                                                    raise_if_incomplete=raise_if_incomplete)
        itr = list(itr)
        self.assertEqual(len(itr), 1)
        for dgroup, sub in itr:
            self.assertEqual(sub.get_value()[0].year, 1900)
            self.assertEqual(sub.get_value()[0].month, 12)


class TestTemporalVariable(AbstractTestTemporal):
    def get_temporalvariable(self, add_bounds=True, start=None, stop=None, days=1, name=None, format_time=True):
        dt = datetime.datetime
        # dt = get_datetime_or_netcdftime
        start = start or dt(1899, 1, 1, 12)
        stop = stop or dt(1901, 12, 31, 12)
        dates = get_date_list(start, stop, days=days)
        if add_bounds:
            delta = datetime.timedelta(hours=12)
            lower = np.array(dates) - delta
            upper = np.array(dates) + delta
            bounds = np.empty((lower.shape[0], 2), dtype=object)
            bounds[:, 0] = lower
            bounds[:, 1] = upper
            bounds = TemporalVariable(name='time_bounds', value=bounds, dimensions=['time', 'bounds'])
        else:
            bounds = None

        # Earlier versions of netcdftime do not support the delta operation.
        dt = get_datetime_or_netcdftime
        dates = np.array(dates)
        dates = dates.flatten()
        for idx in range(len(dates)):
            curr = dates[idx]
            dates[idx] = dt(curr.year, curr.month, curr.day, curr.hour)
        if add_bounds:
            bfill = bounds.get_value().flatten()
            for idx in range(len(bfill)):
                curr = bfill[idx]
                bfill[idx] = dt(curr.year, curr.month, curr.day, curr.hour)
            bfill = bfill.reshape(bounds.shape)
            bounds.set_value(bfill)

        td = TemporalVariable(value=dates, bounds=bounds, name=name, format_time=format_time, dimensions='time')
        return td

    @staticmethod
    def init_temporal_variable(**kwargs):
        # Provide a default time dimension.
        if KeywordArgument.DIMENSIONS not in kwargs:
            kwargs[KeywordArgument.DIMENSIONS] = constants.DimensionName.TEMPORAL
        return TemporalVariable(**kwargs)

    def get_template_units(self):
        units = 'day as %Y%m%d.%f'
        td = self.init_temporal_variable(value=self.value_template_units, units=units, calendar='proleptic_gregorian')
        return td

    def test_init(self):
        # td = TemporalVariable(value=[datetime.datetime(2000, 1, 1)])
        td = self.init_temporal_variable(value=[datetime.datetime(2000, 1, 1)])
        self.assertEqual(td.name, constants.DEFAULT_TEMPORAL_NAME)
        self.assertEqual(td.calendar, constants.DEFAULT_TEMPORAL_CALENDAR)
        self.assertEqual(td.units, constants.DEFAULT_TEMPORAL_UNITS)
        self.assertIsInstance(td, TemporalVariable)
        self.assertFalse(td._has_months_units)
        self.assertTrue(td.format_time)

        td = self.init_temporal_variable(value=[datetime.datetime(2000, 1, 1)], units="months since 1978-12")
        self.assertTrue(td._has_months_units)

        # Test with bounds.
        bounds = self.init_temporal_variable(name='time_bounds', dimensions=['time', 'bounds'], value=[[1, 2], [2, 3]])
        t = self.init_temporal_variable(name='time', dimensions=['time'], value=[1.5, 2.5], bounds=bounds)
        self.assertIsInstance(t.bounds, TemporalVariable)

    @attr('data')
    def test_init_data(self):
        rd = self.get_request_dataset()
        tv = TemporalVariable(name='time', request_dataset=rd)

        self.assertEqual(tv.calendar, '365_day')
        self.assertEqual(tv.units, 'days since 1850-1-1')

        tv = TemporalVariable(name='time', request_dataset=rd, calendar='standard', units='days since 1990-1-1')
        self.assertEqual(tv.calendar, 'standard')
        self.assertEqual(tv.units, 'days since 1990-1-1')

    def test_as_record(self):
        keywords = {KeywordArgument.BOUNDS_NAMES: [None, HeaderName.TEMPORAL_BOUNDS]}

        for k in self.iter_product_keywords(keywords):
            tv = TemporalVariable(value=[1, 2, 3], dtype=float, name='time', dimensions='time')
            tv.set_extrapolated_bounds('time_bounds', 'bounds')

            self.assertIsInstance(tv.bounds, TemporalVariable)

            sub = tv[0]
            kwds = {KeywordArgument.BOUNDS_NAMES: k.bounds_names}
            record = sub.as_record(**kwds)
            for v in list(record.values()):
                self.assertNotIsInstance(v, float)
            self.assertEqual(len(record), 7)

            if k.bounds_names is not None:
                for n in HeaderName.TEMPORAL_BOUNDS:
                    self.assertIn(n, record)

    def test_getitem(self):
        td = self.get_temporalvariable(add_bounds=True)
        self.assertIsNotNone(td.value_datetime)
        self.assertIsNotNone(td.value_numtime)
        self.assertIsNotNone(td.bounds.value_datetime)
        self.assertIsNotNone(td.bounds.value_numtime)
        sub = td[2]
        self.assertEqual(sub.parent.shapes, OrderedDict([('time', (1,)), ('time_bounds', (1, 2))]))
        for target, subtarget in itertools.product([sub, sub.bounds], ['value_datetime', 'value_numtime']):
            real_target = getattr(target, subtarget)
            self.assertEqual(real_target.shape[0], 1)

        # Test with a boolean array.
        var = self.init_temporal_variable(value=[1, 2, 3, 4, 5])
        sub = var[np.array([False, True, True, True, False])]
        self.assertEqual(sub.shape, (3,))

    def test_system_360_day_calendar(self):
        months = list(range(1, 13))
        days = list(range(1, 31))
        vec = []
        for month in months:
            for day in days:
                vec.append(netcdftime.datetime(2000, month, day))
        num = date2num(vec, 'days since 1900-01-01', calendar='360_day')
        td = self.init_temporal_variable(value=num, calendar='360_day', units='days since 1900-01-01')
        self.assertNumpyAll(np.ma.array(vec), td.value_datetime)

    def test_system_bounds_datetime_and_bounds_numtime(self):
        value_datetime = np.array([dt(2000, 1, 15), dt(2000, 2, 15)])
        bounds_datetime = np.array([[dt(2000, 1, 1), dt(2000, 2, 1)],
                                    [dt(2000, 2, 1), dt(2000, 3, 1)]])
        value = date2num(value_datetime, constants.DEFAULT_TEMPORAL_UNITS, calendar=constants.DEFAULT_TEMPORAL_CALENDAR)
        bounds_num = date2num(bounds_datetime, constants.DEFAULT_TEMPORAL_UNITS,
                              calendar=constants.DEFAULT_TEMPORAL_CALENDAR)
        bounds_options = [None, bounds_num, bounds_datetime]
        value_options = [value, value, value_datetime]
        for format_time in [True, False]:
            for value, bounds in zip(value_options, bounds_options):
                if bounds is not None:
                    bounds = TemporalVariable(name='time_bounds', value=bounds, dimensions=['time', 'bounds'])
                td = TemporalVariable(value=value, bounds=bounds, format_time=format_time, dimensions=['time'])
                if bounds is not None:
                    try:
                        self.assertNumpyAll(td.bounds.value_datetime, np.ma.array(bounds_datetime))
                    except CannotFormatTimeError:
                        self.assertFalse(format_time)
                    self.assertNumpyAll(td.bounds.value_numtime, np.ma.array(bounds_num))
                else:
                    self.assertIsNone(td.bounds)
                    try:
                        self.assertIsNotNone(td.value_datetime)
                    except CannotFormatTimeError:
                        self.assertFalse(format_time)

    def test_bounds(self):
        # Test bounds inherits calendar from parent.
        tv = TemporalVariable(name='time', value=[1, 2], calendar='noleap', dimensions='time')
        tvb = TemporalVariable(name='tbounds', value=[[0.5, 1.5], [1.5, 2.5]], dimensions=['time', 'bounds'])
        tv.set_bounds(tvb)
        self.assertEqual(tv.bounds.calendar, tv.calendar)

        # Test MFTime and bad bounds metadata overloading the dimension map.
        paths = create_mftime_nc_files(self, units_on_time_bounds=True)
        dmap = {'time': {'attrs': {'axis': 'T'},
                         'bounds': None,
                         'dimensions': [u'time'],
                         'variable': u'time'}}
        rd = RequestDataset(paths, dimension_map=dmap)
        field = rd.get()
        self.assertIsNone(field.time.bounds)
        self.assertIsNotNone(field.time.value_datetime)

        # Run test same as above but change how dimension map is modified.
        rd = RequestDataset(paths)
        rd.dimension_map.set_bounds(DimensionMapKey.TIME, None)
        field = rd.get()
        self.assertIsNone(field.time.bounds)
        self.assertIsNotNone(field.time.value_datetime)

    @attr('cfunits')
    def test_cfunits(self):
        temporal = self.init_temporal_variable(value=[4, 5, 6], units='days since 1900-1-1')
        self.assertEqual(temporal.cfunits.calendar, temporal.calendar)

    @attr('cfunits')
    def test_cfunits_conform(self):
        # Test with template units.
        td = self.get_template_units()
        value_numtime_original = td.value_numtime.copy()
        value_datetime_original = td.value_datetime.copy()
        td.cfunits_conform(get_units_object('days since 1920-1-1', calendar=td.calendar))
        self.assertLess(td.value_numtime.mean(), value_numtime_original.mean())
        self.assertNumpyAll(value_datetime_original, td.value_datetime)
        self.assertEqual(td.units, 'days since 1920-1-1')
        self.assertEqual(td.calendar, 'proleptic_gregorian')

        d = get_units_object('days since 1900-1-1', calendar=constants.DEFAULT_TEMPORAL_CALENDAR)
        td = self.init_temporal_variable(value=[4, 5, 6], units='days since 1901-1-1')
        td.cfunits_conform(d)
        self.assertTrue(get_are_units_equal((td.cfunits, d)))

        # Test with template units.
        units = get_units_object('days since 1960-1-1', calendar='proleptic_gregorian')
        td = self.get_template_units()
        td.cfunits_conform(units)
        self.assertAlmostEqual(td.value_numtime.mean(), 4020.9375)
        self.assertEqual(str(td.units), str(units).split('calendar=')[0].strip())

    @attr('data', 'cfunits')
    def test_cfunits_conform_data(self):

        def _get_temporal_variable_(t_conform_units_to=None):
            rd = self.get_request_dataset(t_conform_units_to=t_conform_units_to)
            bounds = TemporalVariable(name='time_bnds', request_dataset=rd)
            tv = TemporalVariable(name='time', request_dataset=rd, bounds=bounds)
            return tv

        target = get_units_object('days since 1949-1-1', calendar='365_day')
        temporal = _get_temporal_variable_(t_conform_units_to=target)
        temporal_orig = _get_temporal_variable_()
        self.assertTrue(get_are_units_equivalent((target, temporal.cfunits, temporal_orig.cfunits)))
        self.assertEqual(temporal.calendar, '365_day')
        self.assertEqual(temporal_orig.calendar, '365_day')
        self.assertNumpyNotAll(temporal.get_value(), temporal_orig.get_value())
        self.assertNumpyAll(temporal.value_datetime, temporal_orig.value_datetime)

    def test_extent_datetime_and_extent_numtime(self):
        value_numtime = np.array([6000., 6001., 6002])
        value_datetime = self.init_temporal_variable(value=value_numtime).value_datetime

        for format_time in [True, False]:
            for value in [value_numtime, value_datetime]:
                td = self.init_temporal_variable(value=value, format_time=format_time)
                try:
                    self.assertEqual(td.extent_datetime, (min(value_datetime), max(value_datetime)))
                except CannotFormatTimeError:
                    self.assertFalse(format_time)
                self.assertEqual(td.extent_numtime, (6000., 6002.))

    def test_get_between(self):
        keywords = dict(as_datetime=[False, True])

        for k in self.iter_product_keywords(keywords, as_namedtuple=True):
            td = self.get_temporalvariable()
            if not k.as_datetime:
                td._value = td.value_numtime
                td.bounds._value = td.bounds.value_numtime
                td._value_datetime = None
                td._bounds_datetime = None
                self.assertTrue(get_datetime_conversion_state(td.get_value().flatten()[0]))
            res = td.get_between(dt(1899, 1, 4, 12, 0), dt(1899, 1, 10, 12, 0), return_indices=False)
            self.assertEqual(res.shape, (7,))
            self.assertIsNone(td._value_datetime)
            self.assertIsNone(td._bounds_datetime)

        # test with template units
        td = self.get_template_units()
        lower = datetime.datetime(1971, 1, 2)
        upper = datetime.datetime(1971, 1, 5)
        sub = td.get_between(lower, upper)
        self.assertEqual(sub.shape, (3,))

    def test_get_boolean_groups_from_time_regions(self):
        dates = get_date_list(dt(2012, 1, 1), dt(2013, 12, 31), 1)
        seasons = [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]]
        td = self.init_temporal_variable(value=dates)
        time_regions = get_time_regions(seasons, dates, raise_if_incomplete=False)

        dgroups = list(iter_boolean_groups_from_time_regions(time_regions, td))
        # the last winter season is not complete as it does not have enough years
        self.assertEqual(len(dgroups), 7)

        to_test = []
        for dgroup in dgroups:
            sub = td[dgroup]
            # (upper and lower values of time vector, count of elements in time group, the middle value of the vector)
            to_test.append([sub.extent, sub.shape[0], sub[sub.shape[0] / 2].get_value()[0]])
        correct = [[(datetime.datetime(2012, 3, 1, 0, 0), datetime.datetime(2012, 5, 31, 0, 0)), 92,
                    datetime.datetime(2012, 4, 16, 0, 0)],
                   [(datetime.datetime(2012, 6, 1, 0, 0), datetime.datetime(2012, 8, 31, 0, 0)), 92,
                    datetime.datetime(2012, 7, 17, 0, 0)],
                   [(datetime.datetime(2012, 9, 1, 0, 0), datetime.datetime(2012, 11, 30, 0, 0)), 91,
                    datetime.datetime(2012, 10, 16, 0, 0)],
                   [(datetime.datetime(2012, 12, 1, 0, 0), datetime.datetime(2013, 2, 28, 0, 0)), 90,
                    datetime.datetime(2013, 1, 15, 0, 0)],
                   [(datetime.datetime(2013, 3, 1, 0, 0), datetime.datetime(2013, 5, 31, 0, 0)), 92,
                    datetime.datetime(2013, 4, 16, 0, 0)],
                   [(datetime.datetime(2013, 6, 1, 0, 0), datetime.datetime(2013, 8, 31, 0, 0)), 92,
                    datetime.datetime(2013, 7, 17, 0, 0)],
                   [(datetime.datetime(2013, 9, 1, 0, 0), datetime.datetime(2013, 11, 30, 0, 0)), 91,
                    datetime.datetime(2013, 10, 16, 0, 0)]]
        self.assertEqual(to_test, correct)

    def test_get_datetime(self):
        td = self.init_temporal_variable(value=[5, 6])
        dts = np.array([dt(2000, 1, 15, 12), dt(2000, 2, 15, 12)])
        arr = date2num(dts, 'days since 0001-01-01 00:00:00')
        res = td.get_datetime(arr)
        self.assertNumpyAll(dts, res)

        td = self.init_temporal_variable(value=[5, 6], units='months since 1978-12')
        res = td.get_datetime(td.get_value())
        self.assertEqual(res[0], dt(1979, 5, 16))

        td = self.init_temporal_variable(value=[15, 16], units='months since 1978-12')
        res = td.get_datetime(td.get_value())
        self.assertEqual(res[0], dt(1980, 3, 16))

        units = 'days since 0001-01-01 00:00:00'
        calendar = '365_day'
        ndt = netcdftime.datetime
        ndts = np.array([ndt(0000, 2, 30), ndt(0000, 2, 31)])
        narr = date2num(ndts, units, calendar=calendar)
        td = self.init_temporal_variable(value=narr, units=units, calendar=calendar)
        res = td.get_datetime(td.get_value())
        self.assertTrue(all([element.year == 0000 for element in res.flat]))

    def test_get_datetime_with_template_units(self):
        """Template units are present in some non-CF conforming files."""

        td = self.get_template_units()
        self.assertIsNotNone(td.value_numtime)
        self.assertIsNotNone(td.value_datetime)
        self.assertEqual(td.value_datetime[2], datetime.datetime(1971, 1, 3, 22, 30))
        td2 = self.init_temporal_variable(value=td.value_numtime, units=td.units, calendar='proleptic_gregorian')
        self.assertNumpyAll(td.value_datetime, td2.value_datetime)

    def test_get_numtime(self):
        units_options = [constants.DEFAULT_TEMPORAL_UNITS, 'months since 1960-5']
        value_options = [np.array([5000., 5001]), np.array([5, 6, 7])]
        for units, value in zip(units_options, value_options):
            td = self.init_temporal_variable(value=value, units=units)
            nums = td.get_numtime(td.value_datetime)
            self.assertNumpyAll(nums, value)

        # Test passing in a list.
        value = [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2)]
        tv = TemporalVariable(name='t', value=[1], dimensions=['a'])
        nt = tv.get_numtime(value)
        self.assertIsInstance(nt, np.ndarray)

    def test_get_grouping(self):
        tv = self.get_temporalvariable()
        td = tv.get_between(datetime.datetime(1900, 1, 1), datetime.datetime(1900, 12, 31, 23, 59))
        self.assertEqual(td.bounds.shape, (365, 2))
        tgd = td.get_grouping(['year'])
        self.assertEqual(tgd.get_value(), np.array([datetime.datetime(1900, 7, 1)]))

        # Test with a 360_day calendar and 3-hourly data.
        start = 0.0625
        stop = 719.9375
        step = 0.125
        values = np.arange(start, stop + step, step)
        td = self.init_temporal_variable(value=values, calendar='360_day', units='days since 1960-01-01')
        for g in CalcGrouping.iter_possible():
            tgd = td.get_grouping(g)
            self.assertIsInstance(tgd, TemporalGroupVariable)
            # Test calendar is maintained when creating a group dimension.
            self.assertEqual(tgd.calendar, '360_day')

        value = [dt(2012, 1, 1), dt(2012, 1, 2)]
        td = self.init_temporal_variable(value=value)
        tgd = td.get_grouping(['month'])
        self.assertEqual(tuple(tgd.date_parts[0]), (None, 1, None, None, None, None))
        self.assertTrue(tgd.dgroups[0].all())

    def test_get_grouping_all(self):
        for b in [True, False]:
            td = self.get_temporalvariable(add_bounds=b)
            tgd = td.get_grouping('all')
            self.assertEqual(tgd.dgroups, [slice(None)])
            self.assertEqual(td.get_value()[546], tgd.get_value()[0])

            actual = date2num(tgd.bounds.get_value(), td.units, td.calendar).tolist()

            if b:
                desired = [[693232.0, 694327.0]]
            else:
                desired = [[693232.5, 694326.5]]
            self.assertEqual(actual, desired)

    def test_get_grouping_other(self):
        tdim = self.get_temporalvariable()
        grouping = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], 'year']
        new_bounds, date_parts, repr_dt, dgroups = tdim._get_grouping_other_(grouping)

        repr_dt = date2num(repr_dt, tdim.units, calendar=tdim.calendar).tolist()
        desired = [693247.0, 693337.0, 693428.0, 693520.0, 693612.0, 693702.0, 693793.0, 693885.0, 693977.0, 694067.0,
                   694158.0, 694250.0]
        self.assertEqual(repr_dt, desired)

        new_bounds = date2num(new_bounds, tdim.units, calendar=tdim.calendar).tolist()
        desired = [[693232.0, 693597.0], [693291.0, 693383.0], [693383.0, 693475.0], [693475.0, 693566.0],
                   [693597.0, 693962.0], [693656.0, 693748.0], [693748.0, 693840.0], [693840.0, 693931.0],
                   [693962.0, 694327.0], [694021.0, 694113.0], [694113.0, 694205.0], [694205.0, 694296.0]]
        self.assertEqual(new_bounds, desired)

        desired = [False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True, True, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False]
        self.assertEqual(dgroups[4].tolist(), desired)

        desired = [([12, 1, 2], 1899), ([3, 4, 5], 1899), ([6, 7, 8], 1899), ([9, 10, 11], 1899), ([12, 1, 2], 1900),
                   ([3, 4, 5], 1900), ([6, 7, 8], 1900), ([9, 10, 11], 1900), ([12, 1, 2], 1901), ([3, 4, 5], 1901),
                   ([6, 7, 8], 1901), ([9, 10, 11], 1901)]
        self.assertEqual(date_parts.tolist(), desired)

    @attr('data')
    def test_get_grouping_seasonal(self):
        dates = get_date_list(dt(2012, 4, 1), dt(2012, 10, 31), 1)
        td = self.init_temporal_variable(value=dates)

        # Standard seasonal group.
        calc_grouping = [[6, 7, 8]]
        tg = td.get_grouping(calc_grouping)
        self.assertEqual(len(tg.get_value()), 1)
        selected_months = [s.month for s in td.get_value()[tg.dgroups[0]].flat]
        not_selected_months = [s.month for s in td.get_value()[np.invert(tg.dgroups[0])]]
        self.assertEqual(set(calc_grouping[0]), set(selected_months))
        self.assertFalse(set(not_selected_months).issubset(set(calc_grouping[0])))

        # Seasons with different sizes.
        calc_grouping = [[4, 5, 6, 7], [8, 9, 10]]
        tg = td.get_grouping(calc_grouping)
        self.assertEqual(len(tg.get_value()), 2)
        self.assertNumpyAll(tg.dgroups[0], np.invert(tg.dgroups[1]))

        # Crosses year boundary.
        calc_grouping = [[11, 12, 1]]
        dates = get_date_list(dt(2012, 10, 1), dt(2013, 3, 31), 1)
        td = self.init_temporal_variable(value=dates)
        tg = td.get_grouping(calc_grouping)
        selected_months = [s.month for s in td.get_value()[tg.dgroups[0]].flat]
        self.assertEqual(set(calc_grouping[0]), set(selected_months))
        self.assertEqual(tg.get_value()[0], dt(2012, 12, 16))

        # Use real data.
        td = TemporalVariable(request_dataset=self.get_request_dataset())
        tg = td.get_grouping([[3, 4, 5]])
        self.assertEqual(tg.get_value()[0], dt(2005, 4, 16))

    def test_get_grouping_seasonal_empty_with_year_missing_month(self):
        dt1 = datetime.datetime(1900, 0o1, 0o1)
        dt2 = datetime.datetime(1903, 1, 31)
        dates = get_date_list(dt1, dt2, days=1)
        td = self.init_temporal_variable(value=dates)
        group = [[12, 1, 2], 'unique']
        tg = td.get_grouping(group)
        # There should be a month missing from the last season (february) and it should not be considered complete.
        self.assertEqual(tg.get_value().shape[0], 2)

    @attr('data')
    def test_get_grouping_seasonal_real_data_all_seasons(self):
        """Test with real data and full seasons."""

        calc_grouping = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        rd = self.get_request_dataset()
        tv_bounds = TemporalVariable(name='time_bnds', request_dataset=rd)
        tv = TemporalVariable(request_dataset=rd, bounds=tv_bounds)
        tgd = tv.get_grouping(calc_grouping)
        self.assertEqual(tgd.shape, (4,))
        self.assertEqual([xx[1] for xx in calc_grouping], [xx.month for xx in tgd.get_value().flat])
        self.assertEqual(set([xx.day for xx in tgd.get_value().flat]), {constants.CALC_MONTH_CENTROID})
        self.assertEqual([2006, 2005, 2005, 2005], [xx.year for xx in tgd.get_value().flat])
        self.assertNumpyAll(tgd.bounds.value_numtime.data,
                            np.array([[55115.0, 58765.0], [55174.0, 58551.0], [55266.0, 58643.0], [55358.0, 58734.0]]))

    def test_get_grouping_seasonal_unique_flag(self):
        """Test the unique flag for seasonal groups."""

        dates = get_date_list(dt(2012, 1, 1), dt(2013, 12, 31), 1)
        td = self.init_temporal_variable(value=dates)
        calc_grouping = [[6, 7, 8], 'unique']
        tg = td.get_grouping(calc_grouping)

        time_region = {'year': [2012], 'month': [6, 7, 8]}
        sub1, idx1 = td.get_time_region(time_region, return_indices=True)
        time_region = {'year': [2013], 'month': [6, 7, 8]}
        sub2, idx2 = td.get_time_region(time_region, return_indices=True)
        base_select = np.zeros(td.shape[0], dtype=bool)
        dgroups = deque()

        for software, manual in zip(tg.dgroups, dgroups):
            self.assertNumpyAll(software, manual)
        self.assertEqual(len(tg.dgroups), 2)
        self.assertEqual(tg.get_value().tolist(),
                         [datetime.datetime(2012, 7, 17, 0, 0), datetime.datetime(2013, 7, 17, 0, 0)])
        self.assertEqual(tg.bounds.get_value().tolist(),
                         [[datetime.datetime(2012, 6, 1, 0, 0), datetime.datetime(2012, 8, 31, 0, 0)],
                          [datetime.datetime(2013, 6, 1, 0, 0), datetime.datetime(2013, 8, 31, 0, 0)]])

        dgroup1 = base_select.copy()
        dgroup1[idx1] = True
        dgroup2 = base_select.copy()
        dgroup2[idx2] = True

        dgroups.append(dgroup1)
        dgroups.append(dgroup2)

        tg = td.get_grouping([[6, 7, 8], 'year'])
        for ii in range(len(tg.dgroups)):
            self.assertNumpyAll(tg.dgroups[ii], dgroups[ii])
        self.assertEqual(len(tg.dgroups), len(dgroups))

    def test_get_grouping_seasonal_unique_flag_all_seasons(self):
        """Test unique flag with all seasons."""

        start = datetime.datetime(1900, 1, 1)
        end = datetime.datetime(1902, 12, 31)
        ret = self.get_time_series(start, end)
        calc_grouping = [[12, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], 'unique']
        td = self.init_temporal_variable(value=ret)
        group = td.get_grouping(calc_grouping)

        for idx in range(group.shape[0]):
            bounds_lower = group.bounds.get_value()[idx, 0]
            bounds_upper = group.bounds.get_value()[idx, 1]

            sub = td[group.dgroups[idx]]
            self.assertEqual(sub.get_masked_value().compressed().min(), bounds_lower)
            self.assertEqual(sub.get_masked_value().compressed().max(), bounds_upper)

        self.assertEqual(group.get_value().tolist(),
                         [datetime.datetime(1900, 4, 16, 0, 0), datetime.datetime(1900, 7, 17, 0, 0),
                          datetime.datetime(1900, 10, 16, 0, 0), datetime.datetime(1901, 1, 15, 0, 0),
                          datetime.datetime(1901, 4, 16, 0, 0), datetime.datetime(1901, 7, 17, 0, 0),
                          datetime.datetime(1901, 10, 16, 0, 0), datetime.datetime(1902, 1, 15, 0, 0),
                          datetime.datetime(1902, 4, 16, 0, 0), datetime.datetime(1902, 7, 17, 0, 0),
                          datetime.datetime(1902, 10, 16, 0, 0)])
        self.assertEqual(group.bounds.get_value().tolist(),
                         [[datetime.datetime(1900, 3, 1, 0, 0), datetime.datetime(1900, 5, 31, 0, 0)],
                          [datetime.datetime(1900, 6, 1, 0, 0), datetime.datetime(1900, 8, 31, 0, 0)],
                          [datetime.datetime(1900, 9, 1, 0, 0), datetime.datetime(1900, 11, 30, 0, 0)],
                          [datetime.datetime(1900, 12, 1, 0, 0), datetime.datetime(1901, 2, 28, 0, 0)],
                          [datetime.datetime(1901, 3, 1, 0, 0), datetime.datetime(1901, 5, 31, 0, 0)],
                          [datetime.datetime(1901, 6, 1, 0, 0), datetime.datetime(1901, 8, 31, 0, 0)],
                          [datetime.datetime(1901, 9, 1, 0, 0), datetime.datetime(1901, 11, 30, 0, 0)],
                          [datetime.datetime(1901, 12, 1, 0, 0), datetime.datetime(1902, 2, 28, 0, 0)],
                          [datetime.datetime(1902, 3, 1, 0, 0), datetime.datetime(1902, 5, 31, 0, 0)],
                          [datetime.datetime(1902, 6, 1, 0, 0), datetime.datetime(1902, 8, 31, 0, 0)],
                          [datetime.datetime(1902, 9, 1, 0, 0), datetime.datetime(1902, 11, 30, 0, 0)]])

    def test_get_grouping_seasonal_unique_flag_winter_season(self):
        """Test with a single winter season using the unique flag."""

        dt1 = datetime.datetime(1900, 0o1, 0o1)
        dt2 = datetime.datetime(1902, 12, 31)
        dates = get_date_list(dt1, dt2, days=1)
        td = self.init_temporal_variable(value=dates)
        group = [[12, 1, 2], 'unique']
        tg = td.get_grouping(group)
        self.assertEqual(tg.get_value().shape[0], 2)
        self.assertEqual(tg.bounds.get_value().tolist(),
                         [[datetime.datetime(1900, 12, 1, 0, 0), datetime.datetime(1901, 2, 28, 0, 0)],
                          [datetime.datetime(1901, 12, 1, 0, 0), datetime.datetime(1902, 2, 28, 0, 0)]])

    def test_get_grouping_seasonal_year_flag(self):
        """Test with a year flag."""

        dates = get_date_list(dt(2012, 1, 1), dt(2013, 12, 31), 1)
        td = self.init_temporal_variable(value=dates)
        calc_grouping = [[6, 7, 8], 'year']
        tg = td.get_grouping(calc_grouping)
        self.assertEqual(tg.get_value().shape[0], 2)

        actual = date2num(tg.get_value(), td.units, calendar=td.calendar).tolist()
        desired = [734701.0, 735066.0]
        self.assertEqual(actual, desired)

        actual = tg.dgroups[0].tolist()
        desired = [False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, True, True, True,
                   True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                   True, True, True, True, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                   False, False, False, False, False, False, False, False, False, False, False, False, False, False]
        self.assertEqual(actual, desired)

        actual = date2num(td.get_masked_value()[tg.dgroups[1]], td.units, calendar=td.calendar).tolist()
        desired = [735021.0, 735022.0, 735023.0, 735024.0, 735025.0, 735026.0, 735027.0, 735028.0, 735029.0, 735030.0,
                   735031.0, 735032.0, 735033.0, 735034.0, 735035.0, 735036.0, 735037.0, 735038.0, 735039.0, 735040.0,
                   735041.0, 735042.0, 735043.0, 735044.0, 735045.0, 735046.0, 735047.0, 735048.0, 735049.0, 735050.0,
                   735051.0, 735052.0, 735053.0, 735054.0, 735055.0, 735056.0, 735057.0, 735058.0, 735059.0, 735060.0,
                   735061.0, 735062.0, 735063.0, 735064.0, 735065.0, 735066.0, 735067.0, 735068.0, 735069.0, 735070.0,
                   735071.0, 735072.0, 735073.0, 735074.0, 735075.0, 735076.0, 735077.0, 735078.0, 735079.0, 735080.0,
                   735081.0, 735082.0, 735083.0, 735084.0, 735085.0, 735086.0, 735087.0, 735088.0, 735089.0, 735090.0,
                   735091.0, 735092.0, 735093.0, 735094.0, 735095.0, 735096.0, 735097.0, 735098.0, 735099.0, 735100.0,
                   735101.0, 735102.0, 735103.0, 735104.0, 735105.0, 735106.0, 735107.0, 735108.0, 735109.0, 735110.0,
                   735111.0, 735112.0]
        self.assertEqual(actual, desired)

        # Test crossing year boundary.
        for calc_grouping in [[[12, 1, 2], 'year'], ['year', [12, 1, 2]]]:
            tg = td.get_grouping(calc_grouping)

            actual = tg.value_numtime.tolist()
            desired = [734519.0, 734885.0]
            self.assertEqual(actual, desired)

            actual = tg.bounds.value_numtime.tolist()
            desired = [[734504.0, 734869.0], [734870.0, 735234.0]]
            self.assertEqual(actual, desired)

            actual = td.value_numtime[tg.dgroups[1]].tolist()
            desired = [734870.0, 734871.0, 734872.0, 734873.0, 734874.0, 734875.0, 734876.0, 734877.0, 734878.0,
                       734879.0, 734880.0, 734881.0, 734882.0, 734883.0, 734884.0, 734885.0, 734886.0, 734887.0,
                       734888.0, 734889.0, 734890.0, 734891.0, 734892.0, 734893.0, 734894.0, 734895.0, 734896.0,
                       734897.0, 734898.0, 734899.0, 734900.0, 734901.0, 734902.0, 734903.0, 734904.0, 734905.0,
                       734906.0, 734907.0, 734908.0, 734909.0, 734910.0, 734911.0, 734912.0, 734913.0, 734914.0,
                       734915.0, 734916.0, 734917.0, 734918.0, 734919.0, 734920.0, 734921.0, 734922.0, 734923.0,
                       734924.0, 734925.0, 734926.0, 734927.0, 734928.0, 735204.0, 735205.0, 735206.0, 735207.0,
                       735208.0, 735209.0, 735210.0, 735211.0, 735212.0, 735213.0, 735214.0, 735215.0, 735216.0,
                       735217.0, 735218.0, 735219.0, 735220.0, 735221.0, 735222.0, 735223.0, 735224.0, 735225.0,
                       735226.0, 735227.0, 735228.0, 735229.0, 735230.0, 735231.0, 735232.0, 735233.0, 735234.0]
            self.assertEqual(actual, desired)

    def test_get_subset_by_function(self):

        def _func_(value, bounds=None):
            months = [6, 7]
            indices = []
            for ii, dt in enumerate(value.flat):
                if dt.month in months:
                    if dt.month == 6 and dt.day >= 15:
                        indices.append(ii)
                    elif dt.month == 7 and dt.day <= 15:
                        indices.append(ii)
            return indices

        dates = get_date_list(dt(2002, 1, 31), dt(2003, 12, 31), 1)
        td = self.init_temporal_variable(value=dates)

        ret = td.get_subset_by_function(_func_)
        self.assertEqual(ret.shape, (62,))
        for v in ret.value_datetime:
            self.assertIn(v.month, [6, 7])

        ret2 = td.get_subset_by_function(_func_, return_indices=True)
        self.assertNumpyAll(td[ret2[1]].value_datetime, ret.value_datetime)

    def test_get_time_region_value_only(self):
        dates = get_date_list(dt(2002, 1, 31), dt(2009, 12, 31), 1)
        td = self.init_temporal_variable(value=dates)

        ret, indices = td.get_time_region({'month': [8]}, return_indices=True)
        self.assertEqual(set([8]), set([d.month for d in ret.get_value().flat]))

        ret, indices = td.get_time_region({'year': [2008, 2004]}, return_indices=True)
        self.assertEqual(set([2008, 2004]), set([d.year for d in ret.get_value().flat]))

        ret, indices = td.get_time_region({'day': [20, 31]}, return_indices=True)
        self.assertEqual(set([20, 31]), set([d.day for d in ret.get_value().flat]))

        ret, indices = td.get_time_region({'day': [20, 31], 'month': [9, 10], 'year': [2003]}, return_indices=True)
        self.assertNumpyAll(ret.get_masked_value(),
                            np.ma.array([dt(2003, 9, 20), dt(2003, 10, 20), dt(2003, 10, 31, )]))
        self.assertEqual(ret.shape, indices.shape)

        self.assertEqual(ret.extent, (datetime.datetime(2003, 9, 20), datetime.datetime(2003, 10, 31)))

    def test_get_to_conform_value(self):
        td = self.init_temporal_variable(value=[datetime.datetime(2000, 1, 1)])
        self.assertNumpyAll(td._get_to_conform_value_(), np.ma.array([730121.]))

    def test_has_months_units(self):
        td = self.init_temporal_variable(value=[5, 6], units='months since 1978-12')
        self.assertTrue(td._has_months_units)
        td = self.init_temporal_variable(value=[5, 6])
        self.assertFalse(td._has_months_units)

    def test_months_in_time_units(self):
        units = "months since 1978-12"
        vec = list(range(0, 36))
        datetimes = get_datetime_from_months_time_units(vec, units)
        td = self.init_temporal_variable(value=vec, units=units, calendar='standard')
        self.assertTrue(td._has_months_units)
        self.assertNumpyAll(td.value_datetime, np.ma.array(datetimes))

    def test_months_in_time_units_are_bad_netcdftime(self):
        units = "months since 1978-12"
        vec = list(range(0, 36))
        calendar = "standard"
        with self.assertRaises((TypeError, ValueError)):
            num2date(vec, units, calendar=calendar)

    def test_months_in_time_units_between(self):
        units = "months since 1978-12"
        vec = list(range(0, 36))
        datetimes = get_datetime_from_months_time_units(vec, units)
        td = self.init_temporal_variable(value=vec, units=units, calendar='standard')
        ret = td.get_between(datetimes[0], datetimes[3])
        self.assertNumpyAll(ret.get_masked_value(), np.ma.array([0, 1, 2, 3]))

    def test_months_not_in_time_units(self):
        units = "days since 1900-01-01"
        value = np.array([31])
        td = self.init_temporal_variable(value=value, units=units, calendar='standard')
        self.assertFalse(td._has_months_units)

    @attr('cfunits')
    def test_units(self):
        # Test calendar is maintained when setting units.
        td = self.init_temporal_variable(value=[5])
        td.units = get_units_object('days since 1950-1-1', calendar='360_day')
        self.assertEqual(td.calendar, '360_day')
        self.assertEqual(td.units, 'days since 1950-1-1')

    def test_get_time_regions(self):
        dates = get_date_list(dt(2012, 1, 1), dt(2013, 12, 31), 1)

        # Two simple seasons.
        calc_grouping = [[6, 7, 8], [9, 10, 11]]
        time_regions = get_time_regions(calc_grouping, dates)
        correct = [[{'month': [6, 7, 8], 'year': [2012]}], [{'month': [9, 10, 11], 'year': [2012]}],
                   [{'month': [6, 7, 8], 'year': [2013]}], [{'month': [9, 10, 11], 'year': [2013]}]]
        self.assertEqual(time_regions, correct)

        # Add an interannual season at the back.
        calc_grouping = [[6, 7, 8], [9, 10, 11], [12, 1, 2]]
        with self.assertRaises(IncompleteSeasonError):
            get_time_regions(calc_grouping, dates)
        time_regions = get_time_regions(calc_grouping, dates, raise_if_incomplete=False)
        correct = [[{'month': [6, 7, 8], 'year': [2012]}], [{'month': [9, 10, 11], 'year': [2012]}],
                   [{'month': [12], 'year': [2012]}, {'month': [2, 1], 'year': [2013]}],
                   [{'month': [6, 7, 8], 'year': [2013]}], [{'month': [9, 10, 11], 'year': [2013]}]]
        self.assertEqual(time_regions, correct)

        # Put the interannual season in the middle.
        calc_grouping = [[9, 10, 11], [12, 1, 2], [6, 7, 8]]
        with self.assertRaises(IncompleteSeasonError):
            get_time_regions(calc_grouping, dates)
        time_regions = get_time_regions(calc_grouping, dates, raise_if_incomplete=False)
        correct = [[{'month': [9, 10, 11], 'year': [2012]}],
                   [{'month': [12], 'year': [2012]}, {'month': [2, 1], 'year': [2013]}],
                   [{'month': [6, 7, 8], 'year': [2012]}], [{'month': [9, 10, 11], 'year': [2013]}],
                   [{'month': [6, 7, 8], 'year': [2013]}]]
        self.assertEqual(time_regions, correct)

        # Odd seasons, but covering the whole year.
        calc_grouping = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        time_regions = get_time_regions(calc_grouping, dates)
        correct = [[{'month': [1, 2, 3], 'year': [2012]}], [{'month': [4, 5, 6], 'year': [2012]}],
                   [{'month': [7, 8, 9], 'year': [2012]}], [{'month': [10, 11, 12], 'year': [2012]}],
                   [{'month': [1, 2, 3], 'year': [2013]}], [{'month': [4, 5, 6], 'year': [2013]}],
                   [{'month': [7, 8, 9], 'year': [2013]}], [{'month': [10, 11, 12], 'year': [2013]}]]
        self.assertEqual(time_regions, correct)

        # Standard seasons.
        calc_grouping = [[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 1, 2]]
        time_regions = get_time_regions(calc_grouping, dates, raise_if_incomplete=False)
        correct = [[{'month': [3, 4, 5], 'year': [2012]}], [{'month': [6, 7, 8], 'year': [2012]}],
                   [{'month': [9, 10, 11], 'year': [2012]}],
                   [{'month': [12], 'year': [2012]}, {'month': [2, 1], 'year': [2013]}],
                   [{'month': [3, 4, 5], 'year': [2013]}], [{'month': [6, 7, 8], 'year': [2013]}],
                   [{'month': [9, 10, 11], 'year': [2013]}]]
        self.assertEqual(time_regions, correct)

        # In this case, the time series starts in december. The first season/year combination will not actually be
        # present in the time series and should be removed by the code.
        actual = [[{'month': [3, 4, 5], 'year': [1950]}], [{'month': [3, 4, 5], 'year': [1951]}]]
        raise_if_incomplete = False
        seasons = [[3, 4, 5]]
        dates = get_date_list(dt(1949, 12, 16), dt(1951, 12, 16), 1)
        target = get_time_regions(seasons, dates, raise_if_incomplete=raise_if_incomplete)
        self.assertEqual(actual, target)

    def test_time_range_subset(self):
        dt1 = datetime.datetime(1950, 0o1, 0o1, 12)
        dt2 = datetime.datetime(1950, 12, 31, 12)
        dates = np.array(get_date_list(dt1, dt2, 1))
        r1 = datetime.datetime(1950, 0o1, 0o1)
        r2 = datetime.datetime(1950, 12, 31)
        td = self.init_temporal_variable(value=dates)
        ret = td.get_between(r1, r2)
        self.assertEqual(ret.get_value()[-1], datetime.datetime(1950, 12, 30, 12, 0))
        delta = datetime.timedelta(hours=12)
        lower = dates - delta
        upper = dates + delta
        bounds = np.empty((lower.shape[0], 2), dtype=object)
        bounds[:, 0] = lower
        bounds[:, 1] = upper
        bounds = self.init_temporal_variable(value=bounds, name='time_bounds', dimensions=['time', 'bounds'])
        td = self.init_temporal_variable(value=dates, bounds=bounds, dimensions=['time'])
        self.assertTrue(td.has_bounds)
        ret = td.get_between(r1, r2)
        self.assertEqual(ret.get_value()[-1], datetime.datetime(1950, 12, 31, 12, 0))

    def test_value_datetime_and_value_numtime(self):
        value_datetime = np.array([dt(2000, 1, 15), dt(2000, 2, 15)])
        value = date2num(value_datetime, constants.DEFAULT_TEMPORAL_UNITS, calendar=constants.DEFAULT_TEMPORAL_CALENDAR)
        keywords = dict(value=[value, value_datetime],
                        format_time=[True, False])
        for k in self.iter_product_keywords(keywords, as_namedtuple=True):
            td = self.init_temporal_variable(**k._asdict())
            self.assertNumpyAll(td.get_masked_value(), np.ma.array(k.value))
            try:
                self.assertNumpyAll(td.value_datetime, np.ma.array(value_datetime))
            except CannotFormatTimeError:
                self.assertFalse(k.format_time)
            self.assertNumpyAll(td.value_numtime, np.ma.array(value))

    def test_write(self):
        tv = self.get_temporalvariable()
        path = self.get_temporary_file_path('foo.nc')
        self.assertEqual(tv.units, tv.bounds.units)
        self.assertEqual(tv.calendar, tv.bounds.calendar)
        with self.nc_scope(path, 'w') as ds:
            tv.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path) as ds:
            time = ds.variables['time']
            self.assertEqual(time.ncattrs(), ['bounds', 'calendar', 'units'])
            time_bounds = ds.variables['time_bounds']
            self.assertEqual(time_bounds.ncattrs(), ['calendar', 'units'])
        rd = RequestDataset(uri=path)
        bounds = TemporalVariable(name='time_bounds', request_dataset=rd)
        tv2 = TemporalVariable(name='time', request_dataset=rd, bounds=bounds)
        self.assertNumpyAll(tv.value_datetime, tv2.value_datetime)
        self.assertNumpyAll(tv.bounds.value_datetime, tv2.bounds.value_datetime)
        path2 = self.get_temporary_file_path('foo2.nc')
        with self.nc_scope(path2, 'w') as ds:
            tv2.write(ds)
        self.assertNcEqual(path, path2)


class TestTemporalGroupVariable(AbstractTestInterface):
    def get_tgv(self):
        rd = self.get_request_dataset()
        bounds = TemporalVariable(name='time_bnds', request_dataset=rd)
        tv = TemporalVariable(name='time', bounds=bounds, request_dataset=rd)
        return tv.get_grouping(['month'])

    @attr('data')
    def test_init(self):
        tgd = self.get_tgv()
        self.assertIsInstance(tgd, TemporalGroupVariable)

    @attr('data')
    def test_write_netcdf(self):
        tgd = self.get_tgv()
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(path, 'w') as ds:
            tgd.write(ds)
        with self.nc_scope(path) as ds:
            self.assertIn('climatology_bounds', ds.variables)
            ncvar = ds.variables[tgd.name]
            self.assertEqual(ncvar.climatology, 'climatology_bounds')
            with self.assertRaises(AttributeError):
                ncvar.bounds
