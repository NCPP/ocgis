import netCDF4 as nc
from ocgis.test.base import TestBase
from ocgis.interface.nc.temporal import NcTemporalDimension, \
    get_origin_datetime_from_months_units, get_datetime_from_months_time_units, \
    get_difference_in_months, get_num_from_months_time_units
import datetime
import numpy as np


class TestNcTemporalDimension(TestBase):
    def test_360_day_calendar(self):
        months = range(1, 13)
        days = range(1, 31)
        vec = []
        for month in months:
            for day in days:
                vec.append(nc.netcdftime.datetime(2000, month, day))
        num = nc.date2num(vec, 'days since 1900-01-01', calendar='360_day')
        td = NcTemporalDimension(value=num, calendar='360_day', units='days since 1900-01-01')
        self.assertNumpyAll(np.array(vec), td.value_datetime)

    def test_get_origin_datetime_from_months_units(self):
        units = "months since 1978-12"
        self.assertEqual(get_origin_datetime_from_months_units(units), datetime.datetime(1978, 12, 1))
        units = "months since 1979-1-1 0"
        self.assertEqual(get_origin_datetime_from_months_units(units), datetime.datetime(1979, 1, 1))

    def test_get_datetime_from_months_time_units(self):
        units = "months since 1978-12"
        vec = range(0, 36)
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

    def test_get_difference_in_months(self):
        distance = get_difference_in_months(datetime.datetime(1978, 12, 1), datetime.datetime(1979, 3, 1))
        self.assertEqual(distance, 3)
        distance = get_difference_in_months(datetime.datetime(1978, 12, 1), datetime.datetime(1978, 7, 1))
        self.assertEqual(distance, -5)
        distance = get_difference_in_months(datetime.datetime(1978, 12, 1), datetime.datetime(1978, 12, 1))
        self.assertEqual(distance, 0)

    def test_get_num_from_months_time_units_1d_array(self):
        units = "months since 1978-12"
        vec = range(0, 36)
        datetimes = get_datetime_from_months_time_units(vec, units)
        num = get_num_from_months_time_units(datetimes, units, dtype=np.int32)
        self.assertNumpyAll(num, np.array(vec,dtype=np.int32))
        self.assertEqual(num.dtype, np.int32)

    def test_months_in_time_units_are_bad_netcdftime(self):
        units = "months since 1978-12"
        vec = range(0, 36)
        calendar = "standard"
        with self.assertRaises(ValueError):
            nc.num2date(vec, units, calendar=calendar)

    def test_months_in_time_units(self):
        units = "months since 1978-12"
        vec = range(0, 36)
        datetimes = get_datetime_from_months_time_units(vec, units)
        td = NcTemporalDimension(value=vec, units=units, calendar='standard')
        self.assertTrue(td._has_months_units)
        self.assertNumpyAll(td.value_datetime, datetimes)

    def test_months_in_time_units_between(self):
        units = "months since 1978-12"
        vec = range(0, 36)
        datetimes = get_datetime_from_months_time_units(vec, units)
        td = NcTemporalDimension(value=vec, units=units, calendar='standard')
        ret = td.get_between(datetimes[0], datetimes[3])
        self.assertNumpyAll(ret.value, np.array([0, 1, 2, 3]))

    def test_months_not_in_time_units(self):
        units = "days since 1900-01-01"
        value = np.array([31])
        td = NcTemporalDimension(value=value, units=units, calendar='standard')
        self.assertFalse(td._has_months_units)
