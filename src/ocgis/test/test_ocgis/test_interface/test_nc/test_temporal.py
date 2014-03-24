import netCDF4 as nc
from ocgis.test.base import TestBase
from ocgis.interface.nc.temporal import NcTemporalDimension
import numpy as np


class TestNcTemporalDimension(TestBase):
    
    def test_360_day_calendar(self):
        months = range(1,13)
        days = range(1,31)
        vec = []
        for month in months:
            for day in days:
                vec.append(nc.netcdftime.datetime(2000,month,day))
        num = nc.date2num(vec,'days since 1900-01-01', calendar='360_day')
        td = NcTemporalDimension(value=num,calendar='360_day',units='days since 1900-01-01')
        self.assertNumpyAll(np.array(vec),td.value_datetime)
