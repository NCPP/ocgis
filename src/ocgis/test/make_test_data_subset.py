import subprocess
import netCDF4 as nc
from ocgis.test.base import TestBase
import tempfile
import os
import numpy as np


class SingleYearFile(Exception):
    pass


def subset_first_two_years(in_nc,out_nc):
    ds = nc.Dataset(in_nc,'r')
    try:
        tvar = ds.variables['time']
        dts = nc.num2date(tvar[:],tvar.units,calendar=tvar.calendar)
        years = np.array([dt.year for dt in dts.flat])
        years_to_use = np.unique(years)[0:2]
        try:
            idx_time = np.logical_or(years == years_to_use[0],years == years_to_use[1])
        ## likely a single year in the file
        except IndexError:
            if years_to_use.shape[0] == 1:
                raise(SingleYearFile)
            else:
                raise
        dts_float = tvar[:][idx_time]
        start_float,end_float = dts_float[0],dts_float[-1]
        subprocess.check_call(['ncea','-O','-F','-d','time,{0},{1}'.format(start_float,end_float),in_nc,out_nc])
    finally:
        ds.close()
    
#def test_subset_first_two_years():
#    tdata = TestBase.get_tdata()
#    rd = tdata.get_rd('cancm4_tas')
#    f,out_nc = tempfile.mkstemp(suffix='_test_nc.nc')
#    try:
#        subset_first_two_years(rd.uri,out_nc)
#        ds = nc.Dataset(out_nc,'r')
#        try:
#            tvar = ds.variables['time']
#            dts = nc.num2date(tvar[:],tvar.units,calendar=tvar.calendar)
#            uyears = np.unique([dt.year for dt in dts.flat])
#            assert(uyears.shape[0] == 2)
#        finally:
#            ds.close()
#    finally:
#        os.remove(out_nc)
