import subprocess
import netCDF4 as nc
from ocgis.test.base import TestBase
import tempfile
import os
import numpy as np
from nose.plugins.skip import SkipTest
from argparse import ArgumentParser


class SingleYearFile(Exception):
    pass


def subset(in_nc,out_nc,n_years=None):
    ds = nc.Dataset(in_nc,'r')
    try:
        tvar = ds.variables['time']
        if n_years is not None:
            dts = nc.num2date(tvar[:],tvar.units,calendar=tvar.calendar)
            years = np.array([dt.year for dt in dts.flat])
            years_to_use = np.unique(years)
            years_to_use.sort()
            years_to_use = years_to_use[0:n_years]
            try:
                idx_time = np.zeros(years.shape,dtype=bool)
                for year in years_to_use.flat:
                    idx_time = np.logical_or(idx_time,years == year)
            ## likely a single year in the file
            except IndexError:
                if years_to_use.shape[0] == 1:
                    raise(SingleYearFile)
                else:
                    raise
        else:
            idx_time = 0
        dts_float = tvar[:][idx_time]
        try:
            start_float,end_float = dts_float[0],dts_float[-1]
        except IndexError:
            ## assume scalar variable
            start_float,end_float = dts_float,dts_float
        subprocess.check_call(['ncea','-O','-F','-d','time,{0},{1}'.format(start_float,end_float),in_nc,out_nc])
    finally:
        ds.close()
    
def test_subset_years_two_years():
    raise(SkipTest('dev'))
    tdata = TestBase.get_tdata()
    rd = tdata.get_rd('cancm4_tas')
    f,out_nc = tempfile.mkstemp(suffix='_test_nc.nc')
    try:
        subset(rd.uri,out_nc,2)
        ds = nc.Dataset(out_nc,'r')
        try:
            tvar = ds.variables['time']
            dts = nc.num2date(tvar[:],tvar.units,calendar=tvar.calendar)
            uyears = np.unique([dt.year for dt in dts.flat])
            assert(uyears.shape[0] == 2)
        finally:
            ds.close()
    finally:
        os.remove(out_nc)

def test_subset_years_one_year():
    raise(SkipTest('dev'))
    tdata = TestBase.get_tdata()
    rd = tdata.get_rd('cancm4_tas')
    f,out_nc = tempfile.mkstemp(suffix='_test_nc.nc')
    try:
        subset(rd.uri,out_nc,1)
        ds = nc.Dataset(out_nc,'r')
        try:
            tvar = ds.variables['time']
            dts = nc.num2date(tvar[:],tvar.units,calendar=tvar.calendar)
            uyears = np.unique([dt.year for dt in dts.flat])
            assert(uyears.shape[0] == 1)
        finally:
            ds.close()
    finally:
        os.remove(out_nc)
        
        
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('in_nc')
    parser.add_argument('out_nc')
    parser.add_argument('-y','--years',type=int,default=None)
    
    pargs = parser.parse_args()
    
    subset(pargs.in_nc,pargs.out_nc,n_years=pargs.years)