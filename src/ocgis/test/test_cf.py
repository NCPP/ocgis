import unittest
import os
import netCDF4 as nc
from datetime import datetime
from ocgis.test.base import TestBase
import numpy as np
import ocgis
from ocgis.interface.nc.dataset import NcDataset


class Test(TestBase):

    def test_climatology(self):
        ## http://cf-pcmdi.llnl.gov/documents/cf-conventions/1.6/cf-conventions.html#idp5996336
        
        path = os.path.join(self._new_dir,'climatology.nc')
        ds = nc.Dataset(path,'w')
        try:
            dim_time = ds.createDimension('time',size=None)
            dim_bounds = ds.createDimension('bounds',size=2)
            dim_lat = ds.createDimension('lat',size=2)
            dim_lon = ds.createDimension('lon',size=2)
            
            var_lat = ds.createVariable('lat',float,dimensions=(dim_lat._name,))
            var_lat[:] = [43,42]
            var_lon = ds.createVariable('lon',float,dimensions=(dim_lon._name,))
            var_lon[:] = [-109,-108]
            
            dts = datetime(2000,6,16),datetime(2000,7,16),datetime(2000,8,16)
            dts_bounds = [[datetime(2000,6,1,6),datetime(2000,7,1,6)],
                          [datetime(2000,7,1,6),datetime(2000,8,1,6)],
                          [datetime(2000,8,1,6),datetime(2000,9,1,6)]]
            units = 'hours since 0000-00-00 00:00:00'
            calendar = 'standard'
            var_time = ds.createVariable('time',float,dimensions=(dim_time._name,))
            var_time.units = units
            var_time.calendar = calendar
            var_time.climatology = 'climatology_bounds'
            var_time[:] = nc.date2num(dts,units,calendar=calendar)
            var_cbounds = ds.createVariable('climatology_bounds',float,dimensions=(dim_time._name,dim_bounds._name))
            var_cbounds[:] = nc.date2num(dts_bounds,units,calendar=calendar)
            
            var_tas = ds.createVariable('tas',float,dimensions=(dim_time._name,dim_lat._name,dim_lon._name))
            var_tas[:] = np.random.rand(3,2,2)
        finally:
            ds.close()
        
        rd = ocgis.RequestDataset(path,'tas')
        ods = NcDataset(request_dataset=rd)
        self.assertNotEqual(ods.temporal.bounds,None)
        
        rd = ocgis.RequestDataset(path,'tas',time_region={'month':[8]})
        ret = ocgis.OcgOperations(dataset=rd).execute()
        self.assertEqual(ret[1].variables['tas'].temporal.bounds.shape,(1,2))
        self.assertEqual(ret[1].variables['tas'].temporal.value.shape,(1,))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()