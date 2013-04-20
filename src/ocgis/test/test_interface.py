import unittest
from ocgis.test.base import TestBase
from ocgis.interface.nc import NcRowDimension, NcColumnDimension,\
    NcSpatialDimension, NcGridDimension, NcPolygonDimension, NcGlobalInterface
import numpy as np
import netCDF4 as nc
from ocgis.util.helpers import make_poly


class TestNcInterface(TestBase):
    
    def test_row_dimension(self):
        value = np.arange(30,40,step=0.5)
        value = np.flipud(value).copy()
        bounds = np.empty((value.shape[0],2))
        bounds[:,0] = value + 0.25
        bounds[:,1] = value - 0.25
        
        ri = NcRowDimension(value=value)
        self.assertTrue(np.all(value == ri.value))
        self.assertEqual(ri.bounds,None)
        sri = ri.subset(35,38)
        self.assertEqual(len(sri.value),len(sri.uid))
        self.assertTrue(np.all(sri.value >= 35))
        self.assertTrue(np.all(sri.value <= 38))
        self.assertEqual(id(ri.value.base),id(sri.value.base))
        
        ri = NcRowDimension(value=value,bounds=bounds)
        self.assertTrue(np.all(bounds == ri.bounds))
        sri = ri.subset(30.80,38.70)
        self.assertTrue(np.all(sri.value >= 30.8))
        self.assertTrue(np.all(sri.value <= 38.7))
        
        with self.assertRaises(ValueError):
            NcRowDimension(bounds=bounds)
        
        ri = NcRowDimension(value=value)
        self.assertEqual(ri.extent,(value.min(),value.max()))
        
        ri = NcRowDimension(value=value,bounds=bounds)
        self.assertEqual(ri.extent,(bounds.min(),bounds.max()))
        self.assertTrue(np.all(ri.uid == np.arange(1,21)))
        self.assertEqual(ri.resolution,0.5)
        
    def test_spatial_dimension(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ds = nc.Dataset(rd.uri,'r')
        row_data = ds.variables['lat'][:]
        row_bounds = ds.variables['lat_bnds'][:]
        col_data = ds.variables['lon'][:]
        col_bounds = ds.variables['lon_bnds'][:]
        
        rd = NcRowDimension(value=row_data,bounds=row_bounds)
        cd = NcColumnDimension(value=col_data,bounds=col_bounds)
        gd = NcGridDimension(row=rd,column=cd)
        self.assertEqual(gd.resolution,2.8009135133922354)
        
        sd = NcSpatialDimension(row=rd,column=cd)
        
        sgd = gd.subset()
        self.assertEqual(sgd.shape,(rd.shape[0],cd.shape[0]))
        poly = make_poly((-62,59),(87,244))
        sgd = gd.subset(polygon=poly)
        self.assertEqual(sgd.uid.shape,(sgd.row.shape[0],sgd.column.shape[0]))
        self.assertTrue(sum(sgd.shape) < sum(gd.shape))
        lgd = gd[0:5,0:5]
        self.assertEqual(lgd.shape,(5,5))
        
        vd = NcPolygonDimension(gd)
        
        self.assertEqual(vd.geom.shape,vd.grid.shape)
        ivd = vd.intersects(poly)
        self.assertTrue(sum(ivd.geom.shape) < sum(vd.geom.shape))
        self.assertEqual(ivd.weights.max(),1.0)
        
        cvd = vd.clip(poly)
        self.assertEqual(ivd.shape,cvd.shape)
        self.assertFalse(ivd.weights.sum() == cvd.weights.sum())
        ds.close()
        
    def test_load(self):
        rd = self.test_data.get_rd('cancm4_tas')
        gi = NcGlobalInterface(request_dataset=rd)
        
        spatial = gi.spatial
        self.assertEqual(spatial.grid.shape,(64,128))
        
        level = gi.level
        import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()