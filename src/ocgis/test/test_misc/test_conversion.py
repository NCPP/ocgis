import unittest
import netCDF4 as nc
import os

import fiona

from ocgis.test.base import TestBase
import ocgis


class Test(TestBase):

    def test_nc_projection_writing(self):
        rd = self.test_data.get_rd('daymet_tmax')
        ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='nc')
        ret = ops.execute()
        ds = nc.Dataset(ret)
        self.assertTrue('lambert_conformal_conic' in ds.variables)

    def test_csv_shp(self):
        rd1 = self.test_data.get_rd('cancm4_tasmax_2011')
        rd2 = self.test_data.get_rd('maurer_bccr_1950')
        ops = ocgis.OcgOperations(dataset=[rd1, rd2], snippet=True, output_format='csv-shp', geom='state_boundaries',
                                  agg_selection=True, select_ugid=[32])
        ret = ops.execute()
        meta = os.path.join(os.path.split(ret)[0], 'ocgis_output_source_metadata.txt')

        with open(meta, 'r') as f:
            lines = f.readlines()
        self.assertTrue(len(lines) > 50)

    def test_csv_shp_custom_headers(self):
        rd1 = self.test_data.get_rd('cancm4_tasmax_2011')
        rd2 = self.test_data.get_rd('maurer_bccr_1950')
        headers = ['did', 'ugid', 'gid', 'alias', 'value', 'time']
        ops = ocgis.OcgOperations(dataset=[rd1, rd2], snippet=True, output_format='csv-shp', geom='state_boundaries',
                                  agg_selection=True, select_ugid=[32], headers=headers)
        ret = ops.execute()

        with open(ret, 'r') as f:
            line = f.readline()
        fheaders = [h.strip() for h in line.split(',')]
        self.assertEqual(fheaders, [h.upper() for h in headers])
        
    def test_shp_custom_headers(self):
        rd1 = self.test_data.get_rd('cancm4_tasmax_2011')
        rd2 = self.test_data.get_rd('maurer_bccr_1950')
        headers = ['did','ugid','gid','alias','value','time']
        ops = ocgis.OcgOperations(dataset=[rd1,rd2],snippet=True,output_format='shp',
                                  geom='state_boundaries',agg_selection=True,
                                  select_ugid=[32],headers=headers)
        ret = ops.execute()
        
        with fiona.open(ret) as f:
            self.assertEqual(f.meta['schema']['properties'].keys(),[h.upper() for h in headers])
        
    def test_meta(self):
        rd = self.test_data.get_rd('cancm4_tasmax_2011')
        ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='meta',
                                  geom='state_boundaries',agg_selection=True)
        ret = ops.execute()
        self.assertTrue(isinstance(ret,basestring))  
        
    def test_meta_with_source(self):
        rd = self.test_data.get_rd('cancm4_tasmax_2011')
        ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='csv',
                                  geom='state_boundaries',agg_selection=True)
        ret = ops.execute()
        with open(os.path.join(os.path.split(ret)[0],'ocgis_output_metadata.txt')) as f:
            lines = f.readlines()
        self.assertEqual(lines[3],'This is OpenClimateGIS-related metadata. Data-level metadata may be found in the file named: ocgis_output_source_metadata.txt\n')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()