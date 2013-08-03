import unittest
from ocgis.test.base import TestBase
import ocgis
import netCDF4 as nc
import os
from ocgis.util.helpers import ShpIterator
from ocgis.api.operations import OcgOperations
from collections import OrderedDict
import fiona


class Test(TestBase):

    def test_nc_projection_writing(self):
        rd = self.test_data.get_rd('daymet_tmax')
        ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='nc')
        ret = ops.execute()
        ds = nc.Dataset(ret)
        self.assertTrue('lambert_conformal_conic' in ds.variables)

    def test_csv_plus(self):
        rd1 = self.test_data.get_rd('cancm4_tasmax_2011')
        rd2 = self.test_data.get_rd('maurer_bccr_1950')
        ops = ocgis.OcgOperations(dataset=[rd1,rd2],snippet=True,output_format='csv+',
                                  geom='state_boundaries',agg_selection=True,
                                  select_ugid=[32])
        ret = ops.execute()
        meta = os.path.join(os.path.split(ret)[0],'ocgis_output_source_metadata.txt')
        
        with open(meta,'r') as f:
            lines = f.readlines()
        self.assertTrue(len(lines) > 199)
        
#        import subprocess
#        subprocess.call(['nautilus',os.path.split(ret)[0]])
#        import ipdb;ipdb.set_trace()

    def test_csv_plus_custom_headers(self):
        rd1 = self.test_data.get_rd('cancm4_tasmax_2011')
        rd2 = self.test_data.get_rd('maurer_bccr_1950')
        headers = ['alias','value','time']
        ops = ocgis.OcgOperations(dataset=[rd1,rd2],snippet=True,output_format='csv+',
                                  geom='state_boundaries',agg_selection=True,
                                  select_ugid=[32],headers=headers)
        ret = ops.execute()
        
        with open(ret,'r') as f:
            line = f.readline()
        fheaders = [h.strip() for h in line.split(',')]
        self.assertEqual(fheaders,[h.upper() for h in headers])
        
    def test_shp_custom_headers(self):
        rd1 = self.test_data.get_rd('cancm4_tasmax_2011')
        rd2 = self.test_data.get_rd('maurer_bccr_1950')
        headers = ['alias','value','time']
        ops = ocgis.OcgOperations(dataset=[rd1,rd2],snippet=True,output_format='shp',
                                  geom='state_boundaries',agg_selection=True,
                                  select_ugid=[32],headers=headers)
        ret = ops.execute()
        
        fields = ShpIterator(ret).get_fields()
        self.assertEqual(fields,[h.upper() for h in headers])
        
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

    def test_shp_with_frequency_duration(self):
        rd = self.test_data.get_rd('cancm4_tas',kwds={'time_region':{'year':[2003,2004]}})
        calc=[{'name': 'freq_duration', 'func': 'freq_duration', 'kwds': OrderedDict([('threshold', 250), ('operation', 'gte')])}]
        ops = OcgOperations(dataset=rd,calc_grouping=('year',),output_format="shp",select_ugid=(14, 16),
         aggregate=True,geom='state_boundaries',spatial_operation="clip",calc=calc,
         )
        ret = ops.execute()
        
        years = []
        ugid = []
        with fiona.open(ret,'r') as source:
            for feature in source:
                years.append(int(feature['properties']['YEAR']))
                ugid.append(int(feature['properties']['UGID']))
        self.assertEqual(set(years),set([2003,2004]))
        self.assertEqual(set(ugid),set([14,16]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()