import unittest
from ocgis.test.base import TestBase
import ocgis
from unittest.case import SkipTest


class Test(TestBase):
    
    def setUp(self):
        raise(SkipTest('tests only for development purposes'))

    def test_data_download(self):
        ocgis.env.DIR_TEST_DATA = self._test_dir
        rd1 = self.test_data.get_rd('cancm4_tas')
        ocgis.env.reset()
        rd2 = self.test_data.get_rd('cancm4_tas')
        self.assertEqual(rd1,rd2)
    
    def test_entirely_bad_location(self):
        ocgis.env.DIR_TEST_DATA = self._test_dir
        with self.assertRaises(ValueError):
            self.test_data.get_rd('cancm4_tasmax_2011')
            
    def test_copy_files(self):
#        self.test_data.copy_files(self._test_dir)
        self.test_data.copy_files('/home/local/WX/ben.koziol/htmp/transfer')
        
    def test_multifile(self):
        rd = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        self.assertEqual(len(rd.uri),2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()