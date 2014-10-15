import unittest
from ocgis.test.base import TestBase
import ocgis
from unittest.case import SkipTest
from ocgis import constants
import numpy as np


def longrunning(f):
    if constants.test_run_long_tests:
        ret = f
    else:
        def skip(*args):
            raise SkipTest("long-running test")
        skip.__name__ = f.__name__
        ret = skip
    return(ret)
        
    
def dev(f):
    if constants.test_run_dev_tests:
        ret = f
    else:
        def skip(*args):
            raise SkipTest("development-only test")
        skip.__name__ = f.__name__
        ret = skip
    return(ret)


class Test(TestBase):
    
    def test_assertNumpyAll_bad_mask(self):
        arr = np.ma.array([1,2,3],mask=[True,False,True])
        arr2 = np.ma.array([1,2,3],mask=[False,True,False])
        with self.assertRaises(AssertionError):
            self.assertNumpyAll(arr,arr2)
            
    def test_assertNumpyAll_type_differs(self):
        arr = np.ma.array([1,2,3],mask=[True,False,True])
        arr2 = np.array([1,2,3])
        with self.assertRaises(AssertionError):
            self.assertNumpyAll(arr,arr2)

    @dev
    def test_data_download(self):
        ocgis.env.DIR_TEST_DATA = self.current_dir_output
        rd1 = self.test_data_nc.get_rd('cancm4_tas')
        ocgis.env.reset()
        rd2 = self.test_data_nc.get_rd('cancm4_tas')
        self.assertEqual(rd1,rd2)
    
    @dev
    def test_multifile_data_download(self):
        ocgis.env.DIR_TEST_DATA = self.current_dir_output
        ocgis.env.DEBUG = True
        constants.test_data_download_url_prefix = 'https://dl.dropboxusercontent.com/u/867854/test_data_download/'
        rd = self.test_data_nc.get_rd('narccap_pr_wrfg_ncep')
    
    @dev
    def test_entirely_bad_location(self):
        ocgis.env.DIR_TEST_DATA = self.current_dir_output
        with self.assertRaises(ValueError):
            self.test_data_nc.get_rd('cancm4_tasmax_2011')
    
    @dev
    def test_copy_files(self):
        self.test_data_nc.copy_files('/home/local/WX/ben.koziol/htmp/transfer')
        
    def test_multifile(self):
        rd = self.test_data_nc.get_rd('narccap_pr_wrfg_ncep')
        self.assertEqual(len(rd.uri),2)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()