import unittest
import abc
import tempfile
from ocgis import env
import shutil
from copy import deepcopy, copy
import os
from collections import OrderedDict
import subprocess
import ocgis
from warnings import warn
from subprocess import CalledProcessError
import numpy as np
from ocgis.api.request.base import RequestDataset
import netCDF4 as nc


class TestBase(unittest.TestCase):
    '''All tests should inherit from this. It allows test data to be written to
    a temporary folder and removed easily.'''
    __metaclass__ = abc.ABCMeta
    _reset_env = True
    _create_dir = True
    
    def __init__(self,*args,**kwds):
        self.test_data = self.get_tdata()
        super(TestBase,self).__init__(*args,**kwds)
        
    @property
    def _test_bin_dir(self):
        base_dir = os.path.split(__file__)[0]
        ret = os.path.join(base_dir,'bin')
        return(ret)
        
    def assertNumpyAll(self,arr1,arr2):
        return(self.assertTrue(np.all(arr1 == arr2)))
    
    def assertNumpyAllClose(self,arr1,arr2):
        return(self.assertTrue(np.allclose(arr1,arr2)))
    
    def assertNumpyNotAll(self,arr1,arr2):
        return(self.assertFalse(np.all(arr1 == arr2)))
    
    def assertDictEqual(self,d1,d2,msg=None):
        try:
            unittest.TestCase.assertDictEqual(self,d1,d2,msg=msg)
        except AssertionError:
            for k,v in d1.iteritems():
                self.assertEqual(v,d2[k])
            self.assertEqual(set(d1.keys()),set(d2.keys()))
    
    def assertNcEqual(self,uri_src,uri_dest,check_types=True):
        src = nc.Dataset(uri_src)
        dest = nc.Dataset(uri_dest)
        
        try:
            for dimname,dim in src.dimensions.iteritems():
                self.assertEqual(len(dim),len(dest.dimensions[dimname]))
            self.assertEqual(set(src.dimensions.keys()),set(dest.dimensions.keys()))
            
            for varname,var in src.variables.iteritems():
                dvar = dest.variables[varname]
                try:
                    self.assertNumpyAll(var[:],dvar[:])
                except AssertionError:
                    cmp = var[:] == dvar[:]
                    if cmp.shape == (1,) and cmp.data[0] == True:
                        pass
                    else:
                        raise
                if check_types:
                    self.assertEqual(var[:].dtype,dvar[:].dtype)
                for k,v in var.__dict__.iteritems():
                    self.assertNumpyAll(v,getattr(dvar,k))
                self.assertEqual(var.dimensions,dvar.dimensions)
            self.assertEqual(set(src.variables.keys()),set(dest.variables.keys()))
            
            self.assertDictEqual(src.__dict__,dest.__dict__)
        finally:
            src.close()
            dest.close()
    
    @staticmethod
    def get_tdata():
        test_data = TestData()
        test_data.update(['daymet'],'tmax','tmax.nc',key='daymet_tmax')
        test_data.update(['CanCM4'],'tas','tas_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',key='cancm4_tas')
        test_data.update(['CanCM4'],'tasmax','tasmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',key='cancm4_tasmax_2011')
        test_data.update(['CanCM4'],'tasmax','tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',key='cancm4_tasmax_2001')
        test_data.update(['CanCM4'],'tasmin','tasmin_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc',key='cancm4_tasmin_2001')
        test_data.update(['CanCM4'],'rhs','rhs_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',key='cancm4_rhs')
        test_data.update(['CanCM4'],'rhsmax','rhsmax_day_CanCM4_decadal2010_r2i1p1_20110101-20201231.nc',key='cancm4_rhsmax')
        test_data.update(['maurer','bccr'],'Prcp','bccr_bcm2_0.1.sresa1b.monthly.Prcp.1950.nc',key='maurer_bccr_1950')
        test_data.update(['narccap'],'pr','pr_CRCM_ccsm_1981010103.nc',key='narccap_crcm')
        test_data.update(['narccap'],'pr','pr_RCM3_gfdl_1981010103.nc',key='narccap_rcm3')
        test_data.update(['narccap'],'pr','pr_HRM3_gfdl_1981010103.nc',key='narccap_hrm3')
        test_data.update(['narccap'],'pr','pr_WRFG_ccsm_1986010103.nc',key='narccap_wrfg')
#        test_data.update(['CCSM4'],'albisccp','albisccp_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc',key='ccsm4')
#        test_data.update(['hostetler'],'TG','RegCM3_Daily_srm_GFDL.ncml.nc',key='hostetler')
        test_data.update(['maurer','2010'],'pr',['nldas_met_update.obs.daily.pr.1990.nc','nldas_met_update.obs.daily.pr.1991.nc'],key='maurer_2010_pr')
        test_data.update(['maurer','2010'],'tas',['nldas_met_update.obs.daily.tas.1990.nc','nldas_met_update.obs.daily.tas.1991.nc'],key='maurer_2010_tas')
        test_data.update(['maurer','2010'],'tasmin',['nldas_met_update.obs.daily.tasmin.1990.nc','nldas_met_update.obs.daily.tasmin.1991.nc'],key='maurer_2010_tasmin')
        test_data.update(['maurer','2010'],'tasmax',['nldas_met_update.obs.daily.tasmax.1990.nc','nldas_met_update.obs.daily.tasmax.1991.nc'],key='maurer_2010_tasmax')
        test_data.update(['narccap'],'pr',['pr_WRFG_ncep_1981010103.nc','pr_WRFG_ncep_1986010103.nc'],key='narccap_pr_wrfg_ncep')
        test_data.update(['narccap'],'tas','tas_HRM3_gfdl_1981010103.nc',key='narccap_rotated_pole')
        test_data.update(['narccap'],'pr','pr_WRFG_ccsm_1986010103.nc',key='narccap_lambert_conformal')
        test_data.update(['narccap'],'tas','tas_RCM3_gfdl_1981010103.nc',key='narccap_tas_rcm3_gfdl')
        test_data.update(['snippets'],'dtr','snippet_Maurer02new_OBS_dtr_daily.1971-2000.nc',key='snippet_maurer_dtr')
        test_data.update(['CMIP3'],'Tavg','Extraction_Tavg.nc',key='cmip3_extraction') 
               
        test_data.update(['misc','subset_test'],'Tavg','Tavg_bccr_bcm2_0.1.sresa2.nc',key='subset_test_Tavg')
        test_data.update(['misc','subset_test'],'Tavg','sresa2.bccr_bcm2_0.1.monthly.Tavg.RAW.1950-2099.nc',key='subset_test_Tavg_sresa2')
        test_data.update(['misc','subset_test'],'Prcp','sresa2.ncar_pcm1.3.monthly.Prcp.RAW.1950-2099.nc',key='subset_test_Prcp')
        return(test_data)
    
    def setUp(self):
        if self._reset_env: env.reset()
        if self._create_dir:
            self._test_dir = tempfile.mkdtemp(prefix='ocgis_test_')
            env.DIR_OUTPUT = self._test_dir
        else:
            self._create_dir = None
        
    def tearDown(self):
        try:
            if self._create_dir: shutil.rmtree(self._test_dir)
        finally:
            if self._reset_env: env.reset()
            
            
class TestData(OrderedDict):
    
    def copy_files(self,dest):
        if not os.path.exists(dest):
            raise(IOError('Copy destination does not exist: {0}'.format(dest)))
        for k,v in self.iteritems():
            uri = self.get_uri(k)
            if isinstance(uri,basestring):
                to_copy = [uri]
            else:
                to_copy = uri
            for to_copy_uri in to_copy:
                dest_dir = os.path.join(*([dest] + v['collection']))
                dst = os.path.join(dest_dir,os.path.split(to_copy_uri)[1])
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)
                print('copying: {0}...'.format(dst))
                shutil.copy2(to_copy_uri,dst)
        print('copy completed.')
        
    def copy_file(self,key,dest):
        src = self.get_uri(key)
        dest = os.path.join(dest,self[key]['filename'])
        shutil.copy2(src,dest)
        return(dest)
    
    def get_rd(self,key,kwds=None):
        ref = self[key]
        if kwds is None:
            kwds = {}
        kwds.update({'uri':self.get_uri(key),'variable':ref['variable']})
        rd = RequestDataset(**kwds)
        return(rd)
    
    def get_uri(self,key):
        ref = self[key]
        coll = deepcopy(ref['collection'])
        if env.DIR_TEST_DATA is None:
            raise(ValueError('The TestDataset object requires env.DIR_TEST_DATA have a path value.'))
        coll.insert(0,env.DIR_TEST_DATA)
        ## determine if the filename is a string or a sequence of paths
        filename = ref['filename']
        if isinstance(filename,basestring):
            coll.append(filename)
            uri = os.path.join(*coll)
        else:
            uri = []
            for part in filename:
                copy_coll = copy(coll)
                copy_coll.append(part)
                uri.append(os.path.join(*copy_coll))
        ## ensure the uris exist, if not, we may need to download
        try:
            if isinstance(uri,basestring):
                assert(os.path.exists(uri))
            else:
                for element in uri:
                    assert(os.path.exists(element))
        except AssertionError:
            if isinstance(uri,basestring):
                download_uris = [uri]
            else:
                download_uris = uri
            try:
                os.makedirs(env.DIR_TEST_DATA)
            except OSError:
                if os.path.exists(env.DIR_TEST_DATA):
                    warn('Target download location exists. Files will be written to the existing location: {0}'.format(env.DIR_TEST_DATA))
                else:
                    raise
            for download_uri in download_uris:
                wget_url = ocgis.constants.test_data_download_url_prefix + '/'.join(ref['collection']) + '/' + os.path.split(download_uri)[1]
                wget_dest = os.path.join(*([env.DIR_TEST_DATA] + ref['collection'] + [download_uri]))
                try:
                    os.makedirs(os.path.split(wget_dest)[0])
                except OSError:
                    if os.path.exists(os.path.split(wget_dest)[0]):
                        warn('Data download directory exists: {0}'.format(os.path.split(wget_dest)[0]))
                    else:
                        raise
                try:
                    if env.DEBUG:
                        cmd = ['wget','-O',wget_dest,wget_url]
                    else:
                        cmd = ['wget','--quiet','-O',wget_dest,wget_url]
                    subprocess.check_call(cmd)
                except CalledProcessError:
                    raise(ValueError('"wget" was unable to fetch the test data URL ({0}) to the destination location: {1}. The command list was: {2}'.format(wget_url,wget_dest,cmd)))
        return(uri)
    
    def update(self,collection,variable,filename,key=None):
        OrderedDict.update(self,{key or filename:{'collection':collection,
         'filename':filename,'variable':variable}})
