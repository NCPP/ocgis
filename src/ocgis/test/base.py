import unittest
import abc
import tempfile
from ocgis import env, RequestDataset
import shutil
from copy import deepcopy, copy
import os
from collections import OrderedDict


class TestBase(unittest.TestCase):
    '''All tests should inherit from this. It allows test data to be written to
    a temporary folder and removed easily.'''
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,*args,**kwds):
        self.test_data = self.get_tdata()
        super(TestBase,self).__init__(*args,**kwds)
    
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
#        test_data.update(['CCSM4'],'albisccp','albisccp_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc',key='ccsm4')
#        test_data.update(['hostetler'],'TG','RegCM3_Daily_srm_GFDL.ncml.nc',key='hostetler')
        test_data.update(['maurer','2010'],'pr',['nldas_met_update.obs.daily.pr.1990.nc','nldas_met_update.obs.daily.pr.1991.nc'],key='maurer_2010_pr')
        test_data.update(['maurer','2010'],'tas',['nldas_met_update.obs.daily.tas.1990.nc','nldas_met_update.obs.daily.tas.1991.nc'],key='maurer_2010_tas')
        test_data.update(['maurer','2010'],'tasmin',['nldas_met_update.obs.daily.tasmin.1990.nc','nldas_met_update.obs.daily.tasmin.1991.nc'],key='maurer_2010_tasmin')
        test_data.update(['maurer','2010'],'tasmax',['nldas_met_update.obs.daily.tasmax.1990.nc','nldas_met_update.obs.daily.tasmax.1991.nc'],key='maurer_2010_tasmax')
        return(test_data)
    
    def setUp(self):
        env.reset()
        self._new_dir = tempfile.mkdtemp(prefix='ocgis_test_',dir=env.DIR_OUTPUT)
        env.DIR_OUTPUT = self._new_dir
        env.OVERWRITE = True
        
    def tearDown(self):
        try:
            shutil.rmtree(self._new_dir)
        finally:
            env.reset()
            
            
class TestData(OrderedDict):
    
    def copy_files(self,dest):
        if not os.path.exists(dest):
            raise(IOError('Copy destination does not exist: {0}'.format(dest)))
        for k,v in self.iteritems():
            uri = self.get_uri(k)
            rel_path = self.get_relative_path(k)
            dest_dir = os.path.join(dest,os.path.split(rel_path)[0])
            dst = os.path.join(dest_dir,v['filename'])
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            print('copying: {0}...'.format(dst))
            shutil.copy2(uri,dst)
        print('copy completed.')
            
    def get_relative_path(self,key):
        coll = deepcopy(self[key]['collection'])
        coll.append(self[key]['filename'])
        return(os.path.join(*coll))
    
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
        return(uri)
    
    def update(self,collection,variable,filename,key=None):
        OrderedDict.update(self,{key or filename:{'collection':collection,
         'filename':filename,'variable':variable}})
