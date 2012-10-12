import unittest
import os.path
import netCDF4 as nc
from ocg.meta.interface.interface import GlobalInterface
from ocg.api.interp.iocg.dataset import OcgDataset
from ocg.conv.csv_ import CsvConverter
import numpy as np
from ocg.conv.shp import ShpConverter
from ocg.conv.shpidx import ShpIdxConverter
from ocg.conv.numpy_ import NumpyConverter
from ocg.util.helpers import ShpIterator, get_shp_as_multi
from collections import OrderedDict
import datetime
from ocg.test.misc import gen_descriptor_classes
from ocg.api.interp.interpreter import Interpreter


DIR = ('/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/python/'
       'OpenClimateGIS/bin/climate_data/hostetler')
NCS = [
       ['RegCM3_Daily_srm_GFDL.ncml.nc','TG'],
       ['RegCM3_Daily_srm_NCEP.ncml.nc','RNFS']
      ]
SHP = ('/home/local/WX/ben.koziol/Dropbox/nesii/project/'
       'ocg/python/OpenClimateGIS/bin/shp/hostetler_aoi.shp')

def get_meta():
    meta = []
    for n in NCS:
        path = os.path.join(DIR,n[0])
        meta.append({'uri':path,'variable':n[1]})
    return([meta])

def get_geom():
    y = [None]
    y.append(get_shp_as_multi(SHP,'id'))
    return(y)
    
OPTS = OrderedDict({
    'meta':get_meta(),
    'mode':['ocg'],
    'time_range':[[datetime.datetime(1968,1,17),
                   datetime.datetime(1968,1,17)]],
    'level_range':[None],
    'geom':get_geom(),
    'output_format':['shp'],
    'output_grouping':[None],
    'spatial_operation':['intersects','clip'],
    'aggregate':[True,False],
    'calc_raw':[None],
    'calc_grouping':[None],
    'calc':[None]})


class TestHostetler(unittest.TestCase):
    
    def gen_nc(self):
        for n in NCS:
            yield(os.path.join(DIR,n[0]),n[1])
            
    def pass_test_subset(self):
        for uri,var in self.gen_nc():
            ods = OcgDataset(uri)
            time_range = [ods.i.temporal.time.value[0],
                          ods.i.temporal.time.value[0]]
            coll = ods.subset(var,time_range=time_range)
            coll['vid'] = np.array(1)
            coll['value'] = {var:coll['value']}
            conv = NumpyConverter(coll,None,ods,base_name='ocg',wd=None,
                                cengine=None,write_attr=False,write_agg=False)
            ret = conv.write()
            import ipdb;ipdb.set_trace()
            
    def test_interpreter(self):
        for desc in gen_descriptor_classes(niter=float('inf'),opts=OPTS):
            interp = Interpreter.get_interpreter(desc)
            if desc['aggregate'] is False and desc['geom'] is not None:
                ret = interp.execute()
                import ipdb;ipdb.set_trace()
    
#    def test_interface(self):
#        for n in self.gen_nc():
#            ds = nc.Dataset(n,'r')
#            ii = GlobalInterface(ds)
#            import ipdb;ipdb.set_trace()
        
        
if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestSimpleMultiCalc01.test']
    unittest.main()