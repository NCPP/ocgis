import unittest
from collections import OrderedDict
import numpy as np
import datetime
from types import NoneType
import os.path
from shapely import wkb
from warnings import warn
from ocgis.test.misc import gen_descriptor_classes
from ocgis.util.helpers import get_shp_as_multi
from ocgis.api.interp.interpreter import Interpreter


NC_PATH = ('/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/python/'
           'OpenClimateGIS/bin/climate_data/cmip5/'
           'albisccp_cfDay_CCSM4_1pctCO2_r2i1p1_00200101-00391231.nc')
SHP_DIR = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/python/OpenClimateGIS/bin/shp'
SHPS = {1:'watersheds_4326.shp'}
META = {'uri':NC_PATH,
        'variable':'albisccp'}
OPTS = OrderedDict({
    'meta':[[META]],
    'backend':['ocg'],
    'time_range':[None,
                  [datetime.datetime(20,1,1),
                   datetime.datetime(20,2,28)]],
    'level_range':[None],
    'geom':[None,1],
    'output_format':['numpy','keyed','csv','shp','shpidx'],
    'output_grouping':[None],
    'spatial_operation':['intersects','clip'],
    'aggregate':[True,False],
    'calc_raw':[True,False],
    'calc_grouping':['month',None],
    'calc':[None,[{'func':'median'}]],
    'abstraction':['vector']})


def verify(desc,logical):
    ret = True
    for key,value in logical.iteritems():
        if type(value) in [bool,NoneType,str,int]:
            ret = desc[key] is value
        else:
            ret = value(desc[key])
        if ret is False: break
    return(ret)

def pause(cls):
    print('...{0} paused...'.format(cls.__name__))
    
def v_day_time_range(x):
    ret = True
    if x is None: ret = False
    if not np.all(x == np.array([datetime.datetime(2000,1,1),
                                 datetime.datetime(2000,1,1)])):
        ret = False
    return(ret)

def v_not_none(x):
    return(x is not None)


class TestSimpleBase(object):
    opts = None
    
    def gen_desc(self,opts=None):
        for desc in gen_descriptor_classes(niter=float('inf'),opts=opts):
            yield(desc)
            
    def load_geom(self,id,desc):
        try:
            path = os.path.join(SHP_DIR,SHPS[id])
            data = get_shp_as_multi(path,uid_field='id')
            ret = []
#            import ipdb;ipdb.set_trace()
            for ii in data:
                ii['id'] = ii.pop('id')
                ii['geom'] = wkb.loads(ii['geom'].wkb)
                if ii['geom'].is_valid:
                    ret.append(ii)
                else:
                    warn('not valid geometry found')
        except KeyError:
            ret = None
        desc['geom'] = ret
            
    def test(self):
        for desc in self.gen_desc(opts=self.opts):
            interp = Interpreter.get_interpreter(desc)
            for l in self.logicals:
                if verify(desc,l[0]):
                    self.load_geom(desc['geom'],desc)
                    ret = interp.execute()
                    l[1](desc,ret)

#@pause
class TestCmip(unittest.TestCase,TestSimpleBase):
    opts = OPTS
    
    def __init__(self,*args,**kwds):
        self.logicals = [
         [{'output_format':'csv',
           'time_range':v_not_none,
           'geom':v_not_none,
           'spatial_operation':'clip',
           'aggregate':True,
           'calc':v_not_none,
           'calc_raw':False,
           },self.basic_check]
                        ]
        
        super(TestCmip,self).__init__(*args,**kwds)
        
    def basic_check(self,desc,ret):
        print ret
        import ipdb;ipdb.set_trace()
        
if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestSimpleMultiCalc01.test']
    unittest.main()