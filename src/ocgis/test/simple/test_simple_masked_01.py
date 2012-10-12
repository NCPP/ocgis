import unittest
from shapely.geometry.polygon import Polygon
from collections import OrderedDict
from ocg.test.misc import gen_descriptor_classes
from ocg.api.interp.interpreter import Interpreter
import numpy as np
import datetime
from types import NoneType


import make_simple_masked_01
NC_PATH = '/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/python/OpenClimateGIS/src/ocg/test/bin/test_simple_spatial_masked_01.nc'
TGEOMS = [{'id':1,'geom':Polygon(((-105.5,40.5),(-103.5,40.5),(-103.5,38.5),(-105.5,38.5)))},
          {'id':2,'geom':Polygon(((-103.5,40.5),(-101.5,40.5),(-101.5,38.5),(-103.5,38.5)))},
          {'id':3,'geom':Polygon(((-105.5,38.5),(-103.5,38.5),(-103.5,36.5),(-105.5,36.5)))},
          {'id':4,'geom':Polygon(((-103.5,38.5),(-101.5,38.5),(-101.5,36.5),(-103.5,36.5)))},]
MGEOM = [{'id':100,'geom':Polygon(((-104.5,39.5),(-102.5,39.5),(-102.5,37.5),(-104.5,37.5)))}]
TINY = [{'id':2000,'geom':Polygon(((-103.75,38.75),(-103.25,38.75),(-103.25,38.25),(-103.75,38.25)))}]
META = {'uri':NC_PATH,
         'spatial_row_bounds':'bounds_latitude',
         'spatial_col_bounds':'bounds_longitude',
         'calendar':'proleptic_gregorian',
         'time_units':'days since 2000-01-01 00:00:00',
         'time_name':'time',
         'level_name':'level',
         'variable':'foo'}
OPTS = OrderedDict({
    'meta':[[META]],
    'mode':['ocg'],
    'time_range':[None,
                  [datetime.datetime(2000,1,1),
                   datetime.datetime(2000,1,1)]],
    'level_range':[[1],[1,2]],
    'geom':[None,TGEOMS,MGEOM,TINY],
    'output_format':['numpy','csv','shp'],
    'output_grouping':[None],
    'spatial_operation':['intersects','clip'],
    'aggregate':[True,False],
    'calc_raw':[True,False],
    'calc_grouping':[None,'month','year'],
    'calc':[None,
            [{'func':'median' ,'name':'my_median'},
             {'func':'mean'   ,'name':'the_mean' },
             {'func':'between','kwds':{'lower':2,'upper':8}}]],
    'abstraction':['vector']})
#OPTS_CALC = OPTS.copy()
#OPTS_CALC.update(
#    {'time_range':[None,
#                   [datetime.datetime(2000,1,1),
#                    datetime.datetime(2000,1,1)]],
#     'geom':[None],
#     'calc_raw':[True,False],
#     'calc_grouping':['day','month','year',['month','year']],
#     'calc':[[{'func':'median' ,'name':'my_median'},
#              {'func':'mean'   ,'name':'the_mean' },
#              {'func':'between','kwds':{'lower':2,'upper':8}}]]})
#OPTS_MULTI = OPTS.copy()
#META2 = META.copy()
#META2['variable'] = 'foo2'
#OPTS_MULTI.update(
#    {
#     'meta':[[META,META2]],
#     'output_format':['numpy','csv','shp'],
#     }
#                  )
#OPTS_MULTI_CALC = OPTS_MULTI.copy()
#OPTS_MULTI_CALC.update(
#    {
#     'calc':[[{'func':'foomulti','name':'my_multi'}]],
#     'calc_grouping':['day','month','year',['month','year']]
#     })


def verify(desc,logical):
    ret = True
    for key,value in logical.iteritems():
        if type(value) in [bool,NoneType,str]:
            ret = desc[key] is value
        else:
            ret = value(desc[key])
        if ret is False: break
    return(ret)

def pause(cls):
    print('...{0} paused...'.format(cls.__name__))
    
def v_year_time_range(x):
    ret = True
    if x is None: ret = False
    if not np.all(x == np.array([datetime.datetime(2000,1,1),
                                 datetime.datetime(2000,1,1)])):
        ret = False
    return(ret)


class TestSimpleBase(object):
    opts = None
    run_all = False
    
    def gen_desc(self,opts=None):
        for desc in gen_descriptor_classes(niter=float('inf'),opts=opts):
            yield(desc)
            
    def test(self):
        def _exec(desc):
            interp = Interpreter.get_interpreter(desc)
            ret = interp.execute()
            return(ret)
        
        for desc in self.gen_desc(opts=self.opts):
            ret = None
            if self.run_all:
                ret = _exec(desc)
            for l in self.logicals:
                if verify(desc,l[0]):
                    if ret is None:
                        ret = _exec(desc)
                    l[1](desc,ret)

#@pause
class TestSimpleMasked01(unittest.TestCase,TestSimpleBase):
    opts = OPTS
    run_all = False
    
    def __init__(self,*args,**kwds):
        self.logicals = [
         [{'output_format':'numpy',
           'geom':None,
           'calc':None},self.basic_return],
         [{'output_format':'numpy',
           'geom':None,
           'calc':lambda x: x is not None,
           'time_range':v_year_time_range,
           'calc_raw':True},self.calc_raw_return],
         [{'output_format':'numpy',
           'geom':None,
           'calc':lambda x: x is not None,
           'time_range':v_year_time_range,
           'aggregate':True,
           'level_range':lambda x: len(x) == 2},self.aggregate],
                        ]
        
        super(TestSimpleMasked01,self).__init__(*args,**kwds)
        
    def basic_return(self,desc,ret):
        for value in ret[1]['coll']['value'].itervalues():
            self.assertEqual(4,value.mask[0,0].sum())
            
    def calc_raw_return(self,desc,ret):
        ref = ret[1]['coll']['attr']['foo']
        if desc['level_range'] == [1]:
            self.assertEqual(12,ref['n'].sum())
        else:
            self.assertEqual(24,ref['n'].sum())
            
    def aggregate(self,desc,ret):
        ref = ret[1]['coll']['attr']['foo']['n']
        if desc['calc_raw']:
            self.assertEqual(24,ref.sum())
        else:
            self.assertEqual(2,ref.sum())
          

if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestSimpleMultiCalc01.test']
    unittest.main()