import unittest
from shapely.geometry.polygon import Polygon
from collections import OrderedDict
from ocgis.test.misc import gen_descriptor_classes
from ocgis.api.interp.interpreter import Interpreter
import numpy as np
import datetime
from types import NoneType
import make_simple_01
import os.path


NC_PATH = os.path.join(os.path.split(__file__)[0],make_simple_01.OUTNAME)
TGEOMS = [{'id':1,'geom':Polygon(((-105.5,40.5),(-103.5,40.5),(-103.5,38.5),(-105.5,38.5)))},
          {'id':2,'geom':Polygon(((-103.5,40.5),(-101.5,40.5),(-101.5,38.5),(-103.5,38.5)))},
          {'id':3,'geom':Polygon(((-105.5,38.5),(-103.5,38.5),(-103.5,36.5),(-105.5,36.5)))},
          {'id':4,'geom':Polygon(((-103.5,38.5),(-101.5,38.5),(-101.5,36.5),(-103.5,36.5)))},]
MGEOM = [{'id':100,'geom':Polygon(((-104.5,39.5),(-102.5,39.5),(-102.5,37.5),(-104.5,37.5)))}]
TINY = [{'id':2000,'geom':Polygon(((-103.75,38.75),(-103.25,38.75),(-103.25,38.25),(-103.75,38.25)))}]
META = {'uri':NC_PATH,
        'variable':'foo'}
META2 = {'uri':NC_PATH,
         'variable':'foo2'}
OPTS = OrderedDict({
    'meta':[[META],[META,META2]],
    'backend':['ocg'],
    'time_range':[[datetime.datetime(2000,1,1),
                   datetime.datetime(2000,1,1)],
                  None],
    'level_range':[None,[1]],
    'geom':[None,TGEOMS,MGEOM,TINY],
    'output_format':['keyed','numpy','shpidx','shp','csv','nc'],
    'output_grouping':[None],
    'spatial_operation':['intersects','clip'],
    'aggregate':[False,True],
    'calc_raw':[True,False],
    'calc_grouping':['day','month','year',['month','year']],
    'calc':[None,
            [{'func':'median'  ,'name':'my_median'},
             {'func':'mean'    ,'name':'the_mean' },
             {'func':'between' ,'kwds':{'lower':2,'upper':8}}],
            [{'func':'foomulti','name':'my_multi'}]],
                   })

################ MISC FUNCTIONS ################################################

def pause(cls):
    print('...{0} paused...'.format(cls.__name__))

def add_to(logicals,dct):
    for l in logicals:
        l[0].update(dct)

################################################################################

################ VALIDATOR FUNCTIONS ###########################################

def v_day_time_range(x):
    ret = True
    if x is None: ret = False
    if not np.all(x == np.array([datetime.datetime(2000,1,1),
                                 datetime.datetime(2000,1,1)])):
        ret = False
    return(ret)

def v_not_none(x):
    return(x is not None)

def v_func_name(x,name):
    ret = False
    for d in x:
        if d['func'] == name:
            ret = True
            break
    return(ret)

def v_not_func_name(x,name):
    return(not v_func_name(x,name))

def v_not_multi(x):
    if x is None:
        ret = True
    else:
        ret = True
        for d in x:
            if d['func'] == 'foomulti':
                ret = False
                break
    return(ret)

def v_is_multi(x):
    if x is None:
        ret = False
    else:
        ret = False
        for d in x:
            if d['func'] == 'foomulti':
                ret = True
                break
    return(ret)

def v_not_none_or_multi(x):
    if all([v_not_none(x),v_not_multi(x)]):
        ret = True
    else:
        ret = False
    return(ret)

def v_not_none_len_4(x):
    ret = False
    if x is not None and len(x) == 4:
        ret = True
    return(ret)

################################################################################

################ LOGICALS ######################################################

SIMPLE_SPATIAL = [
         [{'time_range':v_day_time_range,
           'output_format':'numpy',
           'calc':None,
           'meta':lambda x: len(x)==1,
           },
          'ss_geom_check']
                   ]

SIMPLE_CALC = [
         [{'time_range':v_day_time_range},'sc_date_range'],
         [{'geom':None,'calc_raw':True,'time_range':v_day_time_range},
          'sc_calc_raw_true'],
         [{'geom':None,'calc_raw':False,
           'aggregate':False,'time_range':v_day_time_range},
          'sc_calc_raw_false'],
         [{'geom':None,'aggregate':True,'time_range':v_day_time_range},
          'sc_aggregate_true'],
         [{'geom':None,'time_range':None},
          'sc_grouping'],
         [{'time_range':None,
           'geom':None,
           'spatial_operation':'intersects',
           'aggregate':False},'sc_grid_calc'],
                        ]
add_to(SIMPLE_CALC,{'output_format':'numpy'})
add_to(SIMPLE_CALC,{'calc':v_not_none_or_multi})
add_to(SIMPLE_CALC,{'meta':lambda x: len(x) == 1})

SIMPLE_MULTI = [
         [{},'sm_multi_check'],
         [{'time_range':None,
           'geom':None,
           'calc_grouping':'month'},'sm_multi_calc_check_aggregate']
               ]
add_to(SIMPLE_MULTI,{'calc':v_is_multi})
add_to(SIMPLE_MULTI,{'output_format':'numpy'})
add_to(SIMPLE_MULTI,{'meta':lambda x: len(x) == 2})

SIMPLE_OUTPUT = [
#         [{'output_format':'nc',
#           'geom':None,
#           'time_range':v_day_time_range,
#           'calc':None,
#           'spatial_operation':'intersects',
#           'aggregate':False},'so_nc_out'],
         [{'output_format':'numpy',
           'calc':v_is_multi,
           'meta':lambda x: len(x) == 2,
           'time_range':v_day_time_range,
           },'so_out']
                 ]

SIMPLE_ITER = [
               [{'output_format':'keyed',
                 'calc':v_not_none_or_multi,
                 'time_range':None,
                 'meta':lambda x: len(x) == 2,
                 'aggregate':True,
                 'geom':None,
                 'calc_grouping':'month',
                 'level_range':None
                 },'si_basic']
               ]

################################################################################

class TestSimpleBase(object):
    opts = OPTS
    logicals = [
                SIMPLE_SPATIAL,
                SIMPLE_CALC,
#                SIMPLE_MULTI,
#                SIMPLE_OUTPUT,
#                SIMPLE_ITER
                ]
    
    @property
    def lmod(self):
        if self.desc['level_range'] is None:
            ret = 2
        elif len(self.desc['level_range']) > 1:
            ref = self.desc['level_range']
            ret = (ref[1] - ref[0]) + 1
        else:
            ret = 1
        return(ret)
    
    def verify(self,desc,logical):
        ret = True
        for key,value in logical.iteritems():
            if type(value) in [bool,NoneType,str]:
                ret = desc[key] is value
            else:
                ret = value(desc[key])
            if ret is False: break
        return(ret)

    def iter_logicals(self):
        for logical in self.logicals:
            for l in logical:
                yield(l)
    
    def gen_desc(self,opts=None):
        for desc in gen_descriptor_classes(niter=float('inf'),opts=opts):
            yield(desc)
            
    def test(self):
        for desc in self.gen_desc(opts=self.opts):
            self.desc = desc
            interp = Interpreter.get_interpreter(desc)
            for l in self.iter_logicals():
                if self.verify(desc,l[0]):
                    try:
                        self.ret = interp.execute()
                    except:
                        raise
                    getattr(self,l[1])()


class TestSimple(unittest.TestCase,TestSimpleBase):
    
    def ss_geom_check(self):
        if self.desc['geom'] is not None and len(self.desc['geom']) == 4:
            self.assertEqual(len(self.ret.keys()),4)
            ref = self.ret[4]['coll'].variables['foo']
            if self.desc['aggregate'] is True:
                for key,value in self.ret.iteritems():
                    self.assertEqual(float(key),
                                     value['coll'].variables['foo'].agg_value[0,0,0])
            else:
                for key,value in self.ret.iteritems():
                    self.assertEqual(value['coll'].geom.shape,(2,2))
#                    tdata = np.ones((1,1,4),dtype=float)*float(key)
#                    t = np.all(float(key) == tdata[0,0,:])
#                    self.assertTrue(t)
        if self.desc['geom'] is not None and self.desc['geom'][0]['id'] == 100:
            self.assertEqual(len(self.ret.keys()),1)
            if self.desc['spatial_operation'] == 'intersects':
                if self.desc['aggregate']:
                    self.assertEqual(self.ret[100]['coll'].variables['foo'].agg_value.shape,
                                     (1,1*self.lmod,1,1))
                    self.assertEqual(self.ret[100]['coll'].variables['foo'].agg_value[0,0,0],
                                     np.mean([1.,2.,3.,4.]))
                else:
                    self.assertEqual(self.ret[100]['coll'].variables['foo'].raw_value.shape,
                                     (1,1*self.lmod,2,2))
                    self.assertEqual(self.ret[100]['coll'].variables['foo'].raw_value.sum(),
                                     self.lmod*np.sum([1.,2.,3.,4.]))
            if self.desc['spatial_operation'] == 'clip':
                if self.desc['aggregate']:
                    self.assertEqual(self.ret[100]['coll'].variables['foo'].agg_value.shape,
                                     (1,1*self.lmod,1,1))
                    self.assertEqual(self.ret[100]['coll'].variables['foo'].agg_value[0,0,0],
                                     np.mean([1.,2.,3.,4.]))
                else:
                    self.assertEqual(self.ret[100]['coll'].variables['foo'].raw_value.shape,
                                     (1,1*self.lmod,2,2))
                    self.assertEqual(self.ret[100]['coll'].variables['foo'].raw_value.sum(),
                                     self.lmod*np.sum([1.,2.,3.,4.]))
        if self.desc['geom'] is not None and self.desc['geom'][0]['id'] == 2000:
            self.assertEqual(len(self.ret.keys()),1)
            if self.desc['spatial_operation'] == 'clip':
                orig = TINY[0]['geom']
                processed = self.ret[2000]['coll'].geom[0,0]
                if self.desc['aggregate']:
                    self.assertEqual(self.ret[2000]['coll'].variables['foo'].agg_value.shape,
                                     (1,1*self.lmod,1,1))
                    self.assertEqual(self.ret[2000]['coll'].variables['foo'].agg_value[0,0,0],
                                     np.sum([0.25*1.,0.25*2.,0.25*3.,0.25*4.]))
                    self.assertEqual(orig.area,
                                     processed.area)
                    self.assertEqual(orig.bounds,
                                     processed.bounds)
                else:
                    self.assertEqual(self.ret[2000]['coll'].variables['foo'].raw_value.shape,
                                     (1,1*self.lmod,2,2))
                    self.assertEqual(self.ret[2000]['coll'].variables['foo'].raw_value.sum(),
                                     self.lmod*np.sum([1.,2.,3.,4.]))
            if self.desc['spatial_operation'] == 'intersects':
                if self.desc['aggregate']:
                    self.assertEqual(self.ret[2000]['coll'].variables['foo'].agg_value.shape,
                                     (1,1*self.lmod,1,1))
                    self.assertEqual(self.ret[2000]['coll'].variables['foo'].agg_value[0,0,0],
                                     np.mean([1.,2.,3.,4.]))
                else:
                    self.assertEqual(self.ret[2000]['coll'].variables['foo'].raw_value.shape,
                                     (1,1*self.lmod,2,2))
                    self.assertEqual(self.ret[2000]['coll'].variables['foo'].raw_value.sum(),
                                     self.lmod*np.sum([1.,2.,3.,4.]))
        elif self.desc['geom'] is None:
            ref = self.ret[1]['coll'].variables['foo']
            self.assertEqual(len(self.ret.keys()),1)
            if self.desc['aggregate'] is True:
                self.assertEqual(ref.agg_value.shape,
                                 (1,1*self.lmod,1,1))
                self.assertEqual(ref.agg_value[0,0,0],
                                 np.mean([1.,2.,3.,4.]))
            else:
                self.assertEqual(ref.raw_value.shape,
                                 (1,1*self.lmod,4,4))
                self.assertEqual(ref.raw_value.sum(),
                                 self.lmod*np.sum([1.*4,2.*4,3*4.,4*4.]))
                
    def sc_grid_calc(self):
        chk = self.ret[1]['coll'].variables['foo'].calc_value
        gmap = {'day':31,
                'month':12,
                'year':2}
        for value in chk.itervalues():
            try:
                tgroup_check = gmap[self.desc['calc_grouping']]
            except TypeError:
                if self.desc['calc_grouping'] == ['month','year']:
                    tgroup_check = 24
            if self.desc['level_range'] is not None and len(self.desc['level_range']) == 1:
                lcheck = 1
            else:
                lcheck = 2
            self.assertEqual(value.shape,(tgroup_check,lcheck,4,4))
          
    def sc_grouping(self):
        ref = self.ret[1]['coll'].variables['foo'].calc_value['the_mean']
        if self.desc['calc_grouping'] == 'month':
            self.assertEqual(ref.shape[0],12)
        if self.desc['calc_grouping'] == 'year':
            self.assertEqual(ref.shape[0],2)
        if self.desc['calc_grouping'] == 'day':
            self.assertTrue(ref.shape[0] == 31)
        if 'month' in self.desc['calc_grouping'] and 'year' in self.desc['calc_grouping']:
            self.assertTrue(ref.shape[0] == 24)
                
    def sc_date_range(self):
        for value in self.ret.itervalues():
            timevec = value['coll'].timevec
            time_range = self.desc['time_range']
            self.assertTrue(time_range[0] <= timevec[0])
            self.assertTrue(time_range[1] >= timevec[-1])
                
    def sc_calc_raw_true(self):
        ref = self.ret[1]['coll'].variables['foo'].calc_value['n']
        if self.desc['aggregate']:
            self.assertEqual(16,ref[0,0,0,0])
        else:
            self.assertTrue(np.all(np.ones((1,1,4,4),dtype=int) == ref.data))
        
    def sc_calc_raw_false(self):
        ref = self.ret[1]['coll'].variables['foo'].calc_value['n']
        if self.desc['level_range'] is None:
            mod = 2
        else:
            mod = 1
        if self.desc['aggregate'] is True:
            self.assertEqual(1*mod,ref.sum())
        else:
            self.assertEqual(16*mod,ref.sum())
        
    def sc_aggregate_true(self):
        ref = self.ret[1]['coll'].variables['foo'].calc_value['n']
        if self.desc['calc_raw'] is True:
            self.assertEqual(16,ref[0,0,0])
        else:
            self.assertEqual(1,ref[0,0,0])

    def sm_multi_check(self):
        for value in self.ret.itervalues():
            self.assertTrue(len(value['coll'].variables.keys()) == 2)
            if self.desc['aggregate'] is True:
                for value in value['coll'].variables.itervalues():
                    self.assertTrue(value.agg_value is not None)
                
    def sm_multi_calc_check_aggregate(self):
        mod = self.lmod
        for value in self.ret.itervalues():
            nref = value['coll'].calc_multi['n']
            if self.desc['aggregate']:
                if self.desc['calc_raw']:
                    self.assertTrue(np.all(nref > 900))
                else:
                    self.assertTrue(np.all(nref < 900))
            else:
                self.assertTrue(np.all(nref < 900))
            for value2 in value['coll'].calc_multi.itervalues():
                if self.desc['aggregate']:
                    self.assertEqual(value2.shape,(12,1*mod,1,1))
                else:
                    self.assertEqual(value2.shape,(12,1*mod,4,4))
                    
    def so_nc_out(self):
        import ipdb;ipdb.set_trace()
        
    def so_out(self):
        print self.ret
        import ipdb;ipdb.set_trace()
        
    def si_basic(self):
#        ref = self.ret[1]['coll']
#        it = RawIterator(ref)
#        for row in it:
#            print row
        import ipdb;ipdb.set_trace()


if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestSimpleMultiCalc01.test']
    unittest.main()