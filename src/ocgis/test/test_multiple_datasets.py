from ocgis.api.operations import OcgOperations
from itertools import izip
import numpy as np
from ocgis.test.base import TestBase
from unittest.case import SkipTest
from ocgis.interface.shp import ShpDataset
import ocgis
import itertools


class Test(TestBase):
    
    def setUp(self):
        TestBase.setUp(self)
        self.maurer = self.test_data.get_rd('maurer_bccr_1950')
        self.cancm4 = self.test_data.get_rd('cancm4_tasmax_2001')
        self.tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
#        self.albisccp = self.test_data.get_rd('ccsm4')
    
    @property
    def california(self):
        ret = ShpDataset('state_boundaries',attr_filter={'ugid':[25]})
        return(ret)
    
    @property
    def dataset(self):
        dataset = [
                   self.maurer.copy(),
                   self.cancm4.copy()
                   ]
        return(dataset)

    def get_ops(self,kwds={}):
        geom = self.california
        ops = OcgOperations(dataset=self.dataset,
                            snippet=True,
                            geom=geom,
                            output_format='numpy')
        for k,v in kwds.iteritems():
            setattr(ops,k,v)
        return(ops)
    
    def get_ref(self,kwds={}):
        ops = self.get_ops(kwds=kwds)
        ret = ops.execute()
        return(ret[25])
    
    def test_keyed(self):
        raise(SkipTest)
        ds = self.dataset
#        ds.append(self.albisccp.copy())
        ds.append(self.tasmin.copy())
        
        ops = OcgOperations(dataset=ds,geom=self.california,output_format='numpy')
        ret = ops.execute()
        ref = ret[25].variables
        self.assertEqual(ref['tasmax']._use_for_id,['gid','tid'])
        self.assertEqual(ref['tasmin']._use_for_id,[])
#        for key in ['albisccp','Prcp']:
#            self.assertEqual(ret[25].variables[key]._use_for_id,['gid','tid'])
        
        ops = OcgOperations(dataset=ds,geom=self.california,output_format='keyed',snippet=True)
        ret = ops.execute()
    
    def test_default(self):
        ops = self.get_ops()        
        ret = ops.execute()
        
        self.assertEqual(['Prcp','tasmax'],ret[25].variables.keys())
        
        shapes = [(1,1,77,83),(1,1,5,4)]
        for v,shape in izip(ret[25].variables.itervalues(),shapes):
            self.assertEqual(v.value.shape,shape)
            
    def test_vector_wrap(self):
        geom = self.california
        keys = [
                ['maurer_bccr_1950',(12,1,77,83)],
                ['cancm4_tasmax_2011',(3650,1,5,4)]
                ]
        for key in keys:
            prev_value = None
            for vector_wrap in [True,False]:
                rd = self.test_data.get_rd(key[0])
                prefix = 'vw_{0}_{1}'.format(vector_wrap,rd.variable)
                ops = ocgis.OcgOperations(dataset=rd,geom=geom,snippet=False,
                      vector_wrap=vector_wrap,prefix=prefix)
                ret = ops.execute()
                ref = ret[25].variables[rd.variable].value
                self.assertEqual(ref.shape,key[1])
                if prev_value is None:
                    prev_value = ref
                else:
                    self.assertTrue(np.all(ref == prev_value))
    
    def test_aggregate_clip(self):
        kwds = {'aggregate':True,'spatial_operation':'clip'}
        ref = self.get_ref(kwds)
        for v in ref.variables.itervalues():
            self.assertEqual(v.spatial.vector.shape,(1,1))
            self.assertEqual(v.value.shape,(1,1,1,1))
    
    def test_calculation(self):
        calc = [{'func':'mean','name':'mean'},{'func':'std','name':'std'}]
        calc_grouping = ['year']
        kwds = {'aggregate':True,
                'spatial_operation':'clip',
                'calc':calc,
                'calc_grouping':calc_grouping,
                'output_format':'numpy',
                'geom':self.california,
                'dataset':self.dataset,
                'snippet':False}
        ops = OcgOperations(**kwds)
        ret = ops.execute()
        
        ref = ret[25].calc['Prcp']
        self.assertEquals(ref.keys(),['mean', 'std', 'n'])
        for value in ref.itervalues():
            self.assertEqual(value.shape,(1,1,1,1))
            
        ref = ret[25].calc['tasmax']
        self.assertEquals(ref.keys(),['mean', 'std', 'n'])
        for value in ref.itervalues():
            self.assertEqual(value.shape,(10,1,1,1))
            
    def test_same_variable_name(self):
        ds = [self.cancm4.copy(),self.cancm4.copy()]
        
        with self.assertRaises(KeyError):
            OcgOperations(dataset=ds)
        ds[0].alias = 'foo'
        ds[1].alias = 'foo'
        with self.assertRaises(KeyError):
            OcgOperations(dataset=ds)
        
        ds = [self.cancm4.copy(),self.cancm4.copy()]
        ds[0].alias = 'foo_var'
        ops = OcgOperations(dataset=ds,snippet=True)
        ret = ops.execute()
        self.assertEqual(ret[1].variables.keys(),['foo_var','tasmax'])
        values = ret[1].variables.values()
        self.assertTrue(np.all(values[0].value == values[1].value))

    def test_consolidating_projections(self):
        rd1 = self.test_data.get_rd('narccap_rcm3')
        rd1.alias = 'rcm3'
        rd2 = self.test_data.get_rd('narccap_crcm')
        rd2.alias = 'crcm'
        rd = [
              rd1,
#              rd2
              ]
        ops = ocgis.OcgOperations(dataset=rd,snippet=True,output_format='shp',
                                  geom='state_boundaries',agg_selection=False,
                                  select_ugid=[25])
        ret = ops.execute()
        import ipdb;ipdb.set_trace()