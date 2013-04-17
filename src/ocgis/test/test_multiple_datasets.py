from ocgis.util.shp_cabinet import ShpCabinet
from ocgis.api.operations import OcgOperations
from itertools import izip
import numpy as np
from ocgis.test.base import TestBase


class Test(TestBase):
    
    def setUp(self):
        self.maurer = self.test_data.get_rd('maurer_bccr_1950')
        self.cancm4 = self.test_data.get_rd('cancm4_tasmax_2001')
        self.tasmin = self.test_data.get_rd('cancm4_tasmin_2001')
        self.albisccp = self.test_data.get_rd('ccsm4')
    
    @property
    def dataset(self):
        dataset = [
                   self.maurer.copy(),
                   self.cancm4.copy()
                   ]
        return(dataset)
    @property
    def california(self):
        sc = ShpCabinet()
        ret = sc.get_geoms('state_boundaries',{'ugid':[25]})
        return(ret)

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
        ds = self.dataset
        ds.append(self.albisccp.copy())
        ds.append(self.tasmin.copy())
        
        ops = OcgOperations(dataset=ds,geom=self.california,output_format='numpy')
        ret = ops.execute()
        ref = ret[25].variables
        self.assertEqual(ref['tasmax']._use_for_id,['gid','tid'])
        self.assertEqual(ref['tasmin']._use_for_id,[])
        for key in ['albisccp','Prcp']:
            self.assertEqual(ret[25].variables[key]._use_for_id,['gid','tid'])
        
        ops = OcgOperations(dataset=ds,geom=self.california,output_format='keyed',snippet=True)
        ret = ops.execute()
    
    def test_default(self):
        ops = self.get_ops()
        ret = ops.execute()
        
        self.assertEqual(['Prcp','tasmax'],ret[25].variables.keys())
        
        shapes = [(1,1,77,83),(1,1,5,4)]
        for v,shape in izip(ret[25].variables.itervalues(),shapes):
            self.assertEqual(v.value.shape,shape)
    
    def test_aggregate_clip(self):
        kwds = {'aggregate':True,'spatial_operation':'clip'}
        ref = self.get_ref(kwds)
        for v in ref.variables.itervalues():
            self.assertEqual(v.spatial.value.shape,(1,1))
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
        
        ref = ret[25].variables['Prcp'].calc_value
        self.assertEquals(ref.keys(),['mean', 'std', 'n'])
        for value in ref.itervalues():
            self.assertEqual(value.shape,(1,1,1,1))
            
        ref = ret[25].variables['tasmax'].calc_value
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
