import unittest
import numpy as np
from ocgis.calc.library.math import NaturalLogarithm, Divide
from ocgis.interface.base.variable import Variable
import itertools
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
from ocgis.calc.library.thresholds import Threshold
from ocgis.exc import SampleSizeNotImplemented


class Test(AbstractTestField):
    
    def test_NaturalLogarithm(self):
        field = self.get_field(with_value=True,month_count=2)
        ln = NaturalLogarithm(field=field)
        ret = ln.execute()
        self.assertEqual(ret['ln_tmax'].value.shape,(2, 60, 2, 3, 4))
        self.assertNumpyAllClose(ret['ln_tmax'].value,np.log(field.variables['tmax'].value))
        
        ln = NaturalLogarithm(field=field,calc_sample_size=True)
        ret = ln.execute()
        self.assertNotIn('n_ln_tmax',ret.keys())
        
    def test_NaturalLogarithm_grouped(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        ln = NaturalLogarithm(field=field,tgd=tgd)
        ret = ln.execute()
        self.assertEqual(ret['ln_tmax'].value.shape,(2, 2, 2, 3, 4))
        
        to_test = np.log(field.variables['tmax'].value)
        to_test = np.ma.mean(to_test[0,tgd.dgroups[0],0,:,:],axis=0)
        to_test2 = ret['ln_tmax'].value[0,0,0,:,:]
        self.assertNumpyAllClose(to_test,to_test2)
        
        ln = NaturalLogarithm(field=field,tgd=tgd,calc_sample_size=True)
        ret = ln.execute()
        self.assertEqual(ret['ln_tmax'].value.shape,(2, 2, 2, 3, 4))
        self.assertEqual(ret['n_ln_tmax'].value.mean(),30.0)
        
        ln = NaturalLogarithm(field=field,tgd=tgd,calc_sample_size=True,use_raw_values=True)
        ret = ln.execute()
        self.assertEqual(ret['ln_tmax'].value.shape,(2, 2, 2, 3, 4))
        self.assertEqual(ret['n_ln_tmax'].value.mean(),30.0)
        
    def test_Divide(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        dv = Divide(field=field,parms={'arr1':'tmax','arr2':'tmin'})
        ret = dv.execute()
        self.assertNumpyAllClose(ret['divide'].value,
                                 field.variables['tmax'].value/field.variables['tmin'].value)
        
        with self.assertRaises(SampleSizeNotImplemented):
            Divide(field=field,parms={'arr1':'tmax','arr2':'tmin'},calc_sample_size=True)
        
    def test_Divide_grouped(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        dv = Divide(field=field,parms={'arr1':'tmax','arr2':'tmin'},tgd=tgd)
        ret = dv.execute()
        self.assertEqual(ret['divide'].value.shape,(2,2,2,3,4))
        self.assertNumpyAllClose(ret['divide'].value[0,1,1,:,2],
                                 np.ma.array([0.0833001563436,0.0940192653632,0.0916398919876],
                                             mask=False,fill_value=1e20))
        
    def test_Divide_use_raw_values(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        grouping = ['month']
        
        ur = [
              True,
              False
              ]
        agg = [
               True,
               False
               ]
        with_tgd = [
                    True,
                    False
                    ]
        
        for u,a,w in itertools.product(ur,agg,with_tgd):
            if w:
                tgd = field.temporal.get_grouping(grouping)
            else:
                tgd = None
            if a:
                cfield = field.get_spatially_aggregated()
                self.assertNotEqual(cfield.shape,cfield._raw.shape)
                self.assertEqual(set([r.value.shape for r in cfield.variables.values()]),set([(2, 60, 2, 1, 1)]))
                self.assertEqual(cfield.shape,(2,60,2,1,1))
            else:
                cfield = field
                self.assertEqual(set([r.value.shape for r in cfield.variables.values()]),set([(2, 60, 2, 3, 4)]))
                self.assertEqual(cfield.shape,(2,60,2,3,4))
            div = Divide(dtype=np.float32,field=cfield,parms={'arr1':'tmax','arr2':'tmin'},tgd=tgd,use_raw_values=u)
            ret = div.execute()
            if a:
                if w:
                    self.assertEqual(set([r.value.shape for r in ret.values()]),set([(2, 2, 2, 1, 1)]))
                else:
                    self.assertEqual(set([r.value.shape for r in ret.values()]),set([(2, 60, 2, 1, 1)]))
            else:
                if w:
                    self.assertEqual(set([r.value.shape for r in ret.values()]),set([(2, 2, 2, 3, 4)]))
                else:
                    self.assertEqual(set([r.value.shape for r in ret.values()]),set([(2, 60, 2, 3, 4)]))
    
    def test_Treshold(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        dv = Threshold(field=field,parms={'threshold':0.5,'operation':'gte'},tgd=tgd)
        ret = dv.execute()
        self.assertEqual(ret['threshold_tmax'].value.shape,(2,2,2,3,4))
        self.assertNumpyAllClose(ret['threshold_tmax'].value[1,1,1,0,:],
         np.ma.array([13,16,15,12],mask=False,fill_value=1e20))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
