import unittest
from ocgis.calc.library.statistics import Mean, FrequencyPercentile
from ocgis.interface.base.variable import DerivedVariable, Variable
import numpy as np
import itertools
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField


class Test(AbstractTestField):
    
    def test_FrequencyPercentile(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        fp = FrequencyPercentile(field=field,tgd=tgd,parms={'percentile':99})
        ret = fp.execute()
        self.assertNumpyAllClose(ret['freq_perc_tmax'].value[0,1,1,0,:],
         np.ma.array(data=[0.92864656,0.98615474,0.95269281,0.98542988],
                     mask=False,fill_value=1e+20))

    def test_Mean(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean')
        dvc = mu.execute()
        dv = dvc['my_mean_tmax']
        self.assertEqual(dv.name,'mean')
        self.assertEqual(dv.alias,'my_mean_tmax')
        self.assertIsInstance(dv,DerivedVariable)
        self.assertEqual(dv.value.shape,(2,2,2,3,4))
        self.assertNumpyAll(np.ma.mean(field.variables['tmax'].value[1,tgd.dgroups[1],0,:,:],axis=0),
                            dv.value[1,1,0,:,:])
        
    def test_Mean_sample_size(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean',calc_sample_size=True)
        dvc = mu.execute()
        dv = dvc['my_mean_tmax']
        self.assertEqual(dv.name,'mean')
        self.assertEqual(dv.alias,'my_mean_tmax')
        self.assertIsInstance(dv,DerivedVariable)
        self.assertEqual(dv.value.shape,(2,2,2,3,4))
        self.assertNumpyAll(np.ma.mean(field.variables['tmax'].value[1,tgd.dgroups[1],0,:,:],axis=0),
                            dv.value[1,1,0,:,:])
        
        ret = dvc['n_my_mean_tmax']
        self.assertNumpyAll(ret.value[0,0,0],
                            np.ma.array(data=[[31,31,31,31],[31,31,31,31],[31,31,31,31]],
                                        mask=[[False,False,False,False],[False,False,False,False],
                                              [False,False,False,False]],
                                        fill_value=999999))
        
        mu = Mean(field=field,tgd=tgd,alias='my_mean',calc_sample_size=False)
        dvc = mu.execute()
        self.assertNotIn('n_my_mean_tmax',dvc.keys())
        
    def test_Mean_two_variables(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean')
        ret = mu.execute()
        self.assertEqual(len(ret),2)
        self.assertAlmostEqual(5.0,abs(ret['my_mean_tmax'].value.mean() - ret['my_mean_tmin'].value.mean()))
        
    def test_Mean_use_raw_values(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        
        ur = [True,False]
        agg = [
               True,
               False
               ]
        
        for u,a in itertools.product(ur,agg):
            if a:
                cfield = field.get_spatially_aggregated()
                self.assertNotEqual(cfield.shape,cfield._raw.shape)
                self.assertEqual(set([r.value.shape for r in cfield.variables.values()]),set([(2, 60, 2, 1, 1)]))
                self.assertEqual(cfield.shape,(2,60,2,1,1))
            else:
                cfield = field
                self.assertEqual(set([r.value.shape for r in cfield.variables.values()]),set([(2, 60, 2, 3, 4)]))
                self.assertEqual(cfield.shape,(2,60,2,3,4))
            mu = Mean(field=cfield,tgd=tgd,alias='my_mean',use_raw_values=u)
            ret = mu.execute()
            if a:
                self.assertEqual(set([r.value.shape for r in ret.values()]),set([(2, 2, 2, 1, 1)]))
            else:
                self.assertEqual(set([r.value.shape for r in ret.values()]),set([(2, 2, 2, 3, 4)]))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()