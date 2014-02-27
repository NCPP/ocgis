import unittest
from ocgis.calc.library.statistics import Mean, FrequencyPercentile
from ocgis.interface.base.variable import DerivedVariable, Variable
import numpy as np
import itertools
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
from ocgis.test.test_simple.test_simple import ToTest, nc_scope
import ocgis
from cfunits.cfunits import Units


class TestFrequencyPercentile(AbstractTestField):
    
    def test(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        fp = FrequencyPercentile(field=field,tgd=tgd,parms={'percentile':99})
        ret = fp.execute()
        self.assertNumpyAllClose(ret['freq_perc_tmax'].value[0,1,1,0,:],
         np.ma.array(data=[0.92864656,0.98615474,0.95269281,0.98542988],
                     mask=False,fill_value=1e+20))


class TestMean(AbstractTestField):
    
    def test_units_are_maintained(self):
        field = self.get_field(with_value=True,month_count=2)
        self.assertEqual(field.variables['tmax'].cfunits,Units('kelvin'))
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean',calc_sample_size=False,
                  dtype=np.float64)
        dvc = mu.execute()
        self.assertEqual(dvc['my_mean_tmax'].cfunits,Units('kelvin'))
    
    def test(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean',dtype=np.float64)
        dvc = mu.execute()
        dv = dvc['my_mean_tmax']
        self.assertEqual(dv.name,'mean')
        self.assertEqual(dv.alias,'my_mean_tmax')
        self.assertIsInstance(dv,DerivedVariable)
        self.assertEqual(dv.value.shape,(2,2,2,3,4))
        self.assertNumpyAll(np.ma.mean(field.variables['tmax'].value[1,tgd.dgroups[1],0,:,:],axis=0),
                            dv.value[1,1,0,:,:])
        
    def test_sample_size(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean',calc_sample_size=True,
                  dtype=np.float64)
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
        
    def test_two_variables(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean',dtype=np.float64)
        ret = mu.execute()
        self.assertEqual(len(ret),2)
        self.assertAlmostEqual(5.0,abs(ret['my_mean_tmax'].value.mean() - ret['my_mean_tmin'].value.mean()))
        
    def test_file_only(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field = field[:,10:20,:,20:30,40:50]
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        ## value should not be loaded at this point
        self.assertEqual(field.variables['tas']._value,None)
        mu = Mean(field=field,tgd=tgd,alias='my_mean',file_only=True)
        ret = mu.execute()
        ## value should still not be loaded
        self.assertEqual(field.variables['tas']._value,None)
        ## there should be no value in the variable present and attempts to load
        ## it should fail.
        with self.assertRaises(Exception):
            ret['my_mean_tas'].value
            
    def test_output_datatype(self):
        ## ensure the output data type is the same as the input data type of
        ## the variable.
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,calc=[{'func':'mean','name':'mean'}],
                                  calc_grouping=['month'],geom='state_boundaries',
                                  select_ugid=[27])
        ret = ops.execute()
        with nc_scope(rd.uri) as ds:
            var_dtype = ds.variables['tas'].dtype
        self.assertEqual(ret[27]['tas'].variables['mean_tas'].dtype,var_dtype)
            
    def test_file_only_by_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,calc=[{'func':'mean','name':'mean'}],
                                  calc_grouping=['month'],geom='state_boundaries',
                                  select_ugid=[27],file_only=True,output_format='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            var = ds.variables['mean_tas']
            ## all data should be masked since this is file only
            self.assertTrue(var[:].mask.all())
        
    def test_use_raw_values(self):
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
