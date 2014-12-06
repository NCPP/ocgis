import pickle
import unittest
import itertools
import numpy as np
from cfunits.cfunits import Units
from ocgis.api.parms.definition import Calc
from ocgis.calc.library.statistics import Mean, FrequencyPercentile, MovingWindow
from ocgis.interface.base.variable import DerivedVariable, Variable
from ocgis.test.base import nc_scope
from ocgis.test.test_ocgis.test_interface.test_base.test_field import AbstractTestField
import ocgis
from ocgis.util.itester import itr_products_keywords


class TestMovingWindow(AbstractTestField):

    def test_calculate(self):
        ma = MovingWindow()
        np.random.seed(1)
        values = np.ma.array(np.random.rand(10), dtype=float).reshape(1, -1, 1, 1, 1)
        k = 5
        ret = ma.calculate(values, k=k, mode='same', operation='mean')
        self.assertEqual(ret.shape, values.shape)
        actual = pickle.loads('cnumpy.ma.core\n_mareconstruct\np0\n(cnumpy.ma.core\nMaskedArray\np1\ncnumpy\nndarray\np2\n(I0\ntp3\nS\'b\'\np4\ntp5\nRp6\n(I1\n(I1\nI10\nI1\nI1\nI1\ntp7\ncnumpy\ndtype\np8\n(S\'f8\'\np9\nI0\nI1\ntp10\nRp11\n(I3\nS\'<\'\np12\nNNNI-1\nI-1\nI0\ntp13\nbI00\nS\'\\xec\\x80\\\'\\x90\\rD\\xd8?\\xee"\\x1d\\xdad\\t\\xd7?\\x126\\xab\\x0b\\xceN\\xd4?\\xc4\\xcdN\\xdc\\xe1&\\xd0?\\x0e\\xa2\\x12\\x8a\\xb8\\xa1\\xc2?#!\\x9bX\\xa3y\\xcb?\\x83\\xb5\\x00\\xd2\\x86\\xe4\\xcd?*M\\xfa\\xe1\\xf7\\xf6\\xd3?\\xd4\\xd3\\xad\\xd1}z\\xd7? J4q\\xc2T\\xdb?\'\np14\nS\'\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\'\np15\ncnumpy.core.multiarray\n_reconstruct\np16\n(g2\n(I0\ntp17\ng4\ntp18\nRp19\n(I1\n(tg11\nI00\nS\'@\\x8c\\xb5x\\x1d\\xaf\\x15D\'\np20\ntp21\nbtp22\nb.')
        self.assertNumpyAllClose(ret, actual)
        ret = ret.squeeze()
        values = values.squeeze()
        self.assertEqual(ret[4], np.mean(values[2:7]))

    def test_execute(self):
        field = self.get_field(month_count=1, with_value=True)
        field = field[:, 0:4, :, :, :]
        field.variables['tmax'].value[:] = 1
        field.variables['tmax'].value.mask[:, :, :, 1, 1] = True
        for mode in ['same', 'valid']:
            for operation in ('mean', 'min', 'max', 'median', 'var', 'std'):
                parms = {'k': 3, 'mode': mode, 'operation': operation}
                ma = MovingWindow(field=field, parms=parms)
                vc = ma.execute()
                if mode == 'same':
                    self.assertEqual(vc['moving_window'].value.shape, field.shape)
                else:
                    self.assertEqual(vc['moving_window'].value.shape, (2, 2, 2, 3, 4))
                    self.assertEqual(ma.field.shape, (2, 2, 2, 3, 4))

    def test_execute_valid_through_operations(self):
        """Test executing a "valid" convolution mode through operations ensuring the data is appropriately truncated."""

        rd = self.test_data.get_rd('cancm4_tas')
        calc = [{'func': 'moving_window', 'name': 'ma', 'kwds': {'k': 5, 'mode': 'valid', 'operation': 'mean'}}]
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, slice=[None, [0, 365], None, [0, 10], [0, 10]])
        ret = ops.execute()
        self.assertEqual(ret[1]['tas'].shape, (1, 361, 1, 10, 10))
        self.assertAlmostEqual(ret[1]['tas'].variables['ma'].value.mean(), 240.08204986149585)

    def test_registry(self):
        Calc([{'func': 'moving_window', 'name': 'ma'}])

    def test_iter_kernel_values_same(self):
        """Test returning kernel values with the 'same' mode."""

        values = np.arange(2, 11).reshape(-1, 1, 1)
        k = 5
        mode = 'same'
        itr = MovingWindow._iter_kernel_values_(values, k, mode=mode)
        to_test = list(itr)
        actual = pickle.loads("(lp0\n(I0\ncnumpy.core.multiarray\n_reconstruct\np1\n(cnumpy\nndarray\np2\n(I0\ntp3\nS'b'\np4\ntp5\nRp6\n(I1\n(I3\nI1\nI1\ntp7\ncnumpy\ndtype\np8\n(S'i8'\np9\nI0\nI1\ntp10\nRp11\n(I3\nS'<'\np12\nNNNI-1\nI-1\nI0\ntp13\nbI00\nS'\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np14\ntp15\nbtp16\na(I1\ng1\n(g2\n(I0\ntp17\ng4\ntp18\nRp19\n(I1\n(I4\nI1\nI1\ntp20\ng11\nI00\nS'\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x05\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np21\ntp22\nbtp23\na(I2\ng1\n(g2\n(I0\ntp24\ng4\ntp25\nRp26\n(I1\n(I5\nI1\nI1\ntp27\ng11\nI00\nS'\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x05\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np28\ntp29\nbtp30\na(I3\ng1\n(g2\n(I0\ntp31\ng4\ntp32\nRp33\n(I1\n(I5\nI1\nI1\ntp34\ng11\nI00\nS'\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x05\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x07\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np35\ntp36\nbtp37\na(I4\ng1\n(g2\n(I0\ntp38\ng4\ntp39\nRp40\n(I1\n(I5\nI1\nI1\ntp41\ng11\nI00\nS'\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x05\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x07\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x08\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np42\ntp43\nbtp44\na(I5\ng1\n(g2\n(I0\ntp45\ng4\ntp46\nRp47\n(I1\n(I5\nI1\nI1\ntp48\ng11\nI00\nS'\\x05\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x07\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x08\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\t\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np49\ntp50\nbtp51\na(I6\ng1\n(g2\n(I0\ntp52\ng4\ntp53\nRp54\n(I1\n(I5\nI1\nI1\ntp55\ng11\nI00\nS'\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x07\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x08\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\t\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\n\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np56\ntp57\nbtp58\na(I7\ng1\n(g2\n(I0\ntp59\ng4\ntp60\nRp61\n(I1\n(I4\nI1\nI1\ntp62\ng11\nI00\nS'\\x07\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x08\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\t\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\n\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np63\ntp64\nbtp65\na(I8\ng1\n(g2\n(I0\ntp66\ng4\ntp67\nRp68\n(I1\n(I3\nI1\nI1\ntp69\ng11\nI00\nS'\\x08\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\t\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\n\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np70\ntp71\nbtp72\na.")
        for idx in range(len(to_test)):
            self.assertEqual(to_test[idx][1].ndim, 3)
            self.assertEqual(to_test[idx][0], actual[idx][0])
            self.assertNumpyAll(to_test[idx][1], actual[idx][1])

    def test_iter_kernel_values_valid(self):
        """Test returning kernel values with the 'valid' mode."""

        values = np.arange(2, 11).reshape(-1, 1, 1)
        k = 5
        mode = 'valid'
        itr = MovingWindow._iter_kernel_values_(values, k, mode=mode)
        to_test = list(itr)
        actual = pickle.loads("(lp0\n(I2\ncnumpy.core.multiarray\n_reconstruct\np1\n(cnumpy\nndarray\np2\n(I0\ntp3\nS'b'\np4\ntp5\nRp6\n(I1\n(I5\nI1\nI1\ntp7\ncnumpy\ndtype\np8\n(S'i8'\np9\nI0\nI1\ntp10\nRp11\n(I3\nS'<'\np12\nNNNI-1\nI-1\nI0\ntp13\nbI00\nS'\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x05\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np14\ntp15\nbtp16\na(I3\ng1\n(g2\n(I0\ntp17\ng4\ntp18\nRp19\n(I1\n(I5\nI1\nI1\ntp20\ng11\nI00\nS'\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x05\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x07\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np21\ntp22\nbtp23\na(I4\ng1\n(g2\n(I0\ntp24\ng4\ntp25\nRp26\n(I1\n(I5\nI1\nI1\ntp27\ng11\nI00\nS'\\x04\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x05\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x07\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x08\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np28\ntp29\nbtp30\na(I5\ng1\n(g2\n(I0\ntp31\ng4\ntp32\nRp33\n(I1\n(I5\nI1\nI1\ntp34\ng11\nI00\nS'\\x05\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x07\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x08\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\t\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np35\ntp36\nbtp37\na(I6\ng1\n(g2\n(I0\ntp38\ng4\ntp39\nRp40\n(I1\n(I5\nI1\nI1\ntp41\ng11\nI00\nS'\\x06\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x07\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x08\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\t\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\n\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np42\ntp43\nbtp44\na.")
        for idx in range(len(to_test)):
            self.assertEqual(to_test[idx][1].ndim, 3)
            self.assertEqual(to_test[idx][0], actual[idx][0])
            self.assertNumpyAll(to_test[idx][1], actual[idx][1])

    def test_iter_kernel_values_asserts(self):
        """Test assert statements."""

        k = [1, 2, 3, 4]
        values = [np.array([[2, 3], [4, 5]]), np.arange(0, 13).reshape(-1, 1, 1)]
        mode = ['same', 'valid', 'foo']
        for kwds in itr_products_keywords({'k': k, 'values': values, 'mode': mode}, as_namedtuple=True):
            try:
                list(MovingWindow._iter_kernel_values_(kwds.values, kwds.k))
            except AssertionError:
                if kwds.k == 3:
                    if kwds.values.shape == (2, 2):
                        continue
                    else:
                        raise
                else:
                    continue
            except NotImplementedError:
                if kwds.mode == 'foo':
                    continue
                else:
                    raise


class TestFrequencyPercentile(AbstractTestField):
    
    def test(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        fp = FrequencyPercentile(field=field,tgd=tgd,parms={'percentile':99})
        ret = fp.execute()
        self.assertNumpyAllClose(ret['freq_perc'].value[0,1,1,0,:],
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
        self.assertEqual(dvc['my_mean'].cfunits,Units('kelvin'))
    
    def test(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean',dtype=np.float64)
        dvc = mu.execute()
        dv = dvc['my_mean']
        self.assertEqual(dv.name,'mean')
        self.assertEqual(dv.alias,'my_mean')
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
        dv = dvc['my_mean']
        self.assertEqual(dv.name,'mean')
        self.assertEqual(dv.alias,'my_mean')
        self.assertIsInstance(dv,DerivedVariable)
        self.assertEqual(dv.value.shape,(2,2,2,3,4))
        self.assertNumpyAll(np.ma.mean(field.variables['tmax'].value[1,tgd.dgroups[1],0,:,:],axis=0),
                            dv.value[1,1,0,:,:])

        ret = dvc['n_my_mean']
        self.assertNumpyAll(ret.value[0,0,0],
                            np.ma.array(data=[[31,31,31,31],[31,31,31,31],[31,31,31,31]],
                                        mask=[[False,False,False,False],[False,False,False,False],
                                              [False,False,False,False]],
                                        fill_value=999999,
                                        dtype=ret.dtype))
        
        mu = Mean(field=field,tgd=tgd,alias='my_mean',calc_sample_size=False)
        dvc = mu.execute()
        self.assertNotIn('n_my_mean',dvc.keys())
        
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
        
    def test_two_variables_sample_size(self):
        field = self.get_field(with_value=True,month_count=2)
        field.variables.add_variable(Variable(value=field.variables['tmax'].value+5,
                                              name='tmin',alias='tmin'))
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        mu = Mean(field=field,tgd=tgd,alias='my_mean',dtype=np.float64,calc_sample_size=True)
        ret = mu.execute()
        self.assertEqual(len(ret),4)
        self.assertAlmostEqual(5.0,abs(ret['my_mean_tmax'].value.mean() - ret['my_mean_tmin'].value.mean()))
        self.assertEqual(set(['my_mean_tmax', 'n_my_mean_tmax', 'my_mean_tmin', 'n_my_mean_tmin']),
                         set(ret.keys()))
        
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
        self.assertEqual(ret[27]['tas'].variables['mean'].dtype,var_dtype)
            
    def test_file_only_by_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd,calc=[{'func':'mean','name':'mean'}],
                                  calc_grouping=['month'],geom='state_boundaries',
                                  select_ugid=[27],file_only=True,output_format='nc')
        ret = ops.execute()
        with nc_scope(ret) as ds:
            var = ds.variables['mean']
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
