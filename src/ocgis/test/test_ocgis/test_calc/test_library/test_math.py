import unittest
import numpy as np
import ocgis
from ocgis.api.parms.definition import Calc
from ocgis.calc.library.math import NaturalLogarithm, Divide, Sum, Convolve1D
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
        self.assertEqual(ret['ln'].value.shape,(2, 60, 2, 3, 4))
        self.assertNumpyAllClose(ret['ln'].value,np.log(field.variables['tmax'].value))
        
        ln = NaturalLogarithm(field=field,calc_sample_size=True)
        ret = ln.execute()
        self.assertNotIn('n_ln',ret.keys())
        
    def test_NaturalLogarithm_units_dimensionless(self):
        field = self.get_field(with_value=True,month_count=2)
        ln = NaturalLogarithm(field=field,alias='ln')
        dvc = ln.execute()
        self.assertEqual(dvc['ln'].units,None)
        
    def test_NaturalLogarithm_grouped(self):
        field = self.get_field(with_value=True,month_count=2)
        grouping = ['month']
        tgd = field.temporal.get_grouping(grouping)
        ln = NaturalLogarithm(field=field,tgd=tgd)
        ret = ln.execute()
        self.assertEqual(ret['ln'].value.shape,(2, 2, 2, 3, 4))
        
        to_test = np.log(field.variables['tmax'].value)
        to_test = np.ma.mean(to_test[0,tgd.dgroups[0],0,:,:],axis=0)
        to_test2 = ret['ln'].value[0,0,0,:,:]
        self.assertNumpyAllClose(to_test,to_test2)
        
        ln = NaturalLogarithm(field=field,tgd=tgd,calc_sample_size=True)
        ret = ln.execute()
        self.assertEqual(ret['ln'].value.shape,(2, 2, 2, 3, 4))
        self.assertEqual(ret['n_ln'].value.mean(),30.0)
        
        ln = NaturalLogarithm(field=field,tgd=tgd,calc_sample_size=True,use_raw_values=True)
        ret = ln.execute()
        self.assertEqual(ret['ln'].value.shape,(2, 2, 2, 3, 4))
        self.assertEqual(ret['n_ln'].value.mean(),30.0)
        
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
        self.assertEqual(ret['threshold'].value.shape,(2,2,2,3,4))
        self.assertNumpyAllClose(ret['threshold'].value[1,1,1,0,:],
         np.ma.array([13,16,15,12],mask=False,fill_value=1e20))


class TestSum(AbstractTestField):

    def test_calculate(self):
        """Test calculate for the sum function."""

        field = self.get_field(with_value=True)
        tgd = field.temporal.get_grouping(['month'])
        sum = Sum(field=field, tgd=tgd)
        np.random.seed(1)
        values = np.random.rand(2, 2, 2)
        values = np.ma.array(values, mask=False)
        to_test = sum.calculate(values)
        self.assertNumpyAll(to_test, np.ma.sum(values, axis=0))

    def test_registry(self):
        """Test sum function is appropriately registered."""

        c = Calc([{'func': 'sum', 'name': 'sum'}])
        self.assertEqual(c.value[0]['ref'], Sum)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()


class TestConvolve1D(AbstractTestField):

    def get_convolve1d_field(self, slice_stop=3):
        field = self.get_field(month_count=1, with_value=True)
        field = field[:, 0:slice_stop, :, :, :]
        field.variables['tmax'].value[:] = 1
        field.variables['tmax'].value.mask[:, :, :, 1, 1] = True
        return field

    def test_execute_same(self):
        """Test convolution with the 'same' mode (numpy default)."""

        field = self.get_convolve1d_field()
        parms = {'v': np.array([1, 1, 1])}
        cd = Convolve1D(field=field, parms=parms)
        self.assertDictEqual(cd._format_parms_(parms), parms)
        vc = cd.execute()
        self.assertNumpyAll(vc['convolve_1d'].value.mask, field.variables['tmax'].value.mask)
        self.assertEqual(vc['convolve_1d'].value.fill_value, field.variables['tmax'].value.fill_value)
        actual = '\x80\x02cnumpy.ma.core\n_mareconstruct\nq\x01(cnumpy.ma.core\nMaskedArray\nq\x02cnumpy\nndarray\nq\x03K\x00\x85q\x04U\x01btRq\x05(K\x01(K\x02K\x03K\x02K\x03K\x04tcnumpy\ndtype\nq\x06U\x02f4K\x00K\x01\x87Rq\x07(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89T@\x02\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@\x00\x00\x00@U\x90\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00cnumpy.core.multiarray\n_reconstruct\nq\x08h\x03K\x00\x85U\x01b\x87Rq\t(K\x01)h\x06U\x02f8K\x00K\x01\x87Rq\n(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x08@\x8c\xb5x\x1d\xaf\x15Dtbtb.'
        actual = np.ma.loads(actual)
        self.assertNumpyAll(actual, vc['convolve_1d'].value, check_fill_value_dtype=False)

    def test_execute_valid(self):
        """Test convolution with the 'valid' mode."""

        #todo: add to docs
        field = self.get_convolve1d_field(slice_stop=4)
        parms = {'v': np.array([1, 1, 1]), 'mode': 'valid'}
        cd = Convolve1D(field=field, parms=parms)
        self.assertDictEqual(cd._format_parms_(parms), parms)
        vc = cd.execute()
        actual = '\x80\x02cnumpy.ma.core\n_mareconstruct\nq\x01(cnumpy.ma.core\nMaskedArray\nq\x02cnumpy\nndarray\nq\x03K\x00\x85q\x04U\x01btRq\x05(K\x01(K\x02K\x02K\x02K\x03K\x04tcnumpy\ndtype\nq\x06U\x02f4K\x00K\x01\x87Rq\x07(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89T\x80\x01\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00\x00\x00\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@\x00\x00@@U`\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00cnumpy.core.multiarray\n_reconstruct\nq\x08h\x03K\x00\x85U\x01b\x87Rq\t(K\x01)h\x06U\x02f8K\x00K\x01\x87Rq\n(K\x03U\x01<NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK\x00tb\x89U\x08@\x8c\xb5x\x1d\xaf\x15Dtbtb.'
        actual = np.ma.loads(actual)
        self.assertNumpyAll(actual, vc['convolve_1d'].value, check_fill_value_dtype=False)
        self.assertEqual(cd.field.shape, (2, 2, 2, 3, 4))
        actual = np.loads('\x80\x02cnumpy.core.multiarray\n_reconstruct\nq\x01cnumpy\nndarray\nq\x02K\x00\x85U\x01b\x87Rq\x03(K\x01K\x02\x85cnumpy\ndtype\nq\x04U\x02O8K\x00K\x01\x87Rq\x05(K\x03U\x01|NNNJ\xff\xff\xff\xffJ\xff\xff\xff\xffK?tb\x89]q\x06(cdatetime\ndatetime\nq\x07U\n\x07\xd0\x01\x01\x0c\x00\x00\x00\x00\x00\x85Rq\x08h\x07U\n\x07\xd0\x01\x02\x0c\x00\x00\x00\x00\x00\x85Rq\tetb.')
        self.assertNumpyAll(actual, cd.field.temporal.value)

    def test_execute_valid_through_operations(self):
        """Test executing a "valid" convolution mode through operations ensuring the data is appropriately truncated."""

        rd = self.test_data.get_rd('cancm4_tas')
        calc = [{'func': 'convolve_1d', 'name': 'convolve', 'kwds': {'v': np.array([1, 1, 1, 1, 1]), 'mode': 'valid'}}]
        ops = ocgis.OcgOperations(dataset=rd, calc=calc, slice=[None, [0, 365], None, [0, 10], [0, 10]])
        ret = ops.execute()
        self.assertEqual(ret[1]['tas'].shape, (1, 361, 1, 10, 10))
        self.assertAlmostEqual(ret[1]['tas'].variables['convolve'].value.mean(), 1200.4059833795013)

    def test_registry(self):
        Calc([{'func': 'convolve_1d', 'name': 'convolve'}])