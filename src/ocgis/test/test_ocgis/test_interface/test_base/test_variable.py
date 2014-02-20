from ocgis.test.base import TestBase
from ocgis.interface.base.variable import Variable
import numpy as np


class TestVariable(TestBase):

    def test_constructor_without_value_dtype_fill_value(self):
        var = Variable(data='foo')
        with self.assertRaises(ValueError):
            var.dtype
        with self.assertRaises(ValueError):
            var.fill_value
            
    def test_constructor_without_value_with_dtype_fill_value(self):
        var = Variable(data='foo',dtype=np.float,fill_value=9)
        self.assertEqual(var.dtype,np.float)
        self.assertEqual(var.fill_value,9)
        
    def test_constructor_with_value_with_dtype_fill_value(self):
        var = Variable(data='foo',dtype=np.float,fill_value=9,value=np.array([1,2,3,4]))
        self.assertEqual(var.dtype,np.float)
        self.assertEqual(var.fill_value,9)
        
    def test_constructor_with_value_without_dtype_fill_value(self):
        value = np.array([1,2,3,4])
        value = np.ma.array(value)
        var = Variable(data='foo',value=value)
        self.assertEqual(var.dtype,value.dtype)
        self.assertEqual(var.fill_value,value.fill_value)
        