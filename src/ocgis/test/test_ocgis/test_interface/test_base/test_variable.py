from ocgis.exc import VariableInCollectionError
from ocgis.test.base import TestBase
from ocgis.interface.base.variable import Variable, VariableCollection
import numpy as np
from ocgis.util.helpers import get_iter


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


class TestVariableCollection(TestBase):
    _create_dir = False

    def get_variable(self, alias='tas_foo'):
        var = Variable(name='tas', alias=alias, value=np.array([4, 5, 6]))
        return var

    def test_init_variable_none(self):
        vc = VariableCollection()
        self.assertEqual(len(vc), 0)

    def test_init_variable_not_none(self):
        variables = [self.get_variable(), [self.get_variable(), self.get_variable('tas_foo2')]]
        for v in variables:
            vc = VariableCollection(variables=v)
            self.assertEqual(vc.keys(), [iv.alias for iv in get_iter(v, dtype=Variable)])

    def test_add_variable(self):
        vc = VariableCollection()
        var = self.get_variable()
        self.assertEqual(var.uid, None)
        vc.add_variable(var)
        self.assertEqual(var.uid, 1)
        self.assertTrue('tas_foo' in vc)
        self.assertEqual(vc._storage_id, [1])
        var.alias = 'again'
        with self.assertRaises(AssertionError):
            vc.add_variable(var)
        var.uid = 100
        vc.add_variable(var)
        self.assertEqual(vc._storage_id, [1, 100])

    def test_add_variable_already_in_collection(self):
        vc = VariableCollection()
        var = self.get_variable()
        vc.add_variable(var)
        with self.assertRaises(VariableInCollectionError):
            vc.add_variable(var)

    def test_add_variable_already_in_collection_uids_update(self):
        vc = VariableCollection()
        var = self.get_variable()
        vc.add_variable(var)
        self.assertEqual(var.uid, 1)
        var.alias = 'variable_2'
        vc.add_variable(var, assign_new_uid=True)
        self.assertEqual(var.uid, 2)
        self.assertEqual(vc._storage_id, [1, 2])

    def test_get_sliced_variables(self):
        variables = [self.get_variable(), self.get_variable('tas_foo2')]
        vc = VariableCollection(variables=variables)
        ret = vc.get_sliced_variables(slice(1))
        for k, v in ret.iteritems():
            self.assertNumpyAll(v.value, np.ma.array([4]))
        for k, v in ret.iteritems():
            self.assertTrue(np.may_share_memory(v.value, ret[k].value))
