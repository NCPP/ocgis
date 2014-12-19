from collections import OrderedDict

from numpy.ma import MaskedArray
from cfunits import Units
import numpy as np

from ocgis.exc import VariableInCollectionError, NoUnitsError
from ocgis.interface.base.attributes import Attributes
from ocgis.test.base import TestBase
from ocgis.interface.base.variable import Variable, VariableCollection, AbstractSourcedVariable, AbstractValueVariable, \
    DerivedVariable
from ocgis.util.helpers import get_iter
from ocgis.util.itester import itr_products_keywords


class FakeAbstractSourcedVariable(AbstractSourcedVariable):

    def _set_value_from_source_(self):
        self._value = self._src_idx*2


class TestAbstractSourcedVariable(TestBase):

    def iter(self):
        src_idx = [1, 2]
        data = 'foo'
        kwds = dict(src_idx=[src_idx, None],
                    data=[data, None])

        for k in self.iter_product_keywords(kwds):
            yield k

    def test_init(self):
        for k in self.iter():
            FakeAbstractSourcedVariable(k.data, k.src_idx)

        FakeAbstractSourcedVariable(None, None)

    def test_format_src_idx(self):
        aa = FakeAbstractSourcedVariable('foo', src_idx=[1, 2])
        self.assertNumpyAll(aa._format_src_idx_([1, 2]), np.array([1, 2]))

    def test_get_value(self):
        aa = FakeAbstractSourcedVariable('foo', src_idx=[1, 2])
        aa._value = None
        self.assertNumpyAll(aa._get_value_(), np.array([1, 2])*2)

    def test_src_idx(self):
        aa = FakeAbstractSourcedVariable('foo', src_idx=[1, 2])
        self.assertNumpyAll(aa._src_idx, np.array([1, 2]))



class FakeAbstractValueVariable(AbstractValueVariable):

    def _get_value_(self):
        return np.array(self._value)


class TestAbstractValueVariable(TestBase):
    create_dir = False

    def test_init(self):
        kwds = dict(value=[[4, 5, 6]])

        for k in self.iter_product_keywords(kwds):
            av = FakeAbstractValueVariable(value=k.value)
            self.assertEqual(av.value, k.value)


class TestDerivedVariable(TestBase):

    def test_init(self):
        self.assertEqual(DerivedVariable.__bases__, (Variable,))

        fdef = [{'func': 'mean', 'name': 'mean'}]
        dv = DerivedVariable(fdef=fdef)
        self.assertEqual(dv.fdef, fdef)
        self.assertIsNone(dv.parents)


class TestVariable(TestBase):

    def test_init(self):
        self.assertEqual(Variable.__bases__, (AbstractSourcedVariable, AbstractValueVariable, Attributes))

        # test passing attributes
        var = Variable(attrs={'a': 6}, value=np.array([5]))
        self.assertEqual(var.attrs['a'], 6)

    def test_init_with_value_with_dtype_fill_value(self):
        var = Variable(data='foo',dtype=np.float,fill_value=9,value=np.array([1,2,3,4]))
        self.assertEqual(var.dtype,np.float)
        self.assertEqual(var.fill_value,9)

    def test_init_with_value_without_dtype_fill_value(self):
        value = np.array([1,2,3,4])
        value = np.ma.array(value)
        var = Variable(data='foo',value=value)
        self.assertEqual(var.dtype,value.dtype)
        self.assertEqual(var.fill_value,value.fill_value)

    def test_init_without_value_dtype_fill_value(self):
        var = Variable(data='foo')
        with self.assertRaises(ValueError):
            var.dtype
        with self.assertRaises(ValueError):
            var.fill_value

    def test_init_without_value_with_dtype_fill_value(self):
        var = Variable(data='foo',dtype=np.float,fill_value=9)
        self.assertEqual(var.dtype,np.float)
        self.assertEqual(var.fill_value,9)

    def test_str(self):
        var = Variable(name='toon')
        self.assertEqual(str(var), 'Variable(name="toon", alias="toon", units=None)')

    def test_conform_units_to(self):
        """Test using the conform_units_to keyword argument."""

        def _get_units_(v):
            try:
                v = Units(v)
            except AttributeError:
                pass
            return v

        value = np.array([1, 2, 3, 4, 5])
        value_masked = np.ma.array(value, mask=[False, True, False, True, False])

        kwds = dict(units=['celsius', None, 'mm/day'],
                    conform_units_to=['kelvin', Units('kelvin'), None],
                    value=[value, value_masked])

        for k in itr_products_keywords(kwds, as_namedtuple=True):

            try:
                var = Variable(**k._asdict())
            except NoUnitsError:
                # without units defined on the input array, the values may not be conformed
                if k.units is None:
                    continue
                else:
                    raise
            except ValueError:
                # units are not convertible
                if _get_units_(k.units) == _get_units_('mm/day') and _get_units_(k.conform_units_to) == _get_units_('kelvin'):
                    continue
                else:
                    raise

            if k.conform_units_to is not None:
                try:
                    self.assertEqual(var.conform_units_to, Units(k.conform_units_to))
                # may already be a Units object
                except AttributeError:
                    self.assertEqual(var.conform_units_to, k.conform_units_to)
            else:
                self.assertIsNone(var.conform_units_to)

            if k.conform_units_to is not None:
                actual = [274.15, 275.15, 276.15, 277.15, 278.15]
                if isinstance(k.value, MaskedArray):
                    mask = value_masked.mask
                else:
                    mask = False
                actual = np.ma.array(actual, mask=mask, fill_value=var.value.fill_value)
                self.assertNumpyAll(actual, var.value)

    def test_get_empty_like(self):
        kwargs = dict(name='tas', alias='tas2', units='celsius', meta={'foo': 5}, uid=5, data='foo', did=5)
        value = np.array([1, 2, 3, 4, 5])
        value = np.ma.array(value, mask=[False, True, False, True, False])
        kwargs['value'] = value
        kwargs['attrs'] = OrderedDict(foo=5)
        var = Variable(**kwargs)

        for shape in [None, (2, 2)]:
            new_var = var.get_empty_like(shape=shape)
            self.assertDictEqual(new_var.attrs, kwargs['attrs'])
            new_var.attrs['hi'] = 'wow'
            self.assertNotEqual(new_var.attrs, kwargs['attrs'])
            self.assertEqual(new_var.uid, var.uid)
            if shape is None:
                actual = np.ma.array(np.zeros(5), dtype=var.dtype, fill_value=var.fill_value, mask=value.mask)
            else:
                actual = np.ma.array(np.zeros((2, 2)), dtype=var.dtype, fill_value=var.fill_value, mask=False)
            self.assertNumpyAll(actual, new_var.value)
            # the meta dictionary should be deepcopied
            new_var.meta['hi'] = 'there'
            self.assertDictEqual(var.meta, {'foo': 5})

class TestVariableCollection(TestBase):
    create_dir = False

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
