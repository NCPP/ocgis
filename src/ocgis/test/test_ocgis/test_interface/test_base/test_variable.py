from collections import OrderedDict

import numpy as np

from ocgis.constants import NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE
from ocgis.exc import VariableInCollectionError, NoUnitsError
from ocgis.interface.base.attributes import Attributes
from ocgis.interface.base.variable import Variable, VariableCollection, AbstractSourcedVariable, AbstractVariable, \
    DerivedVariable
from ocgis.test.base import TestBase, attr
from ocgis.util.helpers import get_iter
from ocgis.util.units import get_units_object, get_are_units_equivalent


class MockAbstractSourcedVariable(AbstractSourcedVariable):
    def _get_value_from_source_(self):
        return 101


class TestAbstractSourcedVariable(TestBase):
    def iter(self):
        src_idx = [1, 2]
        data = 'foo'
        kwds = dict(src_idx=[src_idx, None],
                    request_dataset=[data, None])

        for k in self.iter_product_keywords(kwds):
            yield k

    def test_init(self):
        for k in self.iter():
            m = MockAbstractSourcedVariable(request_dataset=k.request_dataset, src_idx=k.src_idx)
            if k.src_idx is None:
                self.assertIsNone(m._src_idx)
            else:
                self.assertNumpyAll(m._src_idx, np.array(k.src_idx))

        MockAbstractSourcedVariable(request_dataset=None, src_idx=None)

    def test_value(self):
        for k in self.iter():
            m = MockAbstractSourcedVariable(**k._asdict())
            self.assertEqual(m.value, 101)


class TestAbstractVariable(TestBase):
    create_dir = False

    @property
    def value(self):
        return np.array([5, 5, 5])

    def test_init(self):
        self.assertEqual(AbstractVariable.__bases__, (Attributes,))

        av = AbstractVariable(value=self.value)
        self.assertNumpyAll(av.value, self.value)
        self.assertIsNone(av.alias)
        self.assertIsNone(av.conform_units_to)

        fav = AbstractVariable(name='foo')
        self.assertEqual(fav.alias, 'foo')

        # Test data types also pulled from value if present.
        dtype = float
        fav = AbstractVariable(value=self.value, dtype=dtype)
        self.assertEqual(fav.dtype, self.value.dtype)
        self.assertIsNone(fav._dtype)

        # Use string-based units.
        var = Variable(name='tas', units='celsius', value=self.value)
        self.assertEqual(var.units, 'celsius')
        self.assertEqual(var.cfunits, get_units_object('celsius'))
        self.assertNotEqual(var.cfunits, get_units_object('kelvin'))
        self.assertTrue(get_are_units_equivalent((var.cfunits, get_units_object('kelvin'))))

        # Use constructor with units objects v. string.
        var = Variable(name='tas', units=get_units_object('celsius'), value=self.value)
        self.assertEqual(var.units, 'celsius')
        self.assertEqual(var.cfunits, get_units_object('celsius'))

        # Test without units.
        var = Variable(name='tas', units=None, value=self.value)
        self.assertEqual(var.units, None)
        self.assertEqual(var.cfunits, get_units_object(None))

    def test_cfunits_conform(self):
        units_kelvin = get_units_object('kelvin')

        # Conversion of celsius units to kelvin.
        attrs = {k: 1 for k in NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE}
        var = Variable(name='tas', units='celsius', value=self.value, attrs=attrs)
        self.assertEqual(len(var.attrs), 4)
        var.cfunits_conform(units_kelvin)
        self.assertNumpyAll(var.value, np.ma.array([278.15] * 3))
        self.assertEqual(var.cfunits, units_kelvin)
        self.assertEqual(var.units, 'kelvin')
        self.assertEqual(len(var.attrs), 0)

        # If there are no units associated with a variable, conforming the units should fail.
        var = Variable(name='tas', units=None, value=self.value)
        with self.assertRaises(NoUnitsError):
            var.cfunits_conform(units_kelvin)

        # Conversion should fail for nonequivalent units.
        var = Variable(name='tas', units='kelvin', value=self.value)
        with self.assertRaises(ValueError):
            var.cfunits_conform(get_units_object('grams'))

        # The data type should always be updated to match the output from CF units backend.
        av = Variable(value=np.array([4, 5, 6]), dtype=int)
        self.assertEqual(av.dtype, np.dtype(int))
        with self.assertRaises(NoUnitsError):
            av.cfunits_conform('K')
        av.units = 'celsius'
        av.cfunits_conform('K')
        self.assertIsNone(av._dtype)
        self.assertEqual(av.dtype, av.value.dtype)

    def test_cfunits_conform_masked_array(self):
        # Assert mask is respected by unit conversion.
        value = np.ma.array(data=[5, 5, 5], mask=[False, True, False])
        var = Variable(name='tas', units=get_units_object('celsius'), value=value)
        var.cfunits_conform(get_units_object('kelvin'))
        self.assertNumpyAll(np.ma.array([278.15, 278.15, 278.15], mask=[False, True, False]), var.value)

    def test_conform_units_to(self):
        v = AbstractVariable(value=self.value, units='kelvin', conform_units_to='celsius')
        self.assertEqual(v.conform_units_to, get_units_object('celsius'))

    def test_value(self):
        v = AbstractVariable(value=self.value, units='celsius', conform_units_to='kelvin')
        self.assertNumpyAll(v.value, np.array([278.15, 278.15, 278.15]))


class TestDerivedVariable(TestBase):
    def test_init(self):
        self.assertEqual(DerivedVariable.__bases__, (Variable,))

        fdef = [{'func': 'mean', 'name': 'mean'}]
        dv = DerivedVariable(fdef=fdef)
        self.assertEqual(dv.fdef, fdef)
        self.assertIsNone(dv.parents)

    def test_iter_melted(self):
        fdef = {'func': 'mean', 'name': 'mean_alias'}
        tmax = Variable(name='tmax', alias='tmax_alias')
        parents = VariableCollection(variables=tmax)

        for p in [None, parents]:
            dv = DerivedVariable(fdef=fdef, value=np.array([1, 2]), name='mean', alias='my_mean', parents=p)
            for row in dv.iter_melted():
                self.assertEqual(row['calc_key'], 'mean')
                self.assertEqual(row['calc_alias'], 'mean_alias')
                for key in ['name', 'alias']:
                    if p is None:
                        self.assertIsNone(row[key])
                    else:
                        self.assertEqual(row['name'], 'tmax')
                        self.assertEqual(row['alias'], 'tmax_alias')


class TestVariable(TestBase):
    def test_init(self):
        self.assertEqual(Variable.__bases__, (AbstractSourcedVariable,))

        # Test passing attributes.
        var = Variable(attrs={'a': 6}, value=np.array([5]))
        self.assertEqual(var.attrs['a'], 6)

        var = Variable(value=np.array([4, 5]), name='tas', alias='foo')
        self.assertEqual(var.name, 'tas')
        self.assertEqual(var.alias, 'foo')

        # Test fill value handled appropriately.
        value = [4, 5, 6]
        var = Variable(value=value, fill_value=100)
        self.assertEqual(np.ma.array(value).fill_value, var.fill_value)
        self.assertIsNone(var._fill_value)
        var = Variable(fill_value=100)
        self.assertEqual(var.fill_value, 100)

        # Test with data type and fill value.
        value = np.array([1, 2, 3, 4])
        var = Variable(request_dataset='foo', dtype=np.float, fill_value=9, value=value)
        self.assertEqual(var.dtype, value.dtype)
        self.assertEqual(var.fill_value, var.value.fill_value)

        # Test data type and fill value from value array.
        value = np.array([1, 2, 3, 4])
        value = np.ma.array(value)
        var = Variable(request_dataset='foo', value=value)
        self.assertEqual(var.dtype, value.dtype)
        self.assertEqual(var.fill_value, value.fill_value)

        # Test conform_units_to.
        value = [4, 5, 6]
        var = Variable(value=value, units='celsius', conform_units_to='kelvin')
        self.assertGreater(var.value.mean(), 200)
        self.assertEqual(var.units, 'kelvin')

    def test_str(self):
        var = Variable(name='toon')
        self.assertEqual(str(var), 'Variable(name="toon", alias="toon", units=None)')

    def test_get_empty_like(self):
        kwargs = dict(name='tas', alias='tas2', units='celsius', meta={'foo': 5}, uid=5, request_dataset='foo')
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

    def test_getitem(self):
        var = Variable(value=np.random.rand(1, 1, 1, 1, 51), name='foo')
        slc = [slice(None, None, None), slice(None, None, None), slice(None, None, None), np.array([0]), np.array([14])]
        ret = var.__getitem__(slc)
        self.assertEqual(ret.shape, tuple([1] * 5))

    def test_iter_melted(self):

        def _assert_key_(attr, key, row, actual_none=None):
            key_value = row[key]
            if attr is not None:
                self.assertEqual(key_value, attr)
            else:
                if actual_none is None:
                    self.assertIsNone(key_value)
                else:
                    self.assertEqual(key_value, actual_none)

        keywords = dict(value=[np.ma.array([[4, 5], [6, 7]], mask=[[False, True], [False, False]])],
                        use_mask=[True, False],
                        name=[None, 'tmax'],
                        alias=[None, 'tmax_alias'],
                        units=[None, 'celsius'],
                        uid=[None, 3],
                        did=[None, 7],
                        name_uid=[None, 'vid'],
                        attrs=[None, {'foo': 1, 'foo3': 2}])

        for k in self.iter_product_keywords(keywords):
            var = Variable(value=k.value, name=k.name, alias=k.alias, units=k.units, uid=k.uid, attrs=k.attrs)
            rows = []
            for row in var.iter_melted(use_mask=k.use_mask):
                self.assertAsSetEqual(row.keys(), ['name', 'value', 'alias', 'units', 'uid', 'attrs', 'meta'])

                if k.name is None:
                    if k.alias is None:
                        self.assertIsNone(row['alias'])
                    else:
                        self.assertEqual(row['alias'], k.alias)
                else:
                    if k.alias is None:
                        self.assertEqual(row['alias'], k.name)
                    else:
                        self.assertEqual(row['alias'], k.alias)

                _assert_key_(k.name, 'name', row)
                _assert_key_(k.units, 'units', row)
                _assert_key_(k.uid, 'uid', row)
                _assert_key_(k.attrs, 'attrs', row, actual_none=OrderedDict())

                rows.append(row)
            if k.use_mask:
                self.assertEqual(len(rows), 3)
            else:
                self.assertEqual(len(rows), 4)

    @attr('data')
    def test_value(self):
        # Test spatial masks are appropriate applied.
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()[:, 3:5, :, 5:7, 10:12]
        mask = np.array([[True, False], [False, True]])
        field.spatial.set_mask(mask)
        self.assertIsNone(field.variables['tas']._value)
        value = field.variables['tas'].value
        actual = np.array([[[[[True, False], [False, True]]],
                            [[[True, False], [False, True]]]]])
        self.assertNumpyAll(value.mask, actual)
        self.assertGreater(value.mask.sum(), 0)


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

    def test_iter_columns(self):
        variables = [self.get_variable(), self.get_variable('tas_foo2')]
        variables[1].value *= 2
        variables[1].value.mask[2] = True
        vc = VariableCollection(variables=variables)
        rows = list(vc.iter_columns())
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1][1].keys(), ['tas_foo', 'tas_foo2'])
        self.assertIsInstance(rows[2][1], OrderedDict)
        for row in rows:
            self.assertTrue(row[0], 20)

    def test_iter_melted(self):
        variables = [self.get_variable(), self.get_variable('tas_foo2')]
        vc = VariableCollection(variables=variables)
        test = set()
        for row in vc.iter_melted():
            test.update([row['alias']])
        self.assertAsSetEqual(test, [xx.alias for xx in variables])
