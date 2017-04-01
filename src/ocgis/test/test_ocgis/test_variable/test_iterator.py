from collections import OrderedDict
from unittest import SkipTest

import numpy as np

from ocgis.test.base import TestBase, attr
from ocgis.variable.base import Variable
from ocgis.variable.iterator import Iterator


class TestIterator(TestBase):
    @staticmethod
    def create_base_variable(shape=None, name='foo'):
        if shape is None:
            shape = (10, 20)
        var = Variable(name=name, value=np.random.rand(*shape), dimensions=['a', 'b'])
        return var

    @attr('data')
    def test_system_cf_data(self):
        raise SkipTest('profiling only')
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get().get_field_slice({'time': slice(0, 100)}, strict=False)
        follower_names = ['time', 'lon', 'lat']
        itr = Iterator(field['tas'], followers=[field[ii] for ii in follower_names])
        print(field.shapes)
        list(itr)

    def test_system_profile_iterator(self):
        # 24 seconds last profile
        raise SkipTest('profiling only')
        var1 = self.create_base_variable(shape=(1000, 1000), name='one')
        var2 = self.create_base_variable(shape=(1000, 1000), name='two')
        var3 = self.create_base_variable(shape=(1000, 1000), name='three')
        actual = list(Iterator(var1, followers=[var2, var3]))
        self.assertEqual(len(actual), 1000 * 1000)

    def test_iter(self):
        var = self.create_base_variable()
        itr = Iterator(var)

        for _ in range(2):
            as_list = list(itr)
            self.assertEqual(len(as_list), var.shape[0] * var.shape[1])
            self.assertIsInstance(as_list[0], OrderedDict)
            self.assertEqual(len(as_list[0]), 1)

    def test_iter_followers(self):
        base_var = self.create_base_variable()
        follower_var = Variable(name='follower', value=np.random.rand(base_var.shape[1]),
                                dimensions=base_var.dimensions[1].name)
        itr = Iterator(base_var, followers=[follower_var])

        as_list = list(itr)
        self.assertEqual(len(as_list), base_var.shape[0] * base_var.shape[1])
        actual = as_list[0]
        self.assertIn(base_var.name, actual)
        self.assertIn(follower_var.name, actual)

        # Test follower dimensions are not a subset of the base variable dimensions.
        follower_var = Variable(name='follower', value=np.random.rand(base_var.shape[1]), dimensions='unique')
        with self.assertRaises(ValueError):
            Iterator(base_var, followers=[follower_var])

    def test_iter_follower_with_followers(self):
        leader = Variable(name='data', value=np.arange(6).reshape(2, 3), dimensions=['x', 'y'])

        x = Variable(name='x', value=[30., 40.], dimensions='x')
        x_follower = Variable(name='x_bounds', value=[25., 35.], dimensions='x')

        follower_iterator = Iterator(x, followers=[x_follower])

        itr = Iterator(leader, followers=[follower_iterator])

        desired = [OrderedDict([('data', 0), ('x', 30.0), ('x_bounds', 25.0)]),
                   OrderedDict([('data', 1), ('x', 30.0), ('x_bounds', 25.0)]),
                   OrderedDict([('data', 2), ('x', 30.0), ('x_bounds', 25.0)]),
                   OrderedDict([('data', 3), ('x', 40.0), ('x_bounds', 35.0)]),
                   OrderedDict([('data', 4), ('x', 40.0), ('x_bounds', 35.0)]),
                   OrderedDict([('data', 5), ('x', 40.0), ('x_bounds', 35.0)])]
        self.assertEqual(list(itr), desired)

    def test_iter_mask(self):
        var = self.create_base_variable(shape=(2, 2))
        mask = var.get_mask(create=True)
        mask[0, 1] = True
        var.set_mask(mask)

        for allow_masked in [True, False]:
            itr = Iterator(var, allow_masked=allow_masked)
            as_list = list(itr)
            if allow_masked:
                self.assertEqual(len(as_list), 4)
                self.assertIsNone(as_list[1][var.name])
            else:
                self.assertEqual(len(as_list), 3)

    def test_iter_formatter(self):
        def _formatter_(name, value, mask):
            if value is None:
                modified_value = None
            else:
                modified_value = value * 1000
                value = str(value)
            ret = [(name, value), ('modified', modified_value)]
            return ret

        var = Variable(name='data', value=[1, 2, 3], mask=[False, True, False], dimensions='dim')
        itr = Iterator(var, formatter=_formatter_)

        as_list = list(itr)
        actual = as_list[1][var.name]
        self.assertIsNone(actual)
        self.assertEqual(as_list[2][var.name], str(var.get_value()[2]))
        self.assertEqual(as_list[0]['modified'], 1000)

    def test_iter_melted(self):
        var = self.create_base_variable()
        with self.assertRaises(ValueError):
            Iterator(var, melted=var)

        var = Variable(name='lead', value=[1, 2], dimensions='dim')
        fvar = Variable(name='follower', value=[3, 4], dimensions='dim')
        emelted = Variable(name='just_melted', value=[5, 6], dimensions='dim')
        itr = Iterator(var, followers=[fvar, emelted], melted=[var, emelted])

        desired = [OrderedDict([('follower', 3), ('VARIABLE', 'lead'), ('VALUE', 1)]),
                   OrderedDict([('follower', 3), ('VARIABLE', 'just_melted'), ('VALUE', 5)]),
                   OrderedDict([('follower', 4), ('VARIABLE', 'lead'), ('VALUE', 2)]),
                   OrderedDict([('follower', 4), ('VARIABLE', 'just_melted'), ('VALUE', 6)])]
        self.assertEqual(list(itr), desired)

    def test_iter_primary_mask(self):
        var = Variable('a', [1, 2, 3], 'b', mask=np.ones(3, dtype=bool))
        primary_mask = Variable('c', [4, 5, 6], 'b', mask=[True, False, True])
        itr = Iterator(var, followers=[primary_mask], primary_mask=primary_mask, allow_masked=False)
        actual = list(itr)
        self.assertIsNone(actual[0]['a'])

    def test_iter_repeater(self):
        var1 = Variable(name='var1', value=[1, 2, 3], dimensions='dim')
        var2 = Variable(name='var2', value=[1, 2, 3], dimensions='dim')
        var2.get_value()[:] *= 9
        repeater = ('i_am', 'a_repeater')
        itr = Iterator(var1, followers=[var2], repeaters=[repeater])

        desired = [OrderedDict([('i_am', 'a_repeater'), ('var1', 1), ('var2', 9)]),
                   OrderedDict([('i_am', 'a_repeater'), ('var1', 2), ('var2', 18)]),
                   OrderedDict([('i_am', 'a_repeater'), ('var1', 3), ('var2', 27)])]
        actual = list(itr)
        self.assertEqual(actual, desired)

    def test_iter_repeater_on_follower(self):

        var1 = Variable(name='f', value=[1, 2], dimensions='one', repeat_record=[('same', 'var1 owns this')])
        var2 = Variable(name='g', value=[3, 4], dimensions='one', repeat_record=[('same', 'var2 owns this')])

        itr = Iterator(var1, followers=[var2], melted=[var1, var2])
        for row in itr:
            if row['VARIABLE'] == var2.name:
                self.assertEqual(row['same'], var2.repeat_record[0][1])
            elif row['VARIABLE'] == var1.name:
                self.assertEqual(row['same'], var1.repeat_record[0][1])

    def test_iter_with_bounds(self):
        var = Variable(name='bounded', value=[1, 2, 3, 4], dtype=float, dimensions='dim')
        var.set_extrapolated_bounds('the_bounds', 'bounds')

        lower = Variable(name='lower_bounds', value=var.bounds.get_value()[:, 0], dimensions=var.dimensions)
        upper = Variable(name='upper_bounds', value=var.bounds.get_value()[:, 1], dimensions=var.dimensions)

        itr = Iterator(var, followers=[lower, upper])

        actual = list(itr)

        self.assertEqual(len(actual), var.shape[0])
        self.assertEqual(len(actual[0]), 3)
