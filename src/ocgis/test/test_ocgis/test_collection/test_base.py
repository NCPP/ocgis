from copy import deepcopy

import numpy as np

from ocgis.collection.base import AbstractCollection
from ocgis.test.base import TestBase


class TestAbstractCollection(TestBase):
    create_dir = False

    def get_coll(self):
        coll = AbstractCollection()
        coll[2] = 'a'
        coll[1] = 'b'
        return coll

    def test_init(self):
        self.assertIsInstance(self.get_coll(), AbstractCollection)

    def test_contains(self):
        coll = self.get_coll()
        self.assertTrue(1 in coll)
        self.assertFalse(3 in coll)

    def test_copy(self):
        coll = self.get_coll()
        coll['arr'] = np.array([1, 2, 3])
        coll2 = coll.copy()
        self.assertEqual(coll, coll2)
        coll2[3] = 'c'
        self.assertNotIn(3, coll)
        self.assertTrue(np.may_share_memory(coll['arr'], coll2['arr']))

    def test_deepcopy(self):
        coll = self.get_coll()
        self.assertEqual(coll, deepcopy(coll))

    def test_values(self):
        self.assertEqual(list(self.get_coll().values()), ['a', 'b'])

    def test_keys(self):
        self.assertEqual(list(self.get_coll().keys()), [2, 1])

    def test_pop(self):
        self.assertEqual(self.get_coll().pop('hi', 'there'), 'there')
        coll = self.get_coll()
        val = coll.pop(2)
        self.assertEqual(val, 'a')
        self.assertDictEqual(coll, {1: 'b'})

    def test_getitem(self):
        self.assertEqual(self.get_coll()[2], 'a')

    def test_setitem(self):
        coll = self.get_coll()
        coll['a'] = 400
        self.assertEqual(coll['a'], 400)

    def test_repr(self):
        self.assertEqual(repr(self.get_coll()), "AbstractCollection([(2, 'a'), (1, 'b')])")

    def test_str(self):
        coll = self.get_coll()
        self.assertEqual(repr(coll), str(coll))

    def test_first(self):
        self.assertEqual(self.get_coll().first(), 'a')

    def test_update(self):
        coll = self.get_coll()
        coll.update({'t': 'time'})
        self.assertEqual(list(coll.keys()), [2, 1, 't'])

    def test_iter(self):
        self.assertEqual(list(self.get_coll().__iter__()), [2, 1])
