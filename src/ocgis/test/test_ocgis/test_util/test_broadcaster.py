import itertools
from copy import deepcopy

import numpy as np
from six.moves import zip_longest

from ocgis import Variable
from ocgis.test.base import AbstractTestInterface
from ocgis.util.broadcaster import broadcast_array_by_dimension_names, broadcast_variable


class Test(AbstractTestInterface):
    def test(self):
        arr = np.random.rand(2, 3, 4)
        arr = np.ma.array(arr, mask=False)
        src_names = ['time', 'x', 'y']
        dst_names = ['realization', 'time', 'level', 'y', 'x']
        carr = broadcast_array_by_dimension_names(arr, src_names, dst_names)
        self.assertEqual(carr.shape, (1, 2, 1, 4, 3))
        self.assertTrue(np.may_share_memory(arr, carr))
        self.assertFalse(np.any(arr.mask))
        carr.mask[:] = True
        self.assertTrue(np.all(carr.mask))
        self.assertTrue(np.all(arr.mask))
        self.assertTrue(np.may_share_memory(arr.mask, carr.mask))
        carr.mask[:] = False

        carr2 = broadcast_array_by_dimension_names(carr, dst_names, src_names)

        self.assertNumpyAll(arr, carr2)
        self.assertTrue(np.may_share_memory(arr, carr2))
        # self.assertTrue(np.may_share_memory(arr.mask, carr2.mask))

        carr2[:] = 10
        self.assertEqual(arr.mean(), 10)

    def test_system_additional_dimension(self):
        """Test with a randomly named dimension with no metadata definition."""

        arr = np.random.rand(2, 3, 4, 5, 6)
        src_names = ['time', 'flood', 'x', 'y', 'heat_wave']
        dst_and_field_names = ['realization', 'time', 'level', 'y', 'x']

        extra = [dn for dn in src_names if dn not in dst_and_field_names]
        extra = {e: {'index': src_names.index(e)} for e in extra}
        for v in list(extra.values()):
            v['size'] = arr.shape[v['index']]

        src_names_extra_removed = [dn for dn in src_names if dn not in extra]

        itr = itertools.product(
            *[zip_longest([v['index']], list(range(v['size'])), fillvalue=v['index']) for v in
              list(extra.values())])
        for indices in itr:
            slc = [slice(None)] * arr.ndim
            for ii in indices:
                slc[ii[0]] = ii[1]
            extras_removed = arr.__getitem__(slc)

            carr = broadcast_array_by_dimension_names(extras_removed, src_names_extra_removed, dst_and_field_names)

            self.assertEqual(carr.shape, (1, 2, 1, 5, 4))
            self.assertNumpyMayShareMemory(arr, carr)

    def test_broadcast_variable(self):
        value = np.random.rand(3, 4, 5)
        desired_value = deepcopy(value)
        mask = desired_value > 0.5
        desired_mask = deepcopy(mask)
        original_dimensions = ['time', 'lat', 'lon']
        src = Variable(name='src', value=value, mask=mask, dimensions=original_dimensions)
        dst_names = ['lon', 'lat', 'time']
        broadcast_variable(src, dst_names)
        self.assertEqual(src.shape, (5, 4, 3))
        self.assertEqual(src.get_value().shape, (5, 4, 3))
        self.assertEqual(desired_value.sum(), src.get_value().sum())
        broadcast_variable(src, original_dimensions)
        self.assertNumpyAll(desired_value, src.get_value())
        self.assertNumpyMayShareMemory(value, src.get_value())
        self.assertNumpyAll(desired_mask, src.get_mask())
