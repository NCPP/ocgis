import itertools

import numpy as np

from ocgis.test.base import AbstractTestInterface
from ocgis.util.conformer import conform_array_by_dimension_names


class Test(AbstractTestInterface):
    def test(self):
        arr = np.random.rand(2, 3, 4)
        arr = np.ma.array(arr, mask=False)
        src_names = ['time', 'x', 'y']
        dst_names = ['realization', 'time', 'level', 'y', 'x']
        carr = conform_array_by_dimension_names(arr, src_names, dst_names)
        self.assertEqual(carr.shape, (1, 2, 1, 4, 3))
        self.assertTrue(np.may_share_memory(arr, carr))
        self.assertFalse(np.any(arr.mask))
        carr.mask[:] = True
        self.assertTrue(np.all(carr.mask))
        self.assertTrue(np.all(arr.mask))
        self.assertTrue(np.may_share_memory(arr.mask, carr.mask))
        carr.mask[:] = False

        carr2 = conform_array_by_dimension_names(carr, dst_names, src_names)

        self.assertNumpyAll(arr, carr2)
        self.assertTrue(np.may_share_memory(arr, carr2))
        # self.assertTrue(np.may_share_memory(arr.mask, carr2.mask))

        carr2[:] = 10
        self.assertEqual(arr.mean(), 10)

    def test_additional_dimension(self):
        """Test with a randomly named dimension with no metadata definition."""

        arr = np.random.rand(2, 3, 4, 5, 6)
        src_names = ['time', 'flood', 'x', 'y', 'heat_wave']
        dst_and_field_names = ['realization', 'time', 'level', 'y', 'x']

        extra = [dn for dn in src_names if dn not in dst_and_field_names]
        extra = {e: {'index': src_names.index(e)} for e in extra}
        for v in extra.values():
            v['size'] = arr.shape[v['index']]

        src_names_extra_removed = [dn for dn in src_names if dn not in extra]

        itr = itertools.product(
            *[itertools.izip_longest([v['index']], range(v['size']), fillvalue=v['index']) for v in extra.values()])
        for indices in itr:
            slc = [slice(None)] * arr.ndim
            for ii in indices:
                slc[ii[0]] = ii[1]
            extras_removed = arr.__getitem__(slc)

            carr = conform_array_by_dimension_names(extras_removed, src_names_extra_removed, dst_and_field_names)

            self.assertEqual(carr.shape, (1, 2, 1, 5, 4))
            self.assertNumpyMayShareMemory(arr, carr)
