from unittest import SkipTest

import numpy as np
import six

from ocgis import vm
from ocgis.constants import DataType
from ocgis.exc import EmptyObjectError
from ocgis.test.base import attr, AbstractTestInterface
from ocgis.variable.dimension import Dimension
from ocgis.vmachine.mpi import OcgDist, MPI_SIZE, MPI_RANK, get_nonempty_ranks, barrier_print


class TestDimension(AbstractTestInterface):
    @staticmethod
    def get_dimension(**kwargs):
        name = kwargs.pop('name', 'foo')
        kwargs['size'] = kwargs.get('size', 10)
        return Dimension(name, **kwargs)

    def test_init(self):
        dim = Dimension('foo')
        self.assertEqual(dim.name, 'foo')
        self.assertIsNone(dim.size)
        self.assertIsNone(dim.size_current)
        self.assertTrue(dim.is_unlimited)
        self.assertEqual(len(dim), 0)
        self.assertEqual(dim.bounds_local, (0, len(dim)))
        self.assertEqual(dim.bounds_global, dim.bounds_local)

        dim = Dimension('foo', size=23)
        self.assertEqual(dim.size, 23)

        src_idx = np.arange(0, 10, dtype=DataType.DIMENSION_SRC_INDEX)
        dim = self.get_dimension(src_idx=src_idx)
        self.assertNumpyAll(dim._src_idx, src_idx)
        self.assertEqual(dim._src_idx.shape[0], 10)

        # Test distributed dimensions require and size definition.
        with self.assertRaises(ValueError):
            Dimension('foo', dist=True)

        # Test without size definition and a source index.
        for src_idx in ['auto', 'ato', [0, 1, 2]]:
            try:
                dim = Dimension('foo', src_idx=src_idx)
            except ValueError:
                self.assertNotEqual(src_idx, 'auto')
            else:
                self.assertIsNone(dim._src_idx)

        # Test a unique identifier.
        dim = Dimension('foo', uid=10)
        self.assertEqual(dim.uid, 10)

    def test_init_source_name(self):
        dim = Dimension('foo', source_name='actual_foo')
        self.assertEqual(dim.source_name, 'actual_foo')

        dim = Dimension('foo')
        dim.set_name('new_foo')
        self.assertEqual(dim.source_name, 'foo')

    def test_bounds_local(self):
        dim = Dimension('a', 5)
        dim.bounds_local = [0, 2]
        self.assertEqual(dim.bounds_local, (0, 2))

    def test_convert_to_empty(self):
        dim = Dimension('three', 3, src_idx=[10, 11, 12], dist=True)
        dim.convert_to_empty()
        self.assertFalse(dim.is_unlimited)
        self.assertTrue(dim.is_empty)
        self.assertEqual(len(dim), 0)
        self.assertEqual(dim.size, 0)
        self.assertEqual(dim.size_current, 0)

    def test_copy(self):
        sd = self.get_dimension(src_idx=np.arange(10))
        self.assertIsNotNone(sd._src_idx)
        sd2 = sd.copy()
        self.assertTrue(np.may_share_memory(sd._src_idx, sd2._src_idx))
        sd3 = sd2[2:5]
        self.assertEqual(sd, sd2)
        self.assertNotEqual(sd2, sd3)
        self.assertTrue(np.may_share_memory(sd2._src_idx, sd._src_idx))
        self.assertTrue(np.may_share_memory(sd2._src_idx, sd3._src_idx))
        self.assertTrue(np.may_share_memory(sd3._src_idx, sd._src_idx))

        # Test setting new values on object.
        dim = Dimension('five', 5)
        cdim = dim.copy()
        cdim._name = 'six'
        self.assertEqual(dim.name, 'five')
        self.assertEqual(cdim.name, 'six')

    def test_eq(self):
        lhs = self.get_dimension()
        rhs = self.get_dimension()
        self.assertEqual(lhs, rhs)

    @attr('mpi')
    def test_get_distributed_slice(self):
        self.add_barrier = False
        for d in [True, False]:
            dist = OcgDist()
            dim = dist.create_dimension('five', 5, dist=d, src_idx='auto')
            dist.update_dimension_bounds()
            if not dim.is_empty:
                self.assertEqual(dim.bounds_global, (0, 5))

            if dim.dist:
                if MPI_RANK > 1:
                    self.assertTrue(dim.is_empty)
                else:
                    self.assertFalse(dim.is_empty)
            with vm.scoped_by_emptyable('dim slice', dim):
                if not vm.is_null:
                    sub = dim.get_distributed_slice(slice(1, 3))
                else:
                    sub = None

            if dim.dist:
                if not dim.is_empty:
                    self.assertIsNotNone(dim._src_idx)
                else:
                    self.assertIsNone(sub)

                if MPI_SIZE == 2:
                    desired_emptiness = {0: False, 1: True}[MPI_RANK]
                    desired_bounds_local = {0: (0, 2), 1: (0, 0)}[MPI_RANK]
                    self.assertEqual(sub.is_empty, desired_emptiness)
                    self.assertEqual(sub.bounds_local, desired_bounds_local)
                if MPI_SIZE >= 5 and 0 < MPI_RANK > 2:
                    self.assertTrue(sub is None or sub.is_empty)
            else:
                self.assertEqual(len(dim), 5)
                self.assertEqual(dim.bounds_global, (0, 5))
                self.assertEqual(dim.bounds_local, (0, 5))

        dist = OcgDist()
        dim = dist.create_dimension('five', 5, dist=True, src_idx='auto')
        dist.update_dimension_bounds()
        with vm.scoped_by_emptyable('five slice', dim):
            if not vm.is_null:
                sub2 = dim.get_distributed_slice(slice(2, 4))
            else:
                sub2 = None
        if MPI_SIZE == 3 and MPI_RANK == 2:
            self.assertIsNone(sub2)

    @attr('mpi')
    def test_get_distributed_slice_fancy_indexing(self):
        if vm.size != 2:
            raise SkipTest('vm.size != 2')

        dist = OcgDist()
        dim = dist.create_dimension('dim', size=5, dist=True, src_idx='auto')
        dist.update_dimension_bounds()

        barrier_print('original bounds local', dim._bounds_local)

        slices = {0: [False, True, True],
                  1: [False, True]}
        slc = slices[vm.rank]
        sub = dim.get_distributed_slice(slc)

        desired_bounds_global = (0, sum([np.array(ii).sum() for ii in slices.values()]))
        desired_bounds_local = {0: (0, 2), 1: (2, 3)}
        self.assertEqual(sub.bounds_global, desired_bounds_global)
        self.assertEqual(sub.bounds_local, desired_bounds_local[vm.rank])

    @attr('mpi')
    def test_get_distributed_slice_on_rank_subset(self):
        """Test with some a priori empty dimensions."""
        if MPI_SIZE != 4:
            raise SkipTest('MPI_SIZE != 4')

        ompi = OcgDist()
        dim = ompi.create_dimension('eight', 8, dist=True)
        ompi.update_dimension_bounds()

        sub = dim.get_distributed_slice(slice(2, 6))

        live_ranks = get_nonempty_ranks(sub, vm)
        if MPI_RANK in [1, 2]:
            self.assertFalse(sub.is_empty)
        else:
            self.assertTrue(sub.is_empty)
            self.assertEqual(sub.bounds_local, (0, 0))
            self.assertEqual(sub.bounds_global, (0, 0))

        with vm.scoped('live rank dim subset', live_ranks):
            if not vm.is_null:
                sub2 = sub.get_distributed_slice(slice(2, 4))
            else:
                sub2 = None

        if MPI_RANK == 2:
            self.assertEqual(sub2.bounds_local, (0, 2))
            self.assertEqual(sub2.bounds_global, (0, 2))
            self.assertFalse(sub2.is_empty)
        else:
            self.assertTrue(sub2 is None or sub2.is_empty)

        # Try again w/out scoping.
        if sub.is_empty:
            with self.assertRaises(EmptyObjectError):
                sub.get_distributed_slice(slice(2, 4))

    @attr('mpi')
    def test_get_distributed_slice_simple(self):
        ompi = OcgDist()
        dim = ompi.create_dimension('five', 5, dist=True, src_idx='auto')
        ompi.update_dimension_bounds(min_elements=1)

        with vm.scoped_by_emptyable('simple slice test', dim):
            if not vm.is_null:
                sub = dim.get_distributed_slice(slice(2, 4))
            else:
                sub = None

        if sub is not None and not sub.is_empty:
            self.assertEqual(sub.bounds_global, (0, 2))
        else:
            if dim.is_empty:
                self.assertIsNone(sub)
            else:
                self.assertEqual(sub.bounds_global, (0, 0))
                self.assertEqual(sub.bounds_local, (0, 0))

        # Test global bounds are updated.
        ompi = OcgDist()
        dim = ompi.create_dimension('tester', 768, dist=False)
        ompi.update_dimension_bounds()

        sub = dim.get_distributed_slice(slice(73, 157))
        self.assertEqual(sub.size, 84)
        self.assertEqual(sub.bounds_global, (0, 84))
        self.assertEqual(sub.bounds_local, (0, 84))

    @attr('mpi')
    def test_get_distributed_slice_uneven_boundary(self):
        ompi = OcgDist()
        dim = ompi.create_dimension('the_dim', 360, dist=True, src_idx='auto')
        ompi.update_dimension_bounds()

        sub = dim.get_distributed_slice(slice(12, 112))
        if sub.is_empty:
            self.assertEqual(sub.bounds_local, (0, 0))
            self.assertEqual(sub.bounds_global, (0, 0))
            self.assertIsNone(sub._src_idx)
        else:
            self.assertIsNotNone(sub._src_idx)
            self.assertEqual(sub.bounds_global, (0, 100))

        if MPI_SIZE == 4:
            desired = {0: (0, 78), 1: (78, 100)}
            desired = desired.get(MPI_RANK, (0, 0))
            self.assertEqual(sub.bounds_local, desired)

    def test_getitem(self):
        dim = Dimension('foo', size=50)
        sub = dim[30:40]
        self.assertEqual(len(sub), 10)

        dim = Dimension('foo', size=None)
        with self.assertRaises(IndexError):
            dim[400:500]

        # Test with negative indexing.
        dim = Dimension(name='geom', size=2, src_idx='auto')
        slc = slice(0, -1, None)
        actual = dim[slc]
        self.assertEqual(actual, Dimension('geom', size=1, src_idx='auto'))

        dim = Dimension(name='geom', size=5, src_idx=np.arange(5))
        slc = slice(1, -2, None)
        actual = dim[slc]
        desired = Dimension('geom', size=2, src_idx=[1, 2])
        self.assertEqual(actual, desired)

        dim = self.get_dimension(src_idx=np.arange(10))

        sub = dim[4]
        self.assertEqual(sub.size, 1)

        sub = dim[4:5]
        self.assertEqual(sub.size, 1)

        sub = dim[4:6]
        self.assertEqual(sub.size, 2)

        sub = dim[[4, 5, 6]]
        self.assertEqual(sub.size, 3)

        sub = dim[[2, 4, 6]]
        self.assertEqual(sub.size, 3)
        self.assertNumpyAll(sub._src_idx, dim._src_idx[[2, 4, 6]])

        sub = dim[:]
        self.assertEqual(len(sub), len(dim))

        dim = self.get_dimension()
        sub = dim[2:]
        self.assertEqual(len(sub), 8)

        dim = self.get_dimension(src_idx=np.arange(10))
        sub = dim[3:-1]
        np.testing.assert_equal(sub._src_idx, [3, 4, 5, 6, 7, 8])

        dim = self.get_dimension(src_idx=np.arange(10))
        sub = dim[-3:]
        self.assertEqual(sub._src_idx.shape[0], sub.size)

        dim = self.get_dimension(src_idx=np.arange(10))
        sub = dim[-7:-3]
        self.assertEqual(sub._src_idx.shape[0], sub.size)

        dim = self.get_dimension(src_idx=np.arange(10))
        sub = dim[:-3]
        self.assertEqual(sub._src_idx.shape[0], sub.size)

        # Test source index is None after slicing.
        dim = Dimension('water', 10)
        sub = dim[0:3]
        self.assertIsNone(sub._src_idx)

    def test_is_matched_by_alias(self):
        dim = Dimension('non_standard_time')
        dim.append_alias('time')
        self.assertTrue(dim.is_matched_by_alias('time'))

    def test_len(self):
        dim = Dimension('foo')
        self.assertEqual(len(dim), 0)

        # Test size current is used for length.
        dim = Dimension('unlimited', size=None, size_current=4)
        self.assertEqual(len(dim), 4)

    def test_set_size(self):
        dim = self.get_dimension(src_idx='auto')
        kwds = {'size': [None, 0, 1, 3], 'src_idx': [None, 'auto', np.arange(3)]}
        for k in self.iter_product_keywords(kwds):
            try:
                dim.set_size(k.size, src_idx=k.src_idx)
            except ValueError:
                self.assertTrue(k.size != 3)
                self.assertIsInstance(k.src_idx, np.ndarray)
                continue
            if k.size is None:
                self.assertTrue(dim.is_unlimited)
            else:
                self.assertEqual(len(dim), k.size)
            if k.src_idx is None:
                self.assertIsNone(dim._src_idx)
            elif isinstance(k.src_idx, six.string_types) and k.src_idx == 'auto' and k.size is not None:
                self.assertIsNotNone(dim._src_idx)
