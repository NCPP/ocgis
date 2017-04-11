import itertools
from copy import deepcopy
from unittest import SkipTest

import numpy as np

from ocgis import vm
from ocgis.constants import DataType
from ocgis.test.base import attr, AbstractTestInterface
from ocgis.util.helpers import get_local_to_global_slices
from ocgis.variable.base import Variable, VariableCollection
from ocgis.variable.dimension import Dimension
from ocgis.vmachine.mpi import MPI_SIZE, MPI_COMM, create_nd_slices, hgather, \
    get_optimal_splits, get_rank_bounds, OcgMpi, get_global_to_local_slice, MPI_RANK, variable_scatter, \
    variable_collection_scatter, variable_gather, gather_ranks, get_standard_comm_state, \
    get_nonempty_ranks, barrier_ranks, bcast_ranks


class Test(AbstractTestInterface):
    @attr('mpi')
    def test_barrier_ranks(self):
        if MPI_SIZE != 3 and MPI_SIZE != 1:
            raise SkipTest('serial or mpi-3 only')

        barrier_ranks([0])

        ranks = [0, 2]
        if MPI_RANK != 1:
            barrier_ranks(ranks)

    @attr('mpi')
    def test_groups(self):
        from mpi4py.MPI import COMM_NULL

        if MPI_SIZE != 8:
            raise SkipTest('mpi-8 only')
        world_group = MPI_COMM.Get_group()
        sub_group = world_group.Incl([3, 6])
        new_comm = MPI_COMM.Create(sub_group)
        if new_comm != COMM_NULL:
            if new_comm.Get_rank() == 0:
                data = 'what'
            else:
                data = None
            data = new_comm.bcast(data)
            new_comm.Barrier()

        if new_comm != COMM_NULL:
            sub_group.Free()
            new_comm.Free()

    @attr('mpi')
    def test_gather_ranks(self):
        if MPI_SIZE != 3 and MPI_SIZE != 1:
            raise SkipTest('serial or mpi-3 only')

        value = 33
        for root in range(MPI_SIZE):
            gathered = gather_ranks([0], value, root=root)
            if MPI_RANK == root:
                self.assertEqual(gathered, [33])
            else:
                self.assertIsNone(gathered)

        if MPI_SIZE == 3:
            for root in range(MPI_SIZE):
                ranks = [1, 2]
                gathered = gather_ranks(ranks, MPI_RANK + 10, root=root)
                if MPI_RANK == root:
                    self.assertEqual(gathered, [11, 12])
                else:
                    self.assertIsNone(gathered)

    @attr('mpi')
    def test_get_nonempty_ranks(self):
        from ocgis.variable.dimension import Dimension
        comm, rank, size = get_standard_comm_state()

        if size not in [1, 3]:
            raise SkipTest('MPI_SIZE != 1 or 3')

        target = Dimension('a')
        live_ranks = get_nonempty_ranks(target, vm)
        if MPI_RANK == 0:
            self.assertEqual(live_ranks, tuple(range(size)))

        if MPI_SIZE == 3:
            targets = {0: Dimension('a', is_empty=True, dist=True),
                       1: Dimension('a'),
                       2: Dimension('a')}
            with vm.scoped('ner', vm.ranks):
                live_ranks = get_nonempty_ranks(targets[MPI_RANK], vm)
            self.assertEqual(live_ranks, (1, 2))

    def test_get_optimal_splits(self):
        size = 11
        shape = (4, 3)
        splits = get_optimal_splits(size, shape)
        self.assertEqual(splits, (3, 3))

        size = 2
        shape = (4, 3)
        splits = get_optimal_splits(size, shape)
        self.assertEqual(splits, (2, 1))

    def test_create_nd_slices2(self):

        size = (1, 1)
        shape = (4, 3)
        actual = create_nd_slices(size, shape)
        self.assertEqual(actual, ((slice(0, 4, None), slice(0, 3, None)),))

        size = (2, 1)
        shape = (4, 3)
        actual = create_nd_slices(size, shape)
        self.assertEqual(actual, ((slice(0, 2, None), slice(0, 3, None)), (slice(2, 4, None), slice(0, 3, None))))

        size = (4, 2)
        shape = (4, 3)
        actual = create_nd_slices(size, shape)
        to_test = np.arange(12).reshape(*shape)
        pieces = []
        for a in actual:
            pieces.append(to_test[a].reshape(-1))
        self.assertNumpyAll(hgather(pieces).reshape(*shape), to_test)

    def test_get_rank_bounds(self):

        def _run_(arr, nproc):
            desired = arr.sum()

            actual = 0
            length = len(arr)
            for pet in range(nproc + 3):
                bounds = get_rank_bounds(length, nproc, pet)
                if bounds is None:
                    try:
                        self.assertTrue(pet >= (nproc - length) or (nproc > length and pet >= length))
                        self.assertTrue(length < nproc or pet >= nproc)
                    except AssertionError:
                        self.log.debug('   args: {}, {}, {}'.format(length, nproc, pet))
                        self.log.debug(' bounds: {}'.format(bounds))
                        raise
                else:
                    actual += arr[bounds[0]:bounds[1]].sum()

            try:
                assert np.isclose(actual, desired)
            except AssertionError:
                self.log.debug('   args: {}, {}, {}'.format(length, nproc, pet))
                self.log.debug(' bounds: {}'.format(bounds))
                self.log.debug(' actual: {}'.format(actual))
                self.log.debug('desired: {}'.format(desired))
                raise

        lengths = [1, 2, 3, 4, 5, 6, 8, 100, 333, 1333, 10001]
        nproc = [1, 2, 3, 4, 5, 6, 8, 1000, 1333]

        for l, n in itertools.product(lengths, nproc):
            arr = np.random.rand(l) * 100.0
            _run_(arr, n)

        # Test with Nones.
        res = get_rank_bounds(10, MPI_SIZE, MPI_RANK)
        self.assertEqual(res, (0, 10))

        # Test outside the number of elements.
        res = get_rank_bounds(4, size=1000, rank=900)
        self.assertIsNone(res)

        # Test on the edge.
        ret = get_rank_bounds(5, size=8, rank=5)
        self.assertIsNone(ret)

        # Test with more elements than procs.
        _run_(np.arange(6), 5)

        # Test with rank higher than size.
        res = get_rank_bounds(6, size=5, rank=6)
        self.assertIsNone(res)

    def test_get_local_to_global_slices(self):
        # tdk: consider removing this function
        slices_global = (slice(2, 4, None), slice(0, 2, None))
        slices_local = (slice(0, 1, None), slice(0, 2, None))

        lm = get_local_to_global_slices(slices_global, slices_local)
        self.assertEqual(lm, (slice(2, 3, None), slice(0, 2, None)))

    def test_get_global_to_local_slice(self):
        start_stop = (1, 4)
        bounds_local = (0, 3)
        desired = (1, 3)
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (1, 4)
        bounds_local = (3, 5)
        desired = (0, 1)
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (1, 4)
        bounds_local = (4, 8)
        desired = None
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (3, 4)
        bounds_local = (3, 4)
        desired = (0, 1)
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (10, 20)
        bounds_local = (8, 10)
        desired = None
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (10, 20)
        bounds_local = (12, 15)
        desired = (0, 3)
        actual = get_global_to_local_slice(start_stop, bounds_local)
        self.assertEqual(actual, desired)

        start_stop = (None, None)
        bounds_local = (12, 15)
        with self.assertRaises(ValueError):
            _ = get_global_to_local_slice(start_stop, bounds_local)

    @attr('mpi')
    def test_bcast_ranks(self):
        if MPI_SIZE != 3 and MPI_SIZE != 1:
            raise SkipTest('serial or mpi-3 only')

        actual = bcast_ranks([0], 50)

        if MPI_RANK == 0:
            self.assertEqual(actual, 50)
        else:
            self.assertIsNone(actual)

        actual = bcast_ranks([0, 2], 50)
        if MPI_RANK == 1:
            self.assertEqual(actual, None)
        else:
            self.assertEqual(actual, 50)

        if MPI_RANK != 1:
            actual = bcast_ranks([0, 2], 50)
            self.assertEqual(actual, 50)

    # tdk: wtf
    @attr('mpi')
    def test_variable_collection_scatter(self):
        dest_mpi = OcgMpi()
        five = dest_mpi.create_dimension('five', 5, dist=True)
        ten = dest_mpi.create_dimension('ten', 10)
        dest_mpi.create_variable(name='five', dimensions=five)
        dest_mpi.create_variable(name='all_in', dimensions=ten)
        dest_mpi.create_variable(name='i_could_be_a_coordinate_system')
        dest_mpi.update_dimension_bounds()

        if MPI_RANK == 0:
            var = Variable('holds_five', np.arange(5), dimensions='five')
            var_empty = Variable('i_could_be_a_coordinate_system', attrs={'reality': 'im_not'})
            var_not_dist = Variable('all_in', value=np.arange(10) + 10, dimensions='ten')
            vc = VariableCollection(variables=[var, var_empty, var_not_dist])
        else:
            vc = None

        svc = variable_collection_scatter(vc, dest_mpi)

        self.assertEqual(svc['i_could_be_a_coordinate_system'].attrs['reality'], 'im_not')

        if MPI_RANK < 2:
            self.assertFalse(svc['all_in'].is_empty)
            self.assertNumpyAll(svc['all_in'].get_value(), np.arange(10) + 10)
            self.assertFalse(svc.is_empty)
            self.assertFalse(svc['i_could_be_a_coordinate_system'].is_empty)
        else:
            self.assertTrue(svc['all_in'].is_empty)
            self.assertTrue(svc.is_empty)
            self.assertTrue(svc['i_could_be_a_coordinate_system'].is_empty)

        if MPI_RANK == 0:
            self.assertNumpyAll(var.get_value(), vc[var.name].get_value())

        actual = svc['holds_five'].get_value()
        if MPI_SIZE == 2:
            desired = {0: np.arange(3), 1: np.arange(3, 5)}
            self.assertNumpyAll(actual, desired[MPI_RANK])

        actual = svc['holds_five'].is_empty
        if MPI_RANK > 1:
            self.assertTrue(actual)
        else:
            self.assertFalse(actual)

    @attr('mpi', 'wtf')
    def test_variable_gather(self):
        dist = OcgMpi()
        three = dist.create_dimension('three', 3, src_idx=np.arange(3) * 10)
        four = dist.create_dimension('four', 4, src_idx='auto', dist=True)
        dist.create_variable('four', dimensions=[three, four])
        dist.update_dimension_bounds()

        if MPI_RANK == 0:
            np.random.seed(1)
            mask_value = np.random.random(12).reshape(3, 4)
            mask = Variable('mask', value=mask_value, dimensions=['three', 'four'])
        else:
            mask = None

        mask = variable_scatter(mask, dist)
        with vm.scoped('mask gather', dist.get_empty_ranks(inverse=True)):
            if not vm.is_null:
                mask_gather = variable_gather(mask)
            else:
                mask_gather = None

        if MPI_RANK == 0:
            self.assertNumpyAll(mask_gather.get_value(), mask_value)
            self.assertNumpyAll(mask_gather.dimensions[0]._src_idx, np.arange(3) * 10)
            self.assertNumpyAll(mask_gather.dimensions[1]._src_idx, np.arange(4, dtype=DataType.DIMENSION_SRC_INDEX))
            for dim in mask_gather.dimensions:
                self.assertFalse(dim.dist)
        else:
            self.assertIsNone(mask_gather)

    @attr('mpi')
    def test_variable_scatter(self):
        var_value = np.arange(5, dtype=float) + 50
        var_mask = np.array([True, True, False, True, False])

        dest_dist = OcgMpi()
        five = dest_dist.create_dimension('five', 5, src_idx='auto', dist=True)
        dest_dist.update_dimension_bounds()

        if MPI_RANK == 0:
            local_dim = Dimension('local', 5, src_idx='auto')
            dim_src_idx = local_dim._src_idx.copy()

            var = Variable('the_five', value=var_value, mask=var_mask, dimensions=five.name)
            var.set_extrapolated_bounds('the_five_bounds', 'bounds')
            var_bounds_value = var.bounds.get_value()
        else:
            var, var_bounds_value, dim_src_idx = [None] * 3

        svar = variable_scatter(var, dest_dist)

        var_bounds_value = MPI_COMM.bcast(var_bounds_value)
        dim_src_idx = MPI_COMM.bcast(dim_src_idx)

        if MPI_RANK > 1:
            self.assertIsNone(svar.get_value())
            self.assertTrue(svar.is_empty)
        else:
            dest_dim = dest_dist.get_dimension('five')
            self.assertNumpyAll(var_value[slice(*dest_dim.bounds_local)], svar.get_value())
            self.assertNumpyAll(var_mask[slice(*dest_dim.bounds_local)], svar.get_mask())
            self.assertNumpyAll(var_bounds_value[slice(*dest_dim.bounds_local)], svar.bounds.get_value())
            self.assertNumpyAll(dim_src_idx[slice(*dest_dim.bounds_local)], svar.dimensions[0]._src_idx)
            self.assertNumpyAll(dim_src_idx[slice(*dest_dim.bounds_local)], svar.bounds.dimensions[0]._src_idx)

    @attr('mpi')
    def test_variable_scatter_ndimensions(self):

        r = Dimension('realization', 3)
        t = Dimension('time', 365)
        l = Dimension('level', 10)
        y = Dimension('y', 90, dist=True)
        x = Dimension('x', 360, dist=True)

        dimensions = [r, t, l, y, x]

        dest_mpi = OcgMpi()
        for d in dimensions:
            dest_mpi.add_dimension(d)
        dest_mpi.update_dimension_bounds()

        if MPI_RANK == 0:
            local_dimensions = deepcopy(dimensions)
            for l in local_dimensions:
                l.dist = False
            var = Variable('tas', dimensions=local_dimensions)
        else:
            var = None

        svar = variable_scatter(var, dest_mpi)

        self.assertTrue(svar.dist)
        self.assertIsNotNone(svar)

        if MPI_SIZE == 2:
            self.assertEqual(svar.shape, (3, 365, 10, 90, 180))


class TestOcgMpi(AbstractTestInterface):
    def get_ocgmpi_01(self):
        s = Dimension('first_dist', size=5, dist=True, src_idx='auto')
        ompi = OcgMpi()
        ompi.add_dimension(s)
        ompi.create_dimension('not_dist', size=8, dist=False)
        ompi.create_dimension('another_dist', size=6, dist=True)
        ompi.create_dimension('another_not_dist', size=100, dist=False)
        ompi.create_dimension('coordinate_reference_system', size=0)
        self.assertIsNotNone(s._src_idx)
        return ompi

    def test_init(self):
        ompi = OcgMpi(size=2)
        self.assertEqual(len(ompi.mapping), 2)

        dim_x = Dimension('x', 5, dist=False)
        dim_y = Dimension('y', 11, dist=True)
        var_tas = Variable('tas', value=np.arange(0, 5 * 11).reshape(5, 11), dimensions=(dim_x, dim_y))
        thing = Variable('thing', value=np.arange(11) * 10, dimensions=(dim_y,))

        vc = VariableCollection(variables=[var_tas, thing])
        child = VariableCollection(name='younger')
        vc.add_child(child)
        childer = VariableCollection(name='youngest')
        child.add_child(childer)
        dim_six = Dimension('six', 6)
        hidden = Variable('hidden', value=[6, 7, 8, 9, 0, 10], dimensions=dim_six)
        childer.add_variable(hidden)

        ompi.add_dimensions([dim_x, dim_y])
        ompi.add_dimension(dim_six, group=hidden.group)
        ompi.add_variables([var_tas, thing])
        ompi.add_variable(hidden)

        var = ompi.get_variable(hidden)
        self.assertIsInstance(var, dict)

    @attr('mpi')
    def test(self):
        ompi = OcgMpi()
        src_idx = [2, 3, 4, 5, 6]
        dim = ompi.create_dimension('foo', size=5, group='subroot', dist=True, src_idx=src_idx)
        self.assertEqual(dim, ompi.get_dimension(dim.name, group='subroot'))
        self.assertEqual(dim.bounds_local, (0, len(dim)))
        self.assertFalse(dim.is_empty)
        ompi.update_dimension_bounds()
        with self.assertRaises(ValueError):
            ompi.update_dimension_bounds()

        if ompi.size == 2:
            if MPI_RANK == 0:
                self.assertEqual(dim.bounds_local, (0, 3))
                self.assertEqual(dim._src_idx.tolist(), [2, 3, 4])
            elif MPI_RANK == 1:
                self.assertEqual(dim.bounds_local, (3, 5))
                self.assertEqual(dim._src_idx.tolist(), [5, 6])
        elif ompi.size == 8:
            if MPI_RANK <= 1:
                self.assertTrue(len(dim) >= 2)
            else:
                self.assertTrue(dim.is_empty)

        # Test with multiple dimensions.
        d1 = Dimension('d1', size=5, dist=True, src_idx='auto')
        d2 = Dimension('d2', size=10, dist=False)
        d3 = Dimension('d3', size=3, dist=True)
        dimensions = [d1, d2, d3]
        ompi = OcgMpi()
        for dim in dimensions:
            ompi.add_dimension(dim)
        ompi.update_dimension_bounds()
        bounds_local = ompi.get_bounds_local()
        if ompi.size <= 2:
            desired = {(1, 0): ((0, 5), (0, 10), (0, 3)),
                       (2, 0): ((0, 3), (0, 10), (0, 3)),
                       (2, 1): ((3, 5), (0, 10), (0, 3))}
            self.assertAsSetEqual(bounds_local, desired[(ompi.size, MPI_RANK)])
        else:
            if MPI_RANK <= 1:
                self.assertTrue(dimensions[0]._src_idx.shape[0] <= 3)
            for dim in dimensions:
                if MPI_RANK >= 2:
                    if dim.name == 'd1':
                        self.assertTrue(dim.is_empty)
                    else:
                        self.assertFalse(dim.is_empty)
                else:
                    self.assertFalse(dim.is_empty)

        # Test adding an existing dimension.
        ompi = OcgMpi()
        ompi.create_dimension('one')
        with self.assertRaises(ValueError):
            ompi.create_dimension('one')

    def test_create_or_get_group(self):
        ocmpi = OcgMpi()
        _ = ocmpi._create_or_get_group_(['moon', 'base'])
        _ = ocmpi._create_or_get_group_(None)
        _ = ocmpi._create_or_get_group_('flower')
        ocmpi.create_dimension('foo', group=['moon', 'base'])
        ocmpi.create_dimension('end_of_days')
        ocmpi.create_dimension('start_of_days')

        actual = ocmpi.get_dimension('foo', ['moon', 'base'])
        desired = Dimension('foo')
        self.assertEqual(actual, desired)

        desired = {0: {None: {'variables': {},
                              'dimensions': {'end_of_days': Dimension(name='end_of_days', size=None, size_current=None),
                                             'start_of_days': Dimension(name='start_of_days', size=None,
                                                                        size_current=None)},
                              'groups': {'flower': {'variables': {}, 'dimensions': {}, 'groups': {}},
                                         'moon': {'variables': {}, 'dimensions': {},
                                                  'groups': {'base': {'variables': {},
                                                                      'dimensions': {
                                                                          'foo': Dimension(
                                                                              name='foo',
                                                                              size=None,
                                                                              size_current=None)},
                                                                      'groups': {}}}}}}}}

        self.assertDictEqual(ocmpi.mapping, desired)

    def test_get_empty_ranks(self):
        ompi = OcgMpi(size=5)
        ompi.create_dimension('four', 4, dist=True)
        ompi.update_dimension_bounds()
        self.assertEqual(ompi.get_empty_ranks(), (2, 3, 4))
        self.assertEqual(ompi.get_empty_ranks(inverse=True), (0, 1))

    @attr('mpi')
    def test_update_dimension_bounds(self):
        ompi = OcgMpi()
        dim1 = ompi.create_dimension('five', 5, dist=True)
        ompi.update_dimension_bounds()

        if dim1.is_empty:
            desired = (0, 0)
        else:
            desired = (0, 5)
        self.assertEqual(dim1.bounds_global, desired)
        if MPI_SIZE > 1:
            if MPI_SIZE == 2:
                if MPI_RANK == 0:
                    self.assertEqual(dim1.bounds_local, (0, 3))
                else:
                    self.assertEqual(dim1.bounds_local, (3, 5))

        # Test updating on single processor.
        if MPI_SIZE == 1:
            ompi = OcgMpi(size=2)
            ompi.create_dimension('five', 5, dist=True)
            ompi.update_dimension_bounds()
            dim = ompi.get_dimension('five')
            self.assertEqual(dim.bounds_global, (0, 5))
            for rank in range(2):
                actual = ompi.get_dimension('five', rank=rank)
                self.assertEqual(actual.bounds_global, (0, 5))
                if rank == 0:
                    self.assertEqual(actual.bounds_local, (0, 3))
                else:
                    self.assertEqual(actual.bounds_local, (3, 5))

            # Test two dimensions.
            ompi = OcgMpi(size=2)
            ompi.create_dimension('lat', 64, dist=True)
            ompi.create_dimension('lon', 128, dist=True)
            ompi.update_dimension_bounds()
            for rank in range(2):
                lat = ompi.get_dimension('lat', rank=rank)
                self.assertEqual(lat.bounds_local, (0, 64))
                lon = ompi.get_dimension('lon', rank=rank)
                if rank == 0:
                    self.assertEqual(lon.bounds_local, (0, 64))
                else:
                    self.assertEqual(lon.bounds_local, (64, 128))

    def test_update_dimension_bounds_single_simple_dimension(self):
        ompi = OcgMpi(size=2)
        ompi.create_dimension('d1', 2, dist=True)
        ompi.update_dimension_bounds(min_elements=2)
        d1 = ompi.get_dimension('d1')
        for rank in range(2):
            actual = ompi.get_dimension(d1.name, rank=rank)
            if rank == 0:
                self.assertFalse(actual.is_empty)
                self.assertEqual(actual.bounds_local, (0, 2))
                self.assertEqual(id(d1), id(actual))
            else:
                self.assertTrue(actual.is_empty)
                self.assertEqual(actual.bounds_local, (0, 0))
                self.assertNotEqual(id(d1), id(actual))
