from unittest import SkipTest

from mock import Mock
from ocgis import OcgVM, vm, Dimension, env
from ocgis.driver.request.core import RequestDataset
from ocgis.test.base import TestBase, attr
from ocgis.vmachine.mpi import MPI_SIZE, MPI_RANK, MPI_COMM, DummyMPIComm, DummyRequest


class TestOcgVM(TestBase):

    @attr('mpi')
    def test_init(self):
        for _ in range(50):
            vm = OcgVM()
            vm.finalize()
            self.assertTrue(vm.size_global >= 1)
            self.assertIsNotNone(vm.comm)
            vm.finalize()
            vm.__init__()
            vm.finalize()

    @attr('mpi')
    def test_system_get_field_from_file(self):
        """Test returning a distributed field from file."""

        field = self.get_field(nrow=5, ncol=7)
        if MPI_RANK == 0:
            path = self.get_temporary_file_path('data.nc')
        else:
            path = None
        path = MPI_COMM.bcast(path)

        with vm.scoped('write test field', [0]):
            if MPI_RANK == 0:
                field.write(path)

        MPI_COMM.Barrier()

        rd = RequestDataset(path)
        out_field = rd.get()

        if MPI_SIZE == 8:
            self.assertEqual(vm.size, 8)

        if MPI_RANK == 0:
            path2 = self.get_temporary_file_path('out_field.nc')
        else:
            path2 = None
        path2 = MPI_COMM.bcast(path2)

        with vm.scoped_by_emptyable('out_field write', out_field):
            if not vm.is_null:
                out_field.write(path2)

        MPI_COMM.Barrier()

        with vm.scoped('get actual', [0]):
            if MPI_RANK == 0:
                actual = RequestDataset(path2).get()
                actual = actual.data_variables[0].get_value().sum()
            else:
                actual = None

        actual = MPI_COMM.bcast(actual)

        desired = field.data_variables[0].get_value().sum()
        self.assertAlmostEqual(actual, desired)

    def test_system_dummy_comm_Isend_Irecv(self):
        comm = DummyMPIComm()

        recv_data = [[None], None]
        recv_req = comm.Irecv(recv_data, tag='one')

        self.assertFalse(recv_req.Test())

        send_data = [['foo_send'], None]
        req = comm.Isend(send_data, tag='one')
        self.assertIsInstance(req, DummyRequest)
        req.wait()
        req.Test()

        self.assertTrue(recv_req.Test())
        recv_req.wait()

        self.assertEqual(recv_data[0][0], send_data[0][0])
        self.assertEqual(comm._send_recv, {0: {}})

    @attr('mpi')
    def test_system_raise_exception_subcommunicator(self):
        if vm.size != 4:
            raise (SkipTest('vm.size != 4'))

        raiser = Mock(side_effect=IndexError('oops'))

        with self.assertRaises(IndexError):
            e = None
            with vm.scoped('the sub which will raise', [2]):
                if not vm.is_null:
                    try:
                        raiser()
                    except IndexError as exc:
                        e = exc
            es = vm.gather(e)
            es = vm.bcast(es)
            for e in es:
                if e is not None:
                    raise e

    @attr('mpi')
    def test_abort(self):
        if MPI_SIZE > 1:
            raise SkipTest('dev only for parallel')
            if vm.rank == 0:
                exc = ValueError("test test_abort")
                vm.abort(exc=exc, msg="A TEST MESSAGE")
        else:
            with self.assertRaises(RuntimeError):
                vm.abort()

    @attr('mpi')
    def test_barrier(self):
        if MPI_SIZE != 4:
            raise SkipTest('MPI_SIZE != 4')

        vm = OcgVM()
        live_ranks = [1, 3]
        vm.create_subcomm('for barrier', live_ranks, is_current=True)

        if not vm.is_null:
            self.assertEqual(vm.size, 2)
        else:
            self.assertNotIn(MPI_RANK, live_ranks)

        if MPI_RANK in live_ranks:
            vm.barrier()

        vm.finalize()

    @attr('mpi')
    def test_bcast(self):
        if MPI_SIZE != 8:
            raise SkipTest('MPI_SIZE != 8')

        vm = OcgVM()
        live_ranks = [1, 3, 5]
        vm.create_subcomm('tester', live_ranks, is_current=True)
        # vm.set_live_ranks(live_ranks)

        if vm.rank == 0:
            root_value = 101
        else:
            root_value = None

        if MPI_RANK in live_ranks:
            global_value = vm.bcast(root_value)
            self.assertEqual(global_value, 101)
        else:
            self.assertIsNone(root_value)

        vm.finalize()

    @attr('mpi')
    def test_comm_world(self):
        if MPI_SIZE != 2:
            raise SkipTest('MPI_SIZE != 2')
        self.assertEqual(vm.size, 2)
        self.assertEqual(vm.comm_world.Get_size(), 2)
        with vm.scoped('comm world test', [1]):
            if not vm.is_null:
                self.assertEqual(vm.size, 1)
            self.assertEqual(vm.comm_world.Get_size(), 2)

    @attr('mpi')
    def test_create_subcomm(self):
        vm = OcgVM()

        if vm.size != 2:
            raise SkipTest('vm.size != 2')

        self.assertFalse(vm._is_dummy)
        vm.create_subcomm('test', [], is_current=True)
        self.assertTrue(vm.is_null)
        vm.finalize()
        self.assertFalse(vm.is_null)

    @attr('mpi')
    def test_gather(self):
        if MPI_SIZE != 8:
            raise SkipTest('MPI_SIZE != 8')

        vm = OcgVM()
        live_ranks = [1, 3, 7]
        # vm.set_live_ranks(live_ranks)
        vm.create_subcomm('tester', live_ranks, is_current=True)

        if MPI_RANK in live_ranks:
            value = MPI_RANK

            gathered_value = vm.gather(value)

        if MPI_RANK == 1:
            self.assertEqual(gathered_value, [1, 3, 7])
        elif MPI_RANK in live_ranks:
            self.assertIsNone(gathered_value)

        vm.finalize()

    @attr('mpi')
    def test_get_live_ranks_from_object(self):
        if MPI_SIZE != 4:
            raise SkipTest('MPI_SIZE != 4')

        vm = OcgVM()

        if MPI_RANK == 1:
            dim = Dimension('woot', is_empty=True, dist=True)
        else:
            dim = Dimension('woot', dist=True, size=3)

        actual = vm.get_live_ranks_from_object(dim)
        self.assertEqual(actual, (0, 2, 3))

        vm.finalize()

    def test_get_mpi_type(self):
        if env.USE_MPI4PY:
            from mpi4py import MPI
            actual = OcgVM.get_mpi_type(int)
            self.assertEqual(actual, MPI.LONG_LONG)
        else:
            raise SkipTest('not env.USE_MPI4PY')

    @attr('mpi')
    def test_scoped(self):

        if MPI_SIZE != 8:
            raise SkipTest('MPI_SIZE != 8')

        vm = OcgVM()
        self.assertEqual(vm.size, 8)
        with vm.scoped('test', [2, 3, 4]):
            if not vm.is_null:
                self.assertEqual(vm.size, 3)
                self.assertEqual(vm.ranks, range(3))
                with vm.scoped('nested', [1]):
                    if not vm.is_null:
                        self.assertEqual(vm.size, 1)
                        self.assertEqual(len(vm._subcomms), 2)
        self.assertEqual(vm.size, 8)
        self.assertEqual(len(vm._subcomms), 0)
        vm.finalize()

        vm = OcgVM()
        self.assertEqual(vm.size, 8)
        vm.finalize()
