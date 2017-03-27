import os

import numpy as np
from shapely.geometry import box

from ocgis import RequestDataset
from ocgis.base import get_variable_names
from ocgis.constants import MPIWriteMode
from ocgis.spatial.grid_splitter import GridSplitter, does_contain
from ocgis.test.base import attr, AbstractTestInterface
from ocgis.variable.base import Variable
from ocgis.variable.dimension import Dimension
from ocgis.variable.temporal import TemporalVariable
from ocgis.vm.mpi import MPI_COMM, MPI_RANK


class TestGridSplitter(AbstractTestInterface):
    @staticmethod
    def add_data_variable_to_grid(grid):
        ydim, xdim = grid.dimensions
        tdim = Dimension(name='time', size=None)
        value = np.random.rand(31, ydim.size, xdim.size)
        data = Variable(name='data', dimensions=[tdim, ydim, xdim], value=value)
        tvar = TemporalVariable(name='time', value=range(31), dimensions=tdim, attrs={'axis': 'T'})
        grid.parent.add_variable(data)
        grid.parent.add_variable(tvar)

    def get_grid_splitter(self):
        src_grid = self.get_gridxy_global(wrapped=False, with_bounds=True)
        dst_grid = self.get_gridxy_global(wrapped=False, with_bounds=True, resolution=0.5)

        self.add_data_variable_to_grid(src_grid)
        self.add_data_variable_to_grid(dst_grid)

        gs = GridSplitter(src_grid, dst_grid, (2, 3))
        return gs

    @attr('mpi', 'slow')
    def test(self):
        gs = self.get_grid_splitter()

        desired_dst_grid_sum = gs.dst_grid.parent['data'].get_value().sum()
        desired_dst_grid_sum = MPI_COMM.gather(desired_dst_grid_sum)
        if MPI_RANK == 0:
            desired_sum = np.sum(desired_dst_grid_sum)

        desired = [{'y': slice(0, 180, None), 'x': slice(0, 240, None)},
                   {'y': slice(0, 180, None), 'x': slice(240, 480, None)},
                   {'y': slice(0, 180, None), 'x': slice(480, 720, None)},
                   {'y': slice(180, 360, None), 'x': slice(0, 240, None)},
                   {'y': slice(180, 360, None), 'x': slice(240, 480, None)},
                   {'y': slice(180, 360, None), 'x': slice(480, 720, None)}]
        actual = list(gs.iter_dst_grid_slices())
        self.assertEqual(actual, desired)

        src_template = os.path.join(self.current_dir_output, 'src_{}.nc')
        dst_template = os.path.join(self.current_dir_output, 'dst_{}.nc')
        index_path = os.path.join(self.current_dir_output, 'index_path.nc')

        # barrier_print('before write_subsets')
        gs.write_subsets(src_template, dst_template, 'esmf_weights_{}.nc', index_path)
        # barrier_print('after write_subsets')

        if MPI_RANK == 0:
            rank_sums = []

        for ctr in range(1, gs.nsplits_dst[0] * gs.nsplits_dst[1] + 1):
            src_path = src_template.format(ctr)
            dst_path = dst_template.format(ctr)

            src_field = RequestDataset(src_path).get()
            dst_field = RequestDataset(dst_path).get()

            src_envelope_global = box(*src_field.grid.extent_global)
            dst_envelope_global = box(*dst_field.grid.extent_global)

            self.assertTrue(does_contain(src_envelope_global, dst_envelope_global))

            actual = get_variable_names(src_field.data_variables)
            self.assertIn('data', actual)

            actual = get_variable_names(dst_field.data_variables)
            self.assertIn('data', actual)
            actual_data_sum = dst_field['data'].get_value().sum()
            actual_data_sum = MPI_COMM.gather(actual_data_sum)
            if MPI_RANK == 0:
                actual_data_sum = np.sum(actual_data_sum)
                rank_sums.append(actual_data_sum)

        if MPI_RANK == 0:
            self.assertAlmostEqual(desired_sum, np.sum(rank_sums))
            self.assertTrue(os.path.exists(index_path))

        index_field = RequestDataset(index_path).get()
        self.assertTrue(len(index_field.keys()) > 2)

    @attr('slow')
    def test_insert_weighted(self):
        gs = self.get_grid_splitter()

        dst_master_path = self.get_temporary_file_path('out.nc')
        gs.dst_grid.parent.write(dst_master_path)

        dst_master = RequestDataset(dst_master_path).get()
        desired_sums = {}
        for data_variable in dst_master.data_variables:
            dv_sum = data_variable.get_value().sum()
            desired_sums[data_variable.name] = dv_sum
            self.assertNotEqual(dv_sum, 0)
            data_variable.get_value()[:] = 0
        dst_master.write(dst_master_path, write_mode=MPIWriteMode.FILL)
        dst_master = RequestDataset(dst_master_path).get()
        for data_variable in dst_master.data_variables:
            self.assertEqual(data_variable.get_value().sum(), 0)

        src_template = os.path.join(self.current_dir_output, 'src_{}.nc')
        dst_template = os.path.join(self.current_dir_output, 'dst_{}.nc')
        wgt_template = 'esmf_weights_{}.nc'
        index_path = os.path.join(self.current_dir_output, 'index_path.nc')

        gs.write_subsets(src_template, dst_template, wgt_template, index_path)

        gs.insert_weighted(index_path, self.current_dir_output, dst_master_path)

        actual_sums = {}
        dst_master_inserted = RequestDataset(dst_master_path).get()
        for data_variable in dst_master_inserted.data_variables:
            dv_value = data_variable.get_value()
            dv_sum = dv_value.sum()
            actual_sums[data_variable.name] = dv_sum
        for k, v in actual_sums.items():
            self.assertAlmostEqual(v, desired_sums[k])
