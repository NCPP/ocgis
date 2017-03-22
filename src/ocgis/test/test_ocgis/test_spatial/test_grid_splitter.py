import os

import numpy as np
from shapely.geometry import box

from ocgis import RequestDataset
from ocgis.base import get_variable_names
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

    @attr('mpi')
    def test(self):
        # self.add_barrier = False
        src_grid = self.get_gridxy_global(wrapped=False, with_bounds=True)
        dst_grid = self.get_gridxy_global(wrapped=False, with_bounds=True, resolution=0.5)

        self.add_data_variable_to_grid(src_grid)
        self.add_data_variable_to_grid(dst_grid)

        desired_dst_grid_sum = dst_grid.parent['data'].get_value().sum()
        desired_dst_grid_sum = MPI_COMM.gather(desired_dst_grid_sum)
        if MPI_RANK == 0:
            desired_sum = np.sum(desired_dst_grid_sum)

        gs = GridSplitter(src_grid, dst_grid, (2, 3))

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

        for ctr in range(gs.nsplits_dst[0] * gs.nsplits_dst[1]):
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
