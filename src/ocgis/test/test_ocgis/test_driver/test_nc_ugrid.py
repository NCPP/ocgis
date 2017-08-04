import numpy as np

from ocgis import vm
from ocgis.base import raise_if_empty
from ocgis.driver.nc_ugrid import reduce_reindex_coordinate_variables
from ocgis.test.base import TestBase, attr
from ocgis.variable.base import Variable
from ocgis.vmachine.mpi import OcgDist, variable_scatter, hgather


class Test(TestBase):
    @attr('mpi')
    def test_reduce_reindex_coordinate_variables(self):
        self.add_barrier = False
        dist = OcgDist()
        dist.create_dimension('dim', 12, dist=True)
        dist.update_dimension_bounds()

        global_cindex_arr = np.array([4, 2, 1, 2, 1, 4, 1, 4, 2, 5, 6, 7])

        if vm.rank == 0:
            var_cindex = Variable('cindex', value=global_cindex_arr, dimensions='dim')
        else:
            var_cindex = None
        var_cindex = variable_scatter(var_cindex, dist)

        vm.create_subcomm_by_emptyable('test', var_cindex, is_current=True)
        if vm.is_null:
            return

        raise_if_empty(var_cindex)

        coords = np.array([0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 100, 110, 120, 130, 140, 150])
        coords = Variable(name='coords', value=coords, dimensions='coord_dim')

        new_cindex, u_indices = reduce_reindex_coordinate_variables(var_cindex)

        desired = coords[global_cindex_arr].get_value()

        if len(u_indices) > 0:
            new_coords = coords[u_indices].get_value()
        else:
            new_coords = np.array([])
        gathered_new_coords = vm.gather(new_coords)
        gathered_new_cindex = vm.gather(new_cindex)
        if vm.rank == 0:
            gathered_new_coords = hgather(gathered_new_coords)
            gathered_new_cindex = hgather(gathered_new_cindex)

            actual = gathered_new_coords[gathered_new_cindex]

            self.assertAsSetEqual(gathered_new_cindex.tolist(), [2, 1, 0, 3, 4, 5])
            desired_new_coords = [11, 22, 44, 55, 66, 77]
            self.assertAsSetEqual(gathered_new_coords.tolist(), desired_new_coords)
            self.assertEqual(len(gathered_new_coords), len(desired_new_coords))

            self.assertNumpyAll(actual, desired)
