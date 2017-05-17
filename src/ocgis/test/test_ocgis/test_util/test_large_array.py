import time
from copy import deepcopy

import netCDF4 as nc
import numpy as np

import ocgis
from ocgis import RequestDataset
from ocgis import Variable
from ocgis.calc import tile
from ocgis.test.base import TestBase, attr
from ocgis.util.large_array import compute, set_variable_spatial_mask


class Test(TestBase):
    def get_path_to_2d_grid_netcdf(self):
        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get().get_field_slice({'time': slice(0, 100), 'x': slice(0, 50), 'y': slice(0, 50)})
        path = self.get_temporary_file_path('2d_netcdf.nc')
        with self.nc_scope(path, 'w') as ds:
            field.write(ds)
        return path

    @staticmethod
    def get_random_integer(low=1, high=100):
        return int(np.random.random_integers(low, high))

    @attr('data')
    def test_compute_2d_grid(self):
        path = self.get_path_to_2d_grid_netcdf()
        rd = RequestDataset(path)

        ops = ocgis.OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}], calc_grouping=['month'],
                                  output_format='nc', add_auxiliary_files=False, geom=[33.7, -35.9, 109.1, 9.4])
        ret = compute(ops, 3, verbose=False)

        field = RequestDataset(ret).get()
        self.assertEqual(field['mean'].shape, (4, 17, 28))

    @attr('data')
    def test_with_callback(self):
        """Test callback reports status appropriately."""

        percentages = []

        def callback(a, b):
            percentages.append(a)

        rd = self.test_data.get_rd('cancm4_tas', kwds={'time_region': {'month': [3]}})
        ops = ocgis.OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}],
                                  calc_grouping=['month'], output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[2, 9, 12, 23, 25],
                                  add_auxiliary_files=False,
                                  callback=callback)
        ret = compute(ops, 3, verbose=False)
        hundreds = np.array(percentages)
        hundreds = hundreds >= 100.0
        self.assertEqual(hundreds.sum(), 1)

    @attr('slow')
    def test_timing_use_optimizations(self):
        n = list(range(10))
        t = {True: [], False: []}

        for use_optimizations in [True, False]:
            for ii in n:
                t1 = time.time()
                rd = self.test_data.get_rd('cancm4_tas')
                ops = ocgis.OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}],
                                          calc_grouping=['month'], output_format='nc',
                                          geom='state_boundaries',
                                          select_ugid=[2, 9, 12, 23, 25],
                                          add_auxiliary_files=False,
                                          prefix=str(ii) + str(use_optimizations))
                compute(ops, 5, verbose=False, use_optimizations=use_optimizations)
                t2 = time.time()
                t[use_optimizations].append(t2 - t1)
        tmean = {k: {'mean': np.array(v).mean(), 'stdev': np.array(v).std()} for k, v in t.items()}
        self.assertTrue(tmean[True]['mean'] < tmean[False]['mean'])

    @attr('data')
    def test_multivariate_computation(self):
        rd = self.test_data.get_rd('cancm4_tas', kwds={'time_region': {'month': [3]}})
        rd2 = deepcopy(rd)
        rd2.field_name = 'tas2'
        rd2.rename_variable = 'tas2'
        calc = [{'func': 'divide', 'name': 'ln', 'kwds': {'arr1': 'tas', 'arr2': 'tas2'}}]
        ops = ocgis.OcgOperations(dataset=[rd, rd2], calc=calc, calc_grouping=['month'], output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[2, 9, 12, 23, 25],
                                  add_auxiliary_files=False)
        ret = compute(ops, 5, verbose=False)

        ops.prefix = 'ocgis'
        ret_ocgis = ops.execute()

        self.assertNcEqual(ret, ret_ocgis, check_fill_value=False, ignore_attributes={'global': ['history'],
                                                                                      'ln': ['_FillValue']})

    @attr('data')
    def test_with_no_calc_grouping(self):
        rd = self.test_data.get_rd('cancm4_tas', kwds={'time_region': {'month': [3]}})
        ops = ocgis.OcgOperations(dataset=rd, calc=[{'func': 'ln', 'name': 'ln'}],
                                  calc_grouping=None, output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[2, 9, 12, 23, 25],
                                  add_auxiliary_files=False)
        ret = compute(ops, 5, verbose=False)

        ops.prefix = 'ocgis'
        ret_ocgis = ops.execute()
        self.assertNcEqual(ret, ret_ocgis, check_fill_value=False, ignore_attributes={'global': ['history'],
                                                                                      'ln': ['_FillValue']})

    @attr('data')
    def test_compute_with_time_region(self):
        rd = self.test_data.get_rd('cancm4_tas', kwds={'time_region': {'month': [3]}})
        ops = ocgis.OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}],
                                  calc_grouping=['month'], output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[2, 9, 12, 23, 25],
                                  add_auxiliary_files=False)
        ret = compute(ops, 5, verbose=False)

        ops.prefix = 'ocgis'
        ret_ocgis = ops.execute()
        self.assertNcEqual(ret, ret_ocgis, check_fill_value=False, ignore_attributes={'global': ['history'],
                                                                                      'mean': ['_FillValue']})

    @attr('data')
    def test_compute_with_geom(self):
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, calc=[{'func': 'mean', 'name': 'mean'}],
                                  calc_grouping=['month'], output_format='nc',
                                  geom='state_boundaries',
                                  select_ugid=[2, 9, 12, 23, 25],
                                  add_auxiliary_files=False)
        ret = compute(ops, 5, verbose=False)

        ops.prefix = 'ocgis'
        ret_ocgis = ops.execute()
        self.assertNcEqual(ret, ret_ocgis, check_fill_value=False, ignore_attributes={'global': ['history'],
                                                                                      'mean': ['_FillValue']})

    @attr('data')
    def test_compute_small(self):
        rd = self.test_data.get_rd('cancm4_tas')

        # use a smaller netCDF as target
        ops = ocgis.OcgOperations(dataset=rd,
                                  geom='state_boundaries',
                                  select_ugid=[2, 9, 12, 23, 25],
                                  output_format='nc',
                                  prefix='sub',
                                  add_auxiliary_files=False)
        sub = ops.execute()

        # use the compute function
        rd_sub = ocgis.RequestDataset(sub, 'tas')
        ops = ocgis.OcgOperations(dataset=rd_sub, calc=[{'func': 'mean', 'name': 'mean'}],
                                  calc_grouping=['month'], output_format='nc',
                                  add_auxiliary_files=False)
        ret_compute = compute(ops, 5, verbose=False)

        # now just run normally and ensure the answers are the same!
        ops.prefix = 'ocgis_compare'
        ops.add_auxiliary_files = False
        ret_ocgis = ops.execute()
        self.assertNcEqual(ret_compute, ret_ocgis, check_fill_value=False, ignore_attributes={'global': ['history'],
                                                                                              'mean': ['_FillValue']})

    def test_set_variable_spatial_mask(self):
        value = np.random.rand(10, 3, 4)
        value = np.ma.array(value, mask=False)
        mask_spatial = np.zeros((3, 4), dtype=bool)
        mask_spatial[2, 3] = True
        value.mask[:, 1, 1] = True
        var = Variable(name='var', value=value, dimensions=['time', 'y', 'x'])
        slice_row = slice(None)
        slice_col = slice(None)
        self.assertFalse(var.get_mask()[:, 2, 3].any())
        set_variable_spatial_mask(var, mask_spatial, slice_row, slice_col)
        self.assertTrue(var.get_mask().any())
        self.assertTrue(var.get_mask()[:, 1, 1].all())

        vmask = var.get_mask()
        vmask[:, 1, 1] = False
        vmask[:, 2, 3] = False
        var.set_mask(vmask)
        self.assertFalse(var.get_mask().any())

    def test_tile_get_tile_schema(self):
        schema = tile.get_tile_schema(5, 5, 2)
        self.assertEqual(len(schema), 9)

        schema = tile.get_tile_schema(25, 1, 2)
        self.assertEqual(len(schema), 13)

    def test_tile_sum(self):
        ntests = 1000
        for ii in range(ntests):
            nrow, ncol, tdim = [self.get_random_integer() for ii in range(3)]
            x = np.random.rand(nrow, ncol)
            y = np.empty((nrow, ncol), dtype=float)
            schema = tile.get_tile_schema(nrow, ncol, tdim)
            tidx = schema[0]
            row = tidx['row']
            col = tidx['col']
            self.assertTrue(np.all(x[row[0]:row[1], col[0]:col[1]] == x[0:tdim, 0:tdim]))
            running_sum = 0.0
            for value in schema.values():
                row, col = value['row'], value['col']
                slice = x[row[0]:row[1], col[0]:col[1]]
                y[row[0]:row[1], col[0]:col[1]] = slice
                running_sum += slice.sum()
            self.assertAlmostEqual(running_sum, x.sum())
            self.assertTrue(np.all(x == y))
