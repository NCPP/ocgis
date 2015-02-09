from ocgis import RequestDataset
from ocgis.interface.base.dimension.spatial import SpatialGridDimension
from ocgis.interface.base.variable import AbstractSourcedVariable
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.nc.spatial import NcSpatialGridDimension
from ocgis.test.base import TestBase
import numpy as np


class TestNcSpatialGridDimension(TestBase):

    def test_init(self):
        self.assertEqual(NcSpatialGridDimension.__bases__, (AbstractSourcedVariable, SpatialGridDimension))

        row = VectorDimension(value=[4, 5])
        col = VectorDimension(value=[6, 7, 8])
        NcSpatialGridDimension(row=row, col=col)

    def test_getitem(self):
        src_idx = {'row': np.array([5, 6, 7, 8]), 'col': np.array([9, 10, 11])}
        grid = NcSpatialGridDimension(src_idx=src_idx, data='foo')
        self.assertIsNone(grid._uid)
        sub = grid[1:3, 1]
        self.assertNumpyAll(sub._src_idx['col'], np.array([10]))
        self.assertNumpyAll(sub._src_idx['row'], np.array([6, 7]))
        for k, v in src_idx.iteritems():
            self.assertNumpyAll(grid._src_idx[k], v)

    def test_format_src_idx(self):
        ref = NcSpatialGridDimension._format_src_idx_
        value = {'row': np.array([5]), 'col': np.array([6])}
        self.assertEqual(value, ref(value))

    def test_get_uid(self):
        src_idx = {'row': np.array([5, 6, 7, 8]), 'col': np.array([9, 10, 11])}
        grid = NcSpatialGridDimension(src_idx=src_idx, data='foo')
        uid1 = grid._get_uid_()
        self.assertEqual(uid1.shape, (4, 3))

        value = np.ma.array(np.zeros((2, 4, 3)))
        grid = NcSpatialGridDimension(value=value)
        uid2 = grid._get_uid_()
        self.assertEqual(uid2.shape, (4, 3))

        self.assertNumpyAll(uid1, uid2)

    def test_set_value_from_source(self):
        path = self.get_netcdf_path_no_row_column()
        rd = RequestDataset(path)

        src_idx = {'row': np.array([0, 1]), 'col': np.array([0])}
        grid = NcSpatialGridDimension(data=rd, src_idx=src_idx, name_row='yc', name_col='xc')
        self.assertEqual(grid.value.shape, (2, 2, 1))
        with self.nc_scope(path) as ds:
            var_row = ds.variables[grid.name_row]
            var_col = ds.variables[grid.name_col]
            self.assertNumpyAll(var_row[:, 0].reshape(2, 1), grid.value[0].data)
            self.assertNumpyAll(var_col[:, 0].reshape(2, 1), grid.value[1].data)

        src_idx = {'row': np.array([0]), 'col': np.array([1])}
        grid = NcSpatialGridDimension(data=rd, src_idx=src_idx, name_row='yc', name_col='xc')
        self.assertIsNone(grid._value)
        self.assertIsNone(grid._corners)
        self.assertEqual(grid.value.shape, (2, 1, 1))
        self.assertEqual(grid.corners.shape, (2, 1, 1, 4))
        self.assertEqual(grid.corners_esmf.shape, (2, 2, 2))
        actual = np.ma.array([[[[3.5, 3.5, 4.5, 4.5]]], [[[45.0, 55.0, 55.0, 45.0]]]])
        self.assertNumpyAll(actual, grid.corners)

    def test_shape(self):
        src_idx = {'row': np.array([5, 6, 7, 8]), 'col': np.array([9, 10, 11])}
        grid = NcSpatialGridDimension(src_idx=src_idx, data='foo')
        self.assertEqual(grid.shape, (4, 3))
        self.assertIsNone(grid._value)

        row = VectorDimension(value=[4, 5])
        col = VectorDimension(value=[6, 7, 8])
        grid = NcSpatialGridDimension(row=row, col=col)
        self.assertEqual(grid.shape, (2, 3))


    def test_validate(self):
        with self.assertRaises(ValueError):
            NcSpatialGridDimension()
        NcSpatialGridDimension(data='foo')

    def test_value(self):
        row = VectorDimension(value=[4, 5])
        col = VectorDimension(value=[6, 7, 8])
        grid = NcSpatialGridDimension(row=row, col=col)
        self.assertEqual(grid.shape, (2, 3))

        value = grid.value.copy()
        grid = NcSpatialGridDimension(value=value)
        self.assertNumpyAll(grid.value, value)