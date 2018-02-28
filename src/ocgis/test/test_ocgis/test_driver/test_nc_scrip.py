import itertools

import numpy as np
from mock import mock

from ocgis import RequestDataset, DimensionMap, GridUnstruct, PointGC, Field, Variable, Dimension
from ocgis.constants import DriverKey, DMK, Topology
from ocgis.driver.nc_scrip import DriverNetcdfSCRIP
from ocgis.spatial.grid_chunker import GridChunker
from ocgis.test.base import TestBase
from ocgis.variable.crs import Spherical


class FixtureDriverNetcdfSCRIP(object):

    def fixture_driver_scrip_netcdf_field(self):
        xvalue = np.arange(10., 35., step=5)
        yvalue = np.arange(45., 85., step=10)
        grid_size = xvalue.shape[0] * yvalue.shape[0]

        dim_grid_size = Dimension(name='grid_size', size=grid_size)
        x = Variable(name='grid_center_lon', dimensions=dim_grid_size)
        y = Variable(name='grid_center_lat', dimensions=dim_grid_size)

        for idx, (xv, yv) in enumerate(itertools.product(xvalue, yvalue)):
            x.get_value()[idx] = xv
            y.get_value()[idx] = yv

        gc = PointGC(x=x, y=y, crs=Spherical(), driver=DriverNetcdfSCRIP)
        grid = GridUnstruct(geoms=[gc])
        ret = Field(grid=grid, driver=DriverNetcdfSCRIP)

        grid_dims = Variable(name='grid_dims', value=[yvalue.shape[0], xvalue.shape[0]], dimensions='grid_rank')
        ret.add_variable(grid_dims)

        return ret


class TestDriverNetcdfSCRIP(TestBase, FixtureDriverNetcdfSCRIP):

    def test_init(self):
        rd = mock.create_autospec(RequestDataset)
        d = DriverNetcdfSCRIP(rd)
        self.assertIsInstance(d, DriverNetcdfSCRIP)

        field = self.fixture_driver_scrip_netcdf_field()
        self.assertIsInstance(field, Field)

    def test_array_resolution(self):
        self.assertEqual(DriverNetcdfSCRIP.array_resolution(np.array([5]), None), 0.0)
        self.assertEqual(DriverNetcdfSCRIP.array_resolution(np.array([-5, -10, 10, 5], dtype=float), None), 5.0)

    def test_array_resolution_called(self):
        """Test the driver's array resolution method is called appropriately."""

        m_DriverNetcdfSCRIP = mock.create_autospec(DriverNetcdfSCRIP)
        with mock.patch('ocgis.driver.registry.get_driver_class', return_value=m_DriverNetcdfSCRIP):
            x = Variable(name='x', value=[1, 2, 3], dimensions='dimx')
            y = Variable(name='y', value=[4, 5, 6], dimensions='dimy')
            pgc = PointGC(x=x, y=y)
            _ = pgc.resolution_x
            _ = pgc.resolution_y
        self.assertEqual(m_DriverNetcdfSCRIP.array_resolution.call_count, 2)

    def test_create_field(self):
        meta = {'dimensions': {u'grid_corners': {'isunlimited': False,
                                                 'name': u'grid_corners',
                                                 'size': 4},
                               u'grid_rank': {'isunlimited': False,
                                              'name': u'grid_rank',
                                              'size': 2},
                               u'grid_size': {'isunlimited': False,
                                              'name': u'grid_size',
                                              'size': 55296}},
                'file_format': 'NETCDF3_CLASSIC',
                'global_attributes': {
                    u'input_file': u'/fs/cgd/csm/inputdata/lnd/clm2/griddata/griddata_0.9x1.25_c070928.nc',
                    u'title': u'0.9x1.25_c110307.nc'},
                'groups': {},
                'variables': {u'grid_center_lat': {'attrs': {u'units': u'degrees'},
                                                   'dimensions': (u'grid_size',),
                                                   'dtype': np.dtype('float64'),
                                                   'dtype_packed': None,
                                                   'fill_value': 'auto',
                                                   'fill_value_packed': None,
                                                   'name': u'grid_center_lat'},
                              u'grid_center_lon': {'attrs': {u'units': u'degrees'},
                                                   'dimensions': (u'grid_size',),
                                                   'dtype': np.dtype('float64'),
                                                   'dtype_packed': None,
                                                   'fill_value': 'auto',
                                                   'fill_value_packed': None,
                                                   'name': u'grid_center_lon'},
                              u'grid_corner_lat': {'attrs': {u'units': u'degrees'},
                                                   'dimensions': (u'grid_size',
                                                                  u'grid_corners'),
                                                   'dtype': np.dtype('float64'),
                                                   'dtype_packed': None,
                                                   'fill_value': 'auto',
                                                   'fill_value_packed': None,
                                                   'name': u'grid_corner_lat'},
                              u'grid_corner_lon': {'attrs': {u'units': u'degrees'},
                                                   'dimensions': (u'grid_size',
                                                                  u'grid_corners'),
                                                   'dtype': np.dtype('float64'),
                                                   'dtype_packed': None,
                                                   'fill_value': 'auto',
                                                   'fill_value_packed': None,
                                                   'name': u'grid_corner_lon'},
                              u'grid_dims': {'attrs': {},
                                             'dimensions': (u'grid_rank',),
                                             'dtype': np.dtype('int32'),
                                             'dtype_packed': None,
                                             'fill_value': 'auto',
                                             'fill_value_packed': None,
                                             'name': u'grid_dims'},
                              u'grid_imask': {'attrs': {u'units': u'unitless'},
                                              'dimensions': (u'grid_size',),
                                              'dtype': np.dtype('int32'),
                                              'dtype_packed': None,
                                              'fill_value': 'auto',
                                              'fill_value_packed': None,
                                              'name': u'grid_imask'}}}

        rd = RequestDataset(metadata=meta, driver=DriverNetcdfSCRIP)
        d = DriverNetcdfSCRIP(rd)

        dmap = d.create_dimension_map(meta)
        self.assertIsInstance(dmap, DimensionMap)
        self.assertIsNotNone(dmap.get_spatial_mask())
        self.assertNotIn('ocgis_role', dmap.get_property(DMK.SPATIAL_MASK)['attrs'])

        run_topo_tst(self, dmap)

        actual = dmap.get_property(DMK.IS_ISOMORPHIC)
        self.assertTrue(actual)

        field = d.create_field()

        run_topo_tst(self, field.dimension_map)
        dmap = field.grid.dimension_map
        run_topo_tst(self, dmap)

        self.assertEqual(field.crs, Spherical())
        self.assertEqual(field.driver.key, DriverKey.NETCDF_SCRIP)

        # Test grid structure.
        self.assertIsInstance(field.grid, GridUnstruct)
        desired = meta['dimensions']['grid_size']['size']
        run_topo_tst(self, field.grid.dimension_map)
        actual = field.grid.element_dim.size
        run_topo_tst(self, field.grid.dimension_map)
        self.assertEqual(desired, actual)
        self.assertTrue(field.grid.is_isomorphic)
        self.assertEqual(field.grid.abstraction, Topology.POLYGON)

        # Test grid masking.
        grid = field.grid
        self.assertTrue(grid.has_mask)
        for _ in range(3):
            self.assertIsNotNone(field.grid.get_mask())
            self.assertFalse(field.grid.get_mask().any())

        run_topo_tst(self, field.grid.dimension_map)

    def test_gc_nchunks_dst(self):
        field = self.fixture_driver_scrip_netcdf_field()
        gc = mock.create_autospec(GridChunker, spec_set=True)
        gc.dst_grid = field.grid
        actual = field.grid._gc_nchunks_dst_(gc)
        self.assertEqual(actual, 10)


def run_topo_tst(obj, dmap):
    topo = dmap.get_topology(Topology.POLYGON)
    for k in [DMK.X, DMK.Y]:
        actual = topo.get_dimension(k)
        obj.assertEqual(actual[0], 'grid_size')
