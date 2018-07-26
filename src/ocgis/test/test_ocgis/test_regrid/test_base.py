import numpy as np
from copy import deepcopy

from ocgis import OcgOperations
from ocgis import RequestDataset
from ocgis import Variable
from ocgis.collection.field import Field
from ocgis.exc import RegriddingError, CornersInconsistentError
from ocgis.spatial.grid import Grid
from ocgis.test.base import attr, AbstractTestInterface, create_exact_field, create_gridxy_global
from ocgis.test.test_simple.make_test_data import SimpleNc
from ocgis.test.test_simple.test_simple import TestSimpleBase
from ocgis.util.helpers import make_poly
from ocgis.util.itester import itr_products_keywords
from ocgis.variable.crs import CoordinateReferenceSystem, Spherical, WGS84


class TestRegrid(TestSimpleBase):
    nc_factory = SimpleNc
    fn = 'test_simple_spatial_01.nc'

    def get_ofield(self):
        rd = RequestDataset(**self.get_dataset())

        # Create a field composed of two variables.
        ofield = rd.get()

        new_variable = ofield['foo'].extract()
        new_variable.set_name('foo2')
        new_variable.get_value()[:] = 30
        ofield.add_variable(new_variable, is_data=True)

        ofield = ofield.get_field_slice({'level': 0})
        return ofield

    @attr('esmf')
    def test_system(self):
        from ocgis.regrid.base import get_esmf_grid, iter_esmf_fields, RegridOperation, destroy_esmf_objects
        import ESMF

        yc = Variable(name='yc', value=np.arange(-90 + (45 / 2.), 90, 45), dimensions='ydim', dtype=float)
        xc = Variable(name='xc', value=np.arange(15, 360, 30), dimensions='xdim', dtype=float)
        ogrid = Grid(y=yc, x=xc, crs=Spherical())
        ogrid.set_extrapolated_bounds('xc_bounds', 'yc_bounds', 'bounds')

        np.random.seed(1)
        mask = np.random.rand(*ogrid.shape)
        mask = mask > 0.5
        self.assertTrue(mask.sum() > 3)
        ogrid.set_mask(mask)

        egrid = get_esmf_grid(ogrid)
        actual_shape = egrid.size[0].tolist()
        desired_shape = np.flipud(ogrid.shape).tolist()
        self.assertEqual(actual_shape, desired_shape)
        desired = ogrid.get_value_stacked()
        desired = np.ma.array(desired, mask=False)
        desired.mask[0, :, :] = ogrid.get_mask()
        desired.mask[1, :, :] = ogrid.get_mask()
        desired = desired.sum()

        actual_col = egrid.get_coords(0)
        actual_row = egrid.get_coords(1)
        actual_mask = np.invert(egrid.mask[0].astype(bool))
        actual = np.ma.array(actual_row, mask=actual_mask).sum() + np.ma.array(actual_col, mask=actual_mask).sum()
        self.assertEqual(actual, desired)

        desired = 9900.0
        corners = egrid.coords[ESMF.StaggerLoc.CORNER]
        actual = corners[0].sum() + corners[1].sum()
        self.assertEqual(actual, desired)

        ofield = create_exact_field(ogrid, 'data', ntime=3, crs=Spherical())
        variable_name, efield, tidx = list(iter_esmf_fields(ofield, split=False))[0]
        desired_value = ofield['data'].get_value()
        self.assertAlmostEqual(efield.data.sum(), desired_value.sum(), places=3)

        destroy_esmf_objects([egrid, efield])

        ofield.grid.set_mask(ofield.grid.get_mask(), cascade=True)
        desired_value = ofield['data'].get_masked_value()
        keywords = dict(split=[False, True])
        for k in self.iter_product_keywords(keywords):
            opts = {'split': k.split}
            dofield = ofield.deepcopy()
            dofield['data'].get_value().fill(0)
            ro = RegridOperation(ofield, dofield, regrid_options=opts)
            actual_field = ro.execute()
            actual_value = actual_field['data'].get_masked_value()
            self.assertAlmostEqual(0.0, np.abs(desired_value - actual_value).max())

    @attr('esmf')
    def test_check_fields_for_regridding(self):
        from ESMF import RegridMethod
        from ocgis.regrid.base import check_fields_for_regridding

        source = self.get_ofield()
        destination = deepcopy(source)

        # Test non-spherical coordinate systems.
        new_source = deepcopy(source)
        new_source.update_crs(WGS84())
        with self.assertRaises(RegriddingError):
            check_fields_for_regridding(new_source, destination)

        # Change coordinate systems to spherical.
        source.update_crs(Spherical())
        destination.update_crs(Spherical())

        # Test with different parameters of the sphere.
        new_source = deepcopy(source)
        new_source.update_crs(Spherical(semi_major_axis=100))
        with self.assertRaises(RegriddingError):
            check_fields_for_regridding(new_source, destination)

        # Test corners not available on all inputs #####################################################################

        # Test corners not on one of the sources
        new_source = deepcopy(source)

        new_source.grid.x.set_bounds(None)
        new_source.grid.y.set_bounds(None)

        self.assertFalse(new_source.grid.has_bounds)

        for regrid_method in ['auto', RegridMethod.CONSERVE]:
            if regrid_method == RegridMethod.CONSERVE:
                with self.assertRaises(CornersInconsistentError):
                    check_fields_for_regridding(new_source, destination, regrid_method=regrid_method)
            else:
                check_fields_for_regridding(new_source, destination, regrid_method=regrid_method)

    @attr('data', 'esmf', 'slow')
    def test_regrid_field_different_grid_shapes(self):
        """Test regridding a downscaled dataset to GCM output. The input and output grids have different shapes."""

        downscaled = self.test_data.get_rd('maurer_2010_tas')
        downscaled.time_region = {'month': [2], 'year': [1990]}
        downscaled = downscaled.get()
        poly = make_poly([37, 43], [-104, -94])
        downscaled = downscaled.grid.get_intersects(poly).parent
        downscaled.unwrap()
        downscaled.set_crs(Spherical())

        gcm = self.test_data.get_rd('cancm4_tas')
        gcm = gcm.get()
        poly = make_poly([37, 43], [-104 + 360, -94 + 360])
        gcm = gcm.grid.get_intersects(poly).parent
        gcm.set_crs(Spherical())

        self.assertIsNone(downscaled.grid.get_mask())
        self.assertIsNone(gcm.grid.get_mask())

        from ocgis.regrid.base import regrid_field
        regridded = regrid_field(downscaled, gcm)
        dv = regridded.data_variables[0]
        self.assertEqual(dv.shape, (28, 3, 5))
        self.assertEqual(dv.name, 'tas')
        vmask = dv.get_mask()
        self.assertEqual(vmask.sum(), 252)

    @attr('esmf')
    def test_regrid_field(self):
        """Test with equivalent input and output expectations. The shapes of the grids are equal."""

        source = self.get_ofield()
        source.set_crs(Spherical())
        destination = deepcopy(source)
        desired = deepcopy(source)

        keywords = dict(split=[False, True], fill_value=[None, 1e20])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            from ocgis.regrid.base import regrid_field
            regridded = regrid_field(source, destination, split=k.split)
            self.assertIsInstance(regridded, Field)
            self.assertNumpyAll(regridded.grid.get_value_stacked(), desired.grid.get_value_stacked())
            self.assertEqual(regridded.crs, source.crs)
            for variable in regridded.data_variables:
                self.assertGreater(variable.get_value().mean(), 2.0)
                self.assertNumpyAll(variable.get_value(), source[variable.name].get_value())
                self.assertFalse(np.may_share_memory(variable.get_value(), source[variable.name].get_value()))

    @attr('esmf')
    def test_iter_regridded_field_with_corners(self):
        """Test with_corners as True and False when regridding Fields."""

        source = self.get_ofield()
        source.set_crs(Spherical())
        destination = deepcopy(source)

        self.assertEqual(source.grid.abstraction, 'polygon')

        from ESMF import RegridMethod
        for regrid_method in ['auto', RegridMethod.BILINEAR]:
            from ocgis.regrid.base import regrid_field
            regrid_field(source, destination, regrid_method=regrid_method)

    @attr('esmf')
    def test_regrid_field_differing_crs(self):
        """Test exception raised when source and destination CRS values are not equal."""

        source = self.get_ofield()
        destination = deepcopy(source)
        source.set_crs(CoordinateReferenceSystem(epsg=2136))

        with self.assertRaises(RegriddingError):
            from ocgis.regrid.base import regrid_field
            regrid_field(source, destination)

    @attr('esmf')
    def test_regrid_field_value_mask(self):
        """Test with a value mask on the destination."""
        from ocgis.regrid.base import regrid_field

        source = self.get_ofield()
        source.set_crs(Spherical())
        destination = deepcopy(source)

        value_mask = np.zeros(destination.grid.shape, dtype=bool)
        value_mask[1, 1] = True

        regridded = regrid_field(source, destination, value_mask=value_mask)
        self.assertTrue(np.all(regridded.data_variables[0].get_mask()[:, :, 1, 1]))

    @attr('data', 'esmf')
    def test_system_regrid_field_nonoverlapping_extents(self):
        """Test regridding with fields that do not spatially overlap."""

        rd = self.test_data.get_rd('cancm4_tas')
        # nebraska and california
        coll = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[16, 25], snippet=True,
                             vector_wrap=False).execute()
        source = coll.get_element(container_ugid=25)
        destination = coll.get_element(container_ugid=16)

        with self.assertRaises(RegriddingError):
            from ocgis.regrid.base import regrid_field
            regrid_field(source, destination)

    @attr('data', 'esmf')
    def test_regrid_field_partial_extents(self):
        """Test regridding with fields that partially overlap."""

        rd = self.test_data.get_rd('cancm4_tas')
        # california and nevada
        coll = OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23, 25], snippet=True,
                             vector_wrap=False).execute()

        source = coll.get_element(container_ugid=23)
        destination = coll.get_element(container_ugid=25)

        from ocgis.regrid.base import regrid_field
        res = regrid_field(source, destination)
        self.assertEqual(res['tas'].get_mask().sum(), 11)

    @attr('esmf')
    def test_iter_esmf_fields(self):
        ofield = self.get_ofield()

        # Add a unique spatial and value mask.
        mask = ofield.grid.get_mask(create=True)
        mask[1, 1] = True
        ofield.grid.set_mask(mask)

        dtype = [('variable_alias', object), ('efield', object), ('tidx', object)]
        fill = np.array([], dtype=dtype)
        from ocgis.regrid.base import iter_esmf_fields
        for row in iter_esmf_fields(ofield, value_mask=None, split=False):
            app = np.zeros(1, dtype=dtype)
            app[0] = row
            fill = np.append(fill, app)

        # Without splitting, the time index is a single slice.
        self.assertTrue(all([ii is None for ii in fill['tidx']]))

        # There are two variables.
        self.assertEqual(fill.shape[0], 2)
        self.assertEqual(np.unique(fill[0]['variable_alias']), 'foo')
        self.assertEqual(np.unique(fill[1]['variable_alias']), 'foo2')

        # Test data is properly attributed to the correct variable.
        means = np.array([element.data.mean() for element in fill['efield']])
        self.assertTrue(np.all(means[0] == 2.5))
        self.assertTrue(np.all(means[1] == 30))

        for idx in range(fill.shape[0]):
            efield = fill[idx]['efield']
            # Test all time slices are accounted for.
            self.assertEqual(efield.data.shape[-1], ofield.time.shape[0])
            # Test masks are equivalent. ESMF masks have "1" for false requiring the invert operation.
            egrid = efield.grid
            egrid_mask = np.invert(egrid.mask[0].astype(bool))
            self.assertNumpyAll(egrid_mask, mask)

        # Test splitting the data.
        ofield = self.get_ofield()
        res = list(iter_esmf_fields(ofield))
        self.assertEqual(len(res), ofield.time.shape[0] * len(ofield.data_variables))

    @attr('esmf')
    def test_get_esmf_grid_bilinear_regrid_method(self):
        """Test with a regrid method that does not require corners."""

        from ocgis.regrid.base import get_esmf_grid
        from ESMF import RegridMethod
        import ESMF

        rd = RequestDataset(**self.get_dataset())
        field = rd.get()
        self.assertTrue(field.grid.has_bounds)
        egrid = get_esmf_grid(field.grid, regrid_method=RegridMethod.BILINEAR)
        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        for idx in [0, 1]:
            self.assertIsNone(corner[idx])

    @attr('esmf')
    def test_get_esmf_grid_periodicity(self):
        """Test periodicity parameters generate reasonable output."""
        from ocgis.regrid.base import get_esmf_grid

        lon_in = np.arange(-180, 180, 10)
        lat_in = np.arange(-90, 90.1, 4)

        lon = Variable('lon', lon_in, 'dlon')
        lat = Variable('lat', lat_in, 'dlat')

        ogrid = Grid(x=lon, y=lat, crs=Spherical())
        egrid = get_esmf_grid(ogrid)

        self.assertEqual(egrid.periodic_dim, 0)
        self.assertEqual(egrid.num_peri_dims, 1)
        self.assertEqual(egrid.pole_dim, 1)

    @attr('esmf')
    def test_get_esmf_grid_with_mask(self):
        """Test with masked data."""

        from ocgis.regrid.base import get_esmf_grid

        x = Variable(name='x', value=[1, 2, 3], dimensions='x')
        y = Variable(name='y', value=[4, 5, 6], dimensions='y')
        grid = Grid(x, y, crs=Spherical())

        gmask = grid.get_mask(create=True)
        gmask[1, 1] = True
        grid.set_mask(gmask)
        self.assertEqual(grid.get_mask().sum(), 1)

        egrid = get_esmf_grid(grid)
        egrid_mask_inverted = np.invert(np.array(egrid.mask[0], dtype=bool))
        self.assertNumpyAll(grid.get_mask(), egrid_mask_inverted)

        # Test with a value mask.
        value_mask = np.zeros(grid.shape, dtype=bool)
        value_mask[-1, -1] = True
        egrid = get_esmf_grid(grid, value_mask=value_mask)
        egrid_mask_inverted = np.invert(np.array(egrid.mask[0], dtype=bool))
        self.assertNumpyAll(egrid_mask_inverted, np.logical_or(grid.get_mask(), value_mask))

    @attr('esmf')
    def test_get_ocgis_field_from_esmf_field(self):
        from ocgis.regrid.base import get_esmf_field_from_ocgis_field
        from ocgis.regrid.base import get_ocgis_field_from_esmf_field

        ogrid = create_gridxy_global(crs=Spherical())
        ofield = create_exact_field(ogrid, 'foo', ntime=3)

        ogrid = ofield.grid
        ogrid_mask = ogrid.get_mask(create=True)
        ogrid_mask[1, 0] = True
        ogrid.set_mask(ogrid_mask)

        efield = get_esmf_field_from_ocgis_field(ofield)
        self.assertEqual(efield.data.shape, (360, 180, 3))

        ofield_actual = get_ocgis_field_from_esmf_field(efield, field=ofield)

        actual_dv_mask = ofield_actual.data_variables[0].get_mask()
        self.assertTrue(np.all(actual_dv_mask[:, 1, 0]))
        self.assertEqual(actual_dv_mask.sum(), 3)

        self.assertNumpyAll(ofield.data_variables[0].get_value(), ofield_actual.data_variables[0].get_value())
        self.assertEqual(ofield.data_variables[0].name, efield.name)
        self.assertNumpyAll(ofield.time.get_value(), ofield_actual.time.get_value())

    @attr('esmf')
    def test_get_ocgis_field_from_esmf_spatial_only(self):
        """Test with spatial information only."""

        from ocgis.regrid.base import get_esmf_field_from_ocgis_field
        from ocgis.regrid.base import get_ocgis_field_from_esmf_field

        row = Variable(name='row', value=[5, 6], dimensions='row')
        col = Variable(name='col', value=[7, 8], dimensions='col')
        grid = Grid(col, row)
        ofield = Field(grid=grid, crs=Spherical())

        efield = get_esmf_field_from_ocgis_field(ofield)
        ofield_actual = get_ocgis_field_from_esmf_field(efield)

        self.assertEqual(len(ofield_actual.data_variables), 0)
        self.assertNumpyAll(grid.get_value_stacked(), ofield_actual.grid.get_value_stacked())


class TestRegridOperation(AbstractTestInterface):
    @staticmethod
    def create_regridding_field(grid, name_variable):
        col_shape = grid.shape[1]
        row_shape = grid.shape[0]
        value = np.ones(col_shape * row_shape, dtype=float).reshape(grid.shape) * 15.
        variable = Variable(name=name_variable, value=value, dimensions=grid.dimensions)
        field = Field(is_data=variable, grid=grid, crs=Spherical())
        return field

    @attr('slow', 'esmf')
    def test_system_global_grid_combinations(self):
        """Test regridding with different global grid configurations."""

        from ocgis.regrid.base import RegridOperation

        boolmix = [[True, True], [False, False], [True, False], [False, True]]

        keywords = dict(with_bounds=boolmix, wrapped=boolmix, resolution=[[3.0, 6.0], [6.0, 3.0], [3.0, 3.0]])
        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            # print ctr, k
            # if ctr != 10: continue
            source_grid = self.get_gridxy_global(with_bounds=k.with_bounds[0], wrapped=k.wrapped[0],
                                                 resolution=k.resolution[0])
            destination_grid = self.get_gridxy_global(with_bounds=k.with_bounds[1], wrapped=k.wrapped[1],
                                                      resolution=k.resolution[1])
            source_field = self.create_regridding_field(source_grid, 'source')
            destination_field = self.create_regridding_field(destination_grid, 'destination')
            ro = RegridOperation(source_field, destination_field)
            res = ro.execute()

            actual = res['source'].get_masked_value()
            targets = [actual.min(), actual.mean(), actual.max()]
            for t in targets:
                self.assertAlmostEqual(t, 15.0, places=5)

                # @attr('esmf')
                # def test_system_periodic(self):
                #     """Test a periodic grid is regridded as expected."""
                #
                #     from ocgis.regrid.base import RegridOperation
                #
                #     lon_in = np.arange(-180, 180, 10)
                #     lat_in = np.arange(-90, 90.1, 4)
                #
                #     lon = Variable('lon', lon_in, 'dlon')
                #     lat = Variable('lat', lat_in, 'dlat')
                #     print 'lon.shape', lon.shape, 'lat.shape', lat.shape
                #
                #     grid = Grid(x=lon, y=lat, crs=Spherical())
                #
                #     wave = lambda x, k: np.sin(x * k * np.pi / 180.0)
                #     desired = np.outer(wave(lat_in, 3), wave(lon_in, 3)) + 1
                #
                #     print 'desired.min()', desired.min()
                #     print 'desired.mean()', desired.mean()
                #
                #     dstdata = Variable('data', dimensions=('dlat', 'dlon'), parent=grid.parent)
                #     srcdata = deepcopy(dstdata)
                #     srcdata.get_value()[:] = desired
                #
                #     dstfield = dstdata.parent
                #     dstfield.append_to_tags(TagName.DATA_VARIABLES, 'data')
                #     srcfield = srcdata.parent
                #     srcfield.append_to_tags(TagName.DATA_VARIABLES, 'data')
                #
                #     self.assertEqual(dstfield.data_variables[0].get_value().max(), 0.)
                #
                #     ro = RegridOperation(srcfield, dstfield)
                #     ret = ro.execute()
                #
                #     actual = ret.data_variables[0].get_value()
                #
                #     print 'actual.min()', actual.min()
                #     print 'actual.mean()', actual.mean()
                #
                #     print np.max(np.abs(actual - desired))
