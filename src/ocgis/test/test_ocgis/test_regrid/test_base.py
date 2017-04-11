from copy import deepcopy

import numpy as np

from ocgis import OcgOperations
from ocgis import RequestDataset
from ocgis import Variable
from ocgis.collection.field import OcgField
from ocgis.exc import RegriddingError, CornersInconsistentError
from ocgis.spatial.grid import GridXY
from ocgis.test.base import attr, AbstractTestInterface
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

        keywords = dict(split=[False, True])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            from ocgis.regrid.base import regrid_field
            regridded = regrid_field(source, destination, split=k.split)
            self.assertIsInstance(regridded, OcgField)
            self.assertNumpyAll(regridded.grid.get_value_stacked(), desired.grid.get_value_stacked())
            self.assertEqual(regridded.crs, source.crs)
            for variable in regridded.data_variables:
                self.assertGreater(variable.get_value().mean(), 2.0)
                self.assertNumpyAll(variable.get_value(), source[variable.name].get_value().squeeze())
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
        self.assertTrue(np.all(regridded.data_variables[0].get_mask()[:, 1, 1]))

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
        self.assertTrue(all([ii == slice(None) for ii in fill['tidx']]))

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
            self.assertEqual(efield.data.shape[0], ofield.time.shape[0])
            # Test masks are equivalent. ESMF masks have "1" for false requiring the invert operation.
            egrid = efield.grid
            egrid_mask = np.invert(egrid.mask[0].astype(bool))
            self.assertNumpyAll(egrid_mask, mask)

        # Test splitting the data.
        ofield = self.get_ofield()
        res = list(iter_esmf_fields(ofield))
        self.assertEqual(len(res), ofield.time.shape[0] * len(ofield.data_variables))

    @attr('esmf')
    def test_get_esmf_grid(self):
        import ESMF
        rd = RequestDataset(**self.get_dataset())

        have_ocgis_bounds = [True, False]
        for h in have_ocgis_bounds:
            field = rd.get()
            ogrid = field.grid

            if not h:
                ogrid.remove_bounds()
                self.assertFalse(ogrid.has_bounds)

            from ocgis.regrid.base import get_esmf_grid
            egrid = get_esmf_grid(ogrid)

            # ocgis is row major with esmf being column major (i.e. in ocgis rows are stored in the zero index)
            for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
                coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
                desired = ogrid.get_value_stacked()[idx_ocgis, :]
                self.assertNumpyAll(coords, desired)

            corner = egrid.coords[ESMF.StaggerLoc.CORNER]
            if h:
                corner_row = corner[1]
                corner_row_actual = np.array(
                    [[36.5, 36.5, 36.5, 36.5, 36.5], [37.5, 37.5, 37.5, 37.5, 37.5], [38.5, 38.5, 38.5, 38.5, 38.5],
                     [39.5, 39.5, 39.5, 39.5, 39.5], [40.5, 40.5, 40.5, 40.5, 40.5]],
                    dtype=ogrid.archetype.dtype)
                self.assertNumpyAll(corner_row, corner_row_actual)

                corner = egrid.coords[ESMF.StaggerLoc.CORNER]
                corner_col = corner[0]
                corner_col_actual = np.array(
                    [[-105.5, -104.5, -103.5, -102.5, -101.5], [-105.5, -104.5, -103.5, -102.5, -101.5],
                     [-105.5, -104.5, -103.5, -102.5, -101.5], [-105.5, -104.5, -103.5, -102.5, -101.5],
                     [-105.5, -104.5, -103.5, -102.5, -101.5]], dtype=ogrid.archetype.dtype)
                self.assertNumpyAll(corner_col, corner_col_actual)
            else:
                # No corners should be on the ESMF grid.
                for idx in [0, 1]:
                    self.assertIsNone(corner[idx])

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
    def test_get_esmf_grid_change_origin_col(self):
        """Test with different column grid origin."""

        import ESMF
        from ocgis.regrid.base import get_esmf_grid

        rd = RequestDataset(**self.get_dataset())
        field = rd.get()

        field.grid.x.set_value(np.flipud(field.grid.x.get_value()))
        field.grid.x.bounds.set_value(np.fliplr(np.flipud(field.grid.x.bounds.get_value())))

        egrid = get_esmf_grid(field.grid)

        for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
            coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
            self.assertNumpyAll(coords, field.grid.get_value_stacked()[idx_ocgis, ...])

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_row = corner[1]
        corner_row_actual = np.array(
            [[36.5, 36.5, 36.5, 36.5, 36.5], [37.5, 37.5, 37.5, 37.5, 37.5], [38.5, 38.5, 38.5, 38.5, 38.5],
             [39.5, 39.5, 39.5, 39.5, 39.5], [40.5, 40.5, 40.5, 40.5, 40.5]], dtype=field.grid.archetype.dtype)
        self.assertNumpyAll(corner_row, corner_row_actual)

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_col = corner[0]
        corner_col_actual = np.array(
            [[-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5],
             [-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5],
             [-101.5, -102.5, -103.5, -104.5, -105.5]], dtype=field.grid.archetype.dtype)
        self.assertNumpyAll(corner_col, corner_col_actual)

    @attr('esmf')
    def test_get_esmf_grid_change_origin_row(self):
        """Test with different row grid origin."""

        from ocgis.regrid.base import get_esmf_grid
        import ESMF

        rd = RequestDataset(**self.get_dataset())
        field = rd.get()

        field.grid.y.set_value(np.flipud(field.grid.y.get_value()))
        field.grid.y.bounds.set_value(np.fliplr(np.flipud(field.grid.y.bounds.get_value())))

        egrid = get_esmf_grid(field.grid)

        for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
            coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
            self.assertNumpyAll(coords, field.grid.get_value_stacked()[idx_ocgis, ...])

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_row = corner[1]
        corner_row_actual = np.array(
            [[40.5, 40.5, 40.5, 40.5, 40.5], [39.5, 39.5, 39.5, 39.5, 39.5], [38.5, 38.5, 38.5, 38.5, 38.5],
             [37.5, 37.5, 37.5, 37.5, 37.5], [36.5, 36.5, 36.5, 36.5, 36.5]], dtype=field.grid.archetype.dtype)
        self.assertNumpyAll(corner_row, corner_row_actual)

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_col = corner[0]
        corner_col_actual = np.array(
            [[-105.5, -104.5, -103.5, -102.5, -101.5], [-105.5, -104.5, -103.5, -102.5, -101.5],
             [-105.5, -104.5, -103.5, -102.5, -101.5], [-105.5, -104.5, -103.5, -102.5, -101.5],
             [-105.5, -104.5, -103.5, -102.5, -101.5]], dtype=field.grid.archetype.dtype)
        self.assertNumpyAll(corner_col, corner_col_actual)

    @attr('esmf')
    def test_get_esmf_grid_change_origin_row_and_col(self):
        """Test with different row and column grid origin."""

        from ocgis.regrid.base import get_esmf_grid
        import ESMF

        rd = RequestDataset(**self.get_dataset())
        field = rd.get()

        field.grid.y.set_value(np.flipud(field.grid.y.get_value()))
        field.grid.y.bounds.set_value(np.fliplr(np.flipud(field.grid.y.bounds.get_value())))
        field.grid.x.set_value(np.flipud(field.grid.x.get_value()))
        field.grid.x.bounds.set_value(np.fliplr(np.flipud(field.grid.x.bounds.get_value())))

        egrid = get_esmf_grid(field.grid)

        for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
            coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
            self.assertNumpyAll(coords, field.grid.get_value_stacked()[idx_ocgis, ...])

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_row = corner[1]
        dtype = field.grid.dtype
        corner_row_actual = np.array(
            [[40.5, 40.5, 40.5, 40.5, 40.5], [39.5, 39.5, 39.5, 39.5, 39.5], [38.5, 38.5, 38.5, 38.5, 38.5],
             [37.5, 37.5, 37.5, 37.5, 37.5], [36.5, 36.5, 36.5, 36.5, 36.5]], dtype=dtype)
        self.assertNumpyAll(corner_row, corner_row_actual)

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_col = corner[0]
        corner_col_actual = np.array(
            [[-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5],
             [-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5],
             [-101.5, -102.5, -103.5, -104.5, -105.5]], dtype=dtype)
        self.assertNumpyAll(corner_col, corner_col_actual)

    @attr('esmf')
    def test_get_esmf_grid_with_mask(self):
        """Test with masked data."""

        from ocgis.regrid.base import get_esmf_grid

        x = Variable(name='x', value=[1, 2, 3], dimensions='x')
        y = Variable(name='y', value=[4, 5, 6], dimensions='y')
        grid = GridXY(x, y)

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

        ofield = self.get_field(nlevel=0, nrlz=0)
        ogrid = ofield.grid
        ogrid_mask = ogrid.get_mask(create=True)
        ogrid_mask[1, 0] = True
        ogrid.set_mask(ogrid_mask)

        time = ofield.time.extract()
        efield = get_esmf_field_from_ocgis_field(ofield)
        ofield_actual = get_ocgis_field_from_esmf_field(efield, ofield.data_variables[0].dimensions,
                                                        dimension_map=ofield.dimension_map, time=time)

        actual_dv_mask = ofield_actual.data_variables[0].get_mask()
        self.assertTrue(np.all(actual_dv_mask[:, 1, 0]))
        self.assertEqual(actual_dv_mask.sum(), 2)

        self.assertNumpyAll(ofield.time.get_value(), ofield_actual.time.get_value())
        self.assertNumpyAll(ofield.data_variables[0].get_value(), ofield_actual.data_variables[0].get_value())
        self.assertEqual(ofield.data_variables[0].name, efield.name)

    @attr('esmf')
    def test_get_ocgis_field_from_esmf_spatial_only(self):
        """Test with spatial information only."""

        from ocgis.regrid.base import get_esmf_field_from_ocgis_field
        from ocgis.regrid.base import get_ocgis_field_from_esmf_field

        row = Variable(name='row', value=[5, 6], dimensions='row')
        col = Variable(name='col', value=[7, 8], dimensions='col')
        grid = GridXY(col, row)
        ofield = OcgField(grid=grid)

        efield = get_esmf_field_from_ocgis_field(ofield)
        ofield_actual = get_ocgis_field_from_esmf_field(efield)

        self.assertEqual(len(ofield_actual.data_variables), 0)
        self.assertNumpyAll(grid.get_value_stacked(), ofield_actual.grid.get_value_stacked())

    @attr('esmf')
    def test_get_ocgis_grid_from_esmf_grid(self):
        from ocgis.regrid.base import get_esmf_grid
        from ocgis.regrid.base import get_ocgis_grid_from_esmf_grid

        rd = RequestDataset(**self.get_dataset())

        keywords = dict(has_corners=[True, False],
                        has_mask=[True, False],
                        crs=[None, CoordinateReferenceSystem(epsg=4326)])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            field = rd.get()
            grid = field.grid

            if k.has_mask:
                gmask = grid.get_mask(create=True)
                gmask[1, :] = True
                grid.set_mask(gmask)

            if not k.has_corners:
                grid.x.set_bounds(None)
                grid.y.set_bounds(None)
                self.assertFalse(grid.has_bounds)

            egrid = get_esmf_grid(grid)
            ogrid = get_ocgis_grid_from_esmf_grid(egrid, crs=k.crs)

            if k.has_mask:
                actual_mask = ogrid.get_mask()
                self.assertEqual(actual_mask.sum(), 4)
                self.assertTrue(actual_mask[1, :].all())

            self.assertEqual(ogrid.crs, k.crs)

            self.assertNumpyAll(grid.get_value_stacked(), ogrid.get_value_stacked())
            if k.has_corners:
                desired = grid.x.bounds.get_value()
                actual = ogrid.x.bounds.get_value()
                self.assertNumpyAll(actual, desired)

                desired = grid.y.bounds.get_value()
                actual = ogrid.y.bounds.get_value()
                self.assertNumpyAll(actual, desired)
            else:
                self.assertFalse(ogrid.has_bounds)


class TestRegridOperation(AbstractTestInterface):
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
            source_field = self.get_regridding_field(source_grid, 'source')
            destination_field = self.get_regridding_field(destination_grid, 'destination')
            ro = RegridOperation(source_field, destination_field)
            res = ro.execute()

            actual = res['source'].get_masked_value()
            targets = [actual.min(), actual.mean(), actual.max()]
            for t in targets:
                self.assertAlmostEqual(t, 15.0, places=5)

    @staticmethod
    def get_regridding_field(grid, name_variable):
        col_shape = grid.shape[1]
        row_shape = grid.shape[0]
        value = np.ones(col_shape * row_shape, dtype=float).reshape(grid.shape) * 15.
        variable = Variable(name=name_variable, value=value, dimensions=grid.dimensions)
        field = OcgField(is_data=variable, grid=grid, crs=Spherical())
        return field
