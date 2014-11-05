from copy import deepcopy, copy
import ESMF
from shapely.geometry import Polygon, MultiPolygon
import ocgis
from ocgis.exc import RegriddingError, CornersInconsistentError
from ocgis.interface.base.crs import CoordinateReferenceSystem, WGS84, Spherical
from ocgis.interface.base.dimension.spatial import SpatialGridDimension, SpatialDimension
from ocgis.interface.base.field import Field
from ocgis.interface.base.variable import VariableCollection
from ocgis.regrid.base import check_fields_for_regridding, iter_regridded_fields, get_esmf_grid_from_sdim, \
    iter_esmf_fields, get_sdim_from_esmf_grid
from ocgis.test.test_simple.make_test_data import SimpleNc
from ocgis.test.test_simple.test_simple import TestSimpleBase
import numpy as np
from ocgis.util.helpers import iter_array, make_poly
from ocgis.util.itester import itr_products_keywords


class TestRegrid(TestSimpleBase):
    nc_factory = SimpleNc
    fn = 'test_simple_spatial_01.nc'

    def get_ofield(self):
        rd = ocgis.RequestDataset(**self.get_dataset())
        # create a field composed of two variables
        ofield = rd.get()
        new_variable = deepcopy(ofield.variables['foo'])
        new_variable.alias = 'foo2'
        new_variable.value[:] = 30
        ofield.variables.add_variable(new_variable, assign_new_uid=True)
        # only one realization and level are okay at this time
        ofield = ofield[0, :, 0, :, :]
        return ofield

    def atest_single_to_wgs84(self):

        def get_coords(mpoly):
            all_coords = []

            try:
                it = iter(mpoly)
            except TypeError:
                it = [mpoly]

            for poly in it:
                coords = [c for c in poly.exterior.coords]
                all_coords.append(coords)
            all_coords = np.array(all_coords)
            return all_coords

        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        field.spatial.crs = Spherical()
        odd = field[:, :, :, 32, 64]
        odd.spatial.wrap()
        odd.spatial.write_fiona('/tmp/odd_sphere.shp')
        mpoly_original = deepcopy(odd.spatial.geom.polygon.value[0, 0])
        mpoly_original_coords = get_coords(mpoly_original)
        select = mpoly_original_coords[0] == 180
        mpoly_original_coords[0][select] = 179.9999999999999
        select = mpoly_original_coords[1] == -180
        mpoly_original_coords[1][select] = -179.9999999999999
        p1 = Polygon(mpoly_original_coords[0])
        p2 = Polygon(mpoly_original_coords[1])
        mpoly = MultiPolygon([p1, p2])
        odd.spatial.geom.polygon.value[0, 0] = mpoly

        odd.spatial.update_crs(WGS84())
        odd.spatial.write_fiona('/tmp/odd_wgs84.shp')
        mpoly_updated = deepcopy(odd.spatial.geom.polygon.value[0, 0])
        mpoly_updated_coords = get_coords(mpoly_updated)

        import ipdb;ipdb.set_trace()

    def atest_to_spherical(self):
        rd = self.test_data.get_rd('cancm4_tas')
        coll = ocgis.OcgOperations(dataset=rd, vector_wrap=True).execute() #, geom='state_boundaries', agg_selection=True).execute()
        field = coll[1]['tas']
        grid_original = deepcopy(field.spatial.grid)
        # import ipdb;ipdb.set_trace()
        #todo: need to resolve row/column and corners

        # import ipdb;ipdb.set_trace()
        # field = rd.get()
        # field.spatial.wrap()

        target = 'polygon'

        field.spatial.crs = CoordinateReferenceSystem(
            value={'ellps': 'sphere', 'proj': 'longlat', 'towgs84': '0,0,0,0,0,0,0', 'no_defs': ''})
        ofield = deepcopy(field)
        field.spatial.write_fiona('/tmp/sphere.shp', target=target)
        to_crs = WGS84()
        # to_crs_value = to_crs.value
        # to_crs_value.update({'lon_wrap': '180'})
        field.spatial.update_crs(to_crs)
        # wrapping is handled differently by PROJ!
        # field.spatial.grid.value.data[1] = grid_original.value.data[1]
        # field.spatial.grid.corners.data[1] = grid_original.corners.data[1]
        # grid_new = field.spatial.grid.value.copy()
        field.spatial.write_fiona('/tmp/wgs84.shp', target=target)
        # diff = np.abs(grid_original[0].data - grid_new[0].data).mean()
        import ipdb;ipdb.set_trace()

    def test_check_fields_for_regridding(self):

        for use_sdim in [False, True]:
            ofield = self.get_ofield()
            ofield2 = deepcopy(ofield)
            sources = [ofield, ofield2]
            destination_field = deepcopy(ofield)

            if use_sdim:
                destination = destination_field.spatial
            else:
                destination = destination_field

            # test non spherical coordinate systems
            with self.assertRaises(RegriddingError):
                check_fields_for_regridding(sources, destination)

            # change coordinate systems to spherical
            for source in sources:
                source.spatial.crs = Spherical()
            try:
                destination.spatial.crs = Spherical()
            except AttributeError:
                destination.crs = Spherical()

            # test with different parameters of the sphere
            new_sources = deepcopy(sources)
            new_sources[0].spatial.crs = Spherical(semi_major_axis=100)
            with self.assertRaises(RegriddingError):
                check_fields_for_regridding(new_sources, destination)

            # test wrapping check
            new_sources = deepcopy(sources)
            new_sources[1].spatial.unwrap()
            with self.assertRaises(RegriddingError):
                check_fields_for_regridding(new_sources, destination)

            # test spatial extents
            new_sources = deepcopy(sources)
            ref = new_sources[1].spatial
            ref.grid.value[:] = ref.grid.value[:] + 10
            ref.grid.row = None
            ref.grid.col = None
            ref.grid._corners = None
            ref.grid._geom = None
            with self.assertRaises(RegriddingError):
                check_fields_for_regridding(new_sources, destination)

            # test corners not available on all inputs

            # test corners not on one of the sources
            new_sources = deepcopy(sources)
            ref = new_sources[0]
            ref.spatial.grid.value
            ref.spatial.grid.row = None
            ref.spatial.grid.col = None
            ref.spatial.grid._corners = None
            self.assertIsNone(ref.spatial.grid.corners)
            for with_corners in [True, False]:
                if with_corners:
                    with self.assertRaises(CornersInconsistentError):
                        check_fields_for_regridding(new_sources, destination, with_corners=with_corners)
                else:
                    check_fields_for_regridding(new_sources, destination, with_corners=with_corners)

            # test with corners not available on the destination grid
            ref = deepcopy(destination_field)
            ref.spatial.grid.value
            ref.spatial.grid.row = None
            ref.spatial.grid.col = None
            ref.spatial.grid._corners = None
            self.assertIsNone(ref.spatial.grid.corners)
            for with_corners in [True, False]:
                if with_corners:
                    with self.assertRaises(CornersInconsistentError):
                        check_fields_for_regridding(sources, ref, with_corners=with_corners)
                else:
                    check_fields_for_regridding(sources, ref, with_corners=with_corners)

    def test_iter_regridded_fields_different_grid_shapes(self):
        """Test regridding a downscaled dataset to GCM output. The input and output grids have different shapes."""

        downscaled = self.test_data.get_rd('maurer_2010_tas')
        downscaled.time_region = {'month': [2], 'year': [1990]}
        downscaled = downscaled.get()
        poly = make_poly([37, 43], [-104, -94])
        downscaled = downscaled.get_intersects(poly)
        downscaled.spatial.unwrap()
        downscaled.spatial.crs = Spherical()

        gcm = self.test_data.get_rd('cancm4_tas')
        gcm = gcm.get()
        poly = make_poly([37, 43], [-104+360, -94+360])
        gcm = gcm.get_intersects(poly)
        gcm.spatial.crs = Spherical()

        # add masked values to the source and destination
        self.assertFalse(downscaled.spatial.get_mask().any())
        self.assertFalse(gcm.spatial.get_mask().any())

        mask = gcm.spatial.get_mask()
        mask[1, 3] = True
        gcm.spatial.set_mask(mask)

        dmask = downscaled.spatial.get_mask()
        dmask[:] = True
        downscaled.spatial.set_mask(dmask)
        downscaled.variables.first().value.mask[:] = True

        for regridded in iter_regridded_fields([downscaled], gcm):
            self.assertEqual(regridded.shape, (1, 28, 1, 3, 5))
            self.assertEqual(regridded.variables.keys(), ['tas'])
            self.assertAlmostEqual(regridded.variables['tas'].value.data.mean(), 0.057409391)
            self.assertNumpyAll(gcm.spatial.get_mask(), mask)
            for variable in regridded.variables.itervalues():
                vmask = variable.value.mask
                self.assertTrue(vmask[:, :, :, 1, 3].all())
                self.assertEqual(vmask.sum(), 28)

    def test_iter_regridded_fields_problem_bounds(self):
        """Test a dataset with crap bounds will work when with_corners is False."""

        dst = self.test_data.get_rd('cancm4_tas').get()[:, :, :, 20:25, 30:35]
        dst.spatial.crs = Spherical()
        src = deepcopy(dst[0, 0, 0, :, :])

        egrid_dst = get_esmf_grid_from_sdim(dst.spatial)
        egrid_src = get_esmf_grid_from_sdim(src.spatial)
        self.assertEqual(egrid_dst.mask[0].sum(), 25)
        self.assertEqual(egrid_src.mask[0].sum(), 25)
        self.assertNumpyAll(egrid_dst.coords[0][0], egrid_src.coords[0][0])
        self.assertNumpyAll(egrid_dst.coords[0][1], egrid_src.coords[0][1])

        ret = list(iter_esmf_fields(src))
        self.assertEqual(len(ret), 1)
        tidx, variable_alias, efield = ret[0]
        self.assertNumpyAll(efield.grid.coords[0][0], dst.spatial.grid.value.data[1])
        self.assertNumpyAll(efield.grid.coords[0][1], dst.spatial.grid.value.data[0])

        ret = list(iter_regridded_fields([src], dst, with_corners=False))[0]
        actual = dst[0, 0, 0, :, :].variables.first().value
        to_test = ret.variables.first().value
        self.assertFalse(to_test.mask.any())

        self.assertNumpyAllClose(to_test.data, actual.data)
        self.assertNumpyAll(to_test.mask, actual.mask)

    def test_iter_regridded_fields(self):
        """Test with equivalent input and output expectations. The shapes of the grids are equal."""

        ofield = self.get_ofield()
        ofield.spatial.crs = Spherical()
        ofield2 = deepcopy(ofield)
        sources = [ofield, ofield2]
        destination_field = deepcopy(ofield)

        keywords = dict(use_sdim=[True, False])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            if k.use_sdim:
                destination = destination_field.spatial
                sdim = destination
            else:
                destination = destination_field
                sdim = destination.spatial
            self.assertIsNotNone(sdim.grid.row)
            self.assertIsNotNone(sdim.grid.col)
            for ctr, regridded in enumerate(iter_regridded_fields(sources, destination, with_corners='choose',
                                                                  value_mask=None)):
                self.assertIsInstance(regridded, Field)
                self.assertNumpyAll(regridded.spatial.grid.value, sdim.grid.value)
                self.assertEqual(regridded.spatial.crs, sdim.crs)
                self.assertNumpyAll(regridded.spatial.grid.row.value, sdim.grid.row.value)
                self.assertNumpyAll(regridded.spatial.grid.row.bounds, sdim.grid.row.bounds)
                self.assertNumpyAll(regridded.spatial.grid.col.value, sdim.grid.col.value)
                self.assertNumpyAll(regridded.spatial.grid.col.bounds, sdim.grid.col.bounds)
                for variable in regridded.variables.itervalues():
                    self.assertNumpyAll(variable.value, sources[ctr].variables[variable.alias].value)
                    self.assertFalse(np.may_share_memory(variable.value, sources[ctr].variables[variable.alias].value))
            self.assertEqual(ctr, 1)

    def test_iter_regridded_field_with_corners(self):
        """Test with_corners as True and False when regridding Fields."""

        ofield = self.get_ofield()
        ofield.spatial.crs = Spherical()
        ofield2 = deepcopy(ofield)
        sources = [ofield, ofield2]
        destination_field = deepcopy(ofield)

        self.assertIsNotNone(destination_field.spatial.geom.polygon)
        for regridded in iter_regridded_fields(sources, destination_field, with_corners=False):
            self.assertIsNone(regridded.spatial.grid.row.bounds)
            self.assertIsNone(regridded.spatial.grid.col.bounds)
            self.assertIsNone(regridded.spatial.grid.corners)
            self.assertIsNone(regridded.spatial.geom.polygon)

        # check that the destination grid is not modified
        self.assertIsNotNone(destination_field.spatial.grid.row.bounds)

        # remove corners from the destination and make sure that this is caught when with_corners is True
        dest = deepcopy(destination_field).spatial
        dest.grid.value
        dest.grid.row.bounds
        dest.grid.row.bounds = None
        dest.grid.col.bounds
        dest.grid.col.bounds = None
        dest.grid._corners = None
        self.assertIsNone(dest.grid.corners)
        with self.assertRaises(CornersInconsistentError):
            list(iter_regridded_fields(sources, dest, with_corners=True))
        # if this is now false, then there should be no problem as only centroids are used
        list(iter_regridded_fields(sources, dest, with_corners=False))
        # this is also the case with 'choose'
        list(iter_regridded_fields(sources, dest, with_corners='choose'))

    def test_iter_regridded_fields_differing_crs(self):
        """Test exception raised when source and destination CRS values are not equal."""

        ofield = self.get_ofield()
        ofield2 = deepcopy(ofield)
        sources = [ofield, ofield2]
        destination_field = deepcopy(ofield)

        sources[1].spatial.crs = CoordinateReferenceSystem(epsg=2136)

        with self.assertRaises(RegriddingError):
            list(iter_regridded_fields(sources, destination_field))

    def test_iter_regridded_fields_value_mask(self):
        """Test with a value mask on the destination."""

        ofield = self.get_ofield()
        ofield.spatial.crs = Spherical()
        ofield2 = deepcopy(ofield)
        sources = [ofield, ofield2]
        destination_field = deepcopy(ofield)

        value_mask = np.zeros(destination_field.spatial.shape, dtype=bool)
        value_mask[1, 1] = True

        for regridded in iter_regridded_fields(sources, destination_field, value_mask=value_mask):
            self.assertTrue(np.all(regridded.variables.first().value.mask[:, :, :, 1, 1]))

    def test_iter_regridded_fields_nonoverlapping_extents(self):
        """Test regridding with fields that do not spatially overlap."""

        rd = self.test_data.get_rd('cancm4_tas')
        # nebraska and california
        coll = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[16, 25], snippet=True,
                                   vector_wrap=False).execute()
        source = coll[25]['tas']
        destination = coll[16]['tas']
        source.spatial.crs = Spherical()
        destination.spatial.crs = Spherical()

        with self.assertRaises(RegriddingError):
            list(iter_regridded_fields([source], destination))

    def test_iter_regridded_fields_partial_extents(self):
        """Test regridding with fields that partially overlap."""

        rd = self.test_data.get_rd('cancm4_tas')
        # california and nevada
        coll = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23, 25], snippet=True,
                                   vector_wrap=False).execute()
        source = coll[25]['tas']
        destination = coll[23]['tas']
        source.spatial.crs = Spherical()
        destination.spatial.crs = Spherical()

        with self.assertRaises(RegriddingError):
            list(iter_regridded_fields([source], destination))

    def test_iter_esmf_fields(self):
        ofield = self.get_ofield()

        # add a unique spatial mask and value mask
        mask = ofield.spatial.get_mask()
        mask[1, 1] = True
        ofield.spatial.set_mask(mask)
        for variable in ofield.variables.itervalues():
            variable.value.mask[:, :, :, :, :] = mask.copy()

        dtype = [('tidx', int), ('variable_alias', object), ('efield', object)]
        fill = np.array([], dtype=dtype)
        for row in iter_esmf_fields(ofield, with_corners=True, value_mask=None):
            app = np.zeros(1, dtype=dtype)
            app[0] = row
            fill = np.append(fill, app)

        self.assertEqual(fill.shape[0], 122)
        self.assertEqual(np.unique(fill[0:61]['variable_alias']), 'foo')
        self.assertEqual(np.unique(fill[61:]['variable_alias']), 'foo2')

        means = np.array([element.data.mean() for element in fill['efield']])
        self.assertTrue(np.all(means[0:61] == 2.5))
        self.assertTrue(np.all(means[61:] == 30))

        for idx in range(fill.shape[0]):
            efield = fill['efield'][idx]
            egrid = efield.grid
            egrid_mask = np.invert(egrid.mask[0].astype(bool))
            self.assertNumpyAll(egrid_mask, mask)

    def test_get_sdim_from_esmf_grid(self):
        rd = ocgis.RequestDataset(**self.get_dataset())

        keywords = dict(has_corners=[True, False],
                        has_mask=[True, False])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            field = rd.get()
            sdim = field.spatial
            egrid = get_esmf_grid_from_sdim(sdim)

            if not k.has_mask:
                # set the grid flag to indicate no mask is present
                egrid.item_done[ESMF.StaggerLoc.CENTER][0] = False
                # remove the mask from the grid
                egrid.mask[0] = [1]
            else:
                egrid.mask[0][2, 2] = 0
                sdim.grid.value.mask[:, 2, 2] = True
                sdim.grid.corners.mask[:, 2, 2] = True

            if not k.has_corners:
                egrid.coords[ESMF.StaggerLoc.CORNER] = [np.array(0.0), np.array(0.0)]
                egrid.coords_done[ESMF.StaggerLoc.CORNER] = [False, False]

            nsdim = get_sdim_from_esmf_grid(egrid)

            self.assertNumpyAll(sdim.grid.value, nsdim.grid.value)
            if k.has_corners:
                self.assertNumpyAll(sdim.grid.corners, nsdim.grid.corners)
            else:
                self.assertIsNone(nsdim.grid.corners)

    def test_get_esmf_grid_from_sdim_with_mask(self):
        """Test with masked data."""

        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23], snippet=True, vector_wrap=False)
        ret = ops.execute()
        field = ret[23]['tas']
        egrid = get_esmf_grid_from_sdim(field.spatial)
        actual = np.array([[0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32)
        self.assertNumpyAll(egrid.mask[0], actual)

        sdim = get_sdim_from_esmf_grid(egrid)
        self.assertNumpyAll(sdim.get_mask(), field.spatial.get_mask())
        actual = np.array([[[[True, True, True, True], [True, True, True, True], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]]], [[[True, True, True, True], [True, True, True, True], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]]]])
        self.assertNumpyAll(actual, sdim.grid.corners.mask)

    def test_get_esmf_grid_from_sdim(self):
        rd = ocgis.RequestDataset(**self.get_dataset())

        have_ocgis_bounds = [True, False]
        for h in have_ocgis_bounds:
            field = rd.get()
            sdim = field.spatial

            if not h:
                sdim.grid.row.bounds
                sdim.grid.col.bounds
                sdim.grid.row.bounds = None
                sdim.grid.col.bounds = None
                self.assertIsNone(sdim.grid.row.bounds)
                self.assertIsNone(sdim.grid.col.bounds)

            egrid = get_esmf_grid_from_sdim(sdim)

            # ocgis is row major with esmf being column major (i.e. in ocgis rows are stored in the zero index)
            for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
                coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
                self.assertNumpyAll(coords, field.spatial.grid.value[idx_ocgis, ...].data)

            corner = egrid.coords[ESMF.StaggerLoc.CORNER]
            if h:
                corner_row = corner[1]
                corner_row_actual = np.array(
                    [[36.5, 36.5, 36.5, 36.5, 36.5], [37.5, 37.5, 37.5, 37.5, 37.5], [38.5, 38.5, 38.5, 38.5, 38.5],
                     [39.5, 39.5, 39.5, 39.5, 39.5], [40.5, 40.5, 40.5, 40.5, 40.5]],
                    dtype=field.spatial.grid.value.dtype)
                self.assertNumpyAll(corner_row, corner_row_actual)

                corner = egrid.coords[ESMF.StaggerLoc.CORNER]
                corner_col = corner[0]
                corner_col_actual = np.array(
                    [[-105.5, -104.5, -103.5, -102.5, -101.5], [-105.5, -104.5, -103.5, -102.5, -101.5],
                     [-105.5, -104.5, -103.5, -102.5, -101.5], [-105.5, -104.5, -103.5, -102.5, -101.5],
                     [-105.5, -104.5, -103.5, -102.5, -101.5]], dtype=field.spatial.grid.value.dtype)
                self.assertNumpyAll(corner_col, corner_col_actual)
            else:
                for idx in [0, 1]:
                    self.assertNumpyAll(corner[idx], np.array(0.0))

    def test_get_esmf_grid_from_sdim_real_data(self):
        """Test creating ESMF field from real data using an OCGIS spatial dimension."""

        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        egrid = get_esmf_grid_from_sdim(field.spatial)

        for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
            coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
            self.assertNumpyAll(coords, field.spatial.grid.value[idx_ocgis, ...].data)

    def test_get_esmf_grid_from_sdim_change_origin_row(self):
        """Test with different row grid origin."""

        rd = ocgis.RequestDataset(**self.get_dataset())
        field = rd.get()

        field.spatial.grid.row._value = np.flipud(field.spatial.grid.row.value)
        field.spatial.grid.row.bounds = np.fliplr(np.flipud(field.spatial.grid.row.bounds))

        egrid = get_esmf_grid_from_sdim(field.spatial)

        for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
            coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
            self.assertNumpyAll(coords, field.spatial.grid.value[idx_ocgis, ...].data)

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_row = corner[1]
        corner_row_actual = np.array(
            [[40.5, 40.5, 40.5, 40.5, 40.5], [39.5, 39.5, 39.5, 39.5, 39.5], [38.5, 38.5, 38.5, 38.5, 38.5],
             [37.5, 37.5, 37.5, 37.5, 37.5], [36.5, 36.5, 36.5, 36.5, 36.5]], dtype=field.spatial.grid.value.dtype)
        self.assertNumpyAll(corner_row, corner_row_actual)

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_col = corner[0]
        corner_col_actual = np.array(
            [[-105.5, -104.5, -103.5, -102.5, -101.5], [-105.5, -104.5, -103.5, -102.5, -101.5],
             [-105.5, -104.5, -103.5, -102.5, -101.5], [-105.5, -104.5, -103.5, -102.5, -101.5],
             [-105.5, -104.5, -103.5, -102.5, -101.5]], dtype=field.spatial.grid.value.dtype)
        self.assertNumpyAll(corner_col, corner_col_actual)

    def test_get_esmf_grid_from_sdim_change_origin_col(self):
        """Test with different column grid origin."""

        rd = ocgis.RequestDataset(**self.get_dataset())
        field = rd.get()

        field.spatial.grid.col._value = np.flipud(field.spatial.grid.col.value)
        field.spatial.grid.col.bounds = np.fliplr(np.flipud(field.spatial.grid.col.bounds))

        egrid = get_esmf_grid_from_sdim(field.spatial)

        for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
            coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
            self.assertNumpyAll(coords, field.spatial.grid.value[idx_ocgis, ...].data)

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_row = corner[1]
        corner_row_actual = np.array(
            [[36.5, 36.5, 36.5, 36.5, 36.5], [37.5, 37.5, 37.5, 37.5, 37.5], [38.5, 38.5, 38.5, 38.5, 38.5],
             [39.5, 39.5, 39.5, 39.5, 39.5], [40.5, 40.5, 40.5, 40.5, 40.5]], dtype=field.spatial.grid.value.dtype)
        self.assertNumpyAll(corner_row, corner_row_actual)

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_col = corner[0]
        corner_col_actual = np.array(
            [[-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5],
             [-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5],
             [-101.5, -102.5, -103.5, -104.5, -105.5]], dtype=field.spatial.grid.value.dtype)
        self.assertNumpyAll(corner_col, corner_col_actual)

    def test_get_esmf_grid_from_sdim_change_origin_row_and_col(self):
        """Test with different row and column grid origin."""

        rd = ocgis.RequestDataset(**self.get_dataset())
        field = rd.get()

        field.spatial.grid.row._value = np.flipud(field.spatial.grid.row.value)
        field.spatial.grid.row.bounds = np.fliplr(np.flipud(field.spatial.grid.row.bounds))
        field.spatial.grid.col._value = np.flipud(field.spatial.grid.col.value)
        field.spatial.grid.col.bounds = np.fliplr(np.flipud(field.spatial.grid.col.bounds))

        egrid = get_esmf_grid_from_sdim(field.spatial)

        for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
            coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
            self.assertNumpyAll(coords, field.spatial.grid.value[idx_ocgis, ...].data)

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_row = corner[1]
        corner_row_actual = np.array([[40.5, 40.5, 40.5, 40.5, 40.5], [39.5, 39.5, 39.5, 39.5, 39.5], [38.5, 38.5, 38.5, 38.5, 38.5], [37.5, 37.5, 37.5, 37.5, 37.5], [36.5, 36.5, 36.5, 36.5, 36.5]], dtype=field.spatial.grid.value.dtype)
        self.assertNumpyAll(corner_row, corner_row_actual)

        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        corner_col = corner[0]
        corner_col_actual = np.array([[-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5], [-101.5, -102.5, -103.5, -104.5, -105.5]], dtype=field.spatial.grid.value.dtype)
        self.assertNumpyAll(corner_col, corner_col_actual)

    def test_get_esmf_grid_from_sdim_value_mask(self):
        """Test with an additional mask."""

        rd = ocgis.RequestDataset(**self.get_dataset())
        field = rd.get()
        np.random.seed(1)
        self.assertFalse(np.any(field.spatial.get_mask()))
        value_mask = np.random.randint(0, 2, field.spatial.get_mask().shape)
        egrid = get_esmf_grid_from_sdim(field.spatial, value_mask=value_mask)
        self.assertNumpyAll(egrid.mask[0], np.invert(value_mask.astype(bool)).astype(egrid.mask[0].dtype))

    def test_get_esmf_grid_from_sdim_with_corners(self):
        """Test with the with_corners option set to False."""

        rd = ocgis.RequestDataset(**self.get_dataset())
        field = rd.get()
        self.assertIsNotNone(field.spatial.grid.corners)
        egrid = get_esmf_grid_from_sdim(field.spatial, with_corners=False)
        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        for idx in [0, 1]:
            self.assertNumpyAll(corner[idx], np.array(0.0))
