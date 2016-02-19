import itertools
from copy import deepcopy

import ESMF
import numpy as np

import ocgis
from ocgis.api.collection import SpatialCollection
from ocgis.conv.esmpy import ESMPyConverter
from ocgis.exc import RegriddingError, CornersInconsistentError, CannotFormatTimeError
from ocgis.interface.base.crs import CoordinateReferenceSystem, Spherical
from ocgis.interface.base.dimension.base import VectorDimension
from ocgis.interface.base.dimension.spatial import SpatialGridDimension, SpatialDimension
from ocgis.interface.base.dimension.temporal import TemporalDimension
from ocgis.interface.base.field import Field
from ocgis.interface.base.variable import VariableCollection, Variable
from ocgis.regrid.base import check_fields_for_regridding, iter_regridded_fields, get_esmf_grid_from_sdim, \
    iter_esmf_fields, get_sdim_from_esmf_grid, get_ocgis_field_from_esmf_field, RegridOperation
from ocgis.test.base import attr, TestBase
from ocgis.test.test_simple.make_test_data import SimpleNc
from ocgis.test.test_simple.test_simple import TestSimpleBase
from ocgis.util.helpers import make_poly, set_new_value_mask_for_field
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

    @attr('esmf')
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

    @attr('data', 'esmf', 'slow')
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

        desired = {'std': 3.1385237308556095, 'trace': -11.13192056119442, 'min': -11.858446, 'max': 9.8645229,
                   'shape': (1, 28, 1, 3, 5), 'mean': 0.047387103645169008}

        for regridded in iter_regridded_fields([downscaled], gcm):
            self.assertEqual(regridded.shape, (1, 28, 1, 3, 5))
            self.assertEqual(regridded.variables.keys(), ['tas'])
            self.assertDescriptivesAlmostEqual(desired, regridded.variables['tas'].value)
            self.assertNumpyAll(gcm.spatial.get_mask(), mask)
            for variable in regridded.variables.itervalues():
                vmask = variable.value.mask
                self.assertTrue(vmask[:, :, :, 1, 3].all())
                self.assertEqual(vmask.sum(), 28)

    @attr('data', 'esmf')
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
        variable_alias, efield, tidx = ret[0]
        self.assertNumpyAll(efield.grid.coords[0][0], dst.spatial.grid.value.data[1])
        self.assertNumpyAll(efield.grid.coords[0][1], dst.spatial.grid.value.data[0])

        ret = list(iter_regridded_fields([src], dst, with_corners=False))[0]
        actual = dst[0, 0, 0, :, :].variables.first().value
        to_test = ret.variables.first().value
        self.assertFalse(to_test.mask.any())

        self.assertNumpyAllClose(to_test.data, actual.data)
        self.assertNumpyAll(to_test.mask, actual.mask)

    @attr('esmf')
    def test_iter_regridded_fields(self):
        """Test with equivalent input and output expectations. The shapes of the grids are equal."""

        ofield = self.get_ofield()
        ofield.spatial.crs = Spherical()
        ofield2 = deepcopy(ofield)
        sources = [ofield, ofield2]
        destination_field = deepcopy(ofield)

        keywords = dict(use_sdim=[True, False], split=[False, True])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            if k.use_sdim:
                destination = destination_field.spatial
                sdim = destination
            else:
                destination = destination_field
                sdim = destination.spatial
            self.assertIsNotNone(sdim.grid.row)
            self.assertIsNotNone(sdim.grid.col)
            for ctr, regridded in enumerate(iter_regridded_fields(sources, destination, with_corners='auto',
                                                                  value_mask=None, split=k.split)):
                self.assertIsInstance(regridded, Field)
                self.assertNumpyAll(regridded.spatial.grid.value, sdim.grid.value)
                self.assertEqual(regridded.spatial.crs, sdim.crs)
                self.assertNumpyAll(regridded.spatial.grid.row.value, sdim.grid.row.value)
                self.assertNumpyAll(regridded.spatial.grid.row.bounds, sdim.grid.row.bounds)
                self.assertNumpyAll(regridded.spatial.grid.col.value, sdim.grid.col.value)
                self.assertNumpyAll(regridded.spatial.grid.col.bounds, sdim.grid.col.bounds)
                for variable in regridded.variables.itervalues():
                    self.assertGreater(variable.value.mean(), 2.0)
                    self.assertNumpyAll(variable.value, sources[ctr].variables[variable.alias].value)
                    self.assertFalse(np.may_share_memory(variable.value, sources[ctr].variables[variable.alias].value))
            self.assertEqual(ctr, 1)

    @attr('esmf')
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

        # Check that the destination grid is not modified.
        self.assertIsNotNone(destination_field.spatial.grid.row.bounds)

        # Remove corners from the destination and make sure that this is caught when with_corners is True.
        dest = deepcopy(destination_field).spatial
        dest.grid.row.remove_bounds()
        dest.grid.col.remove_bounds()
        dest.grid._corners = None
        self.assertIsNone(dest.grid.corners)
        with self.assertRaises(CornersInconsistentError):
            list(iter_regridded_fields(sources, dest, with_corners=True))
        # If this is now false, then there should be no problem as only centroids are used.
        list(iter_regridded_fields(sources, dest, with_corners=False))
        # This is also the case with 'auto'.
        list(iter_regridded_fields(sources, dest, with_corners='auto'))

    @attr('esmf')
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

    @attr('data', 'esmf')
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

    @attr('data', 'esmf')
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

        res = list(iter_regridded_fields([source], destination))
        self.assertEqual(res[0].variables['tas'].value.mask.sum(), 6)

    @attr('esmf')
    def test_iter_esmf_fields(self):
        ofield = self.get_ofield()

        # Add a unique spatial and value mask.
        mask = ofield.spatial.get_mask()
        mask[1, 1] = True
        ofield.spatial.set_mask(mask)
        for variable in ofield.variables.itervalues():
            variable.value.mask[:, :, :, :, :] = mask.copy()

        dtype = [('variable_alias', object), ('efield', object), ('tidx', object)]
        fill = np.array([], dtype=dtype)
        for row in iter_esmf_fields(ofield, with_corners=True, value_mask=None, split=False):
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
            self.assertEqual(efield.data.shape[0], ofield.shape[1])
            # Test masks are equivalent. ESMF masks have "1" for false requiring the invert operation.
            egrid = efield.grid
            egrid_mask = np.invert(egrid.mask[0].astype(bool))
            self.assertNumpyAll(egrid_mask, mask)

        # Test splitting the data.
        ofield = self.get_ofield()
        res = list(iter_esmf_fields(ofield))
        self.assertEqual(len(res), ofield.shape[1] * len(ofield.variables))

    @attr('esmf')
    def test_get_sdim_from_esmf_grid(self):
        rd = ocgis.RequestDataset(**self.get_dataset())

        keywords = dict(has_corners=[True, False],
                        has_mask=[True, False],
                        crs=[None, CoordinateReferenceSystem(epsg=4326)])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            field = rd.get()
            sdim = field.spatial

            if not k.has_corners:
                sdim.grid.row.remove_bounds()
                sdim.grid.col.remove_bounds()
                self.assertIsNone(sdim.grid.corners)

            egrid = get_esmf_grid_from_sdim(sdim)

            if not k.has_mask:
                egrid.mask[0][2, 2] = 0
                sdim.grid.value.mask[:, 2, 2] = True
                if k.has_corners:
                    sdim.grid.corners.mask[:, 2, 2] = True

            nsdim = get_sdim_from_esmf_grid(egrid, crs=k.crs)
            self.assertEqual(nsdim.crs, k.crs)

            self.assertNumpyAll(sdim.grid.value, nsdim.grid.value)
            if k.has_corners:
                self.assertNumpyAll(sdim.grid.corners, nsdim.grid.corners)
            else:
                self.assertIsNone(nsdim.grid.corners)

    @attr('data', 'esmf')
    def test_get_esmf_grid_from_sdim_with_mask(self):
        """Test with masked data."""

        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, geom='state_boundaries', select_ugid=[23], snippet=True,
                                  vector_wrap=False)
        ret = ops.execute()
        field = ret[23]['tas']
        egrid = get_esmf_grid_from_sdim(field.spatial)
        actual = np.array([[0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int32)
        self.assertNumpyAll(egrid.mask[0], actual)

        sdim = get_sdim_from_esmf_grid(egrid)
        self.assertNumpyAll(sdim.get_mask(), field.spatial.get_mask())
        actual = np.array([[[[True, True, True, True], [True, True, True, True], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]]], [[[True, True, True, True], [True, True, True, True], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]], [[False, False, False, False], [False, False, False, False], [False, False, False, False]]]])
        self.assertNumpyAll(actual, sdim.grid.corners.mask)

    @attr('esmf')
    def test_get_esmf_grid_from_sdim(self):
        rd = ocgis.RequestDataset(**self.get_dataset())

        have_ocgis_bounds = [True, False]
        for h in have_ocgis_bounds:
            field = rd.get()
            sdim = field.spatial

            if not h:
                sdim.grid.row.remove_bounds()
                sdim.grid.col.remove_bounds()
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
                # No corners should be on the ESMF grid.
                for idx in [0, 1]:
                    self.assertIsNone(corner[idx])

    @attr('data', 'esmf')
    def test_get_esmf_grid_from_sdim_real_data(self):
        """Test creating ESMF field from real data using an OCGIS spatial dimension."""

        rd = self.test_data.get_rd('cancm4_tas')
        field = rd.get()
        egrid = get_esmf_grid_from_sdim(field.spatial)

        for idx_esmf, idx_ocgis in zip([0, 1], [1, 0]):
            coords = egrid.coords[ESMF.StaggerLoc.CENTER][idx_esmf]
            self.assertNumpyAll(coords, field.spatial.grid.value[idx_ocgis, ...].data)

    @attr('esmf')
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

    @attr('esmf')
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

    @attr('esmf')
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

    @attr('esmf')
    def test_get_esmf_grid_from_sdim_value_mask(self):
        """Test with an additional mask."""

        rd = ocgis.RequestDataset(**self.get_dataset())
        field = rd.get()
        np.random.seed(1)
        self.assertFalse(np.any(field.spatial.get_mask()))
        value_mask = np.random.randint(0, 2, field.spatial.get_mask().shape)
        egrid = get_esmf_grid_from_sdim(field.spatial, value_mask=value_mask)
        self.assertNumpyAll(egrid.mask[0], np.invert(value_mask.astype(bool)).astype(egrid.mask[0].dtype))

    @attr('esmf')
    def test_get_ocgis_field_from_esmpy_field(self):
        np.random.seed(1)
        temporal = TemporalDimension(value=[3000., 4000., 5000.])
        level = VectorDimension(value=[10, 20, 30, 40])
        realization = VectorDimension(value=[100, 200])

        kwds = dict(crs=[None, CoordinateReferenceSystem(epsg=4326), Spherical()],
                    with_mask=[False, True],
                    with_corners=[False, True],
                    dimensions=[False, True],
                    drealization=[False, True],
                    dtemporal=[False, True],
                    dlevel=[False, True])

        for k in self.iter_product_keywords(kwds):
            row = VectorDimension(value=[1., 2.])
            col = VectorDimension(value=[3., 4.])
            if k.with_corners:
                row.set_extrapolated_bounds()
                col.set_extrapolated_bounds()

            value_tmin = np.random.rand(2, 3, 4, 2, 2)
            tmin = Variable(value=value_tmin, name='tmin')
            variables = VariableCollection([tmin])
            grid = SpatialGridDimension(row=row, col=col)
            sdim = SpatialDimension(grid=grid, crs=k.crs)
            field = Field(variables=variables, spatial=sdim, temporal=temporal, level=level, realization=realization)
            if k.with_mask:
                mask = np.zeros(value_tmin.shape[-2:], dtype=bool)
                mask[0, 1] = True
                set_new_value_mask_for_field(field, mask)
                sdim.set_mask(mask)
                self.assertTrue(tmin.value.mask.any())
                self.assertTrue(sdim.get_mask().any())
            else:
                self.assertFalse(tmin.value.mask.any())
                self.assertFalse(sdim.get_mask().any())
            coll = SpatialCollection()
            coll[1] = {field.name: field}
            conv = ESMPyConverter([coll])
            efield = conv.write()

            if k.dimensions:
                dimensions = {}
                if k.drealization:
                    dimensions['realization'] = realization
                if k.dtemporal:
                    dimensions['temporal'] = temporal
                if k.dlevel:
                    dimensions['level'] = level
            else:
                dimensions = None

            ofield = get_ocgis_field_from_esmf_field(efield, crs=k.crs, dimensions=dimensions)

            self.assertIsInstance(ofield, Field)
            self.assertEqual(ofield.shape, efield.data.shape)

            # Test a default CRS is applied for the spherical case.
            if k.crs is None:
                self.assertEqual(ofield.spatial.crs, Spherical())

            if k.drealization and k.dimensions:
                target = realization.value
            else:
                target = np.array([1, 2])
            self.assertNumpyAll(ofield.realization.value, target)

            if k.dtemporal and k.dimensions:
                target = temporal.value
            else:
                target = np.array([1, 1, 1])
                with self.assertRaises(CannotFormatTimeError):
                    ofield.temporal.value_datetime
                self.assertFalse(ofield.temporal.format_time)
            self.assertNumpyAll(ofield.temporal.value, target)

            if k.dlevel and k.dimensions:
                target = level.value
            else:
                target = np.array([1, 2, 3, 4])
            self.assertNumpyAll(ofield.level.value, target)

            self.assertNumpyAll(field.spatial.grid.value, ofield.spatial.grid.value)
            if k.with_corners:
                self.assertIsNotNone(ofield.spatial.grid.corners)
                self.assertNumpyAll(field.spatial.grid.corners, ofield.spatial.grid.corners)

            try:
                self.assertEqual(ofield.spatial.crs, sdim.crs)
            except AssertionError:
                # A "None" "crs" argument results in a default coordinate system applied to the output OCGIS field.
                self.assertIsNone(k.crs)

            ofield_tmin_value = ofield.variables[efield.name].value
            for arr1, arr2 in itertools.combinations([tmin.value.data, efield.data, ofield_tmin_value.data], r=2):
                self.assertNumpyAll(arr1, arr2, check_arr_type=False)

            rows = list(ofield.get_iter())
            try:
                self.assertEqual(len(rows), len(value_tmin.flatten()))
            except AssertionError:
                self.assertTrue(k.with_mask)
                self.assertEqual(len(rows), len(tmin.value.compressed()))

            self.assertTrue(np.may_share_memory(ofield_tmin_value, efield.data))
            self.assertFalse(np.may_share_memory(ofield_tmin_value, tmin.value))

    @attr('esmf')
    def test_get_ocgis_field_from_esmpy_spatial_only(self):
        """Test with spatial information only."""

        row = VectorDimension(value=[5, 6])
        col = VectorDimension(value=[7, 8])
        grid = SpatialGridDimension(row=row, col=col)
        sdim = SpatialDimension(grid=grid)
        field = Field(spatial=sdim)
        value = np.random.rand(*field.shape)
        variable = Variable(value=value, name='foo')
        field.variables.add_variable(variable)
        efield = self.get_esmf_field(field=field)
        self.assertIsInstance(efield, ESMF.Field)
        ofield = get_ocgis_field_from_esmf_field(efield)
        for attr in ['realization', 'temporal', 'level']:
            self.assertIsNone(getattr(ofield, attr))

    @attr('esmf')
    def test_get_esmf_grid_from_sdim_with_corners(self):
        """Test with the with_corners option set to False."""

        rd = ocgis.RequestDataset(**self.get_dataset())
        field = rd.get()
        self.assertIsNotNone(field.spatial.grid.corners)
        egrid = get_esmf_grid_from_sdim(field.spatial, with_corners=False)
        corner = egrid.coords[ESMF.StaggerLoc.CORNER]
        for idx in [0, 1]:
            self.assertIsNone(corner[idx])


class TestRegridOperation(TestBase):
    @attr('slow', 'esmf')
    def test_combo_global_grid_combinations(self):
        """Test regridding with different global grid configurations."""

        boolmix = [[True, True], [False, False], [True, False], [False, True]]

        keywords = dict(with_bounds=boolmix, wrapped=boolmix, resolution=[[3.0, 6.0], [6.0, 3.0], [3.0, 3.0]])
        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            # print ctr, k
            # if ctr != 10: continue
            source_grid = self.get_spherical_global_grid(with_bounds=k.with_bounds[0], wrapped=k.wrapped[0],
                                                         resolution=k.resolution[0])
            destination_grid = self.get_spherical_global_grid(with_bounds=k.with_bounds[1], wrapped=k.wrapped[1],
                                                              resolution=k.resolution[1])
            source_field = self.get_regridding_field(source_grid, 'source')
            destination_field = self.get_regridding_field(destination_grid, 'destination')
            ro = RegridOperation(source_field, destination_field)
            res = ro.execute()

            actual = res.variables['source'].value
            desired = destination_field.variables['destination'].value
            try:
                self.assertNumpyAllClose(actual, desired)
            except AssertionError:
                # Without bounds, some data will be unmapped and masked around the exterior of the grid if the source
                # resolution is higher than the destination.
                self.assertEqual(k.resolution, [6.0, 3.0])
                self.assertEqual(actual.mask.sum(), 356)
                self.assertTrue(np.all(np.isclose(actual.compressed(), 15.0)))

    def get_regridding_field(self, grid, name_variable):
        spatial = self.get_spatial(grid)
        col_shape = spatial.grid.shape[1]
        row_shape = spatial.grid.shape[0]
        value = np.ones(col_shape * row_shape, dtype=float).reshape(1, 1, 1, row_shape, col_shape) * 15.
        variable = Variable(name=name_variable, value=value)
        field = Field(variables=variable, spatial=spatial)
        return field

    def get_spatial(self, grid):
        spatial = SpatialDimension(grid=grid, crs=Spherical())
        return spatial

    def get_spherical_global_grid(self, resolution=3.0, with_bounds=True, wrapped=False):
        # Column (longitude) coordinates.
        if wrapped:
            start = -180. + (0.5 * resolution)
            stop = 180. + (0.5 * resolution)
        else:
            start = 0.5 * resolution
            stop = 360. + (0.5 * resolution)
        col = np.arange(start, stop, resolution)

        # Row (latitude) coordinates.
        start = -90. + (0.5 * resolution)
        stop = 90. + (0.5 * resolution)
        row = np.arange(start, stop, resolution)
        # The origin should be the upper left.
        row = np.flipud(row)

        col = VectorDimension(name='col', value=col)
        row = VectorDimension(name='row', value=row)

        if with_bounds:
            col.set_extrapolated_bounds()
            row.set_extrapolated_bounds()

        grid = SpatialGridDimension(row=row, col=col)

        return grid
