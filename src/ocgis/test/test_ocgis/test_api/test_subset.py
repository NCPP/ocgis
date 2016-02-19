import csv
import itertools
import os
import pickle
from copy import deepcopy

import numpy as np
from shapely import wkt

import ocgis
from ocgis import env, constants
from ocgis.api.collection import SpatialCollection
from ocgis.api.operations import OcgOperations
from ocgis.api.parms.definition import OutputFormat
from ocgis.api.subset import SubsetOperation
from ocgis.calc.library.index.duration import FrequencyDuration
from ocgis.conv.numpy_ import NumpyConverter
from ocgis.exc import DefinitionValidationError
from ocgis.interface.base.crs import Spherical, CFWGS84, CFPolarStereographic, WGS84, CoordinateReferenceSystem
from ocgis.interface.base.dimension.spatial import SpatialDimension, SpatialGridDimension
from ocgis.interface.base.field import Field
from ocgis.test.base import TestBase, attr
from ocgis.test.test_ocgis.test_api.test_parms.test_definition import TestGeom
from ocgis.util.itester import itr_products_keywords
from ocgis.util.logging_ocgis import ProgressOcgOperations


class TestSubsetOperation(TestBase):
    def get_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        slc = [None, [0, 100], None, [0, 10], [0, 10]]
        ops = ocgis.OcgOperations(dataset=rd, slice=slc)
        return ops

    def get_subset_operation(self):
        geom = TestGeom.get_geometry_dictionaries()
        rd = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd, geom=geom, select_nearest=True)
        subset = SubsetOperation(ops)
        return subset

    @attr('data')
    def test_init(self):
        for rb, p in itertools.product([True, False], [None, ProgressOcgOperations()]):
            sub = SubsetOperation(self.get_operations(), request_base_size_only=rb, progress=p)
            for ii, coll in enumerate(sub):
                self.assertIsInstance(coll, SpatialCollection)
        self.assertEqual(ii, 0)

    @attr('data')
    def test_process_subsettables(self):
        # test extrapolating spatial bounds with no row and column
        for with_corners in [False, True]:
            field = self.get_field()
            field.spatial.grid.value
            if with_corners:
                field.spatial.grid.set_extrapolated_corners()
            field.spatial.grid.row = None
            field.spatial.grid.col = None
            if with_corners:
                self.assertIsNotNone(field.spatial.grid.corners)
            else:
                self.assertIsNone(field.spatial.grid.corners)
            ops = OcgOperations(dataset=field, interpolate_spatial_bounds=True)
            so = SubsetOperation(ops)
            rds = ops.dataset.values()
            res = list(so._process_subsettables_(rds))
            self.assertEqual(len(res), 1)
            coll = res[0]
            self.assertIsNotNone(coll[1][field.name].spatial.grid.corners)

        # test header assignment
        rd = self.test_data.get_rd('cancm4_tas')
        for melted in [False, True]:
            ops = OcgOperations(dataset=rd, slice=[0, 0, 0, 0, 0], melted=melted)
            rds = ops.dataset.values()
            so = SubsetOperation(ops)
            ret = so._process_subsettables_(rds)
            for coll in ret:
                if melted:
                    self.assertEqual(coll.headers, constants.HEADERS_RAW)
                else:
                    self.assertIsNone(coll.headers)

        # test with value keys
        calc = [{'func': 'freq_duration', 'name': 'freq_duration', 'kwds': {'operation': 'gt', 'threshold': 280}}]
        rd = self.test_data.get_rd('cancm4_tas')
        slc = [0, [0, 600], 0, [10, 20], [10, 20]]
        ops = OcgOperations(dataset=rd, slice=slc, calc=calc, calc_grouping=['month', 'year'])
        rds = ops.dataset.values()
        so = SubsetOperation(ops)
        ret = so._process_subsettables_(rds)
        for coll in ret:
            self.assertIsNone(coll.headers)
        ops = OcgOperations(dataset=rd, slice=slc, calc=calc, calc_grouping=['month', 'year'],
                            output_format=constants.OUTPUT_FORMAT_CSV, melted=True)
        rds = ops.dataset.values()
        so = SubsetOperation(ops)
        ret = so._process_subsettables_(rds)
        for coll in ret:
            self.assertTrue(len(coll.value_keys) == 2)
            for key in FrequencyDuration.structure_dtype['names']:
                self.assertIn(key, coll.headers)

    @attr('data')
    def test_abstraction_not_available(self):
        """Test appropriate exception is raised when a selected abstraction is not available."""

        rd = self.test_data.get_rd('daymet_tmax')
        ops = ocgis.OcgOperations(dataset=rd, abstraction='polygon', geom='state_boundaries', select_ugid=[25])
        with self.assertRaises(ValueError):
            ops.execute()

    @attr('esmf')
    def test_dataset_as_field(self):
        """Test with dataset as field not loaded from file - hence, no metadata."""

        import ESMF

        kwds = dict(output_format=list(OutputFormat.iter_possible()),
                    crs=[None, WGS84()])

        for ii, k in enumerate(self.iter_product_keywords(kwds)):
            field = self.get_field(crs=k.crs)

            ops = OcgOperations(dataset=field)
            ret = ops.execute()
            self.assertNumpyAll(ret.gvu(1, 'foo'), field.variables['foo'].value)

            try:
                ops = OcgOperations(dataset=field, output_format=k.output_format, prefix=str(ii))
            except DefinitionValidationError:
                self.assertEqual(k.output_format, constants.OUTPUT_FORMAT_METADATA_JSON)
                continue
            try:
                ret = ops.execute()
            except ValueError:
                if k.output_format == constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH:
                    self.assertIsNone(field.spatial.geom.polygon)
                    continue
                self.assertIsNone(k.crs)
                self.assertIn(k.output_format, [constants.OUTPUT_FORMAT_CSV, constants.OUTPUT_FORMAT_CSV_SHAPEFILE,
                                                constants.OUTPUT_FORMAT_GEOJSON, constants.OUTPUT_FORMAT_SHAPEFILE])
                continue

            if k.output_format == constants.OUTPUT_FORMAT_NUMPY:
                self.assertIsInstance(ret[1]['foo'], Field)
                continue
            if k.output_format == constants.OUTPUT_FORMAT_METADATA_OCGIS:
                self.assertIsInstance(ret, basestring)
                self.assertTrue(len(ret) > 50)
                continue
            if k.output_format == constants.OUTPUT_FORMAT_ESMPY_GRID:
                self.assertIsInstance(ret, ESMF.Field)
                continue

            folder = os.path.split(ret)[0]

            path_did = os.path.join(folder, '{0}_did.csv'.format(ops.prefix))
            with open(path_did, 'r') as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows, [
                {'ALIAS': 'foo', 'DID': '1', 'URI': '', 'UNITS': '', 'STANDARD_NAME': '', 'VARIABLE': 'foo',
                 'LONG_NAME': ''}])

            path_source_metadata = os.path.join(folder, '{0}_source_metadata.txt'.format(ops.prefix))
            with open(path_source_metadata, 'r') as f:
                rows = f.readlines()
            self.assertEqual(rows, [])

            if k.output_format == 'nc':
                with self.nc_scope(ret) as ds:
                    variables_expected = [u'time', u'row', u'col', u'foo']
                    try:
                        self.assertAsSetEqual(ds.variables.keys(), variables_expected)
                    except AssertionError:
                        self.assertIsNotNone(k.crs)
                        variables_expected.append('latitude_longitude')
                        self.assertAsSetEqual(ds.variables.keys(), variables_expected)
                    self.assertNumpyAll(ds.variables['time'][:], field.temporal.value_numtime)
                    self.assertNumpyAll(ds.variables['row'][:], field.spatial.grid.row.value)
                    self.assertNumpyAll(ds.variables['col'][:], field.spatial.grid.col.value)
                    self.assertNumpyAll(ds.variables['foo'][:], field.variables['foo'].value.data.squeeze())

            contents = os.listdir(folder)

            expected_contents = [xx.format(ops.prefix) for xx in
                                 '{0}_source_metadata.txt', '{0}_did.csv', '{0}.log', '{0}_metadata.txt']
            if k.output_format == 'nc':
                expected_contents.append('{0}.nc'.format(ops.prefix))
                self.assertAsSetEqual(contents, expected_contents)
            elif k.output_format == constants.OUTPUT_FORMAT_CSV_SHAPEFILE:
                expected_contents.append('{0}.csv'.format(ops.prefix))
                expected_contents.append('shp')
                self.assertAsSetEqual(contents, expected_contents)
            elif k.output_format == constants.OUTPUT_FORMAT_SHAPEFILE:
                expected_contents = ['{0}.shp', '{0}.dbf', '{0}.shx', '{0}.cpg', '{0}.log', '{0}_metadata.txt',
                                     '{0}_source_metadata.txt', '{0}_did.csv', '{0}.prj']
                expected_contents = [xx.format(ops.prefix) for xx in expected_contents]
                self.assertAsSetEqual(contents, expected_contents)

    @attr('data')
    def test_dataset_as_field_from_file(self):
        """Test with dataset argument coming in as a field as opposed to a request dataset collection."""

        rd = self.test_data.get_rd('cancm4_tas')
        geom = 'state_boundaries'
        select_ugid = [23]
        field = rd.get()
        ops = OcgOperations(dataset=field, snippet=True, geom=geom, select_ugid=select_ugid)
        ret = ops.execute()
        field_out_from_field = ret[23]['tas']
        self.assertEqual(field_out_from_field.shape, (1, 1, 1, 4, 3))
        ops = OcgOperations(dataset=rd, snippet=True, geom=geom, select_ugid=select_ugid)
        ret = ops.execute()
        field_out_from_rd = ret[23]['tas']
        self.assertNumpyAll(field_out_from_field.variables['tas'].value, field_out_from_rd.variables['tas'].value)

    @attr('data')
    def test_geometry_dictionary(self):
        """Test geometry dictionaries come out properly as collections."""

        subset = self.get_subset_operation()
        conv = NumpyConverter(subset)
        coll = conv.write()
        actual = "ccollections\nOrderedDict\np0\n((lp1\n(lp2\ncnumpy.core.multiarray\nscalar\np3\n(cnumpy\ndtype\np4\n(S'i8'\np5\nI0\nI1\ntp6\nRp7\n(I3\nS'<'\np8\nNNNI-1\nI-1\nI0\ntp9\nbS'\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np10\ntp11\nRp12\nacnumpy.ma.core\n_mareconstruct\np13\n(cnumpy.ma.core\nMaskedArray\np14\ncnumpy\nndarray\np15\n(I0\ntp16\nS'b'\np17\ntp18\nRp19\n(I1\n(I1\ntp20\ng4\n(S'V88'\np21\nI0\nI1\ntp22\nRp23\n(I3\nS'|'\np24\nN(S'COUNTRY'\np25\nS'UGID'\np26\ntp27\n(dp28\ng25\n(g4\n(S'S80'\np29\nI0\nI1\ntp30\nRp31\n(I3\nS'|'\np32\nNNNI80\nI1\nI0\ntp33\nbI0\ntp34\nsg26\n(g7\nI80\ntp35\nsI88\nI1\nI16\ntp36\nbI00\nS'France\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np37\nS'\\x00\\x00'\np38\ncnumpy.core.multiarray\n_reconstruct\np39\n(g15\n(I0\ntp40\ng17\ntp41\nRp42\n(I1\n(tg23\nI00\nS'N/A\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00'\np43\ntp44\nbtp45\nbaa(lp46\ng3\n(g7\nS'\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np47\ntp48\nRp49\nag13\n(g14\ng15\ng16\ng17\ntp50\nRp51\n(I1\n(I1\ntp52\ng23\nI00\nS'Germany\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np53\nS'\\x00\\x00'\np54\ng39\n(g15\n(I0\ntp55\ng17\ntp56\nRp57\n(I1\n(tg23\nI00\nS'N/A\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00'\np58\ntp59\nbtp60\nbaa(lp61\ng3\n(g7\nS'\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np62\ntp63\nRp64\nag13\n(g14\ng15\ng16\ng17\ntp65\nRp66\n(I1\n(I1\ntp67\ng23\nI00\nS'Italy\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np68\nS'\\x00\\x00'\np69\ng39\n(g15\n(I0\ntp70\ng17\ntp71\nRp72\n(I1\n(tg23\nI00\nS'N/A\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00?B\\x0f\\x00\\x00\\x00\\x00\\x00'\np73\ntp74\nbtp75\nbaatp76\nRp77\n."
        actual = pickle.loads(actual)
        self.assertEqual(coll.properties, actual)

    def test_process_geometries(self):
        # test multiple geometries with coordinate system update works as expected

        a = 'POLYGON((-105.21347987288135073 40.21514830508475313,-104.39928495762711691 40.21514830508475313,-104.3192002118643984 39.5677966101694949,-102.37047139830508513 39.61451271186440692,-102.12354343220337682 37.51896186440677639,-105.16009004237288593 37.51896186440677639,-105.21347987288135073 40.21514830508475313))'
        b = 'POLYGON((-104.15235699152542281 39.02722457627118757,-103.71189088983049942 39.44099576271186436,-102.71750529661017026 39.28082627118644155,-102.35712394067796538 37.63908898305084705,-104.13900953389830306 37.63241525423728717,-104.15235699152542281 39.02722457627118757))'
        geom = [{'geom': wkt.loads(xx), 'properties': {'UGID': ugid}} for ugid, xx in enumerate([a, b])]

        grid_value = [
            [[37.0, 37.0, 37.0, 37.0], [38.0, 38.0, 38.0, 38.0], [39.0, 39.0, 39.0, 39.0], [40.0, 40.0, 40.0, 40.0]],
            [[-105.0, -104.0, -103.0, -102.0], [-105.0, -104.0, -103.0, -102.0], [-105.0, -104.0, -103.0, -102.0],
             [-105.0, -104.0, -103.0, -102.0]]]
        grid_value = np.ma.array(grid_value, mask=False)
        output_crs = CoordinateReferenceSystem(
            value={'a': 6370997, 'lon_0': -100, 'y_0': 0, 'no_defs': True, 'proj': 'laea', 'x_0': 0, 'units': 'm',
                   'b': 6370997, 'lat_0': 45})
        grid = SpatialGridDimension(value=grid_value)
        sdim = SpatialDimension(grid=grid, crs=WGS84())
        field = Field(spatial=sdim)

        ops = OcgOperations(dataset=field, geom=geom, output_crs=output_crs)
        ret = ops.execute()

        expected = {0: -502052.79407259845,
                    1: -510391.37909706926}
        for ugid, field_dict in ret.iteritems():
            for field in field_dict.itervalues():
                self.assertAlmostEqual(field.spatial.grid.value.data.mean(), expected[ugid])

    @attr('data', 'esmf')
    def test_regridding_bounding_box_wrapped(self):
        """Test subsetting with a wrapped bounding box with the target as a 0-360 global grid."""

        bbox = [-104, 36, -95, 44]
        rd_global = self.test_data.get_rd('cancm4_tas')
        rd_downscaled = self.test_data.get_rd('maurer_bcca_1991')

        ops = ocgis.OcgOperations(dataset=rd_global, regrid_destination=rd_downscaled, geom=bbox, output_format='nc',
                                  snippet=True)
        ret = ops.execute()
        rd = ocgis.RequestDataset(ret)
        field = rd.get()
        self.assertEqual(field.shape, (1, 1, 1, 64, 72))
        self.assertEqual(field.spatial.grid.value.mean(), -29.75)
        self.assertIsNotNone(field.spatial.grid.corners)
        self.assertAlmostEqual(field.variables.first().value.mean(), 262.08718532986109)

    @attr('data', 'esmf')
    def test_regridding_same_field(self):
        """Test regridding operations with same field used to regrid the source."""

        rd_dest = self.test_data.get_rd('cancm4_tas')

        keywords = dict(regrid_destination=[rd_dest, rd_dest.get().spatial, rd_dest.get()],
                        geom=['state_boundaries'])

        select_ugid = [25, 41]

        for ctr, k in enumerate(itr_products_keywords(keywords, as_namedtuple=True)):

            rd1 = self.test_data.get_rd('cancm4_tas')
            rd2 = self.test_data.get_rd('cancm4_tas', kwds={'alias': 'tas2'})

            # print ctr
            # if ctr != 1: continue

            # if the target is a spatial dimension, change the crs to spherical. otherwise, the program will attempt to
            # convert from wgs84 to spherical
            if isinstance(k.regrid_destination, SpatialDimension):
                # the spatial dimension's crs needs to be updated specifically otherwise it will assume the data is
                # wgs84 and attempt updating. the request datasets must also be assigned spherical to ensure the program
                # so the subsetting comes out okay.
                k = deepcopy(k)
                k.regrid_destination.crs = Spherical()
                rd1 = self.test_data.get_rd('cancm4_tas', kwds={'crs': Spherical()})
                rd2 = self.test_data.get_rd('cancm4_tas', kwds={'alias': 'tas2', 'crs': Spherical()})

            ops = ocgis.OcgOperations(dataset=[rd1, rd2], geom=k.geom, regrid_destination=k.regrid_destination,
                                      time_region={'month': [1], 'year': [2002]}, select_ugid=select_ugid)
            subset = SubsetOperation(ops)
            colls = list(subset)
            self.assertEqual(len(colls), 4)
            for coll in colls:
                for d in coll.get_iter_melted():
                    field = d['field']
                    if isinstance(k.regrid_destination, SpatialDimension):
                        self.assertEqual(field.spatial.crs, Spherical())
                    else:
                        self.assertEqual(field.spatial.crs, env.DEFAULT_COORDSYS)
                    self.assertTrue(d['variable'].value.mean() > 100)
                    self.assertTrue(np.any(field.spatial.get_mask()))
                    self.assertTrue(np.any(d['variable'].value.mask))
                    for to_check in [field.spatial.grid.row.bounds, field.spatial.grid.col.bounds,
                                     field.spatial.grid.corners, field.spatial.geom.polygon.value]:
                        self.assertIsNotNone(to_check)

    @attr('data', 'esmf')
    def test_regridding_same_field_bad_bounds_without_corners(self):
        """Test bad bounds may be regridded with_corners as False."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd1, regrid_destination=rd1, snippet=True,
                                  regrid_options={'with_corners': False})
        subset = SubsetOperation(ops)
        ret = list(subset)
        for coll in ret:
            for dd in coll.get_iter_melted():
                field = dd['field']
                self.assertIsNone(field.spatial.grid.corners)
                for to_test in [field.spatial.grid.row.bounds, field.spatial.grid.col.bounds]:
                    self.assertIsNone(to_test)

    @attr('esmf')
    @attr('data')
    def test_regridding_same_field_value_mask(self):
        """Test with a value_mask."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('cancm4_tas', kwds={'alias': 'tas2'})
        value_mask = np.zeros(rd2.get().spatial.shape, dtype=bool)
        value_mask[30, 45] = True
        regrid_options = {'value_mask': value_mask, 'with_corners': False}
        ops = ocgis.OcgOperations(dataset=rd1, regrid_destination=rd2, snippet=True, regrid_options=regrid_options)
        ret = list(SubsetOperation(ops))
        self.assertEqual(1, ret[0][1]['tas'].variables.first().value.mask.sum())

    @attr('data', 'esmf')
    def test_regridding_different_fields_requiring_wrapping(self):
        """Test with fields requiring wrapping."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('maurer_2010_tas')

        geom = 'state_boundaries'
        select_ugid = [25]

        ops = ocgis.OcgOperations(dataset=rd2, regrid_destination=rd1, geom=geom, select_ugid=select_ugid,
                                  time_region={'month': [2], 'year': [1990]})
        subset = SubsetOperation(ops)
        colls = list(subset)
        self.assertEqual(len(colls), 1)
        for coll in colls:
            for dd in coll.get_iter_melted():
                self.assertEqual(dd['field'].shape, (1, 28, 1, 5, 4))

    @attr('data', 'esmf')
    def test_regridding_different_fields_variable_regrid_targets(self):
        """Test with a request dataset having regrid_source as False."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('maurer_2010_tas', kwds={'time_region': {'year': [1990], 'month': [2]}})
        rd2.regrid_source = False
        rd3 = deepcopy(rd2)
        rd3.regrid_source = True
        rd3.alias = 'maurer2'

        geom = 'state_boundaries'
        select_ugid = [25]

        ops = ocgis.OcgOperations(dataset=[rd2, rd3], regrid_destination=rd1, geom=geom, select_ugid=select_ugid)
        subset = SubsetOperation(ops)
        colls = list(subset)
        self.assertEqual(len(colls), 2)
        for coll in colls:
            for dd in coll.get_iter_melted():
                field = dd['field']
                if field.name == 'tas':
                    self.assertEqual(field.shape, (1, 28, 1, 77, 83))
                elif field.name == 'maurer2':
                    self.assertEqual(field.shape, (1, 28, 1, 5, 4))
                else:
                    raise NotImplementedError

    @attr('data', 'esmf')
    def test_regridding_update_crs(self):
        """Test with different CRS values than spherical on input data."""

        # test regridding lambert conformal to 0 to 360 grid

        geom = 'state_boundaries'
        select_ugid = [25]

        keywords = dict(assign_source_crs=[False, True],
                        assign_destination_crs=[False, True],
                        destination_type=['rd', 'field'])

        for k in itr_products_keywords(keywords, as_namedtuple=True):
            if k.assign_source_crs:
                # assign the coordinate system changes some regridding behavior. in this case the value read from file
                # is the same as the assigned value.
                rd1 = self.test_data.get_rd('narccap_lambert_conformal', kwds={'crs': rd1.crs})
            else:
                rd1 = self.test_data.get_rd('narccap_lambert_conformal')

            if k.assign_destination_crs:
                # assign the coordinate system changes some regridding behavior. in this case the value read from file
                # is the same as the assigned value.
                rd2 = self.test_data.get_rd('cancm4_tas', kwds={'crs': CFWGS84()})
            else:
                rd2 = self.test_data.get_rd('cancm4_tas')

            if k.destination_type == 'rd':
                destination = rd2
            elif k.destination_type == 'field':
                destination = rd2.get()
            elif k.destination_type == 'sdim':
                destination = rd2.get().spatial
            else:
                raise NotImplementedError

            ops = ocgis.OcgOperations(dataset=rd1, regrid_destination=destination, geom=geom, select_ugid=select_ugid,
                                      snippet=True)
            subset = SubsetOperation(ops)
            colls = list(subset)

            self.assertEqual(len(colls), 1)
            for coll in colls:
                for dd in coll.get_iter_melted():
                    field = dd['field']
                    self.assertEqual(field.spatial.crs, rd1.crs)

        # swap source and destination grids

        # test regridding lambert conformal to 0 to 360 grid
        rd1 = self.test_data.get_rd('cancm4_tas')
        rd2 = self.test_data.get_rd('narccap_lambert_conformal')

        desired = {'std': 3.0068276279486397, 'max': 286.00671, 'min': 274.16211, 'trace': 31600.361114501953,
                   'mean': 279.64921529314159}

        geom = 'state_boundaries'
        select_ugid = [25]

        ops = ocgis.OcgOperations(dataset=rd1, regrid_destination=rd2, geom=geom, select_ugid=select_ugid, snippet=True)
        subset = SubsetOperation(ops)
        colls = list(subset)
        self.assertEqual(len(colls), 1)
        for coll in colls:
            for dd in coll.get_iter_melted():
                field = dd['field']
                self.assertEqual(field.spatial.crs, rd1.crs)
                actual = field.variables.first().value
                self.assertEqual(actual.shape, (1, 1, 1, 24, 13))
                self.assertDescriptivesAlmostEqual(desired, actual)

    @attr('data', 'esmf')
    def test_regridding_with_output_crs(self):
        """Test with an output coordinate system."""

        rd1 = self.test_data.get_rd('narccap_lambert_conformal')
        rd2 = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd2, regrid_destination=rd1, output_crs=rd1.get().spatial.crs,
                                  geom='state_boundaries', select_ugid=[16, 25])
        ret = ops.execute()

        self.assertEqual(ret.keys(), [16, 25])
        self.assertEqual(len(list(ret.get_iter_melted())), 2)
        for dd in ret.get_iter_melted():
            field = dd['field']
            self.assertEqual(field.spatial.crs, rd1.get().spatial.crs)
            self.assertEqual(field.shape[1], 3650)

    @attr('data', 'esmf', 'slow')
    def test_combo_regridding_two_projected_coordinate_systems(self):
        """Test with two coordinate systems not in spherical coordinates."""
        desired = {'std': 7.5356346308507805e-05, 'max': 0.0021020665, 'min': 0.0, 'trace': 0.00026310274866682025,
                   'mean': 1.8478344156317514e-05, 'shape': (1, 14600, 1, 24, 13)}

        rd1 = self.test_data.get_rd('narccap_lambert_conformal')
        rd2 = self.test_data.get_rd('narccap_polar_stereographic')

        self.assertIsInstance(rd2.crs, CFPolarStereographic)

        ops = ocgis.OcgOperations(dataset=rd2, regrid_destination=rd1, geom='state_boundaries', select_ugid=[25],
                                  regrid_options={'split': False})
        ret = ops.execute()
        ref = ret[25]['pr']
        path = self.get_temporary_file_path('foo.shp')
        ref[0, 0, 0, :, :].write_fiona(path)
        self.assertEqual(ref.spatial.crs, rd2.crs)
        self.assertEqual(ref.shape, (1, 14600, 1, 24, 13))
        # self.assertAlmostEqual(ref.variables['pr'].value.mean(), 1.8220362330916791e-05)
        self.assertDescriptivesAlmostEqual(desired, ref.variables['pr'].value)
