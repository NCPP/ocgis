from copy import deepcopy
import csv
import os
import pickle
import itertools
import numpy as np

import ESMF
from shapely import wkt

from ocgis.calc.library.index.duration import FrequencyDuration
from ocgis.api.parms.definition import OutputFormat
from ocgis.exc import DefinitionValidationError
from ocgis.interface.base.field import Field
from ocgis.api.operations import OcgOperations
from ocgis.conv.numpy_ import NumpyConverter
from ocgis.interface.base.crs import Spherical, CFWGS84, CFPolarStereographic, WGS84, CoordinateReferenceSystem
from ocgis.interface.base.dimension.spatial import SpatialDimension, SpatialGridDimension
from ocgis.test.base import TestBase
import ocgis
from ocgis.api.subset import SubsetOperation
from ocgis.api.collection import SpatialCollection
from ocgis.test.test_ocgis.test_api.test_parms.test_definition import TestGeom
from ocgis.util.itester import itr_products_keywords
from ocgis.util.logging_ocgis import ProgressOcgOperations
from ocgis import env, constants


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

    def test_init(self):
        for rb, p in itertools.product([True, False], [None, ProgressOcgOperations()]):
            sub = SubsetOperation(self.get_operations(), request_base_size_only=rb, progress=p)
            for ii, coll in enumerate(sub):
                self.assertIsInstance(coll, SpatialCollection)
        self.assertEqual(ii, 0)

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

    def test_abstraction_not_available(self):
        """Test appropriate exception is raised when a selected abstraction is not available."""

        rd = self.test_data.get_rd('daymet_tmax')
        ops = ocgis.OcgOperations(dataset=rd, abstraction='polygon', geom='state_boundaries', select_ugid=[25])
        with self.assertRaises(ValueError):
            ops.execute()

    def test_dataset_as_field(self):
        """Test with dataset as field not loaded from file - hence, no metadata."""

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

    def test_geometry_dictionary(self):
        """Test geometry dictionaries come out properly as collections."""

        subset = self.get_subset_operation()
        conv = NumpyConverter(subset)
        coll = conv.write()
        actual = "ccollections\nOrderedDict\np0\n((lp1\n(lp2\ncnumpy.core.multiarray\nscalar\np3\n(cnumpy\ndtype\np4\n(S'i8'\np5\nI0\nI1\ntp6\nRp7\n(I3\nS'<'\np8\nNNNI-1\nI-1\nI0\ntp9\nbS'\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np10\ntp11\nRp12\nacnumpy.core.multiarray\n_reconstruct\np13\n(cnumpy\nndarray\np14\n(I0\ntp15\nS'b'\np16\ntp17\nRp18\n(I1\n(I1\ntp19\ng4\n(S'V16'\np20\nI0\nI1\ntp21\nRp22\n(I3\nS'|'\np23\nN(S'COUNTRY'\np24\nS'UGID'\np25\ntp26\n(dp27\ng24\n(g4\n(S'O8'\np28\nI0\nI1\ntp29\nRp30\n(I3\nS'|'\np31\nNNNI-1\nI-1\nI63\ntp32\nbI0\ntp33\nsg25\n(g7\nI8\ntp34\nsI16\nI1\nI27\ntp35\nbI00\n(lp36\n(S'France'\np37\nI1\ntp38\natp39\nbaa(lp40\ng3\n(g7\nS'\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np41\ntp42\nRp43\nag13\n(g14\n(I0\ntp44\ng16\ntp45\nRp46\n(I1\n(I1\ntp47\ng22\nI00\n(lp48\n(S'Germany'\np49\nI2\ntp50\natp51\nbaa(lp52\ng3\n(g7\nS'\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np53\ntp54\nRp55\nag13\n(g14\n(I0\ntp56\ng16\ntp57\nRp58\n(I1\n(I1\ntp59\ng22\nI00\n(lp60\n(S'Italy'\np61\nI3\ntp62\natp63\nbaatp64\nRp65\n."
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

    def test_regridding_same_field(self):
        """Test regridding operations with same field used to regrid the source."""

        # todo: what happens with multivariate calculations
        #todo: test with all masked values

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

        actual = np.ma.array([[[[[0.0, 0.0, 0.0, 0.0, 289.309326171875, 288.7110290527344, 287.92108154296875,
                                  287.1899108886719, 286.51715087890625, 285.9024658203125, 0.0, 0.0, 0.0],
                                 [0.0, 288.77825927734375, 288.62823486328125, 288.3404541015625, 287.9151611328125,
                                  287.32000732421875, 286.633544921875, 286.0067138671875, 285.43914794921875,
                                  284.93060302734375, 284.48077392578125, 0.0, 0.0],
                                 [288.4192199707031, 288.18804931640625, 287.8165588378906, 287.30499267578125,
                                  286.65362548828125, 285.86676025390625, 285.28515625, 284.7640686035156,
                                  284.30316162109375, 283.90216064453125, 283.560791015625, 0.0, 0.0],
                                 [288.19488525390625, 287.74169921875, 287.14593505859375, 286.4078063964844,
                                  285.52752685546875, 284.5051574707031, 283.87457275390625, 283.4606628417969,
                                  283.1078186035156, 282.8158264160156, 282.58441162109375, 0.0, 0.0],
                                 [288.023193359375, 287.4422607421875, 286.6193542480469, 285.65179443359375,
                                  284.5396728515625, 283.2830505371094, 282.4002685546875, 282.09503173828125,
                                  281.8517761230469, 281.6702575683594, 281.55029296875, 0.0, 0.0],
                                 [287.8075866699219, 287.2928771972656, 286.2398986816406, 285.0399475097656,
                                  283.6930236816406, 282.19915771484375, 280.86077880859375, 280.66571044921875,
                                  280.5335388183594, 280.4640808105469, 280.4613952636719, 280.4708251953125, 0.0],
                                 [287.591552734375, 287.296875, 286.0108337402344, 284.5754089355469,
                                  282.99066162109375, 281.2564392089844, 279.47003173828125, 279.34307861328125,
                                  279.3382263183594, 279.3432922363281, 279.3581848144531, 279.3829040527344, 0.0],
                                 [287.3750305175781, 287.322265625, 285.8916931152344, 284.12139892578125,
                                  282.3462829589844, 280.566162109375, 278.7807922363281, 278.1846618652344,
                                  278.1950988769531, 278.2154846191406, 278.24578857421875, 278.2860107421875, 0.0],
                                 [286.864013671875, 286.48724365234375, 285.2509460449219, 283.4699401855469,
                                  281.6840515136719, 279.8930358886719, 278.0966796875, 277.01617431640625,
                                  277.0421142578125, 277.07806396484375, 277.1240234375, 277.1799621582031, 0.0],
                                 [286.0535583496094, 285.5471496582031, 284.6158752441406, 282.8240661621094,
                                  281.0272521972656, 279.2252197265625, 277.4177551269531, 275.8373107910156,
                                  275.8789367675781, 275.9306945800781, 275.9925231933594, 276.0644226074219, 0.0],
                                 [285.3349609375, 284.69732666015625, 283.9648132324219, 282.183837890625,
                                  280.3759765625, 278.56280517578125, 276.74407958984375, 274.91961669921875,
                                  274.7053527832031, 274.77313232421875, 274.85107421875, 274.9391784667969,
                                  275.0374450683594],
                                 [284.7100830078125, 283.93963623046875, 283.07275390625, 281.54925537109375,
                                  279.730224609375, 277.90576171875, 276.07568359375, 274.2397155761719,
                                  273.5210266113281, 273.6050720214844, 273.69940185546875, 273.9654235839844,
                                  274.24139404296875],
                                 [284.1809387207031, 283.27606201171875, 282.2731018066406, 280.9204406738281,
                                  279.090087890625, 277.25421142578125, 275.41259765625, 273.7033996582031,
                                  272.687744140625, 272.9641418457031, 273.2394104003906, 273.5135498046875,
                                  273.78662109375],
                                 [283.7496337890625, 282.70855712890625, 281.56787109375, 280.3042907714844,
                                  278.54541015625, 276.8524169921875, 275.22515869140625, 273.6634826660156,
                                  272.24554443359375, 272.5191955566406, 272.7915954589844, 273.0628662109375,
                                  273.3330078125],
                                 [283.39312744140625, 282.1578369140625, 280.91937255859375, 279.67755126953125,
                                  278.1316223144531, 276.5411071777344, 275.017333984375, 273.56024169921875,
                                  272.16973876953125, 272.07550048828125, 272.3450622558594, 272.6134338378906,
                                  272.8805847167969],
                                 [282.7581481933594, 281.516845703125, 280.27227783203125, 279.0242614746094,
                                  277.64892578125, 276.16229248046875, 274.743408203125, 273.3922424316406,
                                  272.1087646484375, 271.63311767578125, 271.8998107910156, 272.1651916503906,
                                  272.4293518066406],
                                 [282.1268615722656, 280.87945556640625, 279.6286926269531, 278.3744201660156,
                                  277.095703125, 275.7143249511719, 274.4017639160156, 273.157958984375,
                                  271.98297119140625, 271.1920471191406, 271.4557800292969, 271.71820068359375,
                                  271.97930908203125],
                                 [281.499267578125, 280.24566650390625, 278.9886779785156, 277.7280578613281,
                                  276.4637145996094, 275.19561767578125, 273.9908142089844, 272.85589599609375,
                                  271.7908630371094, 270.79583740234375, 271.0130615234375, 271.26873779296875,
                                  271.4607238769531],
                                 [280.8753662109375, 279.6155090332031, 278.3522033691406, 277.085205078125,
                                  275.81439208984375, 274.6044921875, 273.50897216796875, 272.4844055175781,
                                  271.58203125, 270.6971130371094, 270.5581359863281, 270.7032165527344,
                                  270.7939453125],
                                 [280.25518798828125, 278.989013671875, 277.71929931640625, 276.4458312988281,
                                  275.1974182128906, 274.173828125, 273.29473876953125, 272.4113464355469,
                                  271.5235290527344, 270.63116455078125, 270.1499938964844, 270.1932678222656,
                                  270.1813049316406],
                                 [0.0, 278.4078063964844, 277.3578186035156, 276.3003234863281, 275.2351379394531,
                                  274.162109375, 273.2556457519531, 272.36480712890625, 271.4695129394531,
                                  270.569580078125, 269.8001403808594, 269.7401428222656, 269.6240234375],
                                 [0.0, 278.4853820800781, 277.42474365234375, 276.3564758300781, 275.2804260253906,
                                  274.1964416503906, 273.2213134765625, 272.3229675292969, 271.4200744628906,
                                  270.5124816894531, 269.6001281738281, 269.3451232910156, 269.1233825683594],
                                 [0.0, 278.5711669921875, 277.49969482421875, 276.4205017089844, 275.33343505859375,
                                  274.23834228515625, 273.1918640136719, 272.2858581542969, 271.3752746582031,
                                  270.4599304199219, 269.53973388671875, 269.00958251953125, 268.6806335449219],
                                 [0.0, 0.0, 277.5827941894531, 276.4925537109375, 275.3943176269531, 274.2879638671875,
                                  273.1732482910156, 272.25360107421875, 271.335205078125, 270.4120178222656,
                                  269.48388671875, 268.73492431640625, 0.0]]]]],
                             mask=[[[[[True, True, True, True, True, True, True, True, True, False, False, False, True],
                                      [True, True, True, True, True, True, True, False, False, False, False, False,
                                       True],
                                      [True, True, True, True, True, True, True, False, False, False, False, False,
                                       True],
                                      [True, True, True, True, True, True, False, False, False, False, False, False,
                                       False],
                                      [True, True, True, True, True, False, False, False, False, False, False, False,
                                       False],
                                      [True, True, True, True, False, False, False, False, False, False, False, False,
                                       True],
                                      [True, True, False, False, False, False, False, False, False, False, False, False,
                                       True],
                                      [True, True, False, False, False, False, False, False, False, False, False, True,
                                       True],
                                      [True, True, False, False, False, False, False, False, False, False, True, True,
                                       True],
                                      [True, True, False, False, False, False, False, False, False, False, True, True,
                                       True],
                                      [True, False, False, False, False, False, False, False, False, True, True, True,
                                       True],
                                      [True, False, False, False, False, False, False, False, True, True, True, True,
                                       True],
                                      [True, False, False, False, False, False, False, False, True, True, True, True,
                                       True],
                                      [True, False, False, False, False, False, False, True, True, True, True, True,
                                       True],
                                      [True, False, False, False, False, False, False, True, True, True, True, True,
                                       True],
                                      [False, False, False, False, False, False, True, True, True, True, True, True,
                                       True],
                                      [False, False, False, False, False, False, True, True, True, True, True, True,
                                       True],
                                      [False, False, False, False, False, False, True, True, True, True, True, True,
                                       True],
                                      [False, False, False, False, False, False, False, True, True, True, True, True,
                                       True],
                                      [False, False, False, False, False, False, False, True, True, True, True, True,
                                       True],
                                      [False, False, False, False, False, False, False, True, True, True, True, True,
                                       True],
                                      [False, False, False, False, False, False, False, False, True, True, True, True,
                                       True],
                                      [True, False, False, False, False, False, True, True, True, True, True, True,
                                       True], [True, False, False, True, True, True, True, True, True, True, True, True,
                                               True]]]]],
                             dtype=np.float32,
                             fill_value=np.float32(1e20))

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
                to_test = field.variables.first().value
                self.assertEqual(to_test.shape, (1, 1, 1, 24, 13))
                self.assertNumpyAll(to_test, actual)

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

    def test_regridding_two_projected_coordinate_systems(self):
        """Test with two coordinate systems not in spherical coordinates."""

        rd1 = self.test_data.get_rd('narccap_lambert_conformal')
        rd2 = self.test_data.get_rd('narccap_polar_stereographic')

        self.assertIsInstance(rd2.crs, CFPolarStereographic)

        ops = ocgis.OcgOperations(dataset=rd2, regrid_destination=rd1, geom='state_boundaries', select_ugid=[25])
        ret = ops.execute()
        ref = ret[25]['pr']
        self.assertEqual(ref.spatial.crs, rd2.crs)
        self.assertEqual(ref.shape, (1, 14600, 1, 24, 13))
        self.assertAlmostEqual(ref.variables['pr'].value.mean(), 1.8220362330916791e-05)
