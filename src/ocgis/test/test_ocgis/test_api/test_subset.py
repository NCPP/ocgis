from collections import OrderedDict
from copy import deepcopy
import os
import pickle
from ocgis.conv.numpy_ import NumpyConverter
from ocgis.exc import CornersUnavailable
from ocgis.interface.base.crs import Spherical, CFWGS84, CFPolarStereographic
from ocgis.interface.base.dimension.spatial import SpatialDimension
from ocgis.test.base import TestBase
import ocgis
from ocgis.api.subset import SubsetOperation
from ocgis.api.collection import SpatialCollection
import itertools
from ocgis.test.test_ocgis.test_api.test_parms.test_definition import TestGeom, TestRegridDestination
from ocgis.util.itester import itr_products_keywords
from ocgis.util.logging_ocgis import ProgressOcgOperations
from ocgis import constants
import numpy as np


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

    def test_bounding_box_wrapped(self):
        """Test subsetting with a wrapped bounding box with the target as a 0-360 global grid."""

        bbox = [-104, 36, -95, 44]
        rd_global = self.test_data.get_rd('cancm4_tas')
        uri = os.path.expanduser('~/climate_data/maurer/bcca/obs/tasmax/1_8deg/gridded_obs.tasmax.OBS_125deg.daily.1991.nc')
        rd_downscaled = ocgis.RequestDataset(uri=uri)
        ops = ocgis.OcgOperations(dataset=rd_global, regrid_destination=rd_downscaled, geom=bbox, output_format='nc',
                                  snippet=True)
        ret = ops.execute()
        rd = ocgis.RequestDataset(ret)
        field = rd.get()
        self.assertEqual(field.shape, (1, 1, 1, 64, 72))
        self.assertEqual(field.spatial.grid.value.mean(), -29.75)
        self.assertIsNotNone(field.spatial.grid.corners)
        self.assertAlmostEqual(field.variables.first().value.mean(), 262.07747395833331)

    def test_geometry_dictionary(self):
        """Test geometry dictionaries come out properly as collections."""

        subset = self.get_subset_operation()
        conv = NumpyConverter(subset, None, None)
        coll = conv.write()
        actual = "ccollections\nOrderedDict\np0\n((lp1\n(lp2\ncnumpy.core.multiarray\nscalar\np3\n(cnumpy\ndtype\np4\n(S'i8'\np5\nI0\nI1\ntp6\nRp7\n(I3\nS'<'\np8\nNNNI-1\nI-1\nI0\ntp9\nbS'\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np10\ntp11\nRp12\nacnumpy.core.multiarray\n_reconstruct\np13\n(cnumpy\nndarray\np14\n(I0\ntp15\nS'b'\np16\ntp17\nRp18\n(I1\n(I1\ntp19\ng4\n(S'V16'\np20\nI0\nI1\ntp21\nRp22\n(I3\nS'|'\np23\nN(S'COUNTRY'\np24\nS'UGID'\np25\ntp26\n(dp27\ng24\n(g4\n(S'O8'\np28\nI0\nI1\ntp29\nRp30\n(I3\nS'|'\np31\nNNNI-1\nI-1\nI63\ntp32\nbI0\ntp33\nsg25\n(g7\nI8\ntp34\nsI16\nI1\nI27\ntp35\nbI00\n(lp36\n(S'France'\np37\nI1\ntp38\natp39\nbaa(lp40\ng3\n(g7\nS'\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np41\ntp42\nRp43\nag13\n(g14\n(I0\ntp44\ng16\ntp45\nRp46\n(I1\n(I1\ntp47\ng22\nI00\n(lp48\n(S'Germany'\np49\nI2\ntp50\natp51\nbaa(lp52\ng3\n(g7\nS'\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np53\ntp54\nRp55\nag13\n(g14\n(I0\ntp56\ng16\ntp57\nRp58\n(I1\n(I1\ntp59\ng22\nI00\n(lp60\n(S'Italy'\np61\nI3\ntp62\natp63\nbaatp64\nRp65\n."
        actual = pickle.loads(actual)
        self.assertEqual(coll.properties, actual)

    def test_regridding_same_field(self):
        """Test regridding operations with same field used to regrid the source."""

        #todo: what happens with multivariate calculations
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
                        self.assertEqual(field.spatial.crs, constants.default_coordinate_system)
                    self.assertTrue(d['variable'].value.mean() > 100)
                    self.assertTrue(np.any(field.spatial.get_mask()))
                    self.assertTrue(np.any(d['variable'].value.mask))
                    for to_check in [field.spatial.grid.row.bounds, field.spatial.grid.col.bounds,
                                     field.spatial.grid.corners, field.spatial.geom.polygon.value]:
                        self.assertIsNotNone(to_check)

    def test_regridding_same_field_bad_bounds_raises(self):
        """Test a regridding error is raised with bad bounds."""

        rd1 = self.test_data.get_rd('cancm4_tas')
        ops = ocgis.OcgOperations(dataset=rd1, regrid_destination=rd1, snippet=True)
        subset = SubsetOperation(ops)
        with self.assertRaises(ValueError):
            list(subset)

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
                with self.assertRaises(CornersUnavailable):
                    field.spatial.grid.corners
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

            ops = ocgis.OcgOperations(dataset=rd1, regrid_destination=destination, geom=geom, select_ugid=select_ugid, snippet=True)
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

        actual = np.ma.array([[[[[0.0, 0.0, 0.0, 0.0, 289.3017883300781, 288.7024230957031, 287.9030456542969, 287.1700439453125, 286.50152587890625, 285.89556884765625, 0.0, 0.0, 0.0], [0.0, 288.76837158203125, 288.6125793457031, 288.323486328125, 287.90484619140625, 287.3174743652344, 286.6184387207031, 285.9873046875, 285.422119140625, 284.9210510253906, 284.4820861816406, 0.0, 0.0], [288.41796875, 288.18145751953125, 287.804443359375, 287.2906799316406, 286.6439514160156, 285.8691101074219, 285.2718505859375, 284.7440185546875, 284.2837219238281, 283.88897705078125, 283.557861328125, 0.0, 0.0], [288.1981506347656, 287.7408142089844, 287.1394348144531, 286.3978271484375, 285.5198669433594, 284.5093994140625, 283.86285400390625, 283.4398193359375, 283.08587646484375, 282.79901123046875, 282.5773010253906, 0.0, 0.0], [288.0224914550781, 287.44732666015625, 286.61846923828125, 285.6458435058594, 284.5333557128906, 283.2848815917969, 282.39093017578125, 282.0742492675781, 281.82818603515625, 281.6507568359375, 281.5400390625, 0.0, 0.0], [287.8061218261719, 287.3020935058594, 286.2425231933594, 285.0356140136719, 283.6852722167969, 282.1954345703125, 280.85565185546875, 280.6468200683594, 280.51019287109375, 280.4438171386719, 280.4490966796875, 280.4698181152344, 0.0], [287.58941650390625, 287.3061218261719, 286.0126037597656, 284.5680847167969, 282.9765319824219, 281.241943359375, 279.4717102050781, 279.3304443359375, 279.31854248046875, 279.322509765625, 279.3420715332031, 279.3770751953125, 0.0], [287.37237548828125, 287.3219299316406, 285.88140869140625, 284.1074523925781, 282.3339538574219, 280.56085205078125, 278.7880554199219, 278.173095703125, 278.17498779296875, 278.1929016113281, 278.2267150878906, 278.27618408203125, 0.0], [286.8515319824219, 286.482421875, 285.2543640136719, 283.4666442871094, 281.67938232421875, 279.8925476074219, 278.1060791015625, 277.00677490234375, 277.0225830078125, 277.0546875, 277.1029052734375, 277.1670227050781, 0.0], [286.04193115234375, 285.54132080078125, 284.6296691894531, 282.8280029296875, 281.0268249511719, 279.2261047363281, 277.4257507324219, 275.83135986328125, 275.86126708984375, 275.9077453613281, 275.9705505371094, 276.04949951171875, 0.0], [285.3255920410156, 284.691650390625, 283.970458984375, 282.1915588378906, 280.3763122558594, 278.5615234375, 276.74713134765625, 274.93310546875, 274.6908874511719, 274.75189208984375, 274.8294677734375, 274.9234619140625, 275.03363037109375], [284.70318603515625, 283.93414306640625, 283.0754699707031, 281.557373046875, 279.72784423828125, 277.8988037109375, 276.0702209472656, 274.2420349121094, 273.51129150390625, 273.5870056152344, 273.6817626953125, 273.9592590332031, 274.23809814453125], [284.17547607421875, 283.2694396972656, 282.2713928222656, 280.92547607421875, 279.08148193359375, 277.2380065917969, 275.3950500488281, 273.7021484375, 272.6849060058594, 272.9571533203125, 273.23040771484375, 273.5047607421875, 273.7804260253906], [283.7431945800781, 282.69830322265625, 281.5589904785156, 280.3041687011719, 278.539306640625, 276.8466491699219, 275.2239685058594, 273.6691589355469, 272.2420654296875, 272.5112609863281, 272.7813720703125, 273.05255126953125, 273.324951171875], [283.3773498535156, 282.1432189941406, 280.9111328125, 279.6810607910156, 278.1356201171875, 276.5419006347656, 275.0201110839844, 273.5680236816406, 272.1833801269531, 272.0677185058594, 272.33465576171875, 272.6025695800781, 272.8716735839844], [282.7506408691406, 281.5077209472656, 280.2668762207031, 279.028076171875, 277.65899658203125, 276.1656799316406, 274.7462158203125, 273.3983459472656, 272.1197814941406, 271.6265563964844, 271.8902282714844, 272.15484619140625, 272.4205627441406], [282.1257629394531, 280.8739318847656, 279.62420654296875, 278.3765563964844, 277.1088562011719, 275.7173767089844, 274.40167236328125, 273.1595458984375, 271.9886474609375, 271.187744140625, 271.4480895996094, 271.7093505859375, 271.97161865234375], [281.5027160644531, 280.2418518066406, 278.9831237792969, 277.7264709472656, 276.4718933105469, 275.1964111328125, 273.98602294921875, 272.85113525390625, 271.7894592285156, 270.7986145019531, 271.00823974609375, 271.2593688964844, 271.4490966796875], [280.8815612792969, 279.61151123046875, 278.3436279296875, 277.077880859375, 275.8142395019531, 274.6021423339844, 273.4985656738281, 272.4725036621094, 271.57916259765625, 270.69873046875, 270.55328369140625, 270.6933898925781, 270.78143310546875], [280.2622985839844, 278.98291015625, 277.7057800292969, 276.4308166503906, 275.1949768066406, 274.1751708984375, 273.2945556640625, 272.4111633300781, 271.5249938964844, 270.635986328125, 270.1463928222656, 270.1847839355469, 270.1696472167969], [0.0, 278.40582275390625, 277.3563232421875, 276.3005676269531, 275.2385559082031, 274.17022705078125, 273.26031494140625, 272.3682861328125, 271.4734802246094, 270.5758056640625, 269.7988586425781, 269.7340087890625, 269.6141357421875], [0.0, 278.4913024902344, 277.4296569824219, 276.3616943359375, 275.2873840332031, 274.2066955566406, 273.2289733886719, 272.3282165527344, 271.4246520996094, 270.5182189941406, 269.6088562011719, 269.341552734375, 269.1153259277344], [0.0, 278.5826110839844, 277.5086975097656, 276.4283447265625, 275.3415832519531, 274.2483825683594, 273.2006530761719, 272.2910461425781, 271.37860107421875, 270.4632568359375, 269.54498291015625, 269.00787353515625, 268.6737365722656], [0.0, 0.0, 277.593505859375, 276.5006408691406, 275.4012451171875, 274.29534912109375, 273.1828918457031, 272.25677490234375, 271.3353271484375, 270.4109802246094, 269.48370361328125, 268.7335205078125, 0.0]]]]],
                             mask=[[[[[True, True, True, True, True, True, True, True, True, False, False, False, True], [True, True, True, True, True, True, True, False, False, False, False, False, True], [True, True, True, True, True, True, True, False, False, False, False, False, True], [True, True, True, True, True, True, False, False, False, False, False, False, False], [True, True, True, True, True, False, False, False, False, False, False, False, False], [True, True, True, True, False, False, False, False, False, False, False, False, True], [True, True, False, False, False, False, False, False, False, False, False, False, True], [True, True, False, False, False, False, False, False, False, False, False, True, True], [True, True, False, False, False, False, False, False, False, False, True, True, True], [True, True, False, False, False, False, False, False, False, False, True, True, True], [True, False, False, False, False, False, False, False, False, True, True, True, True], [True, False, False, False, False, False, False, False, True, True, True, True, True], [True, False, False, False, False, False, False, False, True, True, True, True, True], [True, False, False, False, False, False, False, True, True, True, True, True, True], [True, False, False, False, False, False, False, True, True, True, True, True, True], [False, False, False, False, False, False, True, True, True, True, True, True, True], [False, False, False, False, False, False, True, True, True, True, True, True, True], [False, False, False, False, False, False, True, True, True, True, True, True, True], [False, False, False, False, False, False, False, True, True, True, True, True, True], [False, False, False, False, False, False, False, True, True, True, True, True, True], [False, False, False, False, False, False, False, True, True, True, True, True, True], [False, False, False, False, False, False, False, False, True, True, True, True, True], [True, False, False, False, False, False, True, True, True, True, True, True, True], [True, False, False, True, True, True, True, True, True, True, True, True, True]]]]],
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
                self.assertNumpyAll(to_test,actual)

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
