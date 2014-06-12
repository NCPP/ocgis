from collections import OrderedDict
from ocgis.conv.numpy_ import NumpyConverter
from ocgis.test.base import TestBase
import ocgis
from ocgis.api.subset import SubsetOperation
from ocgis.api.collection import SpatialCollection
import itertools
from ocgis.test.test_ocgis.test_api.test_parms.test_definition import TestGeom
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

    def test_init(self):
        for rb, p in itertools.product([True, False], [None, ProgressOcgOperations()]):
            sub = SubsetOperation(self.get_operations(), request_base_size_only=rb, progress=p)
            for ii, coll in enumerate(sub):
                self.assertIsInstance(coll, SpatialCollection)
        self.assertEqual(ii, 0)

    def test_geometry_dictionary(self):
        """Test geometry dictionaries come out properly as collections."""

        subset = self.get_subset_operation()
        conv = NumpyConverter(subset, None, None)
        coll = conv.write()
        self.assertEqual(coll.properties, OrderedDict([(1, {'COUNTRY': 'France', 'UGID': 1}), (2, {'COUNTRY': 'Germany', 'UGID': 2}), (3, {'COUNTRY': 'Italy', 'UGID': 3})]))
