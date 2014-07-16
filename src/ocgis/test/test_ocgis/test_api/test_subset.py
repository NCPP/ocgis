from collections import OrderedDict
import pickle
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
        actual = "ccollections\nOrderedDict\np0\n((lp1\n(lp2\ncnumpy.core.multiarray\nscalar\np3\n(cnumpy\ndtype\np4\n(S'i8'\np5\nI0\nI1\ntp6\nRp7\n(I3\nS'<'\np8\nNNNI-1\nI-1\nI0\ntp9\nbS'\\x01\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np10\ntp11\nRp12\nacnumpy.core.multiarray\n_reconstruct\np13\n(cnumpy\nndarray\np14\n(I0\ntp15\nS'b'\np16\ntp17\nRp18\n(I1\n(I1\ntp19\ng4\n(S'V16'\np20\nI0\nI1\ntp21\nRp22\n(I3\nS'|'\np23\nN(S'COUNTRY'\np24\nS'UGID'\np25\ntp26\n(dp27\ng24\n(g4\n(S'O8'\np28\nI0\nI1\ntp29\nRp30\n(I3\nS'|'\np31\nNNNI-1\nI-1\nI63\ntp32\nbI0\ntp33\nsg25\n(g7\nI8\ntp34\nsI16\nI1\nI27\ntp35\nbI00\n(lp36\n(S'France'\np37\nI1\ntp38\natp39\nbaa(lp40\ng3\n(g7\nS'\\x02\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np41\ntp42\nRp43\nag13\n(g14\n(I0\ntp44\ng16\ntp45\nRp46\n(I1\n(I1\ntp47\ng22\nI00\n(lp48\n(S'Germany'\np49\nI2\ntp50\natp51\nbaa(lp52\ng3\n(g7\nS'\\x03\\x00\\x00\\x00\\x00\\x00\\x00\\x00'\np53\ntp54\nRp55\nag13\n(g14\n(I0\ntp56\ng16\ntp57\nRp58\n(I1\n(I1\ntp59\ng22\nI00\n(lp60\n(S'Italy'\np61\nI3\ntp62\natp63\nbaatp64\nRp65\n."
        actual = pickle.loads(actual)
        self.assertEqual(coll.properties, actual)
