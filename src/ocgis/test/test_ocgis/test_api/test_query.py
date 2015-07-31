from ocgis import CoordinateReferenceSystem, OcgOperations
from ocgis import GeomCabinetIterator
from ocgis.api.parms.definition import Dataset
from ocgis.api.query import QueryInterface
from ocgis.calc.library.statistics import FrequencyPercentile
from ocgis.test.base import TestBase


class TestQueryInterface(TestBase):

    def test_init(self):
        qs = "foo=a&bar=45"
        qi = QueryInterface(qs)
        self.assertDictEqual(qi.query_dict, {'foo': ['a'], 'bar': ['45']})
        self.assertEqual(qi.query_string, qs)

    def test_get_operations(self):
        rd = self.test_data.get_rd('cancm4_tas')
        qs = "uri={0}&spatial_operation=intersects&geom=state_boundaries&geom_select_uid=20|30&output_format=shp&snippet=true".format(rd.uri)
        qi = QueryInterface(qs)
        ops = qi.get_operations()

        self.assertIsInstance(ops._get_object_('dataset'), Dataset)
        self.assertEqual(ops.dataset.first().uri, rd.uri)
        self.assertIsInstance(ops.geom, GeomCabinetIterator)
        self.assertEqual(ops.spatial_operation, 'intersects')
        self.assertEqual(ops.geom_select_uid, (20, 30))

        qs = "uri={0}&spatial_operation=intersects&geom=state_boundaries&geom_select_uid=20|30".format(rd.uri)
        qs += "&calc=mean~the_mean|median~the_median|freq_perc~the_p!percentile~90&calc_grouping=month|year&alias=calcs"
        qs += "&output_crs=4326&conform_units_to=celsius"
        qs += "&time_region=year~2001"
        qi = QueryInterface(qs)
        ops = qi.get_operations()
        ret = ops.execute()
        self.assertEqual(ret.keys(), [20, 30])
        self.assertEqual(ret[20]["calcs"].variables.keys(), ["the_mean", "the_median", "the_p"])
        self.assertEqual(ret[30]["calcs"].variables["the_mean"].shape, (1, 12, 1, 2, 2))
        self.assertEqual(ops.calc[2]["ref"], FrequencyPercentile)
        self.assertEqual(ops.output_crs, CoordinateReferenceSystem(epsg=4326))

    def test_get_operations_dataset(self):
        """Test creating a dataset parameters from a query string."""

        uri1 = self.test_data.get_uri('cancm4_tas')
        uri2 = self.test_data.get_uri('cancm4_rhsmax')
        qs = 'uri={0}|{1}'.format(uri1, uri2)
        qi = QueryInterface(qs)
        ops = qi.get_operations()
        self.assertIsInstance(ops, OcgOperations)
