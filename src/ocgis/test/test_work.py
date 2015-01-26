from ocgis import ShpCabinet, RequestDataset, OcgOperations
from ocgis.test.base import TestBase

"""
These tests written to guide bug fixing or issue development. Theses tests are typically high-level and block-specific
testing occurs in tandem. It is expected that any issues identified by these tests have a corresponding test in the
package hierarchy. Hence, these tests in theory may be removed...
"""


class Test20150119(TestBase):
    def test_shapefile_through_operations_subset(self):
        path = ShpCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(path)
        field = rd.get()
        self.assertIsNone(field.spatial.properties)
        ops = OcgOperations(dataset=rd, output_format='shp', geom='state_boundaries', select_ugid=[15])
        ret = ops.execute()
        rd2 = RequestDataset(ret)
        field2 = rd2.get()
        self.assertAsSetEqual(field.variables.keys(), field2.variables.keys())
        self.assertEqual(tuple([1] * 5), field2.shape)

    def test_shapefile_through_operations(self):
        path = ShpCabinet().get_shp_path('state_boundaries')
        rd = RequestDataset(path)
        field = rd.get()
        self.assertIsNone(field.spatial.properties)
        ops = OcgOperations(dataset=rd, output_format='shp')
        ret = ops.execute()
        rd2 = RequestDataset(ret)
        field2 = rd2.get()
        self.assertAsSetEqual(field.variables.keys(), field2.variables.keys())
        self.assertEqual(field.shape, field2.shape)