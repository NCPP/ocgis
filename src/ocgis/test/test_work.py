from csv import DictReader
import os

import fiona

from ocgis import constants
from ocgis import ShpCabinet, RequestDataset, OcgOperations, env
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


class Test_20150224(TestBase):
    def test_subset_with_shapefile_no_ugid(self):
        """Test a subset operation using a shapefile without a UGID attribute."""

        output_format = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_CSV_SHAPEFILE]

        geom = self.get_shapefile_path_with_no_ugid()
        geom_select_uid = [8, 11]
        geom_uid = 'ID'
        rd = self.test_data.get_rd('cancm4_tas')

        for of in output_format:
            ops = OcgOperations(dataset=rd, geom=geom, geom_select_uid=geom_select_uid, geom_uid=geom_uid, snippet=True,
                                output_format=of)
            self.assertEqual(len(ops.geom), 2)
            ret = ops.execute()
            if of == constants.OUTPUT_FORMAT_NUMPY:
                for element in geom_select_uid:
                    self.assertIn(element, ret)
                self.assertEqual(ret.properties[8].dtype.names, ('STATE_FIPS', 'ID', 'STATE_NAME', 'STATE_ABBR'))
            else:
                with open(ret) as f:
                    reader = DictReader(f)
                    row = reader.next()
                    self.assertIn(geom_uid, row.keys())
                    self.assertNotIn(env.DEFAULT_GEOM_UID, row.keys())

                shp_path = os.path.split(ret)[0]
                shp_path = os.path.join(shp_path, 'shp', '{0}_gid.shp'.format(ops.prefix))
                with fiona.open(shp_path) as source:
                    record = source.next()
                    self.assertIn(geom_uid, record['properties'])
                    self.assertNotIn(env.DEFAULT_GEOM_UID, record['properties'])