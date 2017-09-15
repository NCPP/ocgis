from copy import deepcopy

import numpy as np

from ocgis import OcgOperations
from ocgis.test.base import TestBase, attr


class TestBlah(TestBase):
    @attr('slow')
    def test_system_many_request_datasets(self):
        """Test numerous request datasets."""

        rd_base = self.test_data.get_rd('cancm4_tas')
        geom = [-74.0, 40.0, -72.0, 42.0]
        rds = [deepcopy(rd_base) for ii in range(500)]
        for rd in rds:
            ops = OcgOperations(dataset=rd, geom=geom, snippet=True)
            ret = ops.execute()
            actual = ret.get_element(variable_name='tas').shape
            self.assertEqual(actual, (1, 2, 1))

    def test_maintaining_global_index_with_subset(self):
        original = np.arange(1, 17).reshape(4, 4)
        slc = [slice(0, 2), slice(0, 2)]
        sliced = original[slc]
        self.assertIsNotNone(sliced)
